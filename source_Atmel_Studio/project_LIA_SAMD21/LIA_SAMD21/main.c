/*
Arduino lock-in amplifier (LIA)
Phase-sensitive detector

Dennis van Gils

TODO: implement https://github.com/mpflaga/Arduino-MemoryFree
*/

#include <atmel_start.h>
#include <math.h>
#include <hpl_dma.h>
#include <stdio.h>
#include <stdlib.h>
#include <hpl_tcc_config.h>
#include "DvG_serial_command_listener.h"

#define FIRMWARE_VERSION "ALIA v0.2.0"
#define VARIANT_MCK (48000000ul)     // Master clock frequency

volatile bool is_running = false;    // Is the lock-in amplifier running?
volatile bool is_serial_txc = false; // Is serial data sent out?
volatile bool is_serial_rxc = false; // Is serial data received?
volatile uint32_t millis = 0;        // Updated by SysTick, once every 1 ms

// Preprocessor trick to ensure enums and strings are in sync, so one can write
// 'WAVEFORM_STRING[Cosine]' to give the string 'Cosine'
#define FOREACH_WAVEFORM(WAVEFORM) \
        WAVEFORM(Cosine)    \
        WAVEFORM(Square)    \
        WAVEFORM(Triangle)  \
        WAVEFORM(END_WAVEFORM_ENUM)
#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

enum WAVEFORM_ENUM {
    FOREACH_WAVEFORM(GENERATE_ENUM)
};

static const char *WAVEFORM_STRING[] = {
    FOREACH_WAVEFORM(GENERATE_STRING)
};

// LIA output reference signal parameters
enum WAVEFORM_ENUM ref_waveform = Cosine;
double ref_freq;    // [Hz] Obtained frequency of reference signal
double ref_offs;    // [V]  Obtained voltage offset of reference signal
double ref_ampl;    // [V]  Voltage amplitude reference signal

// Look-up table (LUT) for fast DAC
#define MIN_N_LUT 20       // Min. allowed number of samples for one full period
#define MAX_N_LUT 1000     // Max. allowed number of samples for one full period
uint16_t LUT_wave[MAX_N_LUT] = {0}; // Look-up table allocation
uint16_t N_LUT;            // Current number of samples for one full period
bool is_LUT_dirty = false; // Does the LUT have to be updated with new settings?

// Analog port
#define A_REF 3.300        // [V] Analog voltage reference Arduino
#define ADC_INPUT_BITS 12
#if defined(_SAMD21_)
  #define DAC_OUTPUT_BITS 10
#elif defined(_SAMD51_)
  #define DAC_OUTPUT_BITS 12
#endif
#define MAX_DAC_OUTPUT_BITVAL ((uint16_t) (pow(2, DAC_OUTPUT_BITS) - 1))

// The number of samples to acquire by the ADC and to subsequently send out
// over serial as a single block of data
//#define BLOCK_SIZE 2500     // 2500 [# samples], where 1 sample takes up 16 bits
//#define BLOCK_SIZE 500     // 2500 [# samples], where 1 sample takes up 16 bits
#define BLOCK_SIZE 2000      // 2500 [# samples], where 1 sample takes up 16 bits

/*------------------------------------------------------------------------------
    TIMER_0
------------------------------------------------------------------------------*/

/* Channel CC[0] of timer TCC0 is set to match mode and will trigger output
events periodically at a fixed sample rate (say 50 kHz). It is the source that
triggers and synchronizes the DAC and ADC conversions.

The DAC buffer is N_LUT samples long, corresponding to a full period of
the output waveform. The frequency of the output waveform is hence changed by
increasing or decreasing N_LUT.

The ADC buffer is BLOCK_SIZE samples long and is independent of the DAC.
Whenever the ADC buffer is completely filled up, it is automatically sent out
over the serial port by the direct memory access controller (DMAC). A double
buffer is employed, called TX_buffer_A and TX_buffer_B.

A digital trigger signal that is in sync with every full period of the output
waveform is made available on a digital output pin, useful for connecting up
an oscilloscope.
*/
#define TCC0_PER_CORRECTION 1  // Might be something wrong in the ASF4 library?
const double SAMPLING_PERIOD_us = (CONF_TCC0_PER + TCC0_PER_CORRECTION) *
                                  (1000000. / (CONF_GCLK_TCC0_FREQUENCY /
                                               CONF_TCC0_PRESCALE));
const double SAMPLING_RATE_Hz = (double) CONF_GCLK_TCC0_FREQUENCY /
                                CONF_TCC0_PRESCALE /
                                (CONF_TCC0_PER + TCC0_PER_CORRECTION);

/*------------------------------------------------------------------------------
    Double buffer: TX_buffer_A & TX_buffer_B
------------------------------------------------------------------------------*/

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

/* The serial transmit buffer will contain a single block of data:
 [SOM,                                              {size = 10 bytes}
  (uint32_t) number of buffer being send            {size =  4 bytes}
  (uint32_t) millis timestamp at start of block     {size =  4 bytes}
  (uint16_t) micros part of timestamp               {size =  2 bytes}
  (uint16_t) phase index LUT_wave at start of block {size =  2 bytes}
  BLOCK_SIZE x (uint16_t) ADC readings 'sig_I'      {size = BLOCK_SIZE * 2 bytes}
  EOM]                                              {size = 10 bytes}
to be transmitted by DMAC
*/
#define N_BYTES_SOM     (sizeof(SOM))
#define N_BYTES_COUNTER (4)
#define N_BYTES_MILLIS  (4)
#define N_BYTES_MICROS  (2)
#define N_BYTES_PHASE   (2)
#define N_BYTES_SIG_I   (BLOCK_SIZE * 2)
#define N_BYTES_EOM     (sizeof(EOM))

#define N_BYTES_TX_BUFFER (N_BYTES_SOM + \
                           N_BYTES_COUNTER + \
                           N_BYTES_MILLIS + \
                           N_BYTES_MICROS + \
                           N_BYTES_PHASE + \
                           N_BYTES_SIG_I + \
                           N_BYTES_EOM)

volatile uint32_t TX_buffer_counter = 0;
volatile bool using_TX_buffer_A = true; // When false: Using TX_buffer_B
static uint8_t TX_buffer_A[N_BYTES_TX_BUFFER] = {};
static uint8_t TX_buffer_B[N_BYTES_TX_BUFFER] = {};

#define TX_BUFFER_OFFSET_COUNTER (N_BYTES_SOM)
#define TX_BUFFER_OFFSET_MILLIS  (TX_BUFFER_OFFSET_COUNTER + N_BYTES_COUNTER)
#define TX_BUFFER_OFFSET_MICROS  (TX_BUFFER_OFFSET_MILLIS  + N_BYTES_MILLIS)
#define TX_BUFFER_OFFSET_PHASE   (TX_BUFFER_OFFSET_MICROS  + N_BYTES_MICROS)
#define TX_BUFFER_OFFSET_SIG_I   (TX_BUFFER_OFFSET_PHASE   + N_BYTES_PHASE)

// Outgoing serial string
#define MAXLEN_STR_BUFFER 96
char str_buffer[MAXLEN_STR_BUFFER];
char usb_buffer[MAXLEN_STR_BUFFER];
struct io_descriptor* io;

// Microcontroller unit (mcu) unique identifier (uid) number
uint8_t mcu_uid[16];

/*
void memset32(void *dest, uint32_t value, uintptr_t size) {
    uintptr_t i;
    for(i = 0; i < (size & (~3)); i+=4) {
        memcpy( ((char*)dest) + i, &value, 4);
    }
    for( ; i < size; i++) {
        ((char *) dest)[i] = ((char *) &value)[i&3];
    }
}
*/

/*
// Calculate free SRAM during run-time (32 KB available on SAMD21)
size_t get_free_RAM() {
    char stack_dummy = 0;
    return (&stack_dummy - sbrk(0));
}
*/

/*------------------------------------------------------------------------------
    SYSTICK
------------------------------------------------------------------------------*/

/* We use SYSTICK for time stamping at a microsecond resolution. The SYSTICK
interrupt service routine is set to fire every 1 millisecond. Anything faster
than this will result in a too heavy a burden on system resources and will
deteriorate the timing accuracy. The microsecond part can be retrieved when
needed, see 'get_systick_timestamp'.

Note: The millis counter will roll over after 49.7 days.
*/

void SysTick_Handler(void) {
    millis++;
}

void get_systick_timestamp(uint32_t *stamp_millis,
                           uint16_t *stamp_micros_part) {
    // Adapted from: https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/delay.c
    uint32_t ticks, ticks2;
    uint32_t pend, pend2;
    uint32_t count, count2;

    ticks2 = SysTick->VAL;
    pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk)  ;
    count2 = millis ;

    do {
        ticks  = ticks2;
        pend   = pend2;
        count  = count2;
        ticks2 = SysTick->VAL;
        pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk)  ;
        count2 = millis;
    } while ((pend != pend2) || (count != count2) || (ticks < ticks2));

    (*stamp_millis) = count2;
    if (pend) {(*stamp_millis)++;}
    (*stamp_micros_part) = (( (SysTick->LOAD - ticks) *
                              (1048576 / (VARIANT_MCK/1000000)) ) >> 20);
}



/*------------------------------------------------------------------------------
    USB print
------------------------------------------------------------------------------*/

void usb_print(const char *str_msg) {
    sprintf(usb_buffer, "%s", str_msg);
    cdcdf_acm_write((uint8_t *) usb_buffer, strlen(usb_buffer));
}



/*------------------------------------------------------------------------------
    USART
------------------------------------------------------------------------------*/

void io_write_blocking(uint8_t *data, uint16_t len) {
    is_serial_txc = false;
    io_write(io, data, len);
    while (!is_serial_txc) {}
}

void io_print(const char *str_msg) {
    // Blocking
    is_serial_txc = false;
    io_write(io, (uint8_t *) str_msg, strlen(str_msg));
    while (!is_serial_txc) {}
}

void io_print_timestamp(void) {
    // Report time in microseconds
    static char buf[16];
    uint32_t millis_copy;
    uint16_t micros_part;

    get_systick_timestamp(&millis_copy, &micros_part);
    sprintf(buf, "%lu %03u\n", millis_copy, micros_part);
    io_print(buf);
}

static void cb_USART_txc(const struct usart_async_descriptor *const io_descr) {
    is_serial_txc = true;
}

static void cb_USART_rxc(const struct usart_async_descriptor *const io_descr) {
    is_serial_rxc = true;
}

static void cb_USART_err(const struct usart_async_descriptor *const io_descr) {
    usb_print("USART ERROR\n");
}



/*------------------------------------------------------------------------------
    get_mcu_uid
------------------------------------------------------------------------------*/

void get_mcu_uid(uint8_t raw_uid[16]) {
	// Return the 128-bits uid (serial number) of the micro controller as a byte
	// array

	#ifdef _SAMD21_
	// SAMD21 from section 9.3.3 of the datasheet
	#define SERIAL_NUMBER_WORD_0	*(volatile uint32_t*)(0x0080A00C)
	#define SERIAL_NUMBER_WORD_1	*(volatile uint32_t*)(0x0080A040)
	#define SERIAL_NUMBER_WORD_2	*(volatile uint32_t*)(0x0080A044)
	#define SERIAL_NUMBER_WORD_3	*(volatile uint32_t*)(0x0080A048)
	#endif
	#ifdef _SAMD51_
	// SAMD51 from section 9.6 of the datasheet
	#define SERIAL_NUMBER_WORD_0	*(volatile uint32_t*)(0x008061FC)
	#define SERIAL_NUMBER_WORD_1	*(volatile uint32_t*)(0x00806010)
	#define SERIAL_NUMBER_WORD_2	*(volatile uint32_t*)(0x00806014)
	#define SERIAL_NUMBER_WORD_3	*(volatile uint32_t*)(0x00806018)
	#endif

	uint32_t pdwUniqueID[4];
	pdwUniqueID[0] = SERIAL_NUMBER_WORD_0;
	pdwUniqueID[1] = SERIAL_NUMBER_WORD_1;
	pdwUniqueID[2] = SERIAL_NUMBER_WORD_2;
	pdwUniqueID[3] = SERIAL_NUMBER_WORD_3;

	for (int i = 0; i < 4; i++) {
		raw_uid[i*4+0] = (uint8_t)(pdwUniqueID[i] >> 24);
		raw_uid[i*4+1] = (uint8_t)(pdwUniqueID[i] >> 16);
		raw_uid[i*4+2] = (uint8_t)(pdwUniqueID[i] >> 8);
		raw_uid[i*4+3] = (uint8_t)(pdwUniqueID[i] >> 0);
	}
}


/*------------------------------------------------------------------------------
    Waveform look-up table (LUT)
------------------------------------------------------------------------------*/
/*
In order to drive the DAC at high sampling speeds, we compute the reference
waveform in advance by a look-up table (LUT). The LUT will contain the samples
for one complete period of the waveform. The LUT is statically allocated and can
fit up to 'MAX_N_LUT' number of samples.
Because the 'SAMPLING_PERIOD_us' is fixed and the LUT can only have an integer
number of samples 'N_LUT', the possible wave frequencies are discrete.
That means that there is a distinction between the wanted frequency and the
obtained frequency 'ref_freq'.
*/

// Align the trigger out signal with the 0-phase position of the waveform being
// output by the DAC. The trigger out signal is controlled by function
// 'cb_DAC_0_conversion_done'
#define LUT_OFFSET_TRIG_OUT 1       // Must be >= 0

// The ADC readings lag behind the DAC output by 1 sample
#define SAMPLE_OFFSET_ADC_DAC 1     // Must be >= 0

void parse_freq(const char const *str_value) {
    ref_freq = atof(str_value);
    N_LUT = (uint16_t) round(SAMPLING_RATE_Hz / ref_freq);
    N_LUT = max(N_LUT, MIN_N_LUT);
    N_LUT = min(N_LUT, MAX_N_LUT);
    ref_freq = SAMPLING_RATE_Hz / N_LUT;
}

void parse_offs(const char const *str_value) {
    ref_offs = atof(str_value);
    ref_offs = max(ref_offs, 0.0);
    ref_offs = min(ref_offs, A_REF);
}

void parse_ampl(const char const *str_value) {
    ref_ampl = atof(str_value);
    ref_ampl = max(ref_ampl, 0.0);
    ref_ampl = min(ref_ampl, A_REF);
}

void compute_LUT(uint16_t *LUT_array) {
    double norm_offs = ref_offs / A_REF;    // Normalized
    double norm_ampl = ref_ampl / A_REF;    // Normalized
    double wave;

    /*
    uint32_t tick = millis;
    io_print("Computing LUT...\n");
    */

    // Generate normalized waveform periods in the range [0, 1].
    // The LUT array is rolled around by LUT_OFFSET_TRIG_OUT.
    for (int16_t i = LUT_OFFSET_TRIG_OUT;
                 i < N_LUT + LUT_OFFSET_TRIG_OUT;
                 i++) {
        float j = i % N_LUT;

        switch (ref_waveform) {
            default:
            case Cosine:
                // N_LUT even: extrema [ 0, 1]
                // N_LUT odd : extrema [>0, 1]
                wave = .5 * (1 + cos(M_TWOPI * j / N_LUT));
                break;

            case Square:
                // Extrema guaranteed  [ 0, 1]
                wave = round(fmod(1.75 * N_LUT - j, N_LUT) / (N_LUT - 1));
                break;

            case Triangle:
                // N_LUT even: extrema [ 0, 1]
                // N_LUT odd : extrema [>0, 1]
                wave = 2 * fabs(j / N_LUT - .5);
                break;
        }

        wave = (norm_offs - norm_ampl) + 2 * norm_ampl * wave;
        wave = max(wave, 0.0);
        wave = min(wave, 1.0);
        LUT_array[i - LUT_OFFSET_TRIG_OUT] =
            (uint16_t) round(MAX_DAC_OUTPUT_BITVAL * wave);
    }

    is_LUT_dirty = false;

    /*
    sprintf(str_buffer, " done in %lu ms\n", millis - tick);
    io_print(str_buffer);
    */
}

void write_time_and_phase_stamp_to_TX_buffer(uint8_t *TX_buffer) {
    /*
    Write timestamp and 'phase'-stamp of the first ADC sample of the buffer that
    is about to be sent out over the serial port. We need to know which
    phase angle was output on the DAC, that corresponds in time to the first ADC
    sample of the TX_buffer. This is the essence of a phase-sensitive detector,
    which is the building block of a lock-in amplifier.
    */

    uint32_t millis_copy;
    uint16_t micros_part;
    uint16_t idx_phase;
    //uint32_t t1;
    //uint32_t t2;
    //volatile uint32_t dt;

    //t2 = SysTick->VAL;
    get_systick_timestamp(&millis_copy, &micros_part);
    //t1 = SysTick->VAL;
    //dt = t2 - t1;

    // Modulo takes more cycles when time increases.
    // Is max 164 clock cycles when (2^24 % N_LUT).
    idx_phase = (TIMER_0.time + N_LUT + SAMPLE_OFFSET_ADC_DAC +
                 LUT_OFFSET_TRIG_OUT) % N_LUT;

    TX_buffer_counter++;
    TX_buffer[TX_BUFFER_OFFSET_COUNTER    ] = TX_buffer_counter;
    TX_buffer[TX_BUFFER_OFFSET_COUNTER + 1] = TX_buffer_counter >> 8;
    TX_buffer[TX_BUFFER_OFFSET_COUNTER + 2] = TX_buffer_counter >> 16;
    TX_buffer[TX_BUFFER_OFFSET_COUNTER + 3] = TX_buffer_counter >> 24;
    TX_buffer[TX_BUFFER_OFFSET_MILLIS     ] = millis_copy;
    TX_buffer[TX_BUFFER_OFFSET_MILLIS  + 1] = millis_copy >> 8;
    TX_buffer[TX_BUFFER_OFFSET_MILLIS  + 2] = millis_copy >> 16;
    TX_buffer[TX_BUFFER_OFFSET_MILLIS  + 3] = millis_copy >> 24;
    TX_buffer[TX_BUFFER_OFFSET_MICROS     ] = micros_part;
    TX_buffer[TX_BUFFER_OFFSET_MICROS  + 1] = micros_part >> 8;
    TX_buffer[TX_BUFFER_OFFSET_PHASE      ] = idx_phase;
    TX_buffer[TX_BUFFER_OFFSET_PHASE   + 1] = idx_phase >> 8;
}



/*------------------------------------------------------------------------------
    DAC
------------------------------------------------------------------------------*/

static void cb_DAC_0_conversion_done(struct dac_async_descriptor *const descr,
                                     const uint8_t ch) {
    // Copy a new full period of the waveform into the DAC output buffer
    if (is_running) {
        gpio_set_pin_level(PIN_D0__TRIG_OUT, true);
        dac_async_write(descr, 0, LUT_wave, N_LUT);
        gpio_set_pin_level(PIN_D0__TRIG_OUT, false);
    }
}



/*------------------------------------------------------------------------------
    ADC
------------------------------------------------------------------------------*/

static void cb_ADC_0_error(const struct adc_dma_descriptor *const descr) {
    usb_print("ADC_0 ERROR\n");
}

static void cb_ADC_0_complete(const struct adc_dma_descriptor *const descr) {
    /*
    if (ADC->INTFLAG.bit.OVERRUN == 1) {
        usb_print("OVERRUN\n");
        ADC->INTENSET.bit.OVERRUN = 1;
    }
    */

    if (is_running) {
        if (using_TX_buffer_A) {
            if (TX_buffer_counter == 0) {
                // Start-up routine. Discard the first read ADC samples.
            } else {
                // Trigger DMA channel 1: Send out TX_buffer_A
                _dma_enable_transaction(1, false);
            }

            // Trigger reading a new block of ADC samples into TX_buffer_B
            adc_dma_read(&ADC_0, &(TX_buffer_B[TX_BUFFER_OFFSET_SIG_I]),
                         BLOCK_SIZE);
            write_time_and_phase_stamp_to_TX_buffer(TX_buffer_B);
        } else {
            // Trigger DMA channel 2: Send out TX_buffer_B
            _dma_enable_transaction(2, false);

            // Trigger reading a new block of ADC samples into TX_buffer_A
            adc_dma_read(&ADC_0, &(TX_buffer_A[TX_BUFFER_OFFSET_SIG_I]),
                         BLOCK_SIZE);
            write_time_and_phase_stamp_to_TX_buffer(TX_buffer_A);
        }
        // Switch buffer around
        using_TX_buffer_A = !using_TX_buffer_A;
    }

    //io_print_timestamp();
}



/*------------------------------------------------------------------------------
    LIA
------------------------------------------------------------------------------*/

void init_LIA(void) {
    // DAC
    dac_async_register_callback(&DAC_0, DAC_ASYNC_CONVERSION_DONE_CB,
                                cb_DAC_0_conversion_done);
    dac_async_enable_channel(&DAC_0, 0);

    // ADC
    adc_dma_register_callback(&ADC_0, ADC_DMA_COMPLETE_CB, cb_ADC_0_complete);
    adc_dma_register_callback(&ADC_0, ADC_DMA_ERROR_CB, cb_ADC_0_error);

    timer_start(&TIMER_0);
}

void start_LIA(void) {
    timer_stop(&TIMER_0);
    TIMER_0.time = 0;

    // DAC
    dac_async_write(&DAC_0, 0, LUT_wave, N_LUT);

    // ADC
    // Because the very first sample read by the ADC is invalid and should be
    // discarded, we only read 2 samples into the ADC buffer (an even length
    // is required). This buffer will be discarded and will not be sent out over
    // serial. This is a start-up routine.
    adc_dma_deinit(&ADC_0);
    adc_dma_init(&ADC_0, ADC);
    adc_dma_read(&ADC_0, &(TX_buffer_A[TX_BUFFER_OFFSET_SIG_I]), 2); // Even length required, not odd
    adc_dma_enable_channel(&ADC_0, 0);

    TX_buffer_counter = 0;
    using_TX_buffer_A = true;

    // Set TRIG_OUT to LOW
    gpio_set_pin_level(PIN_D0__TRIG_OUT, false);

    is_running = true;
    timer_start(&TIMER_0);
}



/*------------------------------------------------------------------------------
    Direct memory access controller (DMAC)
------------------------------------------------------------------------------*/

/*
// Snippet for the transfer done interrupt callback.
static void cb_DMA_1_transfer_done(struct _dma_resource *const resource) {
    // Clear transfer complete flag
    DMAC->CHID.bit.ID = 1;
    DMAC->CHINTENSET.bit.TCMPL = 1;

    //sprintf(str_buffer, "%8lu DMA_1_txc\n", millis);
    //usb_print(str_buffer);
}
*/

void configure_DMA_1(void) {
    // Will send out TX_buffer_A over serial
    _dma_set_source_address(1, (void *) TX_buffer_A);
    _dma_set_destination_address(1, (void *)
                                 &(((Sercom *)(USART_0.device.hw))->
                                   USART.DATA.reg));
    _dma_set_data_amount(1, (uint32_t) N_BYTES_TX_BUFFER);

    /*
    // Snippet to enable the transfer done interrupt callback. This can be
    // useful for debugging, but do not leave on for proper operation as it
    // seems to mess with tight timing and screws the event order around,
    // somehow.
    struct _dma_resource *dma_res;

    // Get DMA channel 1 resource to set the application callback
    _dma_get_channel_resource(&dma_res, 1);

    // Set application callback to handle the DMA channel 1 transfer done
    dma_res->dma_cb.transfer_done = cb_DMA_1_transfer_done;

    // Enable DMA channel 1 transfer complete interrupt
    _dma_set_irq_state(1, DMA_TRANSFER_COMPLETE_CB, true);
    */
}

void configure_DMA_2(void) {
    // Will send out TX_buffer_B over serial
    _dma_set_source_address(2, (void *) TX_buffer_B);
    _dma_set_destination_address(2, (void *)
                                 &(((Sercom *)(USART_0.device.hw))->
                                   USART.DATA.reg));
    _dma_set_data_amount(2, (uint32_t) N_BYTES_TX_BUFFER);
}



/*------------------------------------------------------------------------------
    main
------------------------------------------------------------------------------*/

int main(void) {
    NVIC_SetPriority(TCC0_IRQn   , 0);
    NVIC_SetPriority(DAC_IRQn    , 1);
    NVIC_SetPriority(ADC_IRQn    , 1);
    NVIC_SetPriority(SERCOM5_IRQn, 2);
    NVIC_SetPriority(SysTick_IRQn, 3);
    atmel_start_init();

    // Microcontroller unit (MCU) unique identifier (uid) number
    get_mcu_uid(mcu_uid);

    // USART
    usart_async_get_io_descriptor(&USART_0, &io);
    usart_async_register_callback(&USART_0, USART_ASYNC_TXC_CB  , cb_USART_txc);
    usart_async_register_callback(&USART_0, USART_ASYNC_RXC_CB  , cb_USART_rxc);
    usart_async_register_callback(&USART_0, USART_ASYNC_ERROR_CB, cb_USART_err);
    usart_async_enable(&USART_0);

    DvG_scl scl_1;
    scl_configure(&scl_1, io);

    // Copy SOM and EOM to the transmit buffer TX_buffer
    memcpy(TX_buffer_A                         , SOM, 10);
    memcpy(TX_buffer_B                         , SOM, 10);
    memcpy(&TX_buffer_A[N_BYTES_TX_BUFFER - 10], EOM, 10);
    memcpy(&TX_buffer_B[N_BYTES_TX_BUFFER - 10], EOM, 10);

    // Millis and micros timer
    //SysTick_Config(SystemCoreClock / 1000);
    SysTick->LOAD = (uint32_t) (SystemCoreClock / 1000 - 1UL);
    SysTick->VAL  = 0UL;
    SysTick->CTRL =
		SysTick_CTRL_CLKSOURCE_Msk |
		SysTick_CTRL_TICKINT_Msk |
		SysTick_CTRL_ENABLE_Msk;

    // LUT
    parse_freq("250.0");  // [Hz] Wanted startup frequency
    parse_offs("2.0");    // [V]  Wanted startup offset
    parse_ampl("1.0");    // [V]  Wanted startup amplitude
    compute_LUT(LUT_wave);

    // Will send out TX_buffer over SERCOM when triggered
    configure_DMA_1();
    configure_DMA_2();

    // Init DAC and ADC
    init_LIA();

    char *str_cmd;  // Incoming serial command string
    bool toggle = false;
    uint32_t prev_millis = millis;

    while (1) {
        if (millis - prev_millis >= 250) {
            prev_millis = millis;
            gpio_set_pin_level(PIN_D13__LED, toggle);
            toggle = !toggle;
        }

        // Process commands on the data channel
        if (is_serial_rxc) {
            is_serial_rxc = false;

            if (scl_available(&scl_1)) {

                if (is_running) {

                    // -------------------
                    //  Running
                    // -------------------
                    /* Any command received while running will switch the
                    lock-in amp off. The command string will not be checked in
                    advance, because this causes a lot of overhead, during which
                    time the Arduino's serial-out buffer could potentially flood
                    the serial-in buffer at the PC side. This will happen when
                    the PC is not reading (and depleting) the in-buffer as fast
                    as possible because it is now waiting for the 'off' reply to
                    occur.
                    */

                    is_running = false;

                    // Disable DAC output
                    uint16_t DAC_OUTPUT_OFF[1] = {0};
                    dac_async_write(&DAC_0, 0, DAC_OUTPUT_OFF, 1);
                    gpio_set_pin_level(PIN_D0__TRIG_OUT, false);

                    // Cancel all active DMA transactions
                    CRITICAL_SECTION_ENTER();
                    DMAC->CHID.reg = DMAC_CHID_ID(0);   // ADC
                    DMAC->CHCTRLA.reg = 0;
                    DMAC->CHID.reg = DMAC_CHID_ID(1);   // SERCOM TX_buffer_A
                    DMAC->CHCTRLA.reg = 0;
                    DMAC->CHID.reg = DMAC_CHID_ID(2);   // SERCOM TX_buffer_B
                    DMAC->CHCTRLA.reg = 0;
                    CRITICAL_SECTION_LEAVE();

                    // Flush out any binary buffer data scheduled for sending,
                    // potentially flooding the receiving buffer at the PC side
                    // if 'is_running' was not switched to false fast enough.
                    while (!usart_async_is_tx_empty(&USART_0));

                    // Confirm at the PC side that the lock-in amp is off and is
                    // no longer sending binary data. The 'off' message might
                    // still be preceded with some left-over binary data when
                    // being read at the PC side.
                    //io_write(io, (uint8_t *) "\noff\n", 5);  // non-blocking
                    io_print("\noff\n"); // blocking

                    // Flush out and ignore the command
                    scl_get_command(&scl_1);

                } else {

                    // -------------------
                    //  Not running
                    // -------------------

                    str_cmd = scl_get_command(&scl_1);



                    if (strcmp(str_cmd, "id?") == 0) {
                        // Report identity string
                        io_print("Arduino, Alia\n");



                    } else if (strcmp(str_cmd, "mcu?") == 0) {
                        // Report microcontroller model, serial and firmware
                        char str_model[12];
                        char str_uid[33];

                        snprintf(str_model, sizeof(str_model),
                                 #if defined(__SAMD21G18A__)
                                 "SAMD21G18A"
                                 #elif defined(__SAMD21E18A__)
                                 "SAMD21E18A"
                                 #elif defined(__SAMD51P20A__)
                                 "SAMD51P20A"
                                 #elif defined(__SAMD51J19A__)
                                 "SAMD51J19A"
                                 #elif defined(__SAMD51G19A__)
                                 "SAMD51G19A"
                                 #else
                                 "unknown MCU"
                                 #endif
                                 );

						// Format the uid byte-array to hex representation
						str_uid[32] = 0;
						for (uint8_t j = 0; j < 16; j++)
						sprintf(&str_uid[2*j], "%02X", mcu_uid[j]);

                        snprintf(str_buffer, MAXLEN_STR_BUFFER, "%s\t%s\t%s\n",
                                 FIRMWARE_VERSION, str_model, str_uid);
                        io_print(str_buffer);



                    } else if (strcmp(str_cmd, "const?") == 0) {
                        // Report constants
                        snprintf(str_buffer, MAXLEN_STR_BUFFER,
                                 "%.6f\t%u\t%u\t%u\t%u\t%.3f\t%u\t%u\n",
                                 SAMPLING_PERIOD_us,
                                 BLOCK_SIZE,
                                 N_BYTES_TX_BUFFER,
                                 DAC_OUTPUT_BITS,
                                 ADC_INPUT_BITS,
                                 A_REF,
                                 MIN_N_LUT,
                                 MAX_N_LUT);
                        io_print(str_buffer);



                    } else if (strcmp(str_cmd, "ref?") == 0 ||
                               strcmp(str_cmd, "?") == 0) {
                        // Report reference signal settings
                        sprintf(str_buffer, "%.3f\t%.3f\t%.3f\t%s\t%i\n",
                                ref_freq, ref_offs, ref_ampl,
                                WAVEFORM_STRING[ref_waveform], N_LUT);
                        io_print(str_buffer);



                    } else if (strcmp(str_cmd, "lut?") == 0 ||
                               strcmp(str_cmd, "l?") == 0) {
                        // Report the LUT as a binary stream.
                        // The reported LUT will start at phase = 0 deg.
                        io_write_blocking((uint8_t *) &N_LUT, 2);
                        io_write_blocking((uint8_t *) &is_LUT_dirty, 1);
                        io_write_blocking((uint8_t *)
                                          &LUT_wave[N_LUT - LUT_OFFSET_TRIG_OUT],
                                          LUT_OFFSET_TRIG_OUT * 2);
                        io_write_blocking((uint8_t *) LUT_wave,
                                          (N_LUT - LUT_OFFSET_TRIG_OUT) * 2);



                    } else if (strcmp(str_cmd, "lut_ascii?") == 0 ||
                               strcmp(str_cmd, "la?") == 0) {
                        // Report the LUT as tab-delimited ASCII.
                        // The reported LUT will start at phase = 0 deg.
                        // Convenience function handy for debugging from a
                        // serial console.
                        uint16_t i;

                        sprintf(str_buffer, "%u\t%i\n", N_LUT, is_LUT_dirty);
                        io_print(str_buffer);

                        for (i = N_LUT - LUT_OFFSET_TRIG_OUT; i < N_LUT; i++) {
                            sprintf(str_buffer, "%u\t", LUT_wave[i]);
                            io_print(str_buffer);
                        }
                        for (i = 0; i < N_LUT - LUT_OFFSET_TRIG_OUT - 1; i++) {
                            sprintf(str_buffer, "%u\t", LUT_wave[i]);
                            io_print(str_buffer);
                        }
                        sprintf(str_buffer, "%u\n",
                                LUT_wave[N_LUT - LUT_OFFSET_TRIG_OUT - 1]);
                        io_print(str_buffer);



                    } else if (strcmp(str_cmd, "time?") == 0) {
                        // Report time in microseconds
                        io_print_timestamp();



                    } else if (strcmp(str_cmd, "off") == 0) {
                        // Lock-in amp is already off and we reply with an
                        // acknowledgment
                        io_print("already_off\n");



                    } else if (strcmp(str_cmd, "on") == 0) {
                        // Start lock-in amp
                        start_LIA();



					} else if (strcmp(str_cmd, "_on") == 0) {
						// Start lock-in amp and reset the timestamp to (near) 0
						// for the upcoming new transmit buffer
						NVIC_DisableIRQ(SysTick_IRQn);
						millis = 0;         // Reset millis part
						SysTick->VAL = 0UL; // Reset micros part
						NVIC_EnableIRQ(SysTick_IRQn);

						start_LIA();



                    } else if (strncmp(str_cmd, "freq", 4) == 0) {
                        // Set frequency of the reference signal [Hz].
                        // Automatically recomputes the LUT and replies back the
                        // effective setting.
                        parse_freq(&str_cmd[4]);
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%.3f\n", ref_freq);
                        io_print(str_buffer);

                    } else if (strncmp(str_cmd, "_freq", 5) == 0) {
                        // Set frequency of the reference signal [Hz].
                        // You still have to call 'compute_LUT(LUT_wave)' for it
                        // to become effective.
                        parse_freq(&str_cmd[5]);
                        is_LUT_dirty = true;
                        sprintf(str_buffer, "%.3f\n", ref_freq);
                        io_print(str_buffer);



                    } else if (strncmp(str_cmd, "offs", 4) == 0) {
                        // Set offset of the reference signal [V].
                        // Automatically recomputes the LUT and replies back the
                        // effective setting.
                        parse_offs(&str_cmd[4]);
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%.3f\n", ref_offs);
                        io_print(str_buffer);

                    } else if (strncmp(str_cmd, "_offs", 5) == 0) {
                        // Set offset of the reference signal [V].
                        // You still have to call 'compute_LUT(LUT_wave)' for it
                        // to become effective.
                        parse_offs(&str_cmd[5]);
                        is_LUT_dirty = true;
                        sprintf(str_buffer, "%.3f\n", ref_offs);
                        io_print(str_buffer);



                    } else if (strncmp(str_cmd, "ampl", 4) == 0) {
                        // Set amplitude of the reference signal [V].
                        // Automatically recomputes the LUT and replies back the
                        // effective setting.
                        parse_ampl(&str_cmd[4]);
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%.3f\n", ref_ampl);
                        io_print(str_buffer);

                    } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
                        // Set amplitude of the reference signal [V].
                        // You still have to call 'compute_LUT(LUT_wave)' for it
                        // to become effective.
                        parse_ampl(&str_cmd[5]);
                        is_LUT_dirty = true;
                        sprintf(str_buffer, "%.3f\n", ref_ampl);
                        io_print(str_buffer);



                    } else if (strncmp(str_cmd, "wave", 4) == 0) {
                        // Set the waveform type of the reference signal.
                        // Automatically recomputes the LUT and replies back the
                        // effective setting.
                        ref_waveform = atoi(&str_cmd[4]);
                        ref_waveform = max(ref_waveform, 0);
                        ref_waveform = min(ref_waveform, END_WAVEFORM_ENUM - 1);
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%s\n", WAVEFORM_STRING[ref_waveform]);
                        io_print(str_buffer);

                    } else if (strncmp(str_cmd, "_wave", 5) == 0) {
                        // Set the waveform type of the reference signal.
                        // You still have to call 'compute_LUT(LUT_wave)' for it
                        // to become effective.
                        ref_waveform = atoi(&str_cmd[5]);
                        ref_waveform = max(ref_waveform, 0);
                        ref_waveform = min(ref_waveform, END_WAVEFORM_ENUM - 1);
                        is_LUT_dirty = true;
                        sprintf(str_buffer, "%s\n", WAVEFORM_STRING[ref_waveform]);
                        io_print(str_buffer);



                    } else if (strcmp(str_cmd, "compute_lut") == 0 ||
                               strcmp(str_cmd, "c") == 0) {
                        // (Re)compute the LUT based on the following settings:
                        // ref_freq, ref_offs, ref_ampl, ref_waveform.
                        compute_LUT(LUT_wave);
                        io_print("!\n");    // Reply with OK '!'



                    /* Set the waveform type of the reference signal.
                       Automatically recomputes the LUT and replies back the
                       effective setting. Convenience functions to be called
                       from a serial terminal.
                    */
                    } else if (strcmp(str_cmd, "cos") == 0) {
                        ref_waveform = Cosine;
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%s\n",
                                WAVEFORM_STRING[ref_waveform]);
                        io_print(str_buffer);

                    } else if (strcmp(str_cmd, "sqr") == 0) {
                        ref_waveform = Square;
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%s\n",
                                WAVEFORM_STRING[ref_waveform]);
                        io_print(str_buffer);

                    } else if (strcmp(str_cmd, "tri") == 0) {
                        ref_waveform = Triangle;
                        compute_LUT(LUT_wave);
                        sprintf(str_buffer, "%s\n",
                                WAVEFORM_STRING[ref_waveform]);
                        io_print(str_buffer);
                    }
                }
            }
        }
    }
}
