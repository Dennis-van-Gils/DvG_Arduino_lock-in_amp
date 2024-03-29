/*------------------------------------------------------------------------------
  Arduino lock-in amplifier

  Pins:
    - A0 : Output reference signal `ref_X`, single-ended with respect to GND

    When `ADC_DIFFERENTIAL` = 0
    ------------------------------------------
    - A1 : Input signal `sig_I`, single-ended with respect to GND
    - A2 : Not used

    When `ADC_DIFFERENTIAL` = 1
    ------------------------------------------
    - A1 : Input signal `sig_I`, differential(+)
    - A2 : Input signal `sig_I`, differential(-)

    - D12: A digital trigger-out signal that is in sync with every full period
           of the output reference waveform, useful for connecting up to an
           oscilloscope.

  TODO:
    Implement https://github.com/mpflaga/Arduino-MemoryFree

  Dennis van Gils
  31-08-2021
*/

#include <atmel_start.h>
#include <math.h>
#include <hpl_dma.h>
#include <stdio.h>
#include <stdlib.h>
#include <hpl_tcc_config.h>
#include <string.h>
#include <utils.h>
#include "DvG_serial_command_listener.h"

#define FIRMWARE_VERSION "ALIA v1.0.0 MICROCHIPSTUDIO"
#define VARIANT_MCK (48000000ul)   // Master clock frequency

// OBSERVATION: Single-ended has half the noise compared to differential
#define ADC_DIFFERENTIAL 0  // Leave at 0, differential mode is not implemented

// Microcontroller unit (mcu)
#if defined __SAMD21G18A__
#  define MCU_MODEL "SAMD21G18A"
#elif defined __SAMD21E18A__
#  define MCU_MODEL "SAMD21E18A"
#elif defined __SAMD51P20A__
#  define MCU_MODEL "SAMD51P20A"
#elif defined __SAMD51J19A__
#  define MCU_MODEL "SAMD51J19A"
#elif defined __SAMD51G19A__
#  define MCU_MODEL "SAMD51G19A"
#endif

static void LED_off() {gpio_set_pin_level(PIN_D13__LED, false);}
static void LED_on()  {gpio_set_pin_level(PIN_D13__LED, true);}
static void TRIG_OUT_off() {gpio_set_pin_level(PIN_D12__TRIG_OUT, false);}
static void TRIG_OUT_on() {gpio_set_pin_level(PIN_D12__TRIG_OUT, true);}
// static void TRIG_OUT_toggle() {
//   gpio_set_pin_level(PIN_D12__TRIG_OUT, !gpio_get_pin_level(PIN_D12__TRIG_OUT));
// }

#define ADC_INPUT_BITS 12
#if defined(_SAMD21_)
#  define DAC_OUTPUT_BITS 10
#elif defined(_SAMD51_)
#  define DAC_OUTPUT_BITS 12
#endif

// Preprocessor trick to ensure enums and strings are in sync, so one can write
// 'WAVEFORM_STRING[Sine]' to give the string 'Sine'
#define FOREACH_WAVEFORM(WAVEFORM)                                             \
  WAVEFORM(Sine)                                                               \
  WAVEFORM(Square)                                                             \
  WAVEFORM(Triangle)                                                           \
  WAVEFORM(END_WAVEFORM_ENUM)
#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

enum WAVEFORM_ENUM { FOREACH_WAVEFORM(GENERATE_ENUM) };
static const char *WAVEFORM_STRING[] = {FOREACH_WAVEFORM(GENERATE_STRING)};

// Others
volatile bool is_running = false; // Is the lock-in amplifier running?
volatile bool trigger_reset_time = false;
volatile bool is_serial_txc = false; // Is serial data sent out?
volatile bool is_serial_rxc = false; // Is serial data received?
volatile uint32_t millis = 0;        // Updated by SysTick, once every 1 ms
char mcu_uid[33]; // Serial number

/*------------------------------------------------------------------------------
  TIMER_0
--------------------------------------------------------------------------------

  Channel CC[0] of timer TCC0 is set to match mode and will trigger output
  events periodically at a fixed sample rate (say 20 kHz). It is the source that
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

const double SAMPLING_PERIOD_us =
  (CONF_TCC0_PER + TCC0_PER_CORRECTION) *
  (1000000. / (CONF_GCLK_TCC0_FREQUENCY / CONF_TCC0_PRESCALE));

const double SAMPLING_RATE_Hz =
  (double) CONF_GCLK_TCC0_FREQUENCY / CONF_TCC0_PRESCALE /
  (CONF_TCC0_PER + TCC0_PER_CORRECTION);

/*------------------------------------------------------------------------------
  Sampling
--------------------------------------------------------------------------------

  * Double buffer: TX_buffer_A & TX_buffer_B

    Each acquired `sig_I` sample will get written to a buffer to be transmitted
    over serial once the buffer is full. There are two of these buffers:
    `TX_buffer_A` and `TX_buffer_B`.

    The buffer that will be send each transmission is `BLOCK_SIZE` input samples
    long. We employ a double buffer technique, where alternatingly `TX_buffer_A`
    is being written to and `TX_buffer_B` is being sent, and vice-versa.

    A full transmit buffer will contain a single block of data:

    [
      SOM,                                              {10 bytes}
      (uint32_t) number of block being send             { 4 bytes}
      (uint32_t) millis timestamp at start of block     { 4 bytes}
      (uint16_t) micros part of timestamp               { 2 bytes}
      (uint16_t) `LUT_wave` index at start of block     { 2 bytes}
      BLOCK_SIZE x (int16_t) ADC readings `sig_I`       {BLOCK_SIZE * 2 bytes}
      EOM                                               {10 bytes}
    ]
*/

// Hint: Maintaining `SAMPLING_PERIOD_us x BLOCK_SIZE` = 0.1 seconds long will
// result in a serial transmit rate of 10 blocks / s, which acts nicely with
// the Python GUI.
#define BLOCK_SIZE 2000

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

#define N_BYTES_SOM     (sizeof(SOM))
#define N_BYTES_COUNTER (4)
#define N_BYTES_MILLIS  (4)
#define N_BYTES_MICROS  (2)
#define N_BYTES_PHASE   (2)
#define N_BYTES_SIG_I   (BLOCK_SIZE * 2)
#define N_BYTES_EOM     (sizeof(EOM))

#define N_BYTES_TX_BUFFER (N_BYTES_SOM     + \
                           N_BYTES_COUNTER + \
                           N_BYTES_MILLIS  + \
                           N_BYTES_MICROS  + \
                           N_BYTES_PHASE   + \
                           N_BYTES_SIG_I   + \
                           N_BYTES_EOM)

static uint8_t TX_buffer_A[N_BYTES_TX_BUFFER] = {0};
static uint8_t TX_buffer_B[N_BYTES_TX_BUFFER] = {0};
volatile uint32_t TX_buffer_counter = 0;
volatile bool using_TX_buffer_A = true; // When false: Using TX_buffer_B

#define TX_BUFFER_OFFSET_COUNTER (N_BYTES_SOM)
#define TX_BUFFER_OFFSET_MILLIS  (TX_BUFFER_OFFSET_COUNTER + N_BYTES_COUNTER)
#define TX_BUFFER_OFFSET_MICROS  (TX_BUFFER_OFFSET_MILLIS  + N_BYTES_MILLIS)
#define TX_BUFFER_OFFSET_PHASE   (TX_BUFFER_OFFSET_MICROS  + N_BYTES_MICROS)
#define TX_BUFFER_OFFSET_SIG_I   (TX_BUFFER_OFFSET_PHASE   + N_BYTES_PHASE)

/*------------------------------------------------------------------------------
  Serial
------------------------------------------------------------------------------*/

// Outgoing serial string
#define MAXLEN_STR_BUFFER 100
char str_buffer[MAXLEN_STR_BUFFER];
char usb_buffer[MAXLEN_STR_BUFFER];
struct io_descriptor* io;

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
  Waveform look-up table (LUT)
--------------------------------------------------------------------------------

  In order to drive the DAC at high sampling rates, we compute the reference
  waveform in advance by a look-up table (LUT). The LUT will contain the samples
  for one complete period of the waveform. The LUT is statically allocated and
  can fit up to `MAX_N_LUT` number of samples.

  Because the `SAMPLING_PERIOD_us` is fixed and the LUT can only have an integer
  number of samples `N_LUT`, the possible wave frequencies are discrete. That
  means that there is a distinction between the wanted frequency and the
  obtained frequency `ref_freq`.
*/

// Output reference signal `ref_X` parameters
enum WAVEFORM_ENUM ref_waveform;
double ref_freq;         // [Hz]    Obtained frequency of reference signal
double ref_offs;         // [V]     Voltage offset of reference signal
double ref_ampl;         // [V]     Voltage amplitude reference signal
double ref_VRMS;         // [V_RMS] Voltage amplitude reference signal
double ref_RMS_factor;   // RMS factor belonging to chosen waveform
bool ref_is_clipping_HI; // Output reference signal is clipping high?
bool ref_is_clipping_LO; // Output reference signal is clipping low?

// Look-up table (LUT) for fast DAC
#define MIN_N_LUT 20   // Min. allowed number of samples for one full period
#define MAX_N_LUT 1000 // Max. allowed number of samples for one full period
uint16_t LUT_wave[MAX_N_LUT] = {0}; // Look-up table allocation
uint16_t N_LUT; // Current number of samples for one full period
// Does the LUT need to be recomputed to reflect new settings?
bool is_LUT_dirty = false;

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define MAX_DAC_OUTPUT_BITVAL ((1 << DAC_OUTPUT_BITS) - 1)

void compute_LUT(uint16_t *LUT_array) {
  double norm_offs = ref_offs / A_REF; // Normalized
  double norm_ampl = ref_ampl / A_REF; // Normalized
  double wave;

  // Clear all waveform data for ease of debugging
  for (uint16_t i = 0; i < MAX_N_LUT; i++) {
    LUT_array[i] = 0;
  }

  // Generate normalized waveform periods in the range [0, 1]
  ref_is_clipping_HI = false;
  ref_is_clipping_LO = false;

  for (uint16_t i = 0; i < N_LUT; i++) {

    switch (ref_waveform) {
      default:
      case Sine:
        // N_LUT integer multiple of 4: extrema [0, 1], symmetric
        // N_LUT others               : extrema <0, 1>, symmetric
        wave = (sin(M_TWOPI * i / N_LUT) + 1.) * .5;
        break;

      case Square:
        // N_LUT even                 : extrema [0, 1], symmetric
        // N_LUT odd                  : extrema [0, 1], asymmetric !!!
        wave = ((double)(i) / N_LUT < 0.5 ? 1. : 0.);
        break;

      case Triangle:
        // N_LUT integer multiple of 4: extrema [0, 1], symmetric
        // N_LUT others               : extrema <0, 1>, symmetric
        wave = asin(sin(M_TWOPI * i / N_LUT)) / M_PI + .5;
        break;
    }

    wave = (norm_offs - norm_ampl) + 2 * norm_ampl * wave;

    if (wave < 0.0) {
      ref_is_clipping_LO = true;
      wave = 0.0;
    } else if (wave > 1.0) {
      ref_is_clipping_HI = true;
      wave = 1.0;
    }

    LUT_array[i] = (uint16_t)round(MAX_DAC_OUTPUT_BITVAL * wave);
  }

  is_LUT_dirty = false;
}

void set_wave(int value) {
  /* Set the waveform type, keeping `ref_VRMS` constant and changing `ref_ampl`
  according to the new waveform */
  is_LUT_dirty = true;
  value = max(value, 0);
  value = min(value, END_WAVEFORM_ENUM - 1);
  ref_waveform = value;

  switch (ref_waveform) {
    default:
    case Sine:
      ref_RMS_factor = sqrt(2);
      break;

    case Square:
      ref_RMS_factor = 1.;
      break;

    case Triangle:
      ref_RMS_factor = sqrt(3);
      break;
  }

  ref_ampl = ref_VRMS * ref_RMS_factor;
}

void set_freq(double value) {
  is_LUT_dirty = true;
  N_LUT = (uint16_t)round(SAMPLING_RATE_Hz / value);
  N_LUT = max(N_LUT, MIN_N_LUT);
  N_LUT = min(N_LUT, MAX_N_LUT);
  ref_freq = SAMPLING_RATE_Hz / N_LUT;
}

void set_offs(double value) {
  is_LUT_dirty = true;
  ref_offs = max(value, 0.0);
  // ref_offs = min(ref_offs, A_REF);
}

void set_ampl(double value) {
  is_LUT_dirty = true;
  ref_ampl = max(value, 0.0);
  // ref_ampl = min(ref_ampl, A_REF);
  ref_VRMS = ref_ampl / ref_RMS_factor;
}

void set_VRMS(double value) {
  is_LUT_dirty = true;
  ref_VRMS = max(value, 0.0);
  ref_ampl = ref_VRMS * ref_RMS_factor;
  // ref_ampl = min(ref_ampl, A_REF);
  // ref_VRMS = ref_ampl / ref_RMS_factor;
}

/*------------------------------------------------------------------------------
  get_mcu_uid
------------------------------------------------------------------------------*/

void get_mcu_uid(char mcu_uid_out[33]) {
  /* Return the 128-bit unique identifier (uid) of the microcontroller unit
  (mcu) as a hex string. Aka, the serial number.
  */
  uint8_t raw_uid[16]; // uid as byte array

// SAMD21 from section 9.3.3 of the datasheet
#ifdef _SAMD21_
#  define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x0080A00C)
#  define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x0080A040)
#  define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x0080A044)
#  define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x0080A048)
#endif
// SAMD51 from section 9.6 of the datasheet
#ifdef _SAMD51_
#  define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x008061FC)
#  define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x00806010)
#  define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x00806014)
#  define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x00806018)
#endif

  uint32_t pdw_uid[4];
  pdw_uid[0] = SERIAL_NUMBER_WORD_0;
  pdw_uid[1] = SERIAL_NUMBER_WORD_1;
  pdw_uid[2] = SERIAL_NUMBER_WORD_2;
  pdw_uid[3] = SERIAL_NUMBER_WORD_3;

  for (int i = 0; i < 4; i++) {
    raw_uid[i * 4 + 0] = (uint8_t)(pdw_uid[i] >> 24);
    raw_uid[i * 4 + 1] = (uint8_t)(pdw_uid[i] >> 16);
    raw_uid[i * 4 + 2] = (uint8_t)(pdw_uid[i] >> 8);
    raw_uid[i * 4 + 3] = (uint8_t)(pdw_uid[i] >> 0);
  }

  for (int j = 0; j < 16; j++) {
    sprintf(&mcu_uid_out[2 * j], "%02X", raw_uid[j]);
  }
  mcu_uid_out[32] = 0;
}

/*------------------------------------------------------------------------------
  Time keeping
--------------------------------------------------------------------------------

  We use SYSTICK for time stamping at a microsecond resolution. The SYSTICK
  interrupt service routine is set to fire every 1 millisecond. Anything faster
  than this will result in a too heavy a burden on system resources and will
  deteriorate the timing accuracy. The microsecond part can be retrieved when
  needed, see `get_systick_timestamp()`.
*/

void SysTick_Handler(void) {
  millis++;
}

void get_systick_timestamp(uint32_t *stamp_millis,
                           uint16_t *stamp_micros_part) {
  /* Adapted from:
  https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/delay.c

  Note:
    The millis counter will roll over after 49.7 days.
  */
  uint32_t ticks, ticks2;
  uint32_t pend, pend2;
  uint32_t count, count2;

  ticks2 = SysTick->VAL;
  pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
  count2 = millis ;

  do {
    ticks  = ticks2;
    pend   = pend2;
    count  = count2;
    ticks2 = SysTick->VAL;
    pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
    count2 = millis;
  } while ((pend != pend2) || (count != count2) || (ticks < ticks2));

  (*stamp_millis) = count2;
  if (pend) {(*stamp_millis)++;}
  (*stamp_micros_part) =
    (((SysTick->LOAD - ticks) * (1048576 / (VARIANT_MCK / 1000000))) >> 20);
}

void stamp_TX_buffer(uint8_t *TX_buffer) {
  /* Write timestamp and `phase`-stamp of the first ADC sample of the block
  that is about to be sent out over the serial port. We need to know which
  phase angle was output on the DAC, that corresponds in time to the first ADC
  sample of the `TX_buffer`. This is the essence of a phase-sensitive
  detector, which is the building block of a lock-in amplifier.
  */

  static uint32_t startup_millis = 0; // Time when lock-in amp got turned on
  static uint16_t startup_micros = 0; // Time when lock-in amp got turned on
  uint32_t millis_copy;
  uint16_t micros_part;
  uint16_t idx_phase;

  get_systick_timestamp(&millis_copy, &micros_part);

  // Modulo takes more cycles when time increases.
  // Is max 164 clock cycles when (2^24 % N_LUT).
  // The ADC readings lag behind the DAC output by 1 sample
  #define SAMPLE_OFFSET_ADC_DAC 1   // Must be >= 0
  idx_phase = (TIMER_0.time + N_LUT + SAMPLE_OFFSET_ADC_DAC) % N_LUT;


  if (trigger_reset_time) {
    trigger_reset_time = false;
    startup_millis = millis_copy;
    startup_micros = micros_part;
  }

  TX_buffer_counter++;
  millis_copy -= startup_millis;
  if (micros_part >= startup_micros) {
    micros_part -= startup_micros;
  } else {
    micros_part = micros_part + 1000 - startup_micros;
    millis_copy -= 1;
  }
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
  USB print
------------------------------------------------------------------------------*/
/*
void usb_print(const char *str_msg) {
  sprintf(usb_buffer, "%s", str_msg);
  cdcdf_acm_write((uint8_t *) usb_buffer, strlen(usb_buffer));
}
*/

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
  sprintf(buf, "%lu.%03u\n", millis_copy, micros_part);
  io_print(buf);
}

static void cb_USART_txc(const struct usart_async_descriptor *const io_descr) {
  is_serial_txc = true;
}

static void cb_USART_rxc(const struct usart_async_descriptor *const io_descr) {
  is_serial_rxc = true;
}

/*
static void cb_USART_err(const struct usart_async_descriptor *const io_descr) {
  usb_print("USART ERROR\n");
}
*/

/*------------------------------------------------------------------------------
  DAC
------------------------------------------------------------------------------*/

static void cb_DAC_0_conversion_done(struct dac_async_descriptor *const descr,
                                     const uint8_t ch) {
  // Copy a new full period of the waveform into the DAC output buffer
  if (is_running) {
    TRIG_OUT_on();
    dac_async_write(descr, 0, LUT_wave, N_LUT);
    TRIG_OUT_off();
  }
}

/*------------------------------------------------------------------------------
  ADC
------------------------------------------------------------------------------*/

/*
static void cb_ADC_0_error(const struct adc_dma_descriptor *const descr) {
  usb_print("ADC_0 ERROR\n");
}
*/

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
      adc_dma_read(&ADC_0, &(TX_buffer_B[TX_BUFFER_OFFSET_SIG_I]), BLOCK_SIZE);
      stamp_TX_buffer(TX_buffer_B);
    } else {
      // Trigger DMA channel 2: Send out TX_buffer_B
      _dma_enable_transaction(2, false);

      // Trigger reading a new block of ADC samples into TX_buffer_A
      adc_dma_read(&ADC_0, &(TX_buffer_A[TX_BUFFER_OFFSET_SIG_I]), BLOCK_SIZE);
      stamp_TX_buffer(TX_buffer_A);
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
  //adc_dma_register_callback(&ADC_0, ADC_DMA_ERROR_CB, cb_ADC_0_error);

  timer_start(&TIMER_0);
}

void start_LIA(void) {
  timer_stop(&TIMER_0);
  TIMER_0.time = 0;
  LED_on();

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
  _dma_set_destination_address(
    1, (void *) &(((Sercom *)(USART_0.device.hw))->USART.DATA.reg));
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
  _dma_set_destination_address(
    2, (void *) &(((Sercom *)(USART_0.device.hw))->USART.DATA.reg));
  _dma_set_data_amount(2, (uint32_t) N_BYTES_TX_BUFFER);
}

/*------------------------------------------------------------------------------
  main
------------------------------------------------------------------------------*/

int main(void) {
  char *str_cmd; // Incoming serial command string

  NVIC_SetPriority(TCC0_IRQn   , 0);
  NVIC_SetPriority(DAC_IRQn    , 1);
  NVIC_SetPriority(ADC_IRQn    , 1);
  NVIC_SetPriority(SERCOM5_IRQn, 2);
  NVIC_SetPriority(SysTick_IRQn, 3);
  atmel_start_init();

  // Microcontroller unit (MCU) unique identifier (uid) number
  get_mcu_uid(mcu_uid);

  // Use built-in LED to signal running state of lock-in amp
  LED_off();
  TRIG_OUT_off();

  // USART
  usart_async_get_io_descriptor(&USART_0, &io);
  usart_async_register_callback(&USART_0, USART_ASYNC_TXC_CB, cb_USART_txc);
  usart_async_register_callback(&USART_0, USART_ASYNC_RXC_CB, cb_USART_rxc);
  //usart_async_register_callback(&USART_0, USART_ASYNC_ERROR_CB, cb_USART_err);
  usart_async_enable(&USART_0);

  DvG_scl scl_1;
  scl_configure(&scl_1, io);

  // Prepare SOM and EOM
  memcpy(TX_buffer_A                         , SOM, N_BYTES_SOM);
  memcpy(&TX_buffer_A[N_BYTES_TX_BUFFER - 10], EOM, N_BYTES_EOM);
  memcpy(TX_buffer_B                         , SOM, N_BYTES_SOM);
  memcpy(&TX_buffer_B[N_BYTES_TX_BUFFER - 10], EOM, N_BYTES_EOM);

  // Millis and micros timer
  //SysTick_Config(SystemCoreClock / 1000);
  SysTick->LOAD = (uint32_t) (SystemCoreClock / 1000 - 1UL);
  SysTick->VAL  = 0UL;
  SysTick->CTRL =
    SysTick_CTRL_CLKSOURCE_Msk |
    SysTick_CTRL_TICKINT_Msk |
    SysTick_CTRL_ENABLE_Msk;

  // Initial waveform LUT
  set_wave(Sine);
  set_freq(250.0); // [Hz]    Wanted startup frequency
  set_offs(1.65);  // [V]     Wanted startup offset
  set_VRMS(0.5);   // [V_RMS] Wanted startup amplitude
  compute_LUT(LUT_wave);

  // Will send out TX_buffer over SERCOM when triggered
  configure_DMA_1();
  configure_DMA_2();

  // Init DAC and ADC
  init_LIA();

  while (1) {
    // Process commands on the data channel
    if (is_serial_rxc) {
      is_serial_rxc = false;

      if (scl_available(&scl_1)) {

        if (is_running) { // Atomic read, `noInterrupts()` not required here
          /*--------------
            Running
          ----------------
            Any command received while running will switch the lock-in amp off.
            The command string will not be checked in advance, because this
            causes a lot of overhead, during which time the Arduino's serial-out
            buffer could potentially flood the serial-in buffer at the PC side.
            This will happen when the PC is not reading (and depleting) the
            in-buffer as fast as possible because it is now waiting for the
            'off' reply to occur.
          */
          is_running = false;
          LED_off();
          TRIG_OUT_off();

          // Disable DAC output
          uint16_t DAC_OUTPUT_OFF[1] = {0};
          dac_async_write(&DAC_0, 0, DAC_OUTPUT_OFF, 1);

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
          /*-------------
            Not running
          ---------------
            We are ready to process any incoming commands.
          */
          str_cmd = scl_get_command(&scl_1);

          if (strcmp(str_cmd, "id?") == 0) {
            // Report identity string
            io_print("Arduino, Alia\n");

          } else if (strcmp(str_cmd, "mcu?") == 0) {
            // Report microcontroller information
            snprintf(str_buffer,
                     MAXLEN_STR_BUFFER,
                     "%s\t%s\t%lu\t%s\n",
                     FIRMWARE_VERSION,
                     MCU_MODEL,
                     VARIANT_MCK,
                     mcu_uid);
            io_print(str_buffer);

          } else if (strcmp(str_cmd, "adc?") == 0) {
            // Report ADC registers
            io_print("----------------------------------------\n");
            io_print("CTRLA\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  F%u\n" , ".ENABLE"   , ADC->CTRLA.bit.ENABLE); io_print(str_buffer);
            io_print("INPUTCTRL\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".MUXPOS"   , ADC->INPUTCTRL.bit.MUXPOS); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".MUXNEG"   , ADC->INPUTCTRL.bit.MUXNEG); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".GAIN"     , ADC->INPUTCTRL.bit.GAIN); io_print(str_buffer);
            io_print("REFCTRL\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  F%u\n" , ".REFCOMP"  , ADC->REFCTRL.bit.REFCOMP); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".REFSEL"   , ADC->REFCTRL.bit.REFSEL); io_print(str_buffer);
            io_print("AVGCTRL\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".ADJRES"   , ADC->AVGCTRL.bit.ADJRES); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".SAMPLENUM", ADC->AVGCTRL.bit.SAMPLENUM); io_print(str_buffer);
            io_print("SAMPCTRL\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  %u\n"  , ".SAMPLEN"  , ADC->SAMPCTRL.bit.SAMPLEN); io_print(str_buffer);
            io_print("CTRLB\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".RESSEL"   , ADC->CTRLB.bit.RESSEL); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  F%u\n" , ".CORREN"   , ADC->CTRLB.bit.CORREN); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  F%u\n" , ".LEFTADJ"  , ADC->CTRLB.bit.LEFTADJ); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  F%u\n" , ".DIFFMODE" , ADC->CTRLB.bit.DIFFMODE); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  0x%X\n", ".PRESCALER", ADC->CTRLB.bit.PRESCALER); io_print(str_buffer);
            io_print("CALIB\n");
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  %u\n"  , ".LINEARITY_CAL", ADC->CALIB.bit.LINEARITY_CAL); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "%15s  %u\n"  , ".BIAS_CAL"     , ADC->CALIB.bit.BIAS_CAL); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "OFFSETCORR       %u\n"         , ADC->OFFSETCORR.bit.OFFSETCORR); io_print(str_buffer);
            snprintf(str_buffer, MAXLEN_STR_BUFFER, "GAINCORR         %u\n"         , ADC->GAINCORR.bit.GAINCORR); io_print(str_buffer);
            io_print("----------------------------------------\n");

          } else if (strcmp(str_cmd, "const?") == 0) {
            // Report lock-in amplifier constants
            snprintf(str_buffer, MAXLEN_STR_BUFFER,
                     "%.6f\t%u\t%u\t%u\t%u\t%u\t%.3f\t%u\t%u\n",
                     SAMPLING_PERIOD_us,
                     BLOCK_SIZE,
                     N_BYTES_TX_BUFFER,
                     DAC_OUTPUT_BITS,
                     ADC_INPUT_BITS,
                     ADC_DIFFERENTIAL,
                     A_REF,
                     MIN_N_LUT,
                     MAX_N_LUT);
            io_print(str_buffer);

          } else if (strcmp(str_cmd, "ref?") == 0 ||
                     strcmp(str_cmd, "?") == 0) {
            // (Re)compute the LUT based on the current reference signal settings
            compute_LUT(LUT_wave);

            // Report reference signal `ref_X` settings
            sprintf(str_buffer, "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%i\t%i\t%i\n",
                    WAVEFORM_STRING[ref_waveform],
                    ref_freq,
                    ref_offs,
                    ref_ampl,
                    ref_VRMS,
                    ref_is_clipping_HI,
                    ref_is_clipping_LO,
                    N_LUT);
            io_print(str_buffer);

          } else if (strcmp(str_cmd, "lut?") == 0 ||
                     strcmp(str_cmd, "l?") == 0) {
            // HAS BECOME OBSOLETE
            // Report the LUT as a binary stream. The reported LUT will start at
            // phase = 0 deg.
            io_write_blocking((uint8_t *) &N_LUT, 2);
            io_write_blocking((uint8_t *) &is_LUT_dirty, 1);
            io_write_blocking((uint8_t *) LUT_wave, N_LUT * 2);

          } else if (strcmp(str_cmd, "lut_ascii?") == 0 ||
                     strcmp(str_cmd, "la?") == 0) {
            // Report the LUT as tab-delimited ASCII. The reported LUT will start
            // at phase = 0 deg. Convenience function handy for debugging from a
            // serial console.
            uint16_t i;

            sprintf(str_buffer, "%u\t%i\n", N_LUT, is_LUT_dirty);
            io_print(str_buffer);

            for (i = 0; i < N_LUT - 1; i++) {
              sprintf(str_buffer, "%u\t", LUT_wave[i]);
              io_print(str_buffer);
            }
            sprintf(str_buffer, "%u\n", LUT_wave[N_LUT - 1]);
            io_print(str_buffer);

          } else if (strcmp(str_cmd, "t?") == 0) {
            // Report time in microseconds
            io_print_timestamp();

          } else if (strcmp(str_cmd, "off") == 0) {
            // Lock-in amp is already off and we reply with an acknowledgment
            io_print("already_off\n");

          } else if (strcmp(str_cmd, "on") == 0) {
            // Start lock-in amp
            start_LIA();

          } else if (strcmp(str_cmd, "_on") == 0) {
            // Start lock-in amp and reset the time
            trigger_reset_time = true;
            start_LIA();

          } else if (strncmp(str_cmd, "_wave", 5) == 0) {
            // Set the waveform type of the reference signal.
            // Call 'compute_LUT(LUT_wave)' for it to become effective.
            set_wave(atoi(&str_cmd[5]));
            sprintf(str_buffer, "%s\n", WAVEFORM_STRING[ref_waveform]);
            io_print(str_buffer);

          } else if (strncmp(str_cmd, "_freq", 5) == 0) {
            /// Set frequency of the reference signal [Hz].
            // Call 'compute_LUT(LUT_wave)' for it to become effective.
            set_freq(atof(&str_cmd[5]));
            sprintf(str_buffer, "%.3f\n", ref_freq);
            io_print(str_buffer);

          } else if (strncmp(str_cmd, "_offs", 5) == 0) {
            // Set offset of the reference signal [V].
            // Call 'compute_LUT(LUT_wave)' for it to become effective.
            set_offs(atof(&str_cmd[5]));
            sprintf(str_buffer, "%.3f\n", ref_offs);
            io_print(str_buffer);

          } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
            // Set amplitude of the reference signal [V].
            // Call 'compute_LUT(LUT_wave)' for it to become effective.
            set_ampl(atof(&str_cmd[5]));
            sprintf(str_buffer, "%.3f\n", ref_ampl);
            io_print(str_buffer);

          } else if (strncmp(str_cmd, "_vrms", 5) == 0) {
            // Set amplitude of the reference signal [V_RMS].
            // Call 'compute_LUT(LUT_wave)' for it to become effective.
            set_VRMS(atof(&str_cmd[5]));
            sprintf(str_buffer, "%.3f\n", ref_VRMS);
            io_print(str_buffer);
          }
        }
      }
    }
  }
}
