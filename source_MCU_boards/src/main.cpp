/*------------------------------------------------------------------------------
Arduino lock-in amplifier

Pins:
A0: output reference signal
A1: input signal, differential +
A2: input signal, differential -

Boards                        | MCU        | tested | #define
-----------------------------------------------------------------
M0 family
- Arduino M0                    SAMD21G18A            ARDUINO_SAMD_ZERO
- Arduino M0 Pro                SAMD21G18A   okay     ARDUINO_SAMD_ZERO
- Adafruit Metro M0             SAMD21G18A
- Adafruit Feather M0           SAMD21G18A
- Adafruit ItsyBitsy M0         SAMD21G18A
- Adafruit Trinket M0           SAMD21E18A
- Adafruit Gemma M0             SAMD21E18A

M4 family
- Adafruit Grand Central M4     SAMD51P20A
- Adafruit NeoTrellis M4        SAMD51J19A?
- Adafruit Metro M4             SAMD51J19A            ADAFRUIT_METRO_M4_EXPRESS
- Adafruit Feather M4           SAMD51J19A   okay ADAFRUIT_FEATHER_M4_EXPRESS
- Adafruit ItsyBitsy M4         SAMD51G19A   okay ADAFRUIT_ITSYBITSY_M4_EXPRESS

Dennis van Gils
18-06-2021
------------------------------------------------------------------------------*/

#include "DvG_SerialCommand.h"
#include "Streaming.h"
#include <Arduino.h>


#define FIRMWARE_VERSION "ALIA v0.3.0 VSCODE"

// Define for writing debugging info to the terminal of the second serial port.
// Note: The board needs a second serial port to be used besides the main serial
// port which is assigned to sending buffers of lock-in amp data.
#if defined(ARDUINO_SAMD_ZERO)
//#define DEBUG
#endif

// Interrupt timers
#if defined(__SAMD21G18A__) || defined(__SAMD21E18A__)
#ifndef __SAMD21__
#define __SAMD21__
#endif
#include "ZeroTimer.h"
#elif defined(__SAMD51P20A__) || defined(__SAMD51J19A__) ||                    \
    defined(__SAMD51G19A__)
#ifndef __SAMD51__
#define __SAMD51__
#endif
#include "SAMD51_InterruptTimer.h"
#endif

// No-operation, to burn cycles inside the interrupt service routine
#define NOP __asm("nop");

volatile bool is_running = false; // Is the lock-in amplifier running?
uint8_t
    mcu_uid[16]; // Microcontroller unit (mcu) unique identifier (uid) number

// Preprocessor trick to ensure enums and strings are in sync, so one can write
// 'WAVEFORM_STRING[Cosine]' to give the string 'Cosine'
#define FOREACH_WAVEFORM(WAVEFORM)                                             \
  WAVEFORM(Cosine)                                                             \
  WAVEFORM(Square)                                                             \
  WAVEFORM(Triangle)                                                           \
  WAVEFORM(END_WAVEFORM_ENUM)
#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

enum WAVEFORM_ENUM { FOREACH_WAVEFORM(GENERATE_ENUM) };

static const char *WAVEFORM_STRING[] = {FOREACH_WAVEFORM(GENERATE_STRING)};

/*------------------------------------------------------------------------------
    Waveform look-up table (LUT)
------------------------------------------------------------------------------*/

// Output reference signal parameters
enum WAVEFORM_ENUM ref_waveform = Cosine;
double ref_freq; // [Hz] Obtained frequency of reference signal
double ref_offs; // [V]  Obtained voltage offset of reference signal
double ref_ampl; // [V]  Voltage amplitude reference signal

// Look-up table (LUT) for fast DAC
#define MIN_N_LUT 20   // Min. allowed number of samples for one full period
#define MAX_N_LUT 1000 // Max. allowed number of samples for one full period
uint16_t LUT_wave[MAX_N_LUT] = {0}; // Look-up table allocation
uint16_t N_LUT;            // Current number of samples for one full period
bool is_LUT_dirty = false; // Does the LUT have to be updated with new settings?

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define ADC_INPUT_BITS 12
#if defined(__SAMD21__)
#define DAC_OUTPUT_BITS 10
#elif defined(__SAMD51__)
#define DAC_OUTPUT_BITS 12
#endif
#define MAX_DAC_OUTPUT_BITVAL ((uint16_t)(pow(2, DAC_OUTPUT_BITS) - 1))

/*------------------------------------------------------------------------------
    Timing
------------------------------------------------------------------------------*/

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static __inline__ void syncADC() __attribute__((always_inline, unused));
#if defined(__SAMD21__)
static void syncDAC() {
  while (DAC->STATUS.bit.SYNCBUSY == 1)
    ;
}
static void syncADC() {
  while (ADC->STATUS.bit.SYNCBUSY == 1)
    ;
}
#elif defined(__SAMD51__)
static void syncDAC() {
  while (DAC->STATUS.bit.EOC0 == 1)
    ;
}
static void syncADC() {
  while (ADC0->STATUS.bit.ADCBUSY == 1)
    ;
}
#endif

// Interrupt service routine clock
// Findings using Arduino M0 Pro (legacy notes):
//   SAMPLING_PERIOD_us:
//      min.  40 usec for only writing A0, no serial
//      min.  50 usec for writing A0 and reading A1, no serial
//      min.  80 usec for writing A0 and reading A1, with serial
#define SAMPLING_PERIOD_us 100
const double SAMPLING_RATE_Hz = (double)1.0e6 / SAMPLING_PERIOD_us;

/*------------------------------------------------------------------------------
    Double buffer: TX_buffer_A & TX_buffer_B
------------------------------------------------------------------------------*/

// The buffer that will be send each transmission is BLOCK_SIZE samples long.
// Double the amount of memory is reserved to employ a double buffer technique,
// where alternatingly buffer A is being written to and buffer B is being sent.

// The number of samples to acquire by the ADC and to subsequently send out
// over serial as a single block of data
#define BLOCK_SIZE 1000 // [# samples], where 1 sample takes up 16 bits

/* Tested settings Arduino M0 Pro (legacy notes)
Case A: Turbo and stable on computer Onera, while only graphing and logging in
        Python without FIR filtering
            SAMPLING_PERIOD_us   80
            BLOCK_SIZE          625
            DAQ --> 12500 Hz
Case B: Stable on computer Onera, while graphing, logging and FIR filtering in
        Python
            SAMPLING_PERIOD_us  200
            BLOCK_SIZE          500
            DAQ --> 5000 Hz
*/

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

/* A sent-out serial transmit buffer will contain a single block of data:
 [SOM,                                              {size = 10 bytes}
  (uint32_t) number of block being send             {size =  4 bytes}
  (uint32_t) millis timestamp at start of block     {size =  4 bytes}
  (uint16_t) micros part of timestamp               {size =  2 bytes}
  (uint16_t) phase index LUT_wave at start of block {size =  2 bytes}
  BLOCK_SIZE x (uint16_t) ADC readings 'sig_I'      {size = BLOCK_SIZE * 2
 bytes}
  EOM]                                              {size = 10 bytes}
*/

// clang-format off
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

volatile static uint8_t TX_buffer_A[N_BYTES_TX_BUFFER] = {0};
volatile static uint8_t TX_buffer_B[N_BYTES_TX_BUFFER] = {0};
volatile uint32_t TX_buffer_counter = 0;
volatile bool trigger_send_TX_buffer_A = false;
volatile bool trigger_send_TX_buffer_B = false;

#define TX_BUFFER_OFFSET_COUNTER (N_BYTES_SOM)
#define TX_BUFFER_OFFSET_MILLIS  (TX_BUFFER_OFFSET_COUNTER + N_BYTES_COUNTER)
#define TX_BUFFER_OFFSET_MICROS  (TX_BUFFER_OFFSET_MILLIS  + N_BYTES_MILLIS)
#define TX_BUFFER_OFFSET_PHASE   (TX_BUFFER_OFFSET_MICROS  + N_BYTES_MICROS)
#define TX_BUFFER_OFFSET_SIG_I   (TX_BUFFER_OFFSET_PHASE   + N_BYTES_PHASE)
// clang-format on

/*------------------------------------------------------------------------------
    Serial
------------------------------------------------------------------------------*/

// Arduino M0 Pro
// Serial   : Programming USB port (UART).
// SerialUSB: Native USB port (USART). Baudrate setting gets ignored and is
//            always as fast as possible.
/*
   *** Tested scenarios (legacy notes)
   SAMPLING_PERIOD_us   200 [usec]
   BLOCK_SIZE           500 [samples]
   BAUDRATE             1e6 [only used when #define Ser_data Serial]
   (a)
      #define Ser_data Serial
      Regardless of #define DEBUG
      Only connect USB cable to programming port
      --> Perfect timing. Timestamp jitter 0 usec
   (b)
      #define Ser_data Serial
      Regardless of #define DEBUG
      Both USB cables to programming port and native port
      --> Timestamp jitter +\- 3 usec
   (c)
      #define Ser_data SerialUSB
      Regardless of #define DEBUG
      Only connect USB cable to native port
      --> Timestamp jitter +\- 4 usec
   (d)
      #define Ser_data SerialUSB
      Regardless of #define DEBUG
      Both USB cables to programming port and native port
      --> Timestamp jitter +\- 4 usec
*/
#define SERIAL_DATA_BAUDRATE 1e6 // Only used when '#define Ser_data Serial'

#if defined(ARDUINO_SAMD_ZERO)
#define Ser_data SerialUSB
#ifdef DEBUG
#define Ser_debug Serial
#endif
#else
#define Ser_data Serial
#endif

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);

char str_buffer[96];

/*------------------------------------------------------------------------------
    get_mcu_uid
------------------------------------------------------------------------------*/

void get_mcu_uid(uint8_t raw_uid[16]) {
/* Return the 128-bits uid (serial number) of the micro controller as a byte
array.
*/

#ifdef _SAMD21_
// SAMD21 from section 9.3.3 of the datasheet
#define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x0080A00C)
#define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x0080A040)
#define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x0080A044)
#define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x0080A048)
#endif
#ifdef _SAMD51_
// SAMD51 from section 9.6 of the datasheet
#define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x008061FC)
#define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x00806010)
#define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x00806014)
#define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x00806018)
#endif

  uint32_t pdwUniqueID[4];
  pdwUniqueID[0] = SERIAL_NUMBER_WORD_0;
  pdwUniqueID[1] = SERIAL_NUMBER_WORD_1;
  pdwUniqueID[2] = SERIAL_NUMBER_WORD_2;
  pdwUniqueID[3] = SERIAL_NUMBER_WORD_3;

  for (int i = 0; i < 4; i++) {
    raw_uid[i * 4 + 0] = (uint8_t)(pdwUniqueID[i] >> 24);
    raw_uid[i * 4 + 1] = (uint8_t)(pdwUniqueID[i] >> 16);
    raw_uid[i * 4 + 2] = (uint8_t)(pdwUniqueID[i] >> 8);
    raw_uid[i * 4 + 3] = (uint8_t)(pdwUniqueID[i] >> 0);
  }
}

/*------------------------------------------------------------------------------
    Waveform look-up table (LUT)
------------------------------------------------------------------------------*/
/* In order to drive the DAC at high sampling speeds, we compute the reference
waveform in advance by a look-up table (LUT). The LUT will contain the samples
for one complete period of the waveform. The LUT is statically allocated and can
fit up to 'MAX_N_LUT' number of samples.

Because the 'SAMPLING_PERIOD_us' is fixed and the LUT can only have an integer
number of samples 'N_LUT', the possible wave frequencies are discrete.
That means that there is a distinction between the wanted frequency and the
obtained frequency 'ref_freq'.
*/

void parse_freq(const char *str_value) {
  ref_freq = atof(str_value);
  N_LUT = (uint16_t)round(SAMPLING_RATE_Hz / ref_freq);
  N_LUT = max(N_LUT, MIN_N_LUT);
  N_LUT = min(N_LUT, MAX_N_LUT);
  ref_freq = SAMPLING_RATE_Hz / N_LUT;
}

void parse_offs(const char *str_value) {
  ref_offs = atof(str_value);
  ref_offs = max(ref_offs, 0.0);
  ref_offs = min(ref_offs, A_REF);
}

void parse_ampl(const char *str_value) {
  ref_ampl = atof(str_value);
  ref_ampl = max(ref_ampl, 0.0);
  ref_ampl = min(ref_ampl, A_REF);
}

void compute_LUT(uint16_t *LUT_array) {
  double norm_offs = ref_offs / A_REF; // Normalized
  double norm_ampl = ref_ampl / A_REF; // Normalized
  double wave;

#ifdef DEBUG
  Ser_debug << "Creating LUT...";
#endif

  // Generate normalized waveform periods in the range [0, 1].
  for (int16_t i = 0; i < N_LUT; i++) {
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
    LUT_array[i] = (uint16_t)round(MAX_DAC_OUTPUT_BITVAL * wave);
  }

  is_LUT_dirty = false;

#ifdef DEBUG
  Ser_debug << " done." << endl;
#endif
}

/*------------------------------------------------------------------------------
    Time keeping
------------------------------------------------------------------------------*/

void get_systick_timestamp(uint32_t *stamp_millis,
                           uint16_t *stamp_micros_part) {
  /* Adapted from:
  https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/delay.c
  Note: The millis counter will roll over after 49.7 days.
  */
  uint32_t ticks, ticks2;
  uint32_t pend, pend2;
  uint32_t count, count2;
  uint32_t _ulTickCount = millis();

  ticks2 = SysTick->VAL;
  pend2 = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
  count2 = _ulTickCount;

  do {
    ticks = ticks2;
    pend = pend2;
    count = count2;
    ticks2 = SysTick->VAL;
    pend2 = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
    count2 = _ulTickCount;
  } while ((pend != pend2) || (count != count2) || (ticks < ticks2));

  (*stamp_millis) = count2;
  if (pend) {
    (*stamp_millis)++;
  }
  (*stamp_micros_part) =
      (((SysTick->LOAD - ticks) * (1048576 / (VARIANT_MCK / 1000000))) >> 20);
}

void stamp_TX_buffer(volatile uint8_t *TX_buffer, volatile uint16_t *LUT_idx) {
  /* Write timestamp and 'phase'-stamp of the first ADC sample of the block
  that is about to be sent out over the serial port. We need to know which
  phase angle was output on the DAC, that corresponds in time to the first ADC
  sample of the TX_buffer. This is the essence of a phase-sensitive detector,
  which is the building block of a lock-in amplifier.
  */

  uint32_t millis_copy;
  uint16_t micros_part;
  get_systick_timestamp(&millis_copy, &micros_part);

  // clang-format off
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
  TX_buffer[TX_BUFFER_OFFSET_PHASE      ] = *LUT_idx;
  TX_buffer[TX_BUFFER_OFFSET_PHASE   + 1] = *LUT_idx >> 8;
  // clang-format on
}

/*------------------------------------------------------------------------------
    Interrupt service routine (isr) for phase-sentive detection (psd)
------------------------------------------------------------------------------*/

void isr_psd() {
  static bool is_running_prev = is_running;
  static uint8_t startup_counter = 0;
  static bool using_TX_buffer_A = true; // When false: Using TX_buffer_B
  static uint16_t write_idx;            // Current write index of TX_buffer
  volatile static uint16_t LUT_idx;     // Current read index of LUT
  uint16_t ref_X;
  int16_t sig_I;

  if (is_running != is_running_prev) {
    is_running_prev = is_running;
    if (is_running) {
      digitalWrite(PIN_LED, HIGH); // Indicate lock-in amp is running
      startup_counter = 0;
    } else {
      digitalWrite(PIN_LED, LOW); // Indicate lock-in amp is off
      syncDAC();
#if defined(__SAMD21__)
      DAC->DATA.reg = 0; // Set output voltage to 0
#elif defined(__SAMD51__)
      DAC->DATA[0].reg = 0; // Set output voltage to 0
#endif
      syncDAC();
    }
  }
  if (!is_running) {
    return;
  }

  if (startup_counter == 0) {
    write_idx = 0;
    LUT_idx = N_LUT - 1;
    using_TX_buffer_A = true;
    trigger_send_TX_buffer_A = false;
    trigger_send_TX_buffer_B = false;

    // Nudge DAC and ADC somewhat into synchronization
    uint16_t nop_delay_us = 1; // [us]
    uint32_t nop_cycles_us = SystemCoreClock / 1.0e6 * nop_delay_us;
    syncDAC();
    for (uint32_t i = 0; i < 1 * nop_cycles_us; i++) {
      NOP;
    }
    syncADC();
    for (uint32_t i = 0; i < 2 * nop_cycles_us; i++) {
      NOP;
    }
  }

  // Read input signal corresponding to the DAC output of the previous timestep.
  // This ensures that the previously set DAC output has had enough time to
  // stabilize.
  syncADC();
#if defined(__SAMD21__)
  // ADC->SWTRIG.bit.START = 1;
  // while (ADC->INTFLAG.bit.RESRDY == 0);   // Wait for conversion to complete
  // syncADC();
  // sig_I = ADC->RESULT.reg;
  ADC->SWTRIG.bit.START = 1;
  while (ADC->INTFLAG.bit.RESRDY == 0)
    ; // Wait for conversion to complete
  syncADC();
  sig_I = ADC->RESULT.reg;
#elif defined(__SAMD51__)
  ADC0->SWTRIG.bit.START = 1;
  while (ADC0->INTFLAG.bit.RESRDY == 0)
    ; // Wait for conversion to complete
  syncADC();
  sig_I = ADC0->RESULT.reg;
#endif
  // syncADC(); // NOT NECESSARY

  // Output reference signal
  ref_X = LUT_wave[LUT_idx];
// syncDAC(); // DON'T ENABLE: Causes timing jitter in the output waveform
#if defined(__SAMD21__)
  DAC->DATA.reg = ref_X;
#elif defined(__SAMD51__)
  DAC->DATA[0].reg = ref_X;
#endif
  syncDAC();

  if (startup_counter == 0) {
    // No valid input signal yet, hence return. Next timestep it will be valid.
    startup_counter++;
    return;
  } else if (startup_counter == 1) {
    // DAC and ADC should have become finally synchronized in this step
    startup_counter++;
    stamp_TX_buffer(TX_buffer_A, &LUT_idx);
    LUT_idx = 0;
    return;
  }

  // Store the input signal
  // clang-format off
  if (using_TX_buffer_A) {
    TX_buffer_A[TX_BUFFER_OFFSET_SIG_I + write_idx * 2    ] = sig_I;
    TX_buffer_A[TX_BUFFER_OFFSET_SIG_I + write_idx * 2 + 1] = sig_I >> 8;
  } else {
    TX_buffer_B[TX_BUFFER_OFFSET_SIG_I + write_idx * 2    ] = sig_I;
    TX_buffer_B[TX_BUFFER_OFFSET_SIG_I + write_idx * 2 + 1] = sig_I >> 8;
  }
  write_idx++;
  // clang-format on

  // Ready to send the buffer?
  if (write_idx == BLOCK_SIZE) {
    if (using_TX_buffer_A) {
      trigger_send_TX_buffer_A = true;
      stamp_TX_buffer(TX_buffer_B, &LUT_idx);
    } else {
      trigger_send_TX_buffer_B = true;
      stamp_TX_buffer(TX_buffer_A, &LUT_idx);
    }

    using_TX_buffer_A = !using_TX_buffer_A;
    write_idx = 0;
  }

  // Advance the reference signal waveform
  LUT_idx++;
  if (LUT_idx == N_LUT) {
    LUT_idx = 0;
  }
}

/*------------------------------------------------------------------------------
    Print debug information to the terminal
------------------------------------------------------------------------------*/

#ifdef DEBUG
#if defined(__SAMD21__)
void print_debug_info() {
  Ser_debug << "-------------------------------" << endl;
  Ser_debug << "CTRLA" << endl;
  Ser_debug << "  .RUNSTDBY   : " << _HEX(ADC->CTRLA.bit.RUNSTDBY) << endl;
  Ser_debug << "  .ENABLE     : " << _HEX(ADC->CTRLA.bit.ENABLE) << endl;
  Ser_debug << "  .SWRST      : " << _HEX(ADC->CTRLA.bit.SWRST) << endl;
  Ser_debug << "REFCTRL" << endl;
  Ser_debug << "  .REFCOMP    : " << _HEX(ADC->REFCTRL.bit.REFCOMP) << endl;
  Ser_debug << "  .REFSEL     : " << _HEX(ADC->REFCTRL.bit.REFSEL) << endl;
  Ser_debug << "AVGVTRL" << endl;
  Ser_debug << "  .ADJRES     : " << _HEX(ADC->AVGCTRL.bit.ADJRES) << endl;
  Ser_debug << "  .SAMPLENUM  : " << _HEX(ADC->AVGCTRL.bit.SAMPLENUM) << endl;
  Ser_debug << "SAMPCTRL" << endl;
  Ser_debug << "  .SAMPLEN    : " << _HEX(ADC->SAMPCTRL.bit.SAMPLEN) << endl;
  Ser_debug << "CTRLB" << endl;
  Ser_debug << "  .PRESCALER  : " << _HEX(ADC->CTRLB.bit.PRESCALER) << endl;
  Ser_debug << "  .RESSEL     : " << _HEX(ADC->CTRLB.bit.RESSEL) << endl;
  Ser_debug << "  .CORREN     : " << _HEX(ADC->CTRLB.bit.CORREN) << endl;
  Ser_debug << "  .FREERUN    : " << _HEX(ADC->CTRLB.bit.FREERUN) << endl;
  Ser_debug << "  .LEFTADJ    : " << _HEX(ADC->CTRLB.bit.LEFTADJ) << endl;
  Ser_debug << "  .DIFFMODE   : " << _HEX(ADC->CTRLB.bit.DIFFMODE) << endl;
  Ser_debug << "INPUTCTRL" << endl;
  Ser_debug << "  .GAIN       : " << _HEX(ADC->INPUTCTRL.bit.GAIN) << endl;
  Ser_debug << "  .INPUTOFFSET: " << _HEX(ADC->INPUTCTRL.bit.INPUTOFFSET)
            << endl;
  Ser_debug << "  .INPUTSCAN  : " << _HEX(ADC->INPUTCTRL.bit.INPUTSCAN) << endl;
  Ser_debug << "  .MUXNEG     : " << _HEX(ADC->INPUTCTRL.bit.MUXNEG) << endl;
  Ser_debug << "  .MUXPOS     : " << _HEX(ADC->INPUTCTRL.bit.MUXPOS) << endl;
#elif defined(__SAMD51__)
// TO DO
#endif

  float DAQ_rate = 1.0e6 / SAMPLING_PERIOD_us;
  float block_rate = DAQ_rate / BLOCK_SIZE;
  // 8 data bits + 1 start bit + 1 stop bit = 10 bits per data byte
  uint32_t baud = ceil(N_BYTES_TX_BUFFER * 10 * block_rate);
  Ser_debug << "----------------------------------------" << endl;
  Ser_debug << "DAQ rate     : " << _FLOAT(DAQ_rate, 2) << " Hz" << endl;
  Ser_debug << "ISR clock    : " << SAMPLING_PERIOD_us << " usec" << endl;
  Ser_debug << "Block size   : " << BLOCK_SIZE << " samples" << endl;
  Ser_debug << "Transmit rate          : " << _FLOAT(block_rate, 2)
            << " blocks/s" << endl;
  Ser_debug << "Data bytes per transmit: " << N_BYTES_TX_BUFFER << " bytes"
            << endl;
  Ser_debug << "Lower bound baudrate   : " << baud << endl;
  Ser_debug << "----------------------------------------" << endl;
}
#endif

/*------------------------------------------------------------------------------
    setup
------------------------------------------------------------------------------*/

uint8_t NVM_ADC0_BIASCOMP = 0;
uint8_t NVM_ADC0_BIASREFBUF = 0;
uint8_t NVM_ADC0_BIASR2R = 0;

void setup() {
#ifdef DEBUG
  Ser_debug.begin(9600);
#endif

#if Ser_data == Serial
  Ser_data.begin(SERIAL_DATA_BAUDRATE);
#else
  Ser_data.begin(9600);
#endif

  // Microcontroller unit (MCU) unique identifier (uid) number
  get_mcu_uid(mcu_uid);

  // Use built-in LED to signal running state of lock-in amp
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, is_running);

  // Prepare SOM and EOM
  noInterrupts();
  for (uint8_t i = 0; i < N_BYTES_SOM; i++) {
    TX_buffer_A[i] = SOM[i];
    TX_buffer_B[i] = SOM[i];
  }
  for (uint8_t i = 0; i < N_BYTES_EOM; i++) {
    TX_buffer_A[N_BYTES_TX_BUFFER - N_BYTES_EOM + i] = EOM[i];
    TX_buffer_B[N_BYTES_TX_BUFFER - N_BYTES_EOM + i] = EOM[i];
  }
  interrupts();

  /*
  // Disabled because `memcpy` does not operate on volatiles
  noInterrupts();
  memcpy(TX_buffer_A                         , SOM, N_BYTES_SOM);
  memcpy(&TX_buffer_A[N_BYTES_TX_BUFFER - 10], EOM, N_BYTES_EOM);
  memcpy(TX_buffer_B                         , SOM, N_BYTES_SOM);
  memcpy(&TX_buffer_B[N_BYTES_TX_BUFFER - 10], EOM, N_BYTES_EOM);
  interrupts();
  */

  // DAC
  analogWriteResolution(DAC_OUTPUT_BITS);
  analogWrite(A0, 0);

// ADC
// Increase the ADC clock by setting the divisor from default DIV128 to DIV16.
// Setting smaller divisors than DIV16 results in ADC errors.
#if defined(__SAMD21__)
  ADC->CTRLB.bit.PRESCALER = ADC_CTRLB_PRESCALER_DIV16_Val;
#elif defined(__SAMD51__)
  ADC0->CTRLA.bit.PRESCALER = ADC_CTRLA_PRESCALER_DIV16_Val;
#endif
  analogReadResolution(ADC_INPUT_BITS);
  analogRead(A1); // Differential +
  analogRead(A2); // Differential -

// Set differential mode on A1(+) and A2(-)
#if defined(__SAMD21__)
  ADC->CTRLB.bit.DIFFMODE = 1;
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  ADC->INPUTCTRL.bit.MUXNEG = g_APinDescription[A2].ulADCChannelNumber;
  ADC->INPUTCTRL.bit.GAIN = ADC_INPUTCTRL_GAIN_DIV2_Val;
  ADC->REFCTRL.bit.REFSEL = 2; // 2: INTVCC1 on SAMD21 = 1/2 VDDANA
#elif defined(__SAMD51__)
  /*
  ADC0->CTRLA.bit.ENABLE = 0;
  delay(10);

  // NVM Software Calibration Area: address 0x00800080
  uint16_t *NVM_SCA = NULL;
  NVM_SCA = (uint16_t*) 0x00800080ul;
  NVM_ADC0_BIASCOMP = (*NVM_SCA & 0x1c) >> 2;
  NVM_ADC0_BIASREFBUF = (*NVM_SCA & 0xe0) >> 5;
  NVM_ADC0_BIASR2R = (*NVM_SCA & 0x700) >> 8;
  ADC0->CALIB.bit.BIASCOMP   = NVM_ADC0_BIASCOMP;
  ADC0->CALIB.bit.BIASREFBUF = NVM_ADC0_BIASREFBUF;
  ADC0->CALIB.bit.BIASR2R    = NVM_ADC0_BIASR2R;
  delay(10);
  */

  ADC0->INPUTCTRL.bit.DIFFMODE = 1;
  ADC0->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  ADC0->INPUTCTRL.bit.MUXNEG = g_APinDescription[A2].ulADCChannelNumber;
  // ADC0->INPUTCTRL.bit.GAIN does not exist on SAMD51
  ADC0->REFCTRL.bit.REFSEL = 3; // 3: INTVCC1 on SAMD51 = VDDANA

/*
ADC0->OFFSETCORR.bit.OFFSETCORR = ADC_OFFSETCORR_OFFSETCORR(50);
ADC0->GAINCORR.bit.GAINCORR = ADC_GAINCORR_GAINCORR(2065);
ADC0->CTRLB.bit.CORREN = 1;   // Enable offset and gain correction

ADC0->CTRLA.bit.ENABLE = 1;
delay(10);
*/
#endif

  // Prepare for software-triggered acquisition
  syncADC();
#if defined(__SAMD21__)
  ADC->CTRLA.bit.ENABLE = 0x01;
#elif defined(__SAMD51__)
  ADC0->CTRLA.bit.ENABLE = 0x01;
#endif
  syncADC();
#if defined(__SAMD21__)
  ADC->SWTRIG.bit.START = 1;
  ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY;
#elif defined(__SAMD51__)
  ADC0->SWTRIG.bit.START = 1;
  ADC0->INTFLAG.reg = ADC_INTFLAG_RESRDY;
#endif

#ifdef DEBUG
  print_debug_info();
#endif

  // LUT
  parse_freq("250.0"); // [Hz] Wanted startup frequency
  parse_offs("1.7");   // [V]  Wanted startup offset
  parse_ampl("1.414"); // [V]  Wanted startup amplitude
  compute_LUT(LUT_wave);

  // Start the interrupt timer
  TC.startTimer(SAMPLING_PERIOD_us, isr_psd);

  // Experimental: Output a pulse train on pin 9 to act as clock source for
  // a (future) variable anti-aliasing filter IC placed in front of the ADC
  // input ports.
  // TCC_pulse_train.startTimer(10);
}

/*------------------------------------------------------------------------------
    loop
------------------------------------------------------------------------------*/

void loop() {
  char *str_cmd; // Incoming serial command string
  uint32_t now = millis();
  static uint32_t prev_millis = 0;

  // Copy of volatile
  uint8_t _TX_buffer[N_BYTES_TX_BUFFER];

  // Process commands on the data channel every N milliseconds.
  // Deliberately slowed down to improve timing stability of `isr_psd()`.
  if ((now - prev_millis) > 20) {
    prev_millis = now;

    if (sc_data.available()) {
      if (is_running) { // Atomic read, `noInterrupts()` not required here
        // -------------------
        //  Running
        // -------------------
        /* Any command received while running will switch the lock-in amp off.
           The command string will not be checked in advance, because this
           causes a lot of overhead, during which time the Arduino's serial-out
           buffer could potentially flood the serial-in buffer at the PC side.
           This will happen when the PC is not reading (and depleting) the
           in-buffer as fast as possible because it is now waiting for the 'off'
           reply to occur.
        */
        noInterrupts();
        is_running = false;
        trigger_send_TX_buffer_A = false;
        trigger_send_TX_buffer_B = false;
        interrupts();

        // Flush out any binary buffer data scheduled for sending, potentially
        // flooding the receiving buffer at the PC side if 'is_running' was not
        // switched to false fast enough.
        Ser_data.flush();

        // Confirm at the PC side that the lock-in amp is off and is not longer
        // sending binary data. The 'off' message might still be preceded with
        // some left-over binary data when being read at the PC side.
        Ser_data.print("off\n");

        // Flush out and ignore the command
        sc_data.getCmd();

#ifdef DEBUG
        Ser_debug << "OFF" << endl;
#endif
      } else {
        // -------------------
        //  Not running
        // -------------------
        str_cmd = sc_data.getCmd();

        if (strcmp(str_cmd, "id?") == 0) {
          // Report identity string
          Ser_data.println("Arduino, Alia");

        } else if (strcmp(str_cmd, "mcu?") == 0) {
          // Report microcontroller model, serial and firmware
          char str_buffer[96];
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
                   "unknown MCU");
#endif
                   );

          // Format the uid byte-array to hex representation
          str_uid[32] = 0;
          for (uint8_t j = 0; j < 16; j++)
            sprintf(&str_uid[2 * j], "%02X", mcu_uid[j]);

          snprintf(str_buffer, sizeof(str_buffer), "%s\t%s\t%s\n",
                   FIRMWARE_VERSION, str_model, str_uid);
          Ser_data.print(str_buffer);

        } else if (strcmp(str_cmd, "bias?") == 0) {
#if defined(__SAMD51__)
          Ser_data.println(NVM_ADC0_BIASCOMP);
          Ser_data.println(NVM_ADC0_BIASREFBUF);
          Ser_data.println(NVM_ADC0_BIASR2R);

          Ser_data.println(ADC0->CALIB.bit.BIASCOMP);
          Ser_data.println(ADC0->CALIB.bit.BIASREFBUF);
          Ser_data.println(ADC0->CALIB.bit.BIASR2R);

          Ser_data.println(ADC0->OFFSETCORR.bit.OFFSETCORR);
          Ser_data.println(ADC0->GAINCORR.bit.GAINCORR);
#endif

        } else if (strcmp(str_cmd, "const?") == 0) {
          Ser_data.print(SAMPLING_PERIOD_us);
          Ser_data.print('\t');
          Ser_data.print(BLOCK_SIZE);
          Ser_data.print('\t');
          Ser_data.print(N_BYTES_TX_BUFFER);
          Ser_data.print('\t');
          Ser_data.print(DAC_OUTPUT_BITS);
          Ser_data.print('\t');
          Ser_data.print(ADC_INPUT_BITS);
          Ser_data.print('\t');
          Ser_data.print(A_REF);
          Ser_data.print('\t');
          Ser_data.print(MIN_N_LUT);
          Ser_data.print('\t');
          Ser_data.print(MAX_N_LUT);
          Ser_data.print('\n');

#ifdef DEBUG
          print_debug_info();
#endif

        } else if (strcmp(str_cmd, "ref?") == 0 || strcmp(str_cmd, "?") == 0) {
          // Report reference signal settings
          Ser_data.print(ref_freq, 3);
          Ser_data.print('\t');
          Ser_data.print(ref_offs, 3);
          Ser_data.print('\t');
          Ser_data.print(ref_ampl, 3);
          Ser_data.print('\t');
          Ser_data.print(WAVEFORM_STRING[ref_waveform]);
          Ser_data.print('\t');
          Ser_data.print(N_LUT);
          Ser_data.print('\n');

        } else if (strcmp(str_cmd, "lut?") == 0 || strcmp(str_cmd, "l?") == 0) {
          // Report the LUT as a binary stream.
          // The reported LUT will start at phase = 0 deg.
          Ser_data.write((uint8_t *)&N_LUT, 2);
          Ser_data.write((uint8_t *)&is_LUT_dirty, 1);
          Ser_data.write((uint8_t *)LUT_wave, N_LUT * 2);

        } else if (strcmp(str_cmd, "lut_ascii?") == 0 ||
                   strcmp(str_cmd, "la?") == 0) {
          // Report the LUT as tab-delimited ASCII.
          // The reported LUT will start at phase = 0 deg.
          // Convenience function handy for debugging from a
          // serial console.
          uint16_t i;

          sprintf(str_buffer, "%u\t%i\n", N_LUT, is_LUT_dirty);
          Ser_data.print(str_buffer);
          for (i = 0; i < N_LUT; i++) {
            sprintf(str_buffer, "%u\t", LUT_wave[i]);
            Ser_data.print(str_buffer);
          }

        } else if (strcmp(str_cmd, "time?") == 0) {
          // Report time in microseconds
          static char buf[16];
          uint32_t millis_copy;
          uint16_t micros_part;

          get_systick_timestamp(&millis_copy, &micros_part);
          sprintf(buf, "%lu %03u\n", millis_copy, micros_part);
          Ser_data.print(buf);

        } else if (strcmp(str_cmd, "fcpu?") == 0) {
          // Report processor clock frequency
          static char buf[16];

          sprintf(buf, "%lu Hz\n", SystemCoreClock);
          Ser_data.print(buf);

        } else if (strcmp(str_cmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser_data.print("already_off\n");

#ifdef DEBUG
          Ser_debug << "Already OFF" << endl;
#endif

        } else if (strcmp(str_cmd, "on") == 0 || strcmp(str_cmd, "_on") == 0) {
          // Start lock-in amp
          noInterrupts();
          // NVIC_DisableIRQ(TC3_IRQn);
          is_running = true;
          interrupts();
// NVIC_EnableIRQ(TC3_IRQn);

#ifdef DEBUG
          Ser_debug << "ON" << endl;
#endif

        } else if (strncmp(str_cmd, "_freq", 5) == 0) {
          // Set frequency of the reference signal [Hz].
          // You still have to call 'compute_LUT(LUT_wave)' for it
          // to become effective.
          parse_freq(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_freq, 3);

        } else if (strncmp(str_cmd, "_offs", 5) == 0) {
          // Set offset of the reference signal [V].
          // You still have to call 'compute_LUT(LUT_wave)' for it
          // to become effective.
          parse_offs(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_offs, 3);

        } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
          // Set amplitude of the reference signal [V].
          // You still have to call 'compute_LUT(LUT_wave)' for it
          // to become effective.
          parse_ampl(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_ampl, 3);

        } else if (strncmp(str_cmd, "_wave", 5) == 0) {
          // Set the waveform type of the reference signal.
          // You still have to call 'compute_LUT(LUT_wave)' for it
          // to become effective.
          ref_waveform = static_cast<WAVEFORM_ENUM>(atoi(&str_cmd[5]));
          ref_waveform = static_cast<WAVEFORM_ENUM>(max(ref_waveform, 0));
          ref_waveform = static_cast<WAVEFORM_ENUM>(
              min(ref_waveform, END_WAVEFORM_ENUM - 1));
          is_LUT_dirty = true;
          Ser_data.println(WAVEFORM_STRING[ref_waveform]);

        } else if (strcmp(str_cmd, "compute_lut") == 0 ||
                   strcmp(str_cmd, "c") == 0) {
          // (Re)compute the LUT based on the following settings:
          // ref_freq, ref_offs, ref_ampl, ref_waveform.
          compute_LUT(LUT_wave);
          Ser_data.println("!"); // Reply with OK '!'
        }
      }
    }
  }

  // Send buffers over the data channel
  if (is_running && (trigger_send_TX_buffer_A || trigger_send_TX_buffer_B)) {
    /*
    // DEBUG
    Ser_data.println(TX_buffer_counter);
    Ser_data.println(
      trigger_send_TX_buffer_A ? "\nTX_buffer_A" : "\nTX_buffer_B"
    );
    */

    // Copy the volatile buffers
    noInterrupts();
    if (trigger_send_TX_buffer_A) {
      trigger_send_TX_buffer_A = false;
      // memcpy(_TX_buffer, TX_buffer_A, N_BYTES_TX_BUFFER);
      for (int16_t i = N_BYTES_TX_BUFFER - 1; i >= 0; i--) {
        _TX_buffer[i] = TX_buffer_A[i];
      }
    } else {
      trigger_send_TX_buffer_B = false;
      // memcpy(_TX_buffer, TX_buffer_B, N_BYTES_TX_BUFFER);
      for (int16_t i = N_BYTES_TX_BUFFER - 1; i >= 0; i--) {
        _TX_buffer[i] = TX_buffer_B[i];
      }
    }
    interrupts();

    // Note: `write()` can return -1 as indication of an error, e.g. the
    // receiving side being overrun with data.
    size_t w;
    w = Ser_data.write((uint8_t *)_TX_buffer, N_BYTES_TX_BUFFER);

    /*
    // DEBUG
    Ser_data.println(w);
    */
  }
}
