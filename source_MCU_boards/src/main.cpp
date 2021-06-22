/*------------------------------------------------------------------------------
  Arduino lock-in amplifier

  Pins:
  A0: output reference signal
  A1: input signal, differential +
  A2: input signal, differential -

  Boards                     | MCU        | test | #define
  ---------------------------------------------------------------------
  M0 family, SAMD21
  - Arduino M0                 SAMD21G18A          ARDUINO_SAMD_ZERO
  - Arduino M0 Pro             SAMD21G18A   okay   ARDUINO_SAMD_ZERO
  - Adafruit Metro M0          SAMD21G18A
  - Adafruit Feather M0        SAMD21G18A
  - Adafruit ItsyBitsy M0      SAMD21G18A
  - Adafruit Trinket M0        SAMD21E18A
  - Adafruit Gemma M0          SAMD21E18A

  M4 family, SAMD51
  - Adafruit Grand Central M4  SAMD51P20A
  - Adafruit NeoTrellis M4     SAMD51J19A?
  - Adafruit Metro M4          SAMD51J19A          ADAFRUIT_METRO_M4_EXPRESS
  - Adafruit Feather M4        SAMD51J19A   okay   ADAFRUIT_FEATHER_M4_EXPRESS
  - Adafruit ItsyBitsy M4      SAMD51G19A   okay   ADAFRUIT_ITSYBITSY_M4_EXPRESS

  Dennis van Gils
  22-06-2021
------------------------------------------------------------------------------*/

#include "DvG_SerialCommand.h"
#include "Streaming.h"
#include <Arduino.h>

#define FIRMWARE_VERSION "ALIA v0.3.0 VSCODE"

// Microcontroller unit (mcu)
#if defined(__SAMD21G18A__)
#define MCU_MODEL "SAMD21G18A"
#ifndef __SAMD21__
#define __SAMD21__
#endif
#elif defined(__SAMD21E18A__)
#define MCU_MODEL "SAMD21E18A"
#ifndef __SAMD21__
#define __SAMD21__
#endif
#elif defined(__SAMD51P20A__)
#define MCU_MODEL "SAMD51P20A"
#elif defined(__SAMD51J19A__)
#define MCU_MODEL "SAMD51J19A"
#elif defined(__SAMD51G19A__)
#define MCU_MODEL "SAMD51G19A"
#endif

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static __inline__ void syncADC() __attribute__((always_inline, unused));

#ifdef __SAMD21__
#include "ZeroTimer.h"
// clang-format off
static void syncDAC() {while (DAC->STATUS.bit.SYNCBUSY == 1) ;}
static void syncADC() {while (ADC->STATUS.bit.SYNCBUSY == 1) ;}
// clang-format on
#define DAC_OUTPUT_BITS 10
#define ADC_INPUT_BITS 12
#endif

#ifdef __SAMD51__
#include "SAMD51_InterruptTimer.h"
// clang-format off
static void syncDAC() {while (DAC->STATUS.bit.EOC0 == 1) ;}
static void syncADC() {while (ADC0->STATUS.bit.ADCBUSY == 1) ;}
// clang-format on
#define DAC_OUTPUT_BITS 12
#define ADC_INPUT_BITS 12
#endif

// No-operation, to burn clock cycles
#define NOP __asm("nop");

// Preprocessor trick to ensure enums and strings are in sync, so one can write
// 'WAVEFORM_STRING[Cosine]' to give the string 'Cosine'
#define FOREACH_WAVEFORM(WAVEFORM) \
  WAVEFORM(Cosine)                 \
  WAVEFORM(Square)                 \
  WAVEFORM(Triangle)               \
  WAVEFORM(END_WAVEFORM_ENUM)
#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

enum WAVEFORM_ENUM { FOREACH_WAVEFORM(GENERATE_ENUM) };
static const char *WAVEFORM_STRING[] = {FOREACH_WAVEFORM(GENERATE_STRING)};

// Others
volatile bool is_running = false; // Is the lock-in amplifier running?
char mcu_uid[33];                 // Serial number

/*------------------------------------------------------------------------------
  Sampling period
--------------------------------------------------------------------------------

  * Interrupt service routine

    The interrupt service routine will periodically request samples to be read
    by the ADC and send out a new value to the DAC. I.e., it acquires analog
    signal `sig_I` and outputs a new analog signal `ref_X` per timestep
    `SAMPLING_PERIOD_us` in microseconds.

  * Double buffer: TX_buffer_A & TX_buffer_B

    Each acquired `sig_I` sample will get written to a buffer to be transmitted
    over serial once the buffer is full. There are two of these buffers:
    `TX_buffer_A` and `TX_buffer_B`.

    The buffer that will be send each transmission is `BLOCK_SIZE` input samples
    long. We employ a double buffer technique, where alternatingly `TX_buffer_A`
    is being written to and `TX_buffer_B` is being sent, and vice-versa.

    A full transmit buffer will contain a single block of data:

    [
      SOM,                                                {size = 10 bytes}
      (uint32_t) number of block being send               {size =  4 bytes}
      (uint32_t) millis timestamp at start of block       {size =  4 bytes}
      (uint16_t) micros part of timestamp                 {size =  2 bytes}
      (uint16_t) phase index `LUT_wave` at start of block {size =  2 bytes}
      BLOCK_SIZE x (uint16_t) ADC readings `sig_I`        {size = BLOCK_SIZE * 2 bytes}
      EOM                                                 {size = 10 bytes}
    ]
*/

// Hint: Maintaining `SAMPLING_PERIOD_us x BLOCK_SIZE` = 0.1 seconds long will
// result in a serial transmit rate of 10 blocks / s, which acts nicely with
// the Python GUI.
#ifdef __SAMD21__
#define SAMPLING_PERIOD_us 80
#define BLOCK_SIZE 1250
#else
#define SAMPLING_PERIOD_us 40
#define BLOCK_SIZE 2500
#endif

const double SAMPLING_RATE_Hz = (double)1.0e6 / SAMPLING_PERIOD_us;

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

// clang-format off
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
  Waveform look-up table (LUT)
------------------------------------------------------------------------------*/

// Output reference signal `ref_X` parameters
enum WAVEFORM_ENUM ref_waveform = Cosine;
double ref_freq; // [Hz] Obtained frequency of reference signal
double ref_offs; // [V]  Obtained voltage offset of reference signal
double ref_ampl; // [V]  Voltage amplitude reference signal

// Look-up table (LUT) for fast DAC
#define MIN_N_LUT 20                // Min. allowed number of samples for one full period
#define MAX_N_LUT 1000              // Max. allowed number of samples for one full period
uint16_t LUT_wave[MAX_N_LUT] = {0}; // Look-up table allocation
uint16_t N_LUT;                     // Current number of samples for one full period
bool is_LUT_dirty = false;          // Does the LUT have to be updated with new settings?

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define MAX_DAC_OUTPUT_BITVAL ((uint16_t)(pow(2, DAC_OUTPUT_BITS) - 1))

/*------------------------------------------------------------------------------
  Serial
--------------------------------------------------------------------------------

  Arduino M0 Pro
    Serial    : UART , Programming USB port
    SerialUSB : USART, Native USB port

  Adafruit Feather M4 Express
    Serial    : USART
*/

#define SERIAL_DATA_BAUDRATE 1.2e6 // Only used when Serial is UART

#ifdef ARDUINO_SAMD_ZERO
#define Ser_data Serial // Serial or SerialUSB
#else
#define Ser_data Serial // Only Serial
#endif

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);

/*------------------------------------------------------------------------------
  get_mcu_uid
------------------------------------------------------------------------------*/

void get_mcu_uid(char mcu_uid_out[33]) {
  /* Return the 128-bit unique identifier (uid) of the microcontroller unit
  (mcu) as a hex string. Aka, the serial number.
  */
  uint8_t raw_uid[16]; // uid as byte array

#ifdef __SAMD21__ // SAMD21 from section 9.3.3 of the datasheet
#define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x0080A00C)
#define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x0080A040)
#define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x0080A044)
#define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x0080A048)
#endif
#ifdef __SAMD51__ // SAMD51 from section 9.6 of the datasheet
#define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x008061FC)
#define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x00806010)
#define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x00806014)
#define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x00806018)
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

  // Generate normalized waveform periods in the range [0, 1]
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
}

/*------------------------------------------------------------------------------
  Time keeping
------------------------------------------------------------------------------*/

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
  /* Write timestamp and `phase`-stamp of the first ADC sample of the block
  that is about to be sent out over the serial port. We need to know which
  phase angle was output on the DAC, that corresponds in time to the first ADC
  sample of the `TX_buffer`. This is the essence of a phase-sensitive detector,
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
  static bool starting_up = true;
  static bool using_TX_buffer_A = true; // When false: Using TX_buffer_B
  static uint16_t write_idx;            // Current write index of TX_buffer
  volatile static uint16_t LUT_idx;     // Current read index of LUT
  uint16_t ref_X;
  int16_t sig_I;

  if (is_running != is_running_prev) {
    is_running_prev = is_running;
    if (is_running) {
      digitalWrite(PIN_LED, HIGH); // Indicate lock-in amp is running
      starting_up = true;
    } else {
      digitalWrite(PIN_LED, LOW); // Indicate lock-in amp is off
#if defined(__SAMD21__)
      DAC->DATA.reg = 0; // Set output voltage to 0
#elif defined(__SAMD51__)
      DAC->DATA[0].reg = 0; // Set output voltage to 0
#endif
      // syncDAC(); // NOT NECESSARY
    }
  }

  if (!is_running) {
    return;
  }

  if (starting_up) {
    trigger_send_TX_buffer_A = false;
    trigger_send_TX_buffer_B = false;
    using_TX_buffer_A = true;
    write_idx = 0;
    LUT_idx = 0;
  }

  // Read input signal corresponding to the DAC output of the previous timestep.
  // This ensures that the previously set DAC output has had enough time to
  // stabilize.
  // clang-format off
  syncADC(); // NECESSARY
#if defined(__SAMD21__)
  ADC->SWTRIG.bit.START = 1;
  while (ADC->INTFLAG.bit.RESRDY == 0) {;} // Wait for conversion to complete
  syncADC(); // NECESSARY
  sig_I = ADC->RESULT.reg;
#elif defined(__SAMD51__)
  ADC0->SWTRIG.bit.START = 1;
  while (ADC0->INTFLAG.bit.RESRDY == 0) {;} // Wait for conversion to complete
  syncADC(); // NECESSARY
  sig_I = ADC0->RESULT.reg;
#endif
  // syncADC(); // NOT NECESSARY
  // clang-format on

  // Output reference signal
  ref_X = LUT_wave[LUT_idx];
  // syncDAC(); // DON'T ENABLE: Causes timing jitter in the output waveform
#if defined(__SAMD21__)
  DAC->DATA.reg = ref_X;
#elif defined(__SAMD51__)
  DAC->DATA[0].reg = ref_X;
#endif
  // syncDAC(); // NOT NECESSARY

  if (starting_up) {
    // No valid input signal yet, hence return. Next timestep it will be valid.
    starting_up = false;
    stamp_TX_buffer(TX_buffer_A, &LUT_idx);
    LUT_idx++;
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
  print_debug_info
------------------------------------------------------------------------------*/

void print_debug_info() {
#if defined(__SAMD21__)
  Ser_data << "-------------------------------" << endl;
  Ser_data << "CTRLA" << endl;
  Ser_data << "  .RUNSTDBY   : " << _HEX(ADC->CTRLA.bit.RUNSTDBY) << endl;
  Ser_data << "  .ENABLE     : " << _HEX(ADC->CTRLA.bit.ENABLE) << endl;
  Ser_data << "  .SWRST      : " << _HEX(ADC->CTRLA.bit.SWRST) << endl;
  Ser_data << "REFCTRL" << endl;
  Ser_data << "  .REFCOMP    : " << _HEX(ADC->REFCTRL.bit.REFCOMP) << endl;
  Ser_data << "  .REFSEL     : " << _HEX(ADC->REFCTRL.bit.REFSEL) << endl;
  Ser_data << "AVGVTRL" << endl;
  Ser_data << "  .ADJRES     : " << _HEX(ADC->AVGCTRL.bit.ADJRES) << endl;
  Ser_data << "  .SAMPLENUM  : " << _HEX(ADC->AVGCTRL.bit.SAMPLENUM) << endl;
  Ser_data << "SAMPCTRL" << endl;
  Ser_data << "  .SAMPLEN    : " << _HEX(ADC->SAMPCTRL.bit.SAMPLEN) << endl;
  Ser_data << "CTRLB" << endl;
  Ser_data << "  .PRESCALER  : " << _HEX(ADC->CTRLB.bit.PRESCALER) << endl;
  Ser_data << "  .RESSEL     : " << _HEX(ADC->CTRLB.bit.RESSEL) << endl;
  Ser_data << "  .CORREN     : " << _HEX(ADC->CTRLB.bit.CORREN) << endl;
  Ser_data << "  .FREERUN    : " << _HEX(ADC->CTRLB.bit.FREERUN) << endl;
  Ser_data << "  .LEFTADJ    : " << _HEX(ADC->CTRLB.bit.LEFTADJ) << endl;
  Ser_data << "  .DIFFMODE   : " << _HEX(ADC->CTRLB.bit.DIFFMODE) << endl;
  Ser_data << "INPUTCTRL" << endl;
  Ser_data << "  .GAIN       : " << _HEX(ADC->INPUTCTRL.bit.GAIN) << endl;
  Ser_data << "  .INPUTOFFSET: " << _HEX(ADC->INPUTCTRL.bit.INPUTOFFSET) << endl;
  Ser_data << "  .INPUTSCAN  : " << _HEX(ADC->INPUTCTRL.bit.INPUTSCAN) << endl;
  Ser_data << "  .MUXNEG     : " << _HEX(ADC->INPUTCTRL.bit.MUXNEG) << endl;
  Ser_data << "  .MUXPOS     : " << _HEX(ADC->INPUTCTRL.bit.MUXPOS) << endl;
#elif defined(__SAMD51__)
// TO DO
#endif

  float block_rate = SAMPLING_RATE_Hz / BLOCK_SIZE;
  uint32_t baudrate = ceil(N_BYTES_TX_BUFFER * 10 * block_rate);
  // baudrate: 8 data bits + 1 start bit + 1 stop bit = 10 bits per data byte
  Ser_data << "----------------------------------------" << endl;
  Ser_data << "DAQ rate     : " << _FLOAT(SAMPLING_RATE_Hz, 2) << " Hz" << endl;
  Ser_data << "ISR clock    : " << SAMPLING_PERIOD_us << " usec" << endl;
  Ser_data << "Block size   : " << BLOCK_SIZE << " samples" << endl;
  Ser_data << "Transmit rate          : "
           << _FLOAT(block_rate, 2) << " blocks/s" << endl;
  Ser_data << "Data bytes per transmit: "
           << N_BYTES_TX_BUFFER << " bytes" << endl;
  Ser_data << "Lower-bound baudrate   : "
           << baudrate << endl;
  Ser_data << "----------------------------------------" << endl;
}

/*------------------------------------------------------------------------------
  setup
------------------------------------------------------------------------------*/

// Future support
uint8_t NVM_ADC0_BIASCOMP = 0;
uint8_t NVM_ADC0_BIASREFBUF = 0;
uint8_t NVM_ADC0_BIASR2R = 0;

void setup() {
  Ser_data.begin(SERIAL_DATA_BAUDRATE);
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
  // NOTE: Disabled because `memcpy` does not operate on volatiles
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
  // Increase the ADC clock by setting the divisor from default DIV128 to a
  // smaller divisor. This is needed for DAQ rates larger than ~20 kHz on SAMD51
  // and DAQ rates larger than ~10 kHz on SAMD21. Setting too small divisors
  // will result in ADC errors. Keep as large as possible to increase ADC
  // accuracy.
#if defined(__SAMD21__)
  ADC->CTRLB.bit.PRESCALER = ADC_CTRLB_PRESCALER_DIV32_Val;
#elif defined(__SAMD51__)
  ADC0->CTRLA.bit.PRESCALER = ADC_CTRLA_PRESCALER_DIV64_Val;
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
  // syncADC(); // NOT NECESSARY
#if defined(__SAMD21__)
  ADC->CTRLA.bit.ENABLE = 0x01;
#elif defined(__SAMD51__)
  ADC0->CTRLA.bit.ENABLE = 0x01;
#endif
  syncADC();

  /*
  // NOT NECESSARY
  #if defined(__SAMD21__)
    ADC->SWTRIG.bit.START = 1;
    ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY;
  #elif defined(__SAMD51__)
    ADC0->SWTRIG.bit.START = 1;
    ADC0->INTFLAG.reg = ADC_INTFLAG_RESRDY;
  #endif
  */

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

  // Process incoming serial commands every N milliseconds.
  // Deliberately slowed down to improve timing stability of `isr_psd()`.
  if ((now - prev_millis) > 19) {
    prev_millis = now;

    if (sc_data.available()) {
      if (is_running) { // Atomic read, `noInterrupts()` not required here
        /*--------------
          Running
        ----------------
          Any command received while running will switch the lock-in amp off.
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

#if defined(ARDUINO_SAMD_ZERO) && Ser_data == Serial
        // Simply end the serial connection which directly clears the underlying
        // TX (and RX) ringbuffer. `Flush()` on the other hand, would first send
        // out the full TX buffer and wait for the operation to complete.
        // NOTE: Only works on Arduino M0 Pro debugging port
        Ser_data.end();
        Ser_data.begin(SERIAL_DATA_BAUDRATE);
#else
        // Flush out any binary buffer data scheduled for sending, potentially
        // flooding the receiving buffer at the PC side if `is_running` was not
        // switched to `false` fast enough.
        Ser_data.flush();
#endif

        // Confirm at the PC side that the lock-in amp is off and is no longer
        // sending binary data. The 'off' message might still be preceded with
        // some left-over binary data when being read at the PC side.
        Ser_data.print("off\n");

        // Flush out and ignore the command
        sc_data.getCmd();

      } else {
        /*-------------
          Not running
        ---------------
          We are ready to process any incoming commands.
        */
        str_cmd = sc_data.getCmd();

        if (strcmp(str_cmd, "id?") == 0) {
          // Report identity string
          Ser_data.println("Arduino, Alia");

        } else if (strcmp(str_cmd, "mcu?") == 0) {
          // Report firmware, microcontroller model and serial number
          Ser_data << FIRMWARE_VERSION << "\t" << MCU_MODEL << "\t" << mcu_uid
                   << endl;

        } else if (strcmp(str_cmd, "bias?") == 0) {
          // Report ADC bias and correction information
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

        } else if (strcmp(str_cmd, "debug?") == 0) {
          // Report debug information
          print_debug_info();

        } else if (strcmp(str_cmd, "const?") == 0) {
          // Report lock-in amplifier constants
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

        } else if (strcmp(str_cmd, "ref?") == 0 || strcmp(str_cmd, "?") == 0) {
          // Report reference signal `ref_X` settings
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
          // Report the LUT as a binary stream. The reported LUT will start at
          // phase = 0 deg.
          Ser_data.write((uint8_t *)&N_LUT, 2);
          Ser_data.write((uint8_t *)&is_LUT_dirty, 1);
          Ser_data.write((uint8_t *)LUT_wave, N_LUT * 2);

        } else if (strcmp(str_cmd, "lut_ascii?") == 0 ||
                   strcmp(str_cmd, "la?") == 0) {
          // Report the LUT as tab-delimited ASCII. The reported LUT will start
          // at phase = 0 deg. Convenience function handy for debugging from a
          // serial console.
          Ser_data << N_LUT << "\t" << is_LUT_dirty << endl;
          for (uint16_t i = 0; i < N_LUT - 1; i++) {
            Ser_data << LUT_wave[i] << "\t";
          }
          Ser_data << LUT_wave[N_LUT - 1] << endl;

        } else if (strcmp(str_cmd, "t?") == 0) {
          // Report time in microseconds
          static char buf[16];
          uint32_t millis_copy;
          uint16_t micros_part;

          get_systick_timestamp(&millis_copy, &micros_part);
          sprintf(buf, "%lu %03u\n", millis_copy, micros_part);
          Ser_data.print(buf);

        } else if (strcmp(str_cmd, "fcpu?") == 0) {
          // Report processor clock frequency
          Ser_data << SystemCoreClock << " Hz" << endl;

        } else if (strcmp(str_cmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser_data.print("already_off\n");

        } else if (strcmp(str_cmd, "on") == 0 || strcmp(str_cmd, "_on") == 0) {
          // Start lock-in amp
          noInterrupts();
          is_running = true;
          interrupts();

        } else if (strncmp(str_cmd, "_freq", 5) == 0) {
          // Set frequency of the reference signal [Hz].
          // Call 'compute_LUT(LUT_wave)' for it to become effective.
          parse_freq(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_freq, 3);

        } else if (strncmp(str_cmd, "_offs", 5) == 0) {
          // Set offset of the reference signal [V].
          // Call 'compute_LUT(LUT_wave)' for it to become effective.
          parse_offs(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_offs, 3);

        } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
          // Set amplitude of the reference signal [V].
          // Call 'compute_LUT(LUT_wave)' for it to become effective.
          parse_ampl(&str_cmd[5]);
          is_LUT_dirty = true;
          Ser_data.println(ref_ampl, 3);

        } else if (strncmp(str_cmd, "_wave", 5) == 0) {
          // Set the waveform type of the reference signal.
          // Call 'compute_LUT(LUT_wave)' for it to become effective.
          ref_waveform = static_cast<WAVEFORM_ENUM>(atoi(&str_cmd[5]));
          ref_waveform = static_cast<WAVEFORM_ENUM>(max(ref_waveform, 0));
          ref_waveform = static_cast<WAVEFORM_ENUM>(
              min(ref_waveform, END_WAVEFORM_ENUM - 1));
          is_LUT_dirty = true;
          Ser_data.println(WAVEFORM_STRING[ref_waveform]);

        } else if (strcmp(str_cmd, "compute_lut") == 0 ||
                   strcmp(str_cmd, "c") == 0) {
          // (Re)compute the LUT based on the following known settings:
          // ref_freq, ref_offs, ref_ampl, ref_waveform.
          compute_LUT(LUT_wave);
          Ser_data.println("!"); // Reply with OK '!'
        }
      }
    }
  }

  // Send buffer out over the serial connection

  /* NOTE:
    Copying the volatile buffers `TX_buffer_A` and `TX_buffer_B` into a
    non-volatile extra buffer is not necessary and actually hurts performance,
    especially when it is encapsulated by a `noInterrupts()` and
    `interrupts()` routine. The encapsulation results in unstable DAC output,
    where a single sample gets duplicated intermittently as seen on a
    oscilloscope, when the DAQ rate is very high (> 10kHz).

    The employed double-buffer technique already is sufficient to ensure that
    the serial transmit has all the time to send out stream A, while the other
    stream B gets written to by the ISR.
  */
  // uint8_t _TX_buffer[N_BYTES_TX_BUFFER]; // Will hold copy of volatile buffer

  if (is_running && (trigger_send_TX_buffer_A || trigger_send_TX_buffer_B)) {
    /*
    // DEBUG
    Ser_data.println(TX_buffer_counter);
    Ser_data.println(
      trigger_send_TX_buffer_A ? "\nTX_buffer_A" : "\nTX_buffer_B"
    );
    */

    /*
    // Copy the volatile buffer
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
    */

    // NOTE: `write()` can return -1 as indication of an error, e.g. the
    // receiving side being overrun with data.
    // size_t w;

    if (trigger_send_TX_buffer_A) {
      trigger_send_TX_buffer_A = false;
      // w =
      Ser_data.write((uint8_t *)TX_buffer_A, N_BYTES_TX_BUFFER);
    } else {
      trigger_send_TX_buffer_B = false;
      // w =
      Ser_data.write((uint8_t *)TX_buffer_B, N_BYTES_TX_BUFFER);
    }

    /*
    // DEBUG
    Ser_data.println(w);
    */
  }
}
