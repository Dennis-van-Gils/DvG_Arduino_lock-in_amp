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

  For default hardware startup configuration, see
  SAMD21:
  \.platformio\packages\framework-arduino-samd\cores\arduino\startup.c
  SAMD51:
  \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\startup.c

  Dennis van Gils
  05-08-2021
------------------------------------------------------------------------------*/
#define FIRMWARE_VERSION "ALIA v0.2.0 VSCODE"

// clang-format off
#include "Arduino.h"
#include "DvG_SerialCommand.h"
#include "FlashStorage_SAMD.h"
#include "adc_dac_init.h"
#include "adc_dac_functions.h"
// clang-format on

/*
#include "Streaming.h"  // DO NOT USE
The Streaming library, when used as the default method to send ASCII over the
serial connection, is observed to increase the communication time with the
Python main program. I believe the problem lies at the microcontroller side.
We'll refrain from using the Streaming library and instead rely on
`sprintf(buf, ...)` in combination with `Serial.print(buf)`.

To enable float support in `sprintf()` we must add the following to `setup()`:
  asm(".global _printf_float"); // Enable float support in `sprintf()` and like
*/

// Microcontroller unit (mcu)
#if defined __SAMD21G18A__
#  define MCU_MODEL "SAMD21G18A"
#  ifndef __SAMD21__
#    define __SAMD21__
#  endif
#elif defined __SAMD21E18A__
#  define MCU_MODEL "SAMD21E18A"
#  ifndef __SAMD21__
#    define __SAMD21__
#  endif
#elif defined __SAMD51P20A__
#  define MCU_MODEL "SAMD51P20A"
#elif defined __SAMD51J19A__
#  define MCU_MODEL "SAMD51J19A"
#elif defined __SAMD51G19A__
#  define MCU_MODEL "SAMD51G19A"
#endif

#ifdef __SAMD21__
#  include "ZeroTimer.h"
#elif defined __SAMD51__
#  include "SAMD51_InterruptTimer.h"
#endif
// Use built-in LED to signal the running state of lock-in amp
#ifdef ADAFRUIT_FEATHER_M4_EXPRESS
#  include "Adafruit_NeoPixel.h"
Adafruit_NeoPixel strip =
    Adafruit_NeoPixel(1, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);
#endif

static inline void LED_off() {
#ifdef ADAFRUIT_FEATHER_M4_EXPRESS
  strip.setPixelColor(0, strip.Color(0, 0, 255));
  strip.show();
#else
  digitalWrite(PIN_LED, LOW);
#endif
}

static inline void LED_on() {
#ifdef ADAFRUIT_FEATHER_M4_EXPRESS
  strip.setPixelColor(0, strip.Color(0, 255, 0));
  strip.show();
#else
  digitalWrite(PIN_LED, HIGH);
#endif
}

// Flash storage for the ADC calibration correction parameters
ADC_Calibration calibration;
FlashStorage(flash_storage, ADC_Calibration);
bool flash_was_read_okay = false;

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

// Others
volatile bool is_running = false; // Is the lock-in amplifier running?
volatile bool trigger_reset_time = false;
char mcu_uid[33]; // Serial number

/*------------------------------------------------------------------------------
  Serial
--------------------------------------------------------------------------------

  Arduino M0 Pro
    Serial    : UART , Programming USB port
    SerialUSB : USART, Native USB port

  Adafruit Feather M4 Express
    Serial    : USART
*/

#define SERIAL_DATA_BAUDRATE 1e6 // Only used when Serial is UART

#ifdef ARDUINO_SAMD_ZERO
#  define Ser Serial // Serial or SerialUSB
#else
#  define Ser Serial // Only Serial
#endif

#define MAXLEN_buf 100 // `printf()` string buffer
char buf[MAXLEN_buf];  // `printf()` string buffer

// Instantiate serial command listener
DvG_SerialCommand sc_data(Ser);

/*------------------------------------------------------------------------------
  Sampling
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
      SOM,                                              {10 bytes}
      (uint32_t) number of block being send             { 4 bytes}
      (uint32_t) millis timestamp at start of block     { 4 bytes}
      (uint16_t) micros part of timestamp               { 2 bytes}
      BLOCK_SIZE x (uint16_t) `LUT_wave` indexes        {BLOCK_SIZE * 2 bytes}
      BLOCK_SIZE x (int16_t)  ADC readings `sig_I`      {BLOCK_SIZE * 2 bytes}
      EOM                                               {10 bytes}
    ]
*/

// Hint: Maintaining `SAMPLING_PERIOD_us x BLOCK_SIZE` = 0.1 seconds long will
// result in a serial transmit rate of 10 blocks / s, which acts nicely with
// the Python GUI.
#ifdef __SAMD21__
#  define SAMPLING_PERIOD_us 100
#  define BLOCK_SIZE 1000
#else
#  define SAMPLING_PERIOD_us 50
#  define BLOCK_SIZE 2000
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
#define N_BYTES_PHASE   (BLOCK_SIZE * 2)
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
enum WAVEFORM_ENUM ref_waveform;
double ref_freq;         // [Hz]    Obtained frequency of reference signal
double ref_offs;         // [V]     Voltage offset of reference signal
double ref_ampl;         // [V]     Voltage amplitude reference signal
double ref_VRMS;         // [V_RMS] Voltage amplitude reference signal
double ref_RMS_factor;   // RMS factor belonging to chosen waveform
bool ref_is_clipping_HI; // Output reference signal is clipping high?
bool ref_is_clipping_LO; // Output reference signal is clipping low?

/* N_LUT: Number of samples for one full period.
   MUST BE A POWER OF 2.

In order to calculate the phase of the `REF_X` signal to be output by the
interrupt service routine (ISR), we will calculate the rate at which the
`LUT_idx` should advance for each DAQ iteration, called 'idx_per_iter`.

Naive method:
  Inside the ISR, we should multiply `idx_per_iter` with the DAQ iteration
  counter 'iter' and afterwards take the `N_LUT` modulo and round it to the
  nearest integer. This will give us the `LUT_idx` that should currently be
  output.

Problems with this method:
  `iter` is uint32_t and multiplying its value with `idx_per_iter` will result
  in a overflow when `iter` is already large.

Solution, see:
https://www.geeksforgeeks.org/how-to-avoid-overflow-in-modular-multiplication/
  A routine exists for multiplicative modulo that prevents overflow. This
  routine works best on integers, so we will round `idx_per_iter` towards
  the nearest integer and, hence, we should recalculate the new corresponding
  `ref_freq`.

  Another speed improvement is to fix the modulus (i.e. `N_LUT` in this case)
  to a power of 2, because it allows to calculate the modulo using a single
  bitwise `&` operation. This is super fast! I.e.: x % 2n == x & (2n - 1).

  HINDSIGHT: Do not use powers of 2 for `N_LUT`, because it introduces beating
  in the output REF_X signal. Better set `N_LUT` equal to `SAMPLING_RATE_Hz`,
  because that way the requested `ref_freq` and the obtained 'ideal' one are
  identical to each other for all integer input without beating.
*/
#ifdef __SAMD21__
#  define N_LUT 10000 // SAMD21 has less RAM
#elif defined __SAMD51__
#  define N_LUT 20000 // Suggest setting equal to `SAMPLING_RATE_Hz`
#endif

int16_t LUT_wave[N_LUT] = {0};  // Look-up table allocation
volatile uint16_t idx_per_iter; // LUT_idx per DAQ_iter, depends on `ref_freq`

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define MAX_DAC_OUTPUT_BITVAL ((1 << DAC_OUTPUT_BITS) - 1)

void compute_LUT() {
  double norm_offs = ref_offs / A_REF; // Normalized
  double norm_ampl = ref_ampl / A_REF; // Normalized
  double wave;

  // Generate normalized waveform periods in the range [0, 1]
  ref_is_clipping_HI = false;
  ref_is_clipping_LO = false;

  for (uint16_t i = 0; i < N_LUT; i++) {

    switch (ref_waveform) {
      default:
      case Cosine:
        // N_LUT even: extrema [ 0, 1]
        // N_LUT odd : extrema [>0, 1]
        wave = .5 * (1 + cos(M_TWOPI * i / N_LUT));
        break;

      case Square:
        // Extrema guaranteed  [ 0, 1]
        // wave = cos(M_TWOPI * i / N_LUT) < 0. ? 0. : 1.;
        if ((i < (N_LUT / 4.)) || (i >= (N_LUT / 4. * 3.))) {
          wave = 1.;
        } else {
          wave = 0.;
        }
        break;

      case Triangle:
        // N_LUT even: extrema [ 0, 1]
        // N_LUT odd : extrema [>0, 1]
        wave = 2 * fabs((double)i / N_LUT - .5);
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

    LUT_wave[i] = round(MAX_DAC_OUTPUT_BITVAL * wave);
  }
}

void set_wave(int value) {
  /* Set the waveform type, keeping `ref_VRMS` constant and changing `ref_ampl`
  according to the new waveform */

  // OVERRIDE: Only `Cosine` allowed, because `Square` and `Triangle` can not
  // be garantueed deterministic on both Arduino and Python side due to
  // rounding differences and the problem of computing the correct 90 degrees
  // quadrant `ref_Y`.
  ref_waveform = static_cast<WAVEFORM_ENUM>(0);
  ref_RMS_factor = sqrt(2);
  ref_ampl = ref_VRMS * ref_RMS_factor;

  /*
  value = max(value, 0);
  value = min(value, END_WAVEFORM_ENUM - 1);
  ref_waveform = static_cast<WAVEFORM_ENUM>(value);

  switch (ref_waveform) {
    default:
    case Cosine:
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
  */

  noInterrupts();
  compute_LUT();
  interrupts();
}

void set_freq(double value) {
  ref_freq = max(value, 10.0);
  ref_freq = min(ref_freq, SAMPLING_RATE_Hz / 4.);
  noInterrupts();
  idx_per_iter = round(N_LUT / SAMPLING_RATE_Hz * ref_freq);
  ref_freq = SAMPLING_RATE_Hz * idx_per_iter / N_LUT;
  interrupts();
}

void set_offs(double value) {
  ref_offs = max(value, 0.0);
  // ref_offs = min(ref_offs, A_REF);
  noInterrupts();
  compute_LUT();
  interrupts();
}

void set_ampl(double value) {
  ref_ampl = max(value, 0.0);
  // ref_ampl = min(ref_ampl, A_REF);
  ref_VRMS = ref_ampl / ref_RMS_factor;
  noInterrupts();
  compute_LUT();
  interrupts();
}

void set_VRMS(double value) {
  ref_VRMS = max(value, 0.0);
  ref_ampl = ref_VRMS * ref_RMS_factor;
  // ref_ampl = min(ref_ampl, A_REF);
  // ref_VRMS = ref_ampl / ref_RMS_factor;
  noInterrupts();
  compute_LUT();
  interrupts();
}

uint32_t mulmod_int(uint16_t a, uint32_t b, uint16_t n) {
  // Multiplicative modulo `(a * b) mod n`, safe against overflow of `a * b`.
  // Integers only, /no/ constraint on `n` being a power of 2.
  uint32_t sum = 0;
  while (b) {
    if (b & 1)
      sum = (sum + a) % n;
    a = (a << 1) % n;
    b = b >> 1;
  }
  return sum;
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
#ifdef __SAMD21__
#  define SERIAL_NUMBER_WORD_0 *(volatile uint32_t *)(0x0080A00C)
#  define SERIAL_NUMBER_WORD_1 *(volatile uint32_t *)(0x0080A040)
#  define SERIAL_NUMBER_WORD_2 *(volatile uint32_t *)(0x0080A044)
#  define SERIAL_NUMBER_WORD_3 *(volatile uint32_t *)(0x0080A048)
#endif
// SAMD51 from section 9.6 of the datasheet
#ifdef __SAMD51__
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
------------------------------------------------------------------------------*/

void get_systick_timestamp(uint32_t *stamp_millis,
                           uint16_t *stamp_micros_part) {
  /* Adapted from:
  https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/delay.c

  Note:
    The millis counter will roll over after 49.7 days.
  */
  // clang-format off
  uint32_t ticks, ticks2;
  uint32_t pend, pend2;
  uint32_t count, count2;
  uint32_t _ulTickCount = millis();

  ticks2 = SysTick->VAL;
  pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
  count2 = _ulTickCount;

  do {
    ticks  = ticks2;
    pend   = pend2;
    count  = count2;
    ticks2 = SysTick->VAL;
    pend2  = !!(SCB->ICSR & SCB_ICSR_PENDSTSET_Msk);
    count2 = _ulTickCount;
  } while ((pend != pend2) || (count != count2) || (ticks < ticks2));

  (*stamp_millis) = count2;
  if (pend) {(*stamp_millis)++;}
  (*stamp_micros_part) =
    (((SysTick->LOAD - ticks) * (1048576 / (VARIANT_MCK / 1000000))) >> 20);
  // clang-format on
}

void stamp_TX_buffer(volatile uint8_t *TX_buffer) {
  /* Write timestamp of the first ADC sample of the block that is about to be
  sent out over the serial port.
  */

  static uint32_t startup_millis = 0; // Time when lock-in amp got turned on
  static uint16_t startup_micros = 0; // Time when lock-in amp got turned on
  uint32_t millis_copy;
  uint16_t micros_part;

  get_systick_timestamp(&millis_copy, &micros_part);

  if (trigger_reset_time) {
    trigger_reset_time = false;
    startup_millis = millis_copy;
    startup_micros = micros_part;
  }

  // clang-format off
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
  // clang-format on
}

/*------------------------------------------------------------------------------
  Interrupt service routine (ISR) for phase-sentive detection (PSD)
------------------------------------------------------------------------------*/
// We need a few iterations to get started up correcly in `isr_psd()`
#define N_STARTUP_ITERS 2

#ifndef __SAMD21__
// DEBUG info: Collects execution time durations of `isr_psd()` in microseconds.
// They should always be smaller than the ISR period, otherwise a processor
// lock-up might occur or other unspecified behavior. Tweak the DAC settings
// (predominantly `SAMPLEN`, 'SAMPLENUM` and `PRESCALER`) to alter the execution
// time duration.
volatile static uint16_t isr_duration[BLOCK_SIZE] = {0};
#endif

void isr_psd() {
  static uint32_t DAQ_iter = 0;         // Increments each time step
  static uint16_t write_idx = 0;        // Current write index of TX_buffer
  static bool using_TX_buffer_A = true; // When false: Using TX_buffer_B
  static bool is_running_prev = is_running;
  static uint16_t LUT_idx_prev = 0;
  uint16_t LUT_idx;
  int16_t sig_I;

#ifndef __SAMD21__
  // DEBUG info: Execution time duration
  static uint16_t debug_iter = 0;
  uint32_t tick = SysTick->VAL;
#endif

  // Stamp the next TX buffer, one time step in advance before the current TX
  // buffer will get completely full. Stamp as soon as possible, before other
  // DAC and ADC operations, to get close to the true start time of the current
  // ISR call: Inside an ISR `millis` is stalled, but `micros` is not.
  if (write_idx == BLOCK_SIZE - 1) {
    if (using_TX_buffer_A) {
      stamp_TX_buffer(TX_buffer_B);
    } else {
      stamp_TX_buffer(TX_buffer_A);
    }
  }
  if (DAQ_iter == N_STARTUP_ITERS) {
    stamp_TX_buffer(TX_buffer_A);
  }

  // Start / stop
  if (is_running != is_running_prev) {
    is_running_prev = is_running;

    if (is_running) {
      // Just got turned on
      DAQ_iter = 0;
      write_idx = 0;
      trigger_send_TX_buffer_A = false;
      trigger_send_TX_buffer_B = false;
      using_TX_buffer_A = true;
    } else {
      // Just got turned off
      DAC_set_output(0.);
    }
  }

  if (!is_running) {
    return;
  }

  // Read ADC signal `sig_I` corresponding to the DAC output of the previous
  // time step. This ensures that the previously set DAC output has had enough
  // time to stabilize.
  sig_I = ADC_read_signal();

  // Calculate the phase (i.e. LUT index) of the upcoming DAC output 'ref_X`.
  // Takes between 1.6 and 4.3 ms @ 120 MHz cpu clock, depending on how large
  // `DAQ_iter` is.
  LUT_idx = mulmod_int(idx_per_iter, (DAQ_iter - N_STARTUP_ITERS), N_LUT);

  // Output DAC signal `ref_X`
  // We don't worry about syncing, because the ISR is assumed slow enough for
  // the DAC output to have caught up on the next time step.
  DAC_set_output_nosync(LUT_wave[LUT_idx]);

  // Start-up routine: ADC signal `sig_I` is not valid yet.
  if (DAQ_iter <= N_STARTUP_ITERS) {
    LUT_idx_prev = LUT_idx;
    DAQ_iter++;
    return;
  }

  // Store signals in the TX buffer
  // clang-format off
  if (using_TX_buffer_A) {
    TX_buffer_A[TX_BUFFER_OFFSET_PHASE + write_idx * 2    ] = LUT_idx_prev;
    TX_buffer_A[TX_BUFFER_OFFSET_PHASE + write_idx * 2 + 1] = LUT_idx_prev >> 8;
    TX_buffer_A[TX_BUFFER_OFFSET_SIG_I + write_idx * 2    ] = sig_I;
    TX_buffer_A[TX_BUFFER_OFFSET_SIG_I + write_idx * 2 + 1] = sig_I >> 8;
  } else {
    TX_buffer_B[TX_BUFFER_OFFSET_PHASE + write_idx * 2    ] = LUT_idx_prev;
    TX_buffer_B[TX_BUFFER_OFFSET_PHASE + write_idx * 2 + 1] = LUT_idx_prev >> 8;
    TX_buffer_B[TX_BUFFER_OFFSET_SIG_I + write_idx * 2    ] = sig_I;
    TX_buffer_B[TX_BUFFER_OFFSET_SIG_I + write_idx * 2 + 1] = sig_I >> 8;
  }
  write_idx++;
  // clang-format on

  // Check if the TX buffer is full and ready to be send over serial
  if (write_idx == BLOCK_SIZE) {
    if (using_TX_buffer_A) {
      trigger_send_TX_buffer_A = true;
    } else {
      trigger_send_TX_buffer_B = true;
    }

    using_TX_buffer_A = !using_TX_buffer_A;
    write_idx = 0;
  }

  // Advance the indices
  LUT_idx_prev = LUT_idx;
  DAQ_iter++;

#ifndef __SAMD21__
  // DEBUG info: Execution time duration
  isr_duration[debug_iter] =
      (uint16_t)((tick - SysTick->VAL) * (1048576 / (VARIANT_MCK / 1000000)) >>
                 20); // [us]
  debug_iter++;
  if (debug_iter == BLOCK_SIZE) {
    debug_iter = 0;
  }
#endif
}

/*------------------------------------------------------------------------------
  setup
------------------------------------------------------------------------------*/

void setup() {
  asm(".global _printf_float"); // Enable float support in `sprintf()` and like
  Ser.begin(SERIAL_DATA_BAUDRATE);
  get_mcu_uid(mcu_uid);

  // Use built-in LED to signal running state of lock-in amp
#ifdef ADAFRUIT_FEATHER_M4_EXPRESS
  strip.begin();
  strip.setBrightness(3);
#else
  pinMode(PIN_LED, OUTPUT);
#endif
  LED_off();

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

  // DAC
  DAC_init();

  // ADC
  ADC_init();

  // Retrieve ADC calibration correction parameters from flash
  calibration = flash_storage.read();
  if (calibration.is_valid) {
    ADC_set_calibration_correction(calibration.gaincorr,
                                   calibration.offsetcorr);
    flash_was_read_okay = true;
  }

  // Initial waveform LUT
  set_wave(Cosine);
  set_freq(250.0); // [Hz]    Wanted startup frequency
  set_offs(1.65);  // [V]     Wanted startup offset
  set_VRMS(0.5);   // [V_RMS] Wanted startup amplitude

  // Start the interrupt timer
  TC.startTimer(SAMPLING_PERIOD_us, isr_psd);

  // Experimental: Output a pulse train on pin 9 to act as clock source for
  // a (future) variable anti-aliasing filter IC placed in front of the ADC
  // input ports.
  /*
#ifdef __SAMD51__
  TCC_pulse_train.startTimer(10);
#endif
  */
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
          in-buffer as fast as possible because it is now waiting for the
          'off' reply to occur.
        */
        LED_off();
        noInterrupts();
        is_running = false;
        trigger_send_TX_buffer_A = false;
        trigger_send_TX_buffer_B = false;
        interrupts();

#if defined ARDUINO_SAMD_ZERO && Ser == Serial
        // Simply end the serial connection which directly clears the
        // underlying TX (and RX) ringbuffer. `Flush()` on the other hand,
        // would first send out the full TX buffer and wait for the operation
        // to complete. NOTE: Only works on Arduino M0 Pro debugging port.
        Ser.end();
        Ser.begin(SERIAL_DATA_BAUDRATE);
#else
        // Flush out any binary buffer data scheduled for sending, potentially
        // flooding the receiving buffer at the PC side if `is_running` was
        // not switched to `false` fast enough.
        Ser.flush();
#endif

        // Confirm at the PC side that the lock-in amp is off and is no longer
        // sending binary data. The 'off' message might still be preceded with
        // some left-over binary data when being read at the PC side.
        Ser.print("off\n");

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
          Ser.print("Arduino, Alia\n");

        } else if (strcmp(str_cmd, "mcu?") == 0) {
          // Report microcontroller information
          // clang-format off
          snprintf(buf, MAXLEN_buf, "%s\t%s\t%lu\t%s\n",
                   FIRMWARE_VERSION,
                   MCU_MODEL,
                   VARIANT_MCK,
                   mcu_uid);
          Ser.print(buf);
          // clang-format on

        } else if (strcmp(str_cmd, "adc?") == 0) {
          // Report ADC registers, debug information
          // clang-format off
          Ser.print("----------------------------------------\n");
#if defined __SAMD21__
          Ser.print("CTRLA\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".ENABLE"   , ADC->CTRLA.bit.ENABLE); Ser.print(buf);
          Ser.print("INPUTCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".MUXPOS"   , ADC->INPUTCTRL.bit.MUXPOS); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".MUXNEG"   , ADC->INPUTCTRL.bit.MUXNEG); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".GAIN"     , ADC->INPUTCTRL.bit.GAIN); Ser.print(buf);
          Ser.print("REFCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".REFCOMP"  , ADC->REFCTRL.bit.REFCOMP); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".REFSEL"   , ADC->REFCTRL.bit.REFSEL); Ser.print(buf);
          Ser.print("AVGCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".ADJRES"   , ADC->AVGCTRL.bit.ADJRES); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".SAMPLENUM", ADC->AVGCTRL.bit.SAMPLENUM); Ser.print(buf);
          Ser.print("SAMPCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  %u\n"  , ".SAMPLEN"  , ADC->SAMPCTRL.bit.SAMPLEN); Ser.print(buf);
          Ser.print("CTRLB\n");
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".RESSEL"   , ADC->CTRLB.bit.RESSEL); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".CORREN"   , ADC->CTRLB.bit.CORREN); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".LEFTADJ"  , ADC->CTRLB.bit.LEFTADJ); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".DIFFMODE" , ADC->CTRLB.bit.DIFFMODE); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".PRESCALER", ADC->CTRLB.bit.PRESCALER); Ser.print(buf);
          Ser.print("CALIB\n");
          snprintf(buf, MAXLEN_buf, "%15s  %u\n"  , ".LINEARITY_CAL", ADC->CALIB.bit.LINEARITY_CAL); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  %u\n"  , ".BIAS_CAL"     , ADC->CALIB.bit.BIAS_CAL); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "OFFSETCORR       %u\n"         , ADC->OFFSETCORR.bit.OFFSETCORR); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "GAINCORR         %u\n"         , ADC->GAINCORR.bit.GAINCORR); Ser.print(buf);
#elif defined __SAMD51__
          Ser.print("INPUTCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".DIFFMODE" , ADC0->INPUTCTRL.bit.DIFFMODE); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".MUXPOS"   , ADC0->INPUTCTRL.bit.MUXPOS); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".MUXNEG"   , ADC0->INPUTCTRL.bit.MUXNEG); Ser.print(buf);
          Ser.print("CTRLA\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".ENABLE"   , ADC0->CTRLA.bit.ENABLE); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".PRESCALER", ADC0->CTRLA.bit.PRESCALER); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".R2R"      , ADC0->CTRLA.bit.R2R); Ser.print(buf);
          Ser.print("CTRLB\n");
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".RESSEL"   , ADC0->CTRLB.bit.RESSEL); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".CORREN"   , ADC0->CTRLB.bit.CORREN); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".LEFTADJ"  , ADC0->CTRLB.bit.LEFTADJ); Ser.print(buf);
          Ser.print("REFCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".REFCOMP"  , ADC0->REFCTRL.bit.REFCOMP); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".REFSEL"   , ADC0->REFCTRL.bit.REFSEL); Ser.print(buf);
          Ser.print("AVGCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".ADJRES"   , ADC0->AVGCTRL.bit.ADJRES); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  0x%X\n", ".SAMPLENUM", ADC0->AVGCTRL.bit.SAMPLENUM); Ser.print(buf);
          Ser.print("SAMPCTRL\n");
          snprintf(buf, MAXLEN_buf, "%15s  F%u\n" , ".OFFCOMP"  , ADC0->SAMPCTRL.bit.OFFCOMP); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  %u\n"  , ".SAMPLEN"  , ADC0->SAMPCTRL.bit.SAMPLEN); Ser.print(buf);
          Ser.print("CALIB\n");
          snprintf(buf, MAXLEN_buf, "%15s  %u\n", ".BIASCOMP"   , ADC0->CALIB.bit.BIASCOMP); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  %u\n", ".BIASREFBUF" , ADC0->CALIB.bit.BIASREFBUF); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s  %u\n", ".BIASR2R"    , ADC0->CALIB.bit.BIASR2R); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "OFFSETCORR       %u\n"     , ADC0->OFFSETCORR.bit.OFFSETCORR); Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "GAINCORR         %u\n"     , ADC0->GAINCORR.bit.GAINCORR); Ser.print(buf);
#endif
          Ser.print("----------------------------------------\n");
          // clang-format on

        } else if (strcmp(str_cmd, "debug?") == 0) {
          // Report debug information
          float block_rate = SAMPLING_RATE_Hz / BLOCK_SIZE;
          uint32_t baudrate = ceil(N_BYTES_TX_BUFFER * 10 * block_rate);
          // 8 data bits + 1 start bit + 1 stop bit = 10 bits per data byte
          // clang-format off
          Ser.print("----------------------------------------\n");
          snprintf(buf, MAXLEN_buf, "%15s:%8.0f  %s\n", "DAQ rate", SAMPLING_RATE_Hz, "Hz");
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s:%8d  %s\n"  , "ISR clock", SAMPLING_PERIOD_us, "usec");
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s:%8d  %s\n"  , "Block size", BLOCK_SIZE, "samples");
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s:%8d  %s\n"  , "Block size", N_BYTES_TX_BUFFER, "bytes");
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s:%8.2f  %s\n", "Transmit rate", block_rate, "blocks/s");
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%15s:%8ld  %s\n" , "Baud rate", baudrate, "Bd");
          Ser.print(buf);
          Ser.print("----------------------------------------\n");
          // clang-format on

        } else if (strcmp(str_cmd, "const?") == 0) {
          // Report lock-in amplifier constants
          // clang-format off
          snprintf(buf, MAXLEN_buf, "%u\t%u\t%u\t%u\t%u\t%u\t%.3f\n",
                   SAMPLING_PERIOD_us,
                   BLOCK_SIZE,
                   N_BYTES_TX_BUFFER,
                   DAC_OUTPUT_BITS,
                   ADC_INPUT_BITS,
                   ADC_DIFFERENTIAL,
                   A_REF);
          Ser.print(buf);
          // clang-format on

        } else if (strcmp(str_cmd, "ref?") == 0 || strcmp(str_cmd, "?") == 0) {
          // Report reference signal `ref_X` settings
          // clang-format off
          snprintf(buf, MAXLEN_buf, "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%i\t%i\t%i\n",
                   WAVEFORM_STRING[ref_waveform],
                   ref_freq,
                   ref_offs,
                   ref_ampl,
                   ref_VRMS,
                   ref_is_clipping_HI,
                   ref_is_clipping_LO,
                   N_LUT);
          Ser.print(buf);
          // clang-format on

        } else if (strcmp(str_cmd, "lut_ascii?") == 0 ||
                   strcmp(str_cmd, "la?") == 0) {
          // Report the LUT as tab-delimited ASCII. The reported LUT will start
          // at phase = 0 deg. Convenience function handy for debugging from a
          // serial console.
          snprintf(buf, MAXLEN_buf, "%d\n", N_LUT);
          Ser.print(buf);
          for (uint16_t i = 0; i < N_LUT - 1; i++) {
            snprintf(buf, MAXLEN_buf, "%d\t", LUT_wave[i]);
            Ser.print(buf);
          }
          snprintf(buf, MAXLEN_buf, "%d\n", LUT_wave[N_LUT - 1]);
          Ser.print(buf);

        } else if (strcmp(str_cmd, "t?") == 0) {
          // Report time
          uint32_t millis_copy;
          uint16_t micros_part;

          get_systick_timestamp(&millis_copy, &micros_part);
          snprintf(buf, MAXLEN_buf, "%ld.%03d\n", millis_copy, micros_part);
          Ser.print(buf);

        } else if (strcmp(str_cmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser.print("already_off\n");

        } else if (strcmp(str_cmd, "on") == 0) {
          // Start lock-in amp
          LED_on();
          noInterrupts();
          is_running = true;
          interrupts();

        } else if (strcmp(str_cmd, "_on") == 0) {
          // Start lock-in amp and reset the time
          LED_on();
          noInterrupts();
          trigger_reset_time = true;
          is_running = true;
          interrupts();

        } else if (strncmp(str_cmd, "_wave", 5) == 0) {
          // Set the waveform type of the reference signal.
          set_wave(atoi(&str_cmd[5]));
          snprintf(buf, MAXLEN_buf, "%s\n", WAVEFORM_STRING[ref_waveform]);
          Ser.print(buf);

        } else if (strncmp(str_cmd, "_freq", 5) == 0) {
          // Set frequency of the output reference signal [Hz].
          set_freq(atof(&str_cmd[5]));
          snprintf(buf, MAXLEN_buf, "%.3f\n", ref_freq);
          Ser.print(buf);

        } else if (strncmp(str_cmd, "_offs", 5) == 0) {
          // Set offset of the reference signal [V].
          set_offs(atof(&str_cmd[5]));
          snprintf(buf, MAXLEN_buf, "%.3f\n", ref_offs);
          Ser.print(buf);

        } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
          // Set amplitude of the reference signal [V].
          set_ampl(atof(&str_cmd[5]));
          snprintf(buf, MAXLEN_buf, "%.3f\n", ref_ampl);
          Ser.print(buf);

        } else if (strncmp(str_cmd, "_vrms", 5) == 0) {
          // Set amplitude of the reference signal [V_RMS].
          set_VRMS(atof(&str_cmd[5]));
          snprintf(buf, MAXLEN_buf, "%.3f\n", ref_VRMS);
          Ser.print(buf);

        } else if (strcmp(str_cmd, "isr?") == 0) {
#ifndef __SAMD21__
          // Report the execution time durations of the previous `isr_psd()`
          // calls in microseconds.
          for (uint16_t i = 0; i < BLOCK_SIZE - 1; i++) {
            snprintf(buf, MAXLEN_buf, "%d\t", isr_duration[i]);
            Ser.print(buf);
          }
          snprintf(buf, MAXLEN_buf, "%d\n", isr_duration[BLOCK_SIZE - 1]);
          Ser.print(buf);
#endif
        } else if (strcmp(str_cmd, "flash?") == 0) {
          // Report if the flash containing the ADC calibration correction
          // parameters was read succesfully.
          snprintf(buf, MAXLEN_buf, "%d\n", flash_was_read_okay);
          Ser.print(buf);

        } else if (strcmp(str_cmd, "store_autocal") == 0) {
          // Write the ADC autocalibration results to flash. This will wear out
          // the flash, so don't call it unnecessarily. Replies with "1\n".
          flash_storage.write(calibration);
          Ser.print("1\n");

        } else if (strcmp(str_cmd, "autocal") == 0) {
          // Perform an ADC autocalibration. The results will /not/ be stored
          // into flash automatically. Replies with multiple lines of ASCII
          // text, describing the calibration results. The last line will
          // read "Done.\n".
          calibration = ADC_autocalibrate();

          // Report findings
          snprintf(buf, MAXLEN_buf, "%5.3f V reads as %5.3f V, %6.1f bitval\n",
                   calibration.V_LO,
                   calibration.sig_LO * A_REF / ((1 << ADC_INPUT_BITS) - 1),
                   calibration.sig_LO);
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "%5.3f V reads as %5.3f V, %6.1f bitval\n",
                   calibration.V_HI,
                   calibration.sig_HI * A_REF / ((1 << ADC_INPUT_BITS) - 1),
                   calibration.sig_HI);
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "errors: gain = %6.4f, offset = %d\n",
                   calibration.gain_error, calibration.offset_error);
          Ser.print(buf);
          snprintf(buf, MAXLEN_buf, "gaincorr = %d, offsetcorr = %d\n",
                   calibration.gaincorr, calibration.offsetcorr);
          Ser.print(buf);
          Ser.print("Done.\n");

        } else if (strcmp(str_cmd, "autocal?") == 0) {
          // Report the ADC calibration correction parameters.
          snprintf(buf, MAXLEN_buf, "%d\t%d\t%d\n", calibration.is_valid,
                   calibration.gaincorr, calibration.offsetcorr);
          Ser.print(buf);
        }
      }
    }
  }

  // Send buffer out over the serial connection
  if (is_running && (trigger_send_TX_buffer_A || trigger_send_TX_buffer_B)) {
    // NOTE: `write()` can return -1 as indication of an error, e.g. the
    // receiving side being overrun with data.
    if (trigger_send_TX_buffer_A) {
      trigger_send_TX_buffer_A = false;
      Ser.write((uint8_t *)TX_buffer_A, N_BYTES_TX_BUFFER);
    } else {
      trigger_send_TX_buffer_B = false;
      Ser.write((uint8_t *)TX_buffer_B, N_BYTES_TX_BUFFER);
    }
  }
}
