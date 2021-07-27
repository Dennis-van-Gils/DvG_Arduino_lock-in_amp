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
  23-07-2019
------------------------------------------------------------------------------*/

#include "Arduino.h"
#include "DvG_SerialCommand.h"
#include "Streaming.h"

#define FIRMWARE_VERSION "ALIA v0.2.0 VSCODE"

// OBSERVATION: Single-ended has half the noise compared to differential
#define ADC_DIFFERENTIAL 0

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
static inline void syncADC() {
  // Taken from `hri_adc_d21.h`
  while (ADC->STATUS.bit.SYNCBUSY == 1) {}
}
#  define DAC_OUTPUT_BITS 10
#  define ADC_INPUT_BITS 12
#  include "ZeroTimer.h"
#endif

#ifdef __SAMD51__
static inline void syncADC(const Adc *hw, uint32_t reg) {
  // Taken from `hri_adc_d51.h`
  while (((Adc *)hw)->SYNCBUSY.reg & reg) {}
}
#  define DAC_OUTPUT_BITS 12
#  define ADC_INPUT_BITS 12
#  include "SAMD51_InterruptTimer.h"
#endif

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

#ifdef DEBUG
volatile uint16_t N_buffers_scheduled_to_be_sent = 0;
uint16_t N_sent_buffers = 0;
#endif

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
#define SAMPLING_PERIOD_us 50
#define BLOCK_SIZE 2000

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
  Serial
--------------------------------------------------------------------------------

  Arduino M0 Pro
    Serial    : UART , Programming USB port
    SerialUSB : USART, Native USB port

  Adafruit Feather M4 Express
    Serial    : USART
*/

// Define for writing debugging info to the terminal of the second serial port.
// Note: The board needs a second serial port to be used besides the main serial
// port which is assigned to sending buffers of lock-in amp data.

#define SERIAL_DATA_BAUDRATE 1e6 // Only used when Serial is UART

#ifdef ARDUINO_SAMD_ZERO
#  define DEBUG
#endif

#ifdef ARDUINO_SAMD_ZERO
#  define Ser_data SerialUSB
#  ifdef DEBUG
#    define Ser_debug Serial
#  endif
#else
#  define Ser_data Serial
#endif

// Instantiate serial command listener
DvG_SerialCommand sc_data(Ser_data);

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
#define N_LUT 20000              // Suggest setting equal to `SAMPLING_RATE_Hz`
uint16_t LUT_array[N_LUT] = {0}; // Look-up table allocation
volatile uint16_t idx_per_iter;  // LUT_idx per DAQ_iter, depends on `ref_freq`

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define MAX_DAC_OUTPUT_BITVAL ((1 << DAC_OUTPUT_BITS) - 1)

void compute_LUT() {
  double norm_offs = ref_offs / A_REF; // Normalized
  double norm_ampl = ref_ampl / A_REF; // Normalized
  double wave;

#ifdef DEBUG
  Ser_debug << "Creating LUT...";
#endif

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

    LUT_array[i] = (uint16_t)round(MAX_DAC_OUTPUT_BITVAL * wave);
  }

#ifdef DEBUG
  Ser_debug << " done." << endl;
#endif
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
  idx_per_iter = (uint16_t)round(N_LUT / SAMPLING_RATE_Hz * ref_freq);
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

void isr_psd() {
  static uint32_t DAQ_iter = 0;
  static bool is_running_prev = is_running;
  static uint8_t startup_counter = 0;
  static bool using_TX_buffer_A = true; // When false: Using TX_buffer_B
  static uint16_t write_idx = 0;        // Current write index of TX_buffer
  static uint16_t LUT_idx_prev = 0;
  uint16_t LUT_idx;
  uint16_t ref_X;
  int16_t sig_I = 0;

  uint32_t millis_copy;
  uint16_t micros_part;
  get_systick_timestamp(&millis_copy, &micros_part);

  if (is_running != is_running_prev) {
    is_running_prev = is_running;

    if (is_running) {
      startup_counter = 0;
      DAQ_iter = 0;
      digitalWrite(PIN_LED, HIGH);
    } else {
      // Set output voltage to 0
#if defined(__SAMD21__)
      DAC->DATA.reg = 0;
#elif defined(__SAMD51__)
      DAC->DATA[0].reg = 0;
#endif
      digitalWrite(PIN_LED, LOW);
    }
  }

  if (!is_running) {
    return;
  }

  if (startup_counter == 0) {
    trigger_send_TX_buffer_A = false;
    trigger_send_TX_buffer_B = false;
    using_TX_buffer_A = true;
    write_idx = 0;
  }

  if (startup_counter <= 4) {
    // now_offset = now;
    LUT_idx_prev = 0;
  }

  // Generate reference signal
  LUT_idx = mulmod_int(idx_per_iter, DAQ_iter, N_LUT);
  ref_X = LUT_array[LUT_idx];

  // Read input signal corresponding to the DAC output of the previous
  // timestep. This ensures that the previously set DAC output has had enough
  // time to stabilize.
#ifdef __SAMD21__
  ADC->SWTRIG.bit.START = 1;
  syncADC();
  sig_I = ADC->RESULT.reg;
#elif defined __SAMD51__
  ADC0->SWTRIG.bit.START = 1;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  sig_I = ADC0->RESULT.reg;
#endif

  // Output reference signal
#ifdef __SAMD21__
  DAC->DATA.reg = ref_X;
#elif defined __SAMD51__
  DAC->DATA[0].reg = ref_X;
#endif

  if (startup_counter < 4) {
    startup_counter++;
    return;
  } else if (startup_counter == 4) {
    startup_counter++;
    stamp_TX_buffer(TX_buffer_A);
    // LUT_idx++;
    return;
  }

  // Store the signals
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

  if (write_idx == BLOCK_SIZE) {
    if (using_TX_buffer_A) {
      trigger_send_TX_buffer_A = true;
      stamp_TX_buffer(TX_buffer_B);
    } else {
      trigger_send_TX_buffer_B = true;
      stamp_TX_buffer(TX_buffer_A);
    }

    using_TX_buffer_A = !using_TX_buffer_A;
    write_idx = 0;
  }

  LUT_idx_prev = LUT_idx;
  DAQ_iter++;
}

/*------------------------------------------------------------------------------
  setup
------------------------------------------------------------------------------*/

void setup() {
  Ser_data.begin(SERIAL_DATA_BAUDRATE);
#ifdef DEBUG
  Ser_debug.begin(9600);
#endif
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

  // DAC
  analogWriteResolution(DAC_OUTPUT_BITS);
  analogWrite(A0, 0);

  // ADC
  // Increase the ADC clock by setting the PRESCALER from default DIV128 to a
  // smaller divisor. This is needed for DAQ rates larger than ~20 kHz on
  // SAMD51 and DAQ rates larger than ~10 kHz on SAMD21. Setting too small
  // divisors will result in ADC errors. Keep as large as possible to increase
  // ADC accuracy.

  // analogReadResolution(ADC_INPUT_BITS);
  // analogRead(A1); // Differential(+)
  // analogRead(A2); // Differential(-)

#ifdef __SAMD21__
  // ADC source clock is default at 48 MHz (Generic Clock Generator 0)
  // DAC source clock is default at 48 MHz (Generic Clock Generator 0)
  // See
  // \.platformio\packages\framework-arduino-samd\cores\arduino\wiring.c
  //
  // Handy calculator:
  // https://blog.thea.codes/getting-the-most-out-of-the-samd21-adc/

  ADC->CTRLA.reg = 0;
  syncADC();

  // Load the factory calibration
  uint32_t bias =
      (*((uint32_t *)ADC_FUSES_BIASCAL_ADDR) & ADC_FUSES_BIASCAL_Msk) >>
      ADC_FUSES_BIASCAL_Pos;
  uint32_t linearity =
      (*((uint32_t *)ADC_FUSES_LINEARITY_0_ADDR) & ADC_FUSES_LINEARITY_0_Msk) >>
      ADC_FUSES_LINEARITY_0_Pos;
  linearity |= ((*((uint32_t *)ADC_FUSES_LINEARITY_1_ADDR) &
                 ADC_FUSES_LINEARITY_1_Msk) >>
                ADC_FUSES_LINEARITY_1_Pos)
               << 5;

  ADC->CALIB.reg =
      ADC_CALIB_BIAS_CAL(bias) | ADC_CALIB_LINEARITY_CAL(linearity);
  // No sync needed according to `hri_adc_d21.h`

  // The ADC clock must remain below 2.1 MHz, see SAMD21 datasheet Table
  // 37-24. Hence, don't go below DIV32 @ 48 MHz.
  ADC->CTRLB.bit.PRESCALER = ADC_CTRLB_PRESCALER_DIV32_Val;
  syncADC();

  // AnalogRead resolution
  ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
  syncADC();

  // Sample averaging
  ADC->AVGCTRL.bit.SAMPLENUM = ADC_AVGCTRL_SAMPLENUM_4_Val;
  // No sync needed according to `hri_adc_d21.h`
  ADC->AVGCTRL.bit.ADJRES = 2; // 2^N, must match `ADC0->AVGCTRL.bit.SAMPLENUM`
  // No sync needed according to `hri_adc_d21.h`

  // Sampling length, larger means increased max input impedance
  // default 63, stable 15 @ DIV32 & SAMPLENUM_4
  ADC->SAMPCTRL.bit.SAMPLEN = 15;
  // No sync needed according to `hri_adc_d21.h`

#  if ADC_DIFFERENTIAL == 1
  ADC->CTRLB.bit.DIFFMODE = 1;
  syncADC();
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC();
  ADC->INPUTCTRL.bit.MUXNEG = g_APinDescription[A2].ulADCChannelNumber;
  syncADC();
#  else
  ADC->CTRLB.bit.DIFFMODE = 0;
  syncADC();
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC();
  ADC->INPUTCTRL.bit.MUXNEG = ADC_INPUTCTRL_MUXNEG_GND_Val;
  syncADC();
#  endif

  ADC->REFCTRL.bit.REFSEL = ADC_REFCTRL_REFSEL_INTVCC1_Val; // 1/2 VDDANA
  // No sync needed according to `hri_adc_d21.h`
  ADC->REFCTRL.bit.REFCOMP = 0;
  // No sync needed according to `hri_adc_d21.h`

  ADC->INPUTCTRL.bit.GAIN = ADC_INPUTCTRL_GAIN_DIV2_Val;
  syncADC();

  // Prepare for software-triggered acquisition
  ADC->CTRLA.bit.ENABLE = 1;
  syncADC();

#elif defined __SAMD51__
  // ADC source clock is default at 48 MHz (Generic Clock Generator 1)
  // DAC source clock is default at 12 MHz (Generic Clock Generator 4)
  // See
  // \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\wiring.c

  ADC0->CTRLA.reg = 0;
  syncADC(ADC0, ADC_SYNCBUSY_SWRST | ADC_SYNCBUSY_ENABLE);

  // Load the factory calibration
  uint32_t biascomp =
      (*((uint32_t *)ADC0_FUSES_BIASCOMP_ADDR) & ADC0_FUSES_BIASCOMP_Msk) >>
      ADC0_FUSES_BIASCOMP_Pos;
  uint32_t biasr2r =
      (*((uint32_t *)ADC0_FUSES_BIASR2R_ADDR) & ADC0_FUSES_BIASR2R_Msk) >>
      ADC0_FUSES_BIASR2R_Pos;
  uint32_t biasref =
      (*((uint32_t *)ADC0_FUSES_BIASREFBUF_ADDR) & ADC0_FUSES_BIASREFBUF_Msk) >>
      ADC0_FUSES_BIASREFBUF_Pos;

  ADC0->CALIB.reg = ADC_CALIB_BIASREFBUF(biasref) | ADC_CALIB_BIASR2R(biasr2r) |
                    ADC_CALIB_BIASCOMP(biascomp);
  // No sync needed according to `hri_adc_d51.h`

  // The ADC clock must remain below 12 MHz, see SAMD51 datasheet Table 54-28.
  ADC0->CTRLA.bit.PRESCALER = ADC_CTRLA_PRESCALER_DIV16_Val;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

  // AnalogRead resolution
  ADC0->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

  // Sample averaging
  ADC0->AVGCTRL.bit.SAMPLENUM = ADC_AVGCTRL_SAMPLENUM_4_Val;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->AVGCTRL.bit.ADJRES = 2; // 2^N, must match `ADC0->AVGCTRL.bit.SAMPLENUM`
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

  // Sampling length, larger means increased max input impedance
  // default 5, stable 15 @ DIV16 & SAMPLENUM_4
  ADC0->SAMPCTRL.bit.SAMPLEN = 15;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

#  if ADC_DIFFERENTIAL == 1
  ADC0->INPUTCTRL.bit.DIFFMODE = 1;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->INPUTCTRL.bit.MUXNEG = g_APinDescription[A2].ulADCChannelNumber;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

  // Rail-2-rail operation, needed for proper diffmode
  ADC0->CTRLA.bit.R2R = 1;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
#  else
  ADC0->INPUTCTRL.bit.DIFFMODE = 0;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->INPUTCTRL.bit.MUXNEG = ADC_INPUTCTRL_MUXNEG_GND_Val;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
#  endif

  ADC0->REFCTRL.bit.REFSEL = ADC_REFCTRL_REFSEL_INTVCC1_Val; // VDDANA
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->REFCTRL.bit.REFCOMP = 0;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);

  /*
  ADC0->GAINCORR.reg = (1 << 11) - 8;
  syncADC(ADC0, ADC_SYNCBUSY_GAINCORR);
  ADC0->OFFSETCORR.reg = 18;
  syncADC(ADC0, ADC_SYNCBUSY_OFFSET);
  ADC0->CTRLB.bit.CORREN = 1;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  */

  // Prepare for software-triggered acquisition
  ADC0->CTRLA.bit.ENABLE = 1;
  syncADC(ADC0, ADC_SYNCBUSY_SWRST | ADC_SYNCBUSY_ENABLE);

#endif

  // Initial waveform LUT
  set_wave(Cosine);
  set_freq(110.0); // [Hz]    Wanted startup frequency
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
  uint32_t prev_millis = 0;

  // Process commands on the data channel
  // Deliberately slowed down to once every 1 ms to improve timing stability of
  // 'isr_psd()'.
  if ((millis() - prev_millis) > 1) {
    prev_millis = millis();

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
        /*-------------
          Not running
        ---------------
          We are ready to process any incoming commands.
        */
        str_cmd = sc_data.getCmd();

        if (strcmp(str_cmd, "id?") == 0) {
          // Report identity string
          Ser_data << "Arduino, Alia" << endl;

        } else if (strcmp(str_cmd, "mcu?") == 0) {
          // Report microcontroller information
          // clang-format off
          Ser_data << FIRMWARE_VERSION << '\t'
                   << MCU_MODEL << '\t'
                   << SystemCoreClock << '\t'
                   << mcu_uid << endl;
          // clang-format on

        } else if (strcmp(str_cmd, "adc?") == 0) {
          // Report ADC registers, debug information
          // clang-format off
          uint8_t w = 15;
          Ser_data << _PAD(40, '-') << endl
#if defined __SAMD21__
                   << "CTRLA" << endl
                   << _WIDTH(".ENABLE", w) << "  F" << _DEC(ADC->CTRLA.bit.ENABLE) << endl
                   << "INPUTCTRL" << endl
                   << _WIDTH(".MUXPOS", w) << "  0x" << _HEX(ADC->INPUTCTRL.bit.MUXPOS) << endl
                   << _WIDTH(".MUXNEG", w) << "  0x" << _HEX(ADC->INPUTCTRL.bit.MUXNEG) << endl
                   << _WIDTH(".GAIN", w) << "  0x" << _HEX(ADC->INPUTCTRL.bit.GAIN) << endl
                   << "REFCTRL" << endl
                   << _WIDTH(".REFCOMP", w) << "  F" << _DEC(ADC->REFCTRL.bit.REFCOMP) << endl
                   << _WIDTH(".REFSEL", w) << "  0x" << _HEX(ADC->REFCTRL.bit.REFSEL) << endl
                   << "AVGCTRL" << endl
                   << _WIDTH(".ADJRES", w) << "  0x" << _HEX(ADC->AVGCTRL.bit.ADJRES) << endl
                   << _WIDTH(".SAMPLENUM", w) << "  0x" << _HEX(ADC->AVGCTRL.bit.SAMPLENUM) << endl
                   << "SAMPCTRL" << endl
                   << _WIDTH(".SAMPLEN", w) << "  " << _DEC(ADC->SAMPCTRL.bit.SAMPLEN) << endl
                   << "CTRLB" << endl
                   << _WIDTH(".RESSEL", w) << "  0x" << _HEX(ADC->CTRLB.bit.RESSEL) << endl
                   << _WIDTH(".CORREN", w) << "  F" << _DEC(ADC->CTRLB.bit.CORREN) << endl
                   << _WIDTH(".LEFTADJ", w) << "  F" <<  _DEC(ADC->CTRLB.bit.LEFTADJ) << endl
                   << _WIDTH(".DIFFMODE", w) << "  F" << _DEC(ADC->CTRLB.bit.DIFFMODE) << endl
                   << _WIDTH(".PRESCALER", w) << "  0x" << _HEX(ADC->CTRLB.bit.PRESCALER) << endl
                   << "CALIB" << endl
                   << _WIDTH(".LINEARITY_CAL", w) << "  " << _DEC(ADC->CALIB.bit.LINEARITY_CAL) << endl
                   << _WIDTH(".BIAS_CAL", w) << "  " << _DEC(ADC->CALIB.bit.BIAS_CAL) << endl
                   << "OFFSETCORR       " <<  _DEC(ADC->OFFSETCORR.bit.OFFSETCORR) << endl
                   << "GAINCORR         " <<  _DEC(ADC->GAINCORR.bit.GAINCORR) << endl
#elif defined __SAMD51__
                   << "INPUTCTRL" << endl
                   << _WIDTH(".DIFFMODE", w) << "  F" << _DEC(ADC0->INPUTCTRL.bit.DIFFMODE) << endl
                   << _WIDTH(".MUXPOS", w) << "  0x" << _HEX(ADC0->INPUTCTRL.bit.MUXPOS) << endl
                   << _WIDTH(".MUXNEG", w) << "  0x" << _HEX(ADC0->INPUTCTRL.bit.MUXNEG) << endl
                   << "CTRLA" << endl
                   << _WIDTH(".ENABLE", w) << "  F" << _DEC(ADC0->CTRLA.bit.ENABLE) << endl
                   << _WIDTH(".PRESCALER", w) << "  0x" << _HEX(ADC0->CTRLA.bit.PRESCALER) << endl
                   << _WIDTH(".R2R", w) << "  F" << _DEC(ADC0->CTRLA.bit.R2R) << endl
                   << "CTRLB" << endl
                   << _WIDTH(".RESSEL", w) << "  0x" << _HEX(ADC0->CTRLB.bit.RESSEL) << endl
                   << _WIDTH(".CORREN", w) << "  F" << _DEC(ADC0->CTRLB.bit.CORREN) << endl
                   << _WIDTH(".LEFTADJ", w) << "  F" <<  _DEC(ADC0->CTRLB.bit.LEFTADJ) << endl
                   << "REFCTRL" << endl
                   << _WIDTH(".REFCOMP", w) << "  F" << _DEC(ADC0->REFCTRL.bit.REFCOMP) << endl
                   << _WIDTH(".REFSEL", w) << "  0x" << _HEX(ADC0->REFCTRL.bit.REFSEL) << endl
                   << "AVGCTRL" << endl
                   << _WIDTH(".ADJRES", w) << "  0x" << _HEX(ADC0->AVGCTRL.bit.ADJRES) << endl
                   << _WIDTH(".SAMPLENUM", w) << "  0x" << _HEX(ADC0->AVGCTRL.bit.SAMPLENUM) << endl
                   << "SAMPCTRL" << endl
                   << _WIDTH(".OFFCOMP", w) << "  F" << _DEC(ADC0->SAMPCTRL.bit.OFFCOMP) << endl
                   << _WIDTH(".SAMPLEN", w) << "  " << _DEC(ADC0->SAMPCTRL.bit.SAMPLEN) << endl
                   << "CALIB" << endl
                   << _WIDTH(".BIASCOMP", w) << "  " << _DEC(ADC0->CALIB.bit.BIASCOMP) << endl
                   << _WIDTH(".BIASREFBUF", w) << "  " << _DEC(ADC0->CALIB.bit.BIASREFBUF) << endl
                   << _WIDTH(".BIASR2R", w) << "  " << _DEC(ADC0->CALIB.bit.BIASR2R) << endl
                   << "OFFSETCORR       " <<  _DEC(ADC0->OFFSETCORR.bit.OFFSETCORR) << endl
                   << "GAINCORR         " <<  _DEC(ADC0->GAINCORR.bit.GAINCORR) << endl
#endif
                   << _PAD(40, '-') << endl;
          // clang-format on

        } else if (strcmp(str_cmd, "debug?") == 0) {
          // Report debug information
          float block_rate = SAMPLING_RATE_Hz / BLOCK_SIZE;
          uint32_t baudrate = ceil(N_BYTES_TX_BUFFER * 10 * block_rate);
          // 8 data bits + 1 start bit + 1 stop bit = 10 bits per data byte
          // clang-format off
          uint8_t w1 = 15;
          uint8_t w2 = 6;
          Ser_data << _PAD(40, '-') << endl
                   << _WIDTH("DAQ rate", w1) << ":  "
                   << _FLOATW(SAMPLING_RATE_Hz, 0, w2) << "  Hz" << endl
                   << _WIDTH("ISR clock", w1) << ":  "
                   << _FLOATW(SAMPLING_PERIOD_us, 0, w2) << "  usec" << endl
                   << _WIDTH("Block size", w1) << ":  "
                   << _FLOATW(BLOCK_SIZE, 0, w2) << "  samples" << endl
                   << _WIDTH("Block size", w1) << ":  "
                   << _FLOATW(N_BYTES_TX_BUFFER, 0, w2) << "  bytes" << endl
                   << _WIDTH("Transmit rate", w1) << ":  "
                   << _FLOATW(block_rate, 2, w2) << "  blocks/s" << endl
                   << _WIDTH("Baudrate", w1) << ":  "
                   << _FLOATW(baudrate, 0, w2) << endl
                   << _PAD(40, '-') << endl;
          // clang-format on

        } else if (strcmp(str_cmd, "const?") == 0) {
          // Report lock-in amplifier constants
          // clang-format off
          Ser_data << SAMPLING_PERIOD_us << '\t'
                   << BLOCK_SIZE << '\t'
                   << N_BYTES_TX_BUFFER << '\t'
                   << DAC_OUTPUT_BITS << '\t'
                   << ADC_INPUT_BITS << '\t'
                   << ADC_DIFFERENTIAL << '\t'
                   << A_REF << endl;
          // clang-format on

        } else if (strcmp(str_cmd, "ref?") == 0 || strcmp(str_cmd, "?") == 0) {
          // Report reference signal `ref_X` settings
          // clang-format off
          Ser_data << WAVEFORM_STRING[ref_waveform] << '\t'
                   << _FLOAT(ref_freq, 3) << '\t'
                   << _FLOAT(ref_offs, 3) << '\t'
                   << _FLOAT(ref_ampl, 3) << '\t'
                   << _FLOAT(ref_VRMS, 3) << '\t'
                   << ref_is_clipping_HI << '\t'
                   << ref_is_clipping_LO << '\t'
                   << N_LUT << endl;
          // clang-format on

        } else if (strcmp(str_cmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser_data.print("already_off\n");

        } else if (strcmp(str_cmd, "on") == 0) {
          // Start lock-in amp
          noInterrupts();
          is_running = true;
          interrupts();

        } else if (strcmp(str_cmd, "_on") == 0) {
          // Start lock-in amp and reset the time
          noInterrupts();
          trigger_reset_time = true;
          is_running = true;
          interrupts();

        } else if (strncmp(str_cmd, "_wave", 5) == 0) {
          // Set the waveform type of the reference signal.
          set_wave(atoi(&str_cmd[5]));
          Ser_data.println(WAVEFORM_STRING[ref_waveform]);

        } else if (strncmp(str_cmd, "_freq", 5) == 0) {
          // Set frequency of the output reference signal [Hz]
          set_freq(atof(&str_cmd[5]));
          Ser_data.println(ref_freq, 3);

        } else if (strncmp(str_cmd, "_offs", 5) == 0) {
          // Set voltage offset of cosine reference signal [V]
          set_offs(atof(&str_cmd[5]));
          Ser_data.println(ref_offs, 3);

        } else if (strncmp(str_cmd, "_ampl", 5) == 0) {
          // Set voltage amplitude of cosine reference signal [V]
          set_ampl(atof(&str_cmd[5]));
          Ser_data.println(ref_ampl, 3);

        } else if (strncmp(str_cmd, "_vrms", 5) == 0) {
          // Set amplitude of the reference signal [V_RMS].
          set_VRMS(atof(&str_cmd[5]));
          Ser_data.println(ref_VRMS, 3);
        }
      }
    }
  }

  // Send buffer out over the serial connection
  if (is_running && (trigger_send_TX_buffer_A || trigger_send_TX_buffer_B)) {
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
  }
}