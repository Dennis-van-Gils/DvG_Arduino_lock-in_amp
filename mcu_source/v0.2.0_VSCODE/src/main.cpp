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

  For default hardware startup configuration, see
  SAMD21:
  \.platformio\packages\framework-arduino-samd\cores\arduino\startup.c
  SAMD51:
  \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\startup.c

  Dennis van Gils
  22-07-2019
------------------------------------------------------------------------------*/

#include "Arduino.h"
#include "DvG_SerialCommand.h"
#include "Streaming.h"

#define FIRMWARE_VERSION "ALIA v0.2.0 VSCODE"

// OBSERVATION: Single-ended has half the noise compared to differential
#define ADC_DIFFERENTIAL 1 // Leave at 1. Single-ended is not implemented

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
char mcu_uid[33];                 // Serial number
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

  * Double buffer

    The buffer that will be send each transmission is BLOCK_SIZE samples long
    for each variable. Double the amount of memory is reserved to employ a
    double buffer technique, where alternatingly the first buffer half
    (buffer A) is being written to and the second buffer half (buffer B) is
    being sent, and vice-versa.
*/

// Hint: Maintaining `SAMPLING_PERIOD_us x BLOCK_SIZE` = 0.1 seconds long will
// result in a serial transmit rate of 10 blocks / s, which acts nicely with
// the Python GUI.
#define SAMPLING_PERIOD_us 200
#define BLOCK_SIZE 500

const double SAMPLING_RATE_Hz = (double)1.0e6 / SAMPLING_PERIOD_us;

// clang-format off
const uint16_t DOUBLE_BLOCK_SIZE = 2 * BLOCK_SIZE;
volatile uint32_t buffer_time       [DOUBLE_BLOCK_SIZE] = {0};
volatile uint16_t buffer_ref_X_phase[DOUBLE_BLOCK_SIZE] = {0};
volatile int16_t  buffer_sig_I      [DOUBLE_BLOCK_SIZE] = {0};

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

const uint8_t  N_BYTES_SOM = sizeof(SOM);
const uint16_t N_BYTES_TIME        = BLOCK_SIZE*sizeof(buffer_time[0]);
const uint16_t N_BYTES_REF_X_PHASE = BLOCK_SIZE*sizeof(buffer_ref_X_phase[0]);
const uint16_t N_BYTES_SIG_I       = BLOCK_SIZE*sizeof(buffer_sig_I[0]);
const uint8_t  N_BYTES_EOM = sizeof(EOM);
const uint32_t N_BYTES_TX_BUFFER = N_BYTES_SOM +
                                   N_BYTES_TIME +
                                   N_BYTES_REF_X_PHASE +
                                   N_BYTES_SIG_I +
                                   N_BYTES_EOM;
// clang-format on

volatile bool trigger_send_TX_buffer_A = false;
volatile bool trigger_send_TX_buffer_B = false;

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
#ifdef ARDUINO_SAMD_ZERO
#  define DEBUG
#endif

/*
   *** Tested scenarios
   SAMPLING_PERIOD_us   200 [usec]
   BLOCK_SIZE           500 [samples]
   BAUDRATE             1e6 [only used when Serial is USART]
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

#define SERIAL_DATA_BAUDRATE 1e6 // Only used when Serial is UART

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

#define N_LUT 9000 // (9000 --> 0.04 deg) Number of samples for one full period
uint16_t LUT_array[N_LUT] = {0}; // Look-up table allocation
volatile double LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
volatile double T_period_micros_dbl = 1.0 / ref_freq * 1e6;

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
  LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
  T_period_micros_dbl = 1.0 / ref_freq * 1e6;
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
  Interrupt service routine (ISR) for phase-sentive detection (PSD)
------------------------------------------------------------------------------*/

void isr_psd() {
  static bool is_running_prev = is_running;
  static uint8_t startup_counter = 0;
  static uint16_t write_idx = 0; // Current write index in double buffer
  static uint32_t now_offset = 0;
  static uint16_t LUT_idx_prev = 0;
  uint32_t now = micros();
  uint16_t LUT_idx;
  uint16_t ref_X;
  int16_t sig_I;

  if (is_running != is_running_prev) {
    is_running_prev = is_running;

    if (is_running) {
      startup_counter = 0;
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
    write_idx = 0;
  }

  if (startup_counter <= 4) {
    now_offset = now;
    LUT_idx_prev = 0;
  }

  // Generate reference signal
  // NOTE: `fmod()` takes a significant time, so calculate it /before/ the ADC
  // such that the ADC and DAC conversions are near simultaneous
  LUT_idx = (uint16_t)round(fmod(now - now_offset, T_period_micros_dbl) *
                            LUT_micros2idx_factor);
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
    // stamp_TX_buffer(TX_buffer_A, &LUT_idx);
    // LUT_idx++;
    return;
  }

  // Store the signals
  buffer_time[write_idx] = now;
  buffer_ref_X_phase[write_idx] = LUT_idx_prev;
  buffer_sig_I[write_idx] = sig_I;
  write_idx++;

  // Ready to send the buffer?
  if (write_idx == BLOCK_SIZE) {
#ifdef DEBUG
    N_buffers_scheduled_to_be_sent++;
#endif
    trigger_send_TX_buffer_A = true;
  } else if (write_idx == DOUBLE_BLOCK_SIZE) {
#ifdef DEBUG
    N_buffers_scheduled_to_be_sent++;
#endif
    trigger_send_TX_buffer_B = true;
    write_idx = 0;
  }

  LUT_idx_prev = LUT_idx;
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

#ifdef DEBUG
          Ser_debug << "Already OFF" << endl;
#endif

        } else if ((strcmp(str_cmd, "on") == 0) ||
                   (strcmp(str_cmd, "_on") == 0)) {
          // Start lock-in amp
          noInterrupts();
          is_running = true;
#ifdef DEBUG
          N_buffers_scheduled_to_be_sent = 0;
          N_sent_buffers = 0;
#endif
          interrupts();

#ifdef DEBUG
          Ser_debug << "ON" << endl;
#endif

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
    uint16_t idx;
#ifdef DEBUG
    int32_t bytes_sent = 0;
    uint16_t dropped_buffers = 0;
    bool fError = false;
#endif

    // NOTE: `write()` can return -1 as indication of an error, e.g. the
    // receiving side being overrun with data.
    // size_t w;

    if (trigger_send_TX_buffer_A) {
      trigger_send_TX_buffer_A = false;
      idx = 0;
    } else {
      trigger_send_TX_buffer_B = false;
      idx = BLOCK_SIZE;
    }

#ifdef DEBUG
    int32_t w = Ser_data.write((uint8_t *)&SOM, N_BYTES_SOM);
    if (w == -1) {
      fError = true;
    } else {
      bytes_sent += w;
    }
#else
    Ser_data.write((uint8_t *)&SOM, N_BYTES_SOM);
#endif

#ifdef DEBUG
    w = Ser_data.write((uint8_t *)&buffer_time[idx], N_BYTES_TIME);
    if (w == -1) {
      fError = true;
    } else {
      bytes_sent += w;
    }
#else
    Ser_data.write((uint8_t *)&buffer_time[idx], N_BYTES_TIME);
#endif

#ifdef DEBUG
    w = Ser_data.write((uint8_t *)&buffer_ref_X_phase[idx],
                       N_BYTES_REF_X_PHASE);
    if (w == -1) {
      fError = true;
    } else {
      bytes_sent += w;
    }
#else
    Ser_data.write((uint8_t *)&buffer_ref_X_phase[idx], N_BYTES_REF_X_PHASE);
#endif

#ifdef DEBUG
    w = Ser_data.write((uint8_t *)&buffer_sig_I[idx], N_BYTES_SIG_I);
    if (w == -1) {
      fError = true;
    } else {
      bytes_sent += w;
    }
#else
    Ser_data.write((uint8_t *)&buffer_sig_I[idx], N_BYTES_SIG_I);
#endif

#ifdef DEBUG
    w = Ser_data.write((uint8_t *)&EOM, N_BYTES_EOM);
    if (w == -1) {
      fError = true;
    } else {
      bytes_sent += w;
    }
#else
    Ser_data.write((uint8_t *)&EOM, N_BYTES_EOM);
#endif
    //}

    /*
    #ifdef DEBUG
      Ser_debug << micros() - tick << endl;
    #endif
    //interrupts();   // Uncomment only for debugging purposes
    */

#ifdef DEBUG
    N_sent_buffers++;

    noInterrupts();
    N_buffers_scheduled_to_be_sent--;
    if (N_buffers_scheduled_to_be_sent != 0) {
      dropped_buffers = N_buffers_scheduled_to_be_sent;
      N_buffers_scheduled_to_be_sent = 0;
    }
    interrupts();

    if ((dropped_buffers == 0) && (bytes_sent == N_BYTES_TX_BUFFER)) {
      // Ser_debug << N_sent_buffers << ((idx == 0)?" A ":" B ") << bytes_sent;
      // Ser_debug << " OK" << endl;
    } else {
      Ser_debug << N_sent_buffers << ((idx == 0) ? " A " : " B ") << bytes_sent;
      if (dropped_buffers != 0) {
        Ser_debug << " DROPPED " << dropped_buffers;
      }
      if (fError) {
        Ser_debug << " CAN'T WRITE";
      } else if (bytes_sent != N_BYTES_TX_BUFFER) {
        Ser_debug << " WRONG N_BYTES SENT";
      }
      Ser_debug << endl;
    }
#endif
  }
}