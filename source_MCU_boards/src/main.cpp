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
- Adafruit Feather M4           SAMD51J19A   okay     ADAFRUIT_FEATHER_M4_EXPRESS
- Adafruit ItsyBitsy M4         SAMD51G19A   okay     ADAFRUIT_ITSYBITSY_M4_EXPRESS

When using Visual Studio Code as IDE set the following user settings to have
proper bindings to strncmpi
"C_Cpp.intelliSenseEngine": "Tag Parser"
"C_Cpp.intelliSenseEngineFallback": "Disabled"

Dennis van Gils
28-03-2019
------------------------------------------------------------------------------*/

#include <Arduino.h>
#include "DvG_SerialCommand.h"
#include "Streaming.h"

#if defined(__SAMD21G18A__) || \
    defined(__SAMD21E18A__)
  #ifndef __SAMD21__
    #define __SAMD21__
  #endif
  #include "ZeroTimer.h"
#elif defined(__SAMD51P20A__) || \
      defined(__SAMD51J19A__) || \
      defined(__SAMD51G19A__)
  #ifndef __SAMD51__
    #define __SAMD51__
  #endif
  #include "SAMD51_InterruptTimer.h"
#endif

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static __inline__ void syncADC() __attribute__((always_inline, unused));
#if defined(__SAMD21__)
  static void syncDAC() {while (DAC->STATUS.bit.SYNCBUSY == 1);}
  static void syncADC() {while (ADC->STATUS.bit.SYNCBUSY == 1);}
#elif defined(__SAMD51__)
  static void syncDAC() {while (DAC->STATUS.bit.EOC0 == 1);}
  static void syncADC() {while (ADC0->STATUS.bit.ADCBUSY == 1);}
#endif

// Define for writing debugging info to the terminal of the second serial port.
// Note: The board needs a second serial port to be used besides the main serial
// port which is assigned to sending buffers of lock-in amp data.
#if defined(ARDUINO_SAMD_ZERO)
  #define DEBUG
#endif

// Arduino M0 Pro
// Serial   : Programming USB port (UART).
// SerialUSB: Native USB port (USART). Baudrate setting gets ignored and is
//            always as fast as possible.
/*
   *** Tested scenarios
   ISR_CLOCK   200 [usec]
   BUFFER_SIZE 500 [samples]
   BAUDRATE    1e6 [only used when #define Ser_data Serial]
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
#define SERIAL_DATA_BAUDRATE 1e6  // Only used when '#define Ser_data Serial'

#if defined(ARDUINO_SAMD_ZERO)
  #define Ser_data    SerialUSB
  #ifdef DEBUG
    #define Ser_debug Serial
  #endif
#else
  #define Ser_data    Serial
#endif

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);

// Interrupt service routine clock
// Findings using Arduino M0 Pro:
//   ISR_CLOCK: min.  40 usec for only writing A0, no serial
//              min.  50 usec for writing A0 and reading A1, no serial
//              min.  80 usec for writing A0 and reading A1, with serial
#define ISR_CLOCK 200     // [usec]

// Buffers
// The buffer that will be send each transmission is BUFFER_SIZE samples long
// for each variable. Double the amount of memory is reserved to employ a double
// buffer technique, where alternatingly the first buffer half (buffer A) is
// being written to and the second buffer half (buffer B) is being sent.
#define BUFFER_SIZE 500   // [samples]

/* Tested settings Arduino M0 Pro
Case A: turbo and stable on computer Onera, while only graphing and logging in
        Python without FIR filtering
  ISR_CLOCK   80
  BUFFER_SIZE 625
  DAQ --> 12500 Hz
Case B: stable on computer Onera, while graphing, logging and FIR filtering in
        Python
  ISR_CLOCK   200
  BUFFER_SIZE 500
  DAQ --> 5000 Hz
*/

const uint16_t DOUBLE_BUFFER_SIZE = 2 * BUFFER_SIZE;
volatile uint32_t buffer_time       [DOUBLE_BUFFER_SIZE] = {0};
volatile uint16_t buffer_ref_X_phase[DOUBLE_BUFFER_SIZE] = {0};
volatile int16_t  buffer_sig_I      [DOUBLE_BUFFER_SIZE] = {0};

// Serial transmission sentinels: start and end of message
const char SOM[] = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
const char EOM[] = {0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x00, 0xff, 0x7f};

const uint8_t  N_BYTES_SOM = sizeof(SOM);
const uint16_t N_BYTES_TIME        = BUFFER_SIZE*sizeof(buffer_time[0]);
const uint16_t N_BYTES_REF_X_PHASE = BUFFER_SIZE*sizeof(buffer_ref_X_phase[0]);
const uint16_t N_BYTES_SIG_I       = BUFFER_SIZE*sizeof(buffer_sig_I[0]);
const uint8_t  N_BYTES_EOM = sizeof(EOM);
const uint32_t N_BYTES_TRANSMIT_BUFFER = N_BYTES_SOM +
                                         N_BYTES_TIME +
                                         N_BYTES_REF_X_PHASE +
                                         N_BYTES_SIG_I +
                                         N_BYTES_EOM;

volatile bool fSend_buffer_A = false;
volatile bool fSend_buffer_B = false;

// Analog port resolutions
#if defined(__SAMD21__)
  #define ANALOG_WRITE_RESOLUTION 10  // [bits] DAC
#elif defined(__SAMD51__)
  #define ANALOG_WRITE_RESOLUTION 12  // [bits] DAC
#endif
#define ANALOG_READ_RESOLUTION 12     // [bits] ADC

/*------------------------------------------------------------------------------
    Cosine wave look-up table (LUT)
------------------------------------------------------------------------------*/

double ref_freq = 110.0;      // [Hz], aka f_R

// Tip: Limiting the output voltage range to slighty above 0.0 V will improve
// the shape of the cosine wave at its minimum. Apparently, the analog out port
// has difficulty in cleanly dropping the output voltage completely to 0.0 V.
#define A_REF        3.300    // [V] Analog voltage reference Arduino
double ref_V_offset = 1.7;    // [V] Voltage offset of cos. reference signal
double ref_V_ampl   = 1.414;  // [V] Voltage amplitude of cos. reference signal

#define N_LUT 9000  // (9000 --> 0.04 deg) Number of samples for one full period
volatile double LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
volatile double T_period_micros_dbl = 1.0 / ref_freq * 1e6;
uint16_t LUT_cos[N_LUT] = {0};
//uint16_t LUT_sin[N_LUT] = {0};

void create_LUT() {
  uint16_t ANALOG_WRITE_MAX_BITVAL = pow(2, ANALOG_WRITE_RESOLUTION) - 1;
  double norm_offset = ref_V_offset / A_REF;    // Normalized
  double norm_ampl   = ref_V_ampl   / A_REF;    // Normalized
  double cosine_value;

  #ifdef DEBUG
    Ser_debug << "Creating LUT...";
  #endif

  for (uint16_t i = 0; i < N_LUT; i++) {
    cosine_value = norm_offset + norm_ampl * cos(2*PI*i/N_LUT);
    cosine_value = max(cosine_value, 0.0);
    cosine_value = min(cosine_value, 1.0);
    LUT_cos[i] = (uint16_t) round(ANALOG_WRITE_MAX_BITVAL * cosine_value);
  }

  /*
  for (uint16_t i = 0; i < N_LUT; i++) {
    LUT_sin[i] = LUT_cos[(i + N_LUT/4*3) % N_LUT];
  }
  */

  #ifdef DEBUG
    Ser_debug << " done." << endl;
  #endif
}

/*------------------------------------------------------------------------------
    Interrupt service routine (isr) for phase-sentive detection (psd)
------------------------------------------------------------------------------*/
volatile bool fRunning = false;
volatile uint16_t N_buffers_scheduled_to_be_sent = 0;
uint16_t N_sent_buffers = 0;

void isr_psd() {
  static bool fPrevRunning = fRunning;
  static bool fStartup = true;
  static uint16_t write_idx1 = 0;   // Current write index in double buffer
  static uint16_t write_idx2 = 0;   // Current write index in double buffer

  if (fRunning != fPrevRunning) {
    fPrevRunning = fRunning;
    if (fRunning) {
      digitalWrite(PIN_LED, HIGH);  // Indicate lock-in amp is running
      fStartup = true;
    } else {
      digitalWrite(PIN_LED, LOW);   // Indicate lock-in amp is off
      syncDAC();
      #if defined(__SAMD21__)
        DAC->DATA.reg = 0;          // Set output voltage to 0
      #elif defined(__SAMD51__)
        DAC->DATA[0].reg = 0;       // Set output voltage to 0
      #endif
    }
  }
  if (!fRunning) {return;}

  // Generate reference signals
  uint32_t now = micros();
  static uint32_t now_offset = 0;
  if (fStartup) {now_offset = now;} // Force cosine to start at phase = 0 deg
  uint16_t LUT_idx = round(fmod(now - now_offset, T_period_micros_dbl) * \
                           LUT_micros2idx_factor);
  uint16_t ref_X = LUT_cos[LUT_idx];

  // Output reference signal
  syncDAC();
  #if defined(__SAMD21__)
    DAC->DATA.reg = ref_X;
  #elif defined(__SAMD51__)
    DAC->DATA[0].reg = ref_X;
  #endif

  // Read input signal
  syncADC();
  #if defined(__SAMD21__)
    ADC->SWTRIG.bit.START = 1;
    while (ADC->INTFLAG.bit.RESRDY == 0);   // Wait for conversion to complete
    int16_t sig_I = ADC->RESULT.reg;
  #elif defined(__SAMD51__)
    ADC0->SWTRIG.bit.START = 1;
    while (ADC0->INTFLAG.bit.RESRDY == 0);  // Wait for conversion to complete
    int16_t sig_I = ADC0->RESULT.reg;
  #endif

  // Store in buffers
  if (fStartup) {
    buffer_time[0] = now;
    buffer_ref_X_phase[0] = LUT_idx;
    buffer_sig_I[0] = sig_I;
    write_idx1 = 1;
    write_idx2 = 0;
    fStartup = false;
  } else {
    buffer_time[write_idx1] = now;
    buffer_ref_X_phase[write_idx1] = LUT_idx;
    buffer_sig_I[write_idx2] = sig_I;
    write_idx1++;
    write_idx2++;
  }

  // Ready to send the buffer?
  if (write_idx1 == BUFFER_SIZE) {
    N_buffers_scheduled_to_be_sent++;
    fSend_buffer_A = true;
  } else if (write_idx1 == DOUBLE_BUFFER_SIZE) {
    N_buffers_scheduled_to_be_sent++;
    fSend_buffer_B = true;
    write_idx1 = 0;
  }

  if (write_idx2 == DOUBLE_BUFFER_SIZE) {write_idx2 = 0;}
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
        Ser_debug << "  .INPUTOFFSET: " << _HEX(ADC->INPUTCTRL.bit.INPUTOFFSET) << endl;
        Ser_debug << "  .INPUTSCAN  : " << _HEX(ADC->INPUTCTRL.bit.INPUTSCAN) << endl;
        Ser_debug << "  .MUXNEG     : " << _HEX(ADC->INPUTCTRL.bit.MUXNEG) << endl;
        Ser_debug << "  .MUXPOS     : " << _HEX(ADC->INPUTCTRL.bit.MUXPOS) << endl;
      #elif defined (__SAMD51__)
      // TO DO
      #endif

      float DAQ_rate = 1.0e6 / ISR_CLOCK;
      float buffer_rate = DAQ_rate / BUFFER_SIZE;
      // 8 data bits + 1 start bit + 1 stop bit = 10 bits per data byte
      uint32_t baud = ceil(N_BYTES_TRANSMIT_BUFFER * 10 * buffer_rate);
      Ser_debug << "----------------------------------------" << endl;
      Ser_debug << "ISR clock    : " << ISR_CLOCK << " usec" << endl;
      Ser_debug << "DAQ rate     : " << _FLOAT(DAQ_rate, 2) << " Hz" << endl;
      Ser_debug << "Buffer size  : " << BUFFER_SIZE << " samples" << endl;
      Ser_debug << "Transmit rate          : " << _FLOAT(buffer_rate, 2)
                << " buffers/s" << endl;
      Ser_debug << "Data bytes per transmit: " << N_BYTES_TRANSMIT_BUFFER
                << " bytes" << endl;
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

  // Use built-in LED to signal running state of lock-in amp
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, fRunning);

  // DAC
  analogWriteResolution(ANALOG_WRITE_RESOLUTION);
  analogWrite(A0, 0);

  // ADC
  // Increase the ADC clock by setting the divisor from default DIV128 to DIV16.
  // Setting smaller divisors than DIV16 results in ADC errors.
  #if defined(__SAMD21__)
    ADC->CTRLB.bit.PRESCALER = ADC_CTRLB_PRESCALER_DIV16_Val;
  #elif defined(__SAMD51__)
    ADC0->CTRLA.bit.PRESCALER = ADC_CTRLA_PRESCALER_DIV16_Val;
  #endif
  analogReadResolution(ANALOG_READ_RESOLUTION);
  analogRead(A1); // Differential +
  analogRead(A2); // Differential -

  // Set differential mode on A1(+) and A2(-)
  #if defined (__SAMD21__)
    ADC->CTRLB.bit.DIFFMODE = 1;
    ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[PIN_A1].ulADCChannelNumber;
    ADC->INPUTCTRL.bit.MUXNEG = g_APinDescription[PIN_A2].ulADCChannelNumber;
    ADC->INPUTCTRL.bit.GAIN = ADC_INPUTCTRL_GAIN_DIV2_Val;
    ADC->REFCTRL.bit.REFSEL = 2;  // 2: INTVCC1 on SAMD21 = 1/2 VDDANA
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
    //*/

    ADC0->INPUTCTRL.bit.DIFFMODE = 1;
    ADC0->INPUTCTRL.bit.MUXPOS = g_APinDescription[PIN_A1].ulADCChannelNumber;
    ADC0->INPUTCTRL.bit.MUXNEG = g_APinDescription[PIN_A2].ulADCChannelNumber;
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
  #if defined (__SAMD21__)
    ADC->CTRLA.bit.ENABLE = 0x01;
  #elif defined(__SAMD51__)
    ADC0->CTRLA.bit.ENABLE = 0x01;
  #endif
  syncADC();
  #if defined (__SAMD21__)
    ADC->SWTRIG.bit.START = 1;
    ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY;
  #elif defined(__SAMD51__)
    ADC0->SWTRIG.bit.START = 1;
    ADC0->INTFLAG.reg = ADC_INTFLAG_RESRDY;
  #endif

  #ifdef DEBUG
    print_debug_info();
  #endif

  // Create the cosine lookup table
  create_LUT();

  // Start the interrupt timer
  TC.startTimer(ISR_CLOCK, isr_psd);
}

/*------------------------------------------------------------------------------
    loop
------------------------------------------------------------------------------*/

void loop() {
  char* strCmd; // Incoming serial command string
  uint32_t prev_millis = 0;

  // Process commands on the data channel
  // Deliberately slowed down to once every 1 ms to improve timing stability of
  // 'isr_psd()'.
  if ((millis() - prev_millis) > 1) {
    prev_millis = millis();

    if (sc_data.available()) {
      if (fRunning) {
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
        fRunning = false;
        fSend_buffer_A = false;
        fSend_buffer_B = false;
        interrupts();

        // Flush out any binary buffer data scheduled for sending, potentially
        // flooding the receiving buffer at the PC side if 'fRunning' was not
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
        strCmd = sc_data.getCmd();

        if (strcmpi(strCmd, "id?") == 0) {
          // Reply identity string
          Ser_data.println("Arduino lock-in amp");

        } else if (strcmpi(strCmd, "mcu?") == 0) {
          // Reply microcontroller type string
          #if defined(__SAMD21G18A__)
            Ser_data.println("SAMD21G18A");
          #elif defined(__SAMD21E18A__)
            Ser_data.println("SAMD21E18A");
          #elif defined(__SAMD51P20A__)
            Ser_data.println("SAMD51P20A");
          #elif defined(__SAMD51J19A__)
            Ser_data.println("SAMD51J19A");
          #elif defined(__SAMD51G19A__)
            Ser_data.println("SAMD51G19A");
          #else
            Ser_data.println("unknown MCU");
          #endif

        } else if (strcmpi(strCmd, "bias?") == 0) {
          #if defined (__SAMD51__)
            Ser_data.println(NVM_ADC0_BIASCOMP);
            Ser_data.println(NVM_ADC0_BIASREFBUF);
            Ser_data.println(NVM_ADC0_BIASR2R);

            Ser_data.println(ADC0->CALIB.bit.BIASCOMP);
            Ser_data.println(ADC0->CALIB.bit.BIASREFBUF);
            Ser_data.println(ADC0->CALIB.bit.BIASR2R);

            Ser_data.println(ADC0->OFFSETCORR.bit.OFFSETCORR);
            Ser_data.println(ADC0->GAINCORR.bit.GAINCORR);
          #endif

        } else if (strcmpi(strCmd, "config?") == 0) {
          Ser_data.print(ISR_CLOCK);
          Ser_data.print('\t');
          Ser_data.print(BUFFER_SIZE);
          Ser_data.print('\t');
          Ser_data.print(N_BYTES_TRANSMIT_BUFFER);
          Ser_data.print('\t');
          Ser_data.print(N_LUT);
          Ser_data.print('\t');
          Ser_data.print(ANALOG_WRITE_RESOLUTION);
          Ser_data.print('\t');
          Ser_data.print(ANALOG_READ_RESOLUTION);
          Ser_data.print('\t');
          Ser_data.print(A_REF);
          Ser_data.print('\t');
          Ser_data.print(ref_V_offset, 3);
          Ser_data.print('\t');
          Ser_data.print(ref_V_ampl, 3);
          Ser_data.print('\t');
          Ser_data.print(ref_freq, 2);
          Ser_data.print('\n');

          #ifdef DEBUG
            print_debug_info();
          #endif

        } else if (strcmpi(strCmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser_data.print("already_off\n");

          #ifdef DEBUG
            Ser_debug << "Already OFF" << endl;
          # endif

        } else if (strcmpi(strCmd, "on") == 0) {
          // Start lock-in amp
          noInterrupts();
          fRunning = true;
          N_buffers_scheduled_to_be_sent = 0;
          N_sent_buffers = 0;
          fSend_buffer_A = false;
          fSend_buffer_B = false;
          interrupts();

          #ifdef DEBUG
            Ser_debug << "ON" << endl;
          # endif

        } else if (strncmpi(strCmd, "ref_freq", 8) == 0) {
          // Set frequency of the output reference signal [Hz]
          ref_freq = parseFloatInString(strCmd, 8);
          noInterrupts();
          LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
          T_period_micros_dbl = 1.0 / ref_freq * 1e6;
          interrupts();
          Ser_data.println(ref_freq, 2);

        } else if (strncmpi(strCmd, "ref_V_offset", 12) == 0) {
          // Set voltage offset of cosine reference signal [V]
          ref_V_offset = parseFloatInString(strCmd, 12);
          ref_V_offset = max(ref_V_offset, 0.0);
          ref_V_offset = min(ref_V_offset, A_REF);
          noInterrupts();
          create_LUT();
          interrupts();
          Ser_data.println(ref_V_offset, 3);

        } else if (strncmpi(strCmd, "ref_V_ampl", 10) == 0) {
          // Set voltage amplitude of cosine reference signal [V]
          ref_V_ampl = parseFloatInString(strCmd, 10);
          ref_V_ampl = max(ref_V_ampl, 0.0);
          ref_V_ampl = min(ref_V_ampl, A_REF);
          noInterrupts();
          create_LUT();
          interrupts();
          Ser_data.println(ref_V_ampl, 3);
        }

      }
    }
  }

  // Send buffers over the data channel
  if (fRunning && (fSend_buffer_A || fSend_buffer_B)) {
    int32_t  bytes_sent = 0;
    uint16_t dropped_buffers = 0;
    uint16_t idx;
    bool fError = false;

    if (fSend_buffer_A) {idx = 0;} else {idx = BUFFER_SIZE;}

    // Uncomment 'noInterrupts()' and 'interrupts()' only for debugging purposes
    // to get the execution time of a complete buffer transmission without being
    // disturbed by 'isr_psd()'. Will suspend 'isr_psd()' for the duration of
    // below 'Ser_data.write()'.
    /*
    #ifdef DEBUG
      //noInterrupts(); // Uncomment only for debugging purposes
      uint32_t tick = micros();
    #endif
    */

    // Contrary to Arduino documentation, 'write' can return -1 as indication
    // of an error, e.g. the receiving side being overrun with data.
    int32_t w = Ser_data.write((uint8_t *) &SOM, N_BYTES_SOM);
    if (w == -1) {fError = true;} else {bytes_sent += w;}

    w = Ser_data.write((uint8_t *) &buffer_time[idx], N_BYTES_TIME);
    if (w == -1) {fError = true;} else {bytes_sent += w;}

    w = Ser_data.write((uint8_t *) &buffer_ref_X_phase[idx], N_BYTES_REF_X_PHASE);
    if (w == -1) {fError = true;} else {bytes_sent += w;}

    w = Ser_data.write((uint8_t *) &buffer_sig_I[idx], N_BYTES_SIG_I);
    if (w == -1) {fError = true;} else {bytes_sent += w;}

    w = Ser_data.write((uint8_t *) &EOM, N_BYTES_EOM);
    if (w == -1) {fError = true;} else {bytes_sent += w;}

    /*
    #ifdef DEBUG
      Ser_debug << micros() - tick << endl;
    #endif
    //interrupts();   // Uncomment only for debugging purposes
    */

    N_sent_buffers++;
    if (fSend_buffer_A) {fSend_buffer_A = false;}
    if (fSend_buffer_B) {fSend_buffer_B = false;}

    noInterrupts();
    N_buffers_scheduled_to_be_sent--;
    if (N_buffers_scheduled_to_be_sent != 0) {
      dropped_buffers = N_buffers_scheduled_to_be_sent;
      N_buffers_scheduled_to_be_sent = 0;
    }
    interrupts();

    #ifdef DEBUG
      if ((dropped_buffers == 0) && (bytes_sent == N_BYTES_TRANSMIT_BUFFER)) {
        //Ser_debug << N_sent_buffers << ((idx == 0)?" A ":" B ") << bytes_sent;
        //Ser_debug << " OK" << endl;
      } else {
        Ser_debug << N_sent_buffers << ((idx == 0)?" A ":" B ") << bytes_sent;
        if (dropped_buffers != 0 ) {
          Ser_debug << " DROPPED " << dropped_buffers;
        }
        if (fError) {
          Ser_debug << " CAN'T WRITE";
        } else if (bytes_sent != N_BYTES_TRANSMIT_BUFFER) {
          Ser_debug << " WRONG N_BYTES SENT";
        }
        Ser_debug << endl;
      }
    #endif
  }
}
