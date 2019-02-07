/*------------------------------------------------------------------------------
Arduino lock-in amplifier

Pins:
A0: output reference signal
A1: input signal, differential +
A2: input signal, differential -

Dennis van Gils
21-01-2019
------------------------------------------------------------------------------*/

#include <Arduino.h>
#include "DvG_SerialCommand.h"
#include "Adafruit_ZeroTimer.h"
//#include "ZeroTimer.h"
#include "Streaming.h"

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static void syncDAC() {while (DAC->STATUS.bit.EOC0 == 1);}

// Wait for synchronization of registers between the clock domains
static __inline__ void syncADC() __attribute__((always_inline, unused));
static void syncADC() {while (ADC0->STATUS.bit.ADCBUSY == 1);}

Adafruit_ZeroTimer zt3 = Adafruit_ZeroTimer(3);

// Define for writing debugging info to the terminal
//#define DEBUG

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
#define Ser_data    Serial
#ifdef DEBUG
  #define Ser_debug Serial
#endif

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);

// Interrupt service routine clock
// ISR_CLOCK: min.  40 usec for only writing A0, no serial
//            min.  50 usec for writing A0 and reading A1, no serial
//            min.  80 usec for writing A0 and reading A1, with serial
#define ISR_CLOCK 100     // [usec]

// Buffers
// The buffer that will be send each transmission is BUFFER_SIZE samples long
// for each variable. Double the amount of memory is reserved to employ a double
// buffer technique, where alternatingly the first buffer half (buffer A) is
// being written to and the second buffer half (buffer B) is being sent.
#define BUFFER_SIZE 500   // [samples]

/* Tested settings
Case A: turbo and stable on computer Onera
  ISR_CLOCK   80
  BUFFER_SIZE 625
  DAQ --> 12500 Hz
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
#define ANALOG_WRITE_RESOLUTION 10      // [bits] Fixed to 10 on M0 Pro
#define ANALOG_READ_RESOLUTION  12      // [bits] 10 or 12 on M0 Pro

/*------------------------------------------------------------------------------
    Cosine wave look-up table (LUT)
------------------------------------------------------------------------------*/

double ref_freq = 137.0;      // [Hz], aka f_R

// Tip: Limiting the output voltage range to slighty above 0.0 V will improve
// the shape of the sine wave at its minimum. Apparently, the analog out port
// has difficulty in cleanly dropping the output voltage completely to 0.0 V.
#define A_REF        3.300    // [V] Analog voltage reference Arduino
double ref_V_center = 2.0;    // [V] Center voltage of cosine reference signal
double ref_V_p2p    = 0.4;    // [V] Peak-to-peak voltage of cosine ref. signal

#define N_LUT 9000  // (9000 --> 0.04 deg) Number of samples for one full period
volatile double LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
volatile double T_period_micros_dbl = 1.0 / ref_freq * 1e6;
uint16_t LUT_cos[N_LUT] = {0};
//uint16_t LUT_sin[N_LUT] = {0};

void create_LUT() {
  uint16_t ANALOG_WRITE_MAX_BITVAL = pow(2, ANALOG_WRITE_RESOLUTION) - 1;
  double offset = ref_V_center / A_REF;
  double amplitude = 0.5 / A_REF * ref_V_p2p;
  double cosine_value;

  #ifdef DEBUG
    Ser_debug << "Creating LUT...";
  #endif

  for (uint16_t i = 0; i < N_LUT; i++) {
    cosine_value = offset + amplitude * cos(2*PI*i/N_LUT);
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
      DAC->DATA[0].reg = 0 & 0x3FF;    // Set output voltage to 0
    }
  }
  if (!fRunning) {return;}

  // Generate reference signals
  uint32_t now = micros();
  static uint32_t now_offset = 0;
  if (fStartup) {now_offset = now;} // Force cosine to start at phase = 0 deg
  uint16_t LUT_idx = round(fmod(now - now_offset, T_period_micros_dbl) * \
                           LUT_micros2idx_factor);
  uint16_t ref_X = LUT_cos[LUT_idx];    // aka v_RX

  // Output reference signal
  syncDAC();
  DAC->DATA[0].reg = ref_X & 0x3FF;

  // Read input signal
  syncADC();
  ADC0->SWTRIG.bit.START = 1;
  while (ADC0->INTFLAG.bit.RESRDY == 0); // Wait for conversion to complete
  int16_t sig_I = ADC0->RESULT.reg;      // aka V_I without amplification (g = 1)

  // Store in buffers
  if (fStartup) {
    buffer_time [0] = now;
    buffer_ref_X_phase[0] = LUT_idx;
    buffer_sig_I[0] = sig_I;
    write_idx1 = 1;
    write_idx2 = 0;
    fStartup = false;
  } else {
    buffer_time [write_idx1] = now;
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
  void print_debug_info() {
      Ser_debug << "-------------------------------" << endl;
      Ser_debug << "CTRLA" << endl;
      Ser_debug << "  .RUNSTDBY   : " << _HEX(ADC0->CTRLA.bit.RUNSTDBY) << endl;
      Ser_debug << "  .ENABLE     : " << _HEX(ADC0->CTRLA.bit.ENABLE) << endl;
      Ser_debug << "  .SWRST      : " << _HEX(ADC0->CTRLA.bit.SWRST) << endl;
      Ser_debug << "REFCTRL" << endl;
      Ser_debug << "  .REFCOMP    : " << _HEX(ADC0->REFCTRL.bit.REFCOMP) << endl;
      Ser_debug << "  .REFSEL     : " << _HEX(ADC0->REFCTRL.bit.REFSEL) << endl;
      Ser_debug << "AVGVTRL" << endl;
      Ser_debug << "  .ADJRES     : " << _HEX(ADC0->AVGCTRL.bit.ADJRES) << endl;
      Ser_debug << "  .SAMPLENUM  : " << _HEX(ADC0->AVGCTRL.bit.SAMPLENUM) << endl;
      Ser_debug << "SAMPCTRL" << endl;
      Ser_debug << "  .SAMPLEN    : " << _HEX(ADC0->SAMPCTRL.bit.SAMPLEN) << endl;
      Ser_debug << "CTRLB" << endl;
      Ser_debug << "  .PRESCALER  : " << _HEX(ADC0->CTRLB.bit.PRESCALER) << endl;
      Ser_debug << "  .RESSEL     : " << _HEX(ADC0->CTRLB.bit.RESSEL) << endl;
      Ser_debug << "  .CORREN     : " << _HEX(ADC0->CTRLB.bit.CORREN) << endl;
      Ser_debug << "  .FREERUN    : " << _HEX(ADC0->CTRLB.bit.FREERUN) << endl;
      Ser_debug << "  .LEFTADJ    : " << _HEX(ADC0->CTRLB.bit.LEFTADJ) << endl;
      Ser_debug << "  .DIFFMODE   : " << _HEX(ADC0->CTRLB.bit.DIFFMODE) << endl;
      Ser_debug << "INPUTCTRL" << endl;
      Ser_debug << "  .GAIN       : " << _HEX(ADC0->INPUTCTRL.bit.GAIN) << endl;
      Ser_debug << "  .INPUTOFFSET: " << _HEX(ADC0->INPUTCTRL.bit.INPUTOFFSET) << endl;
      Ser_debug << "  .INPUTSCAN  : " << _HEX(ADC0->INPUTCTRL.bit.INPUTSCAN) << endl;
      Ser_debug << "  .MUXNEG     : " << _HEX(ADC0->INPUTCTRL.bit.MUXNEG) << endl;
      Ser_debug << "  .MUXPOS     : " << _HEX(ADC0->INPUTCTRL.bit.MUXPOS) << endl;

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
  ADC0->CTRLA.bit.PRESCALER = ADC_CTRLA_PRESCALER_DIV16_Val;
  analogReadResolution(ANALOG_READ_RESOLUTION);
  analogRead(A1); // Differential +
  analogRead(A2); // Differential -

  // Set differential mode on A1(+) and A2(-)
  ADC0->INPUTCTRL.bit.DIFFMODE = 1;
  ADC0->INPUTCTRL.bit.MUXPOS = 2; // 2 == AIN2 on SAMD21 = A1 on Arduino board
  ADC0->INPUTCTRL.bit.MUXNEG = 3; // 3 == AIN3 on SAMD21 = A2 on Arduino board
  //ADC0->INPUTCTRL.bit.GAIN = ADC_INPUTCTRL_GAIN_DIV2_Val;
  ADC0->REFCTRL.bit.REFSEL = ADC_REFCTRL_REFSEL_INTVCC1_Val;

  // Prepare for software-triggered acquisition
  syncADC();
  ADC0->CTRLA.bit.ENABLE = 0x01;
  syncADC();
  ADC0->SWTRIG.bit.START = 1;
  ADC0->INTFLAG.reg = ADC_INTFLAG_RESRDY;

  #ifdef DEBUG
    print_debug_info();
  #endif

  // Create the cosine lookup table
  create_LUT();

  // Start the interrupt timer
  //TC.startTimer(ISR_CLOCK, isr_psd);
  /********************* Timer #3, 16 bit, two PWM outs, period = 65535 */
  /*
  zt3.configure(TC_CLOCK_PRESCALER_DIV2, // prescaler
                TC_COUNTER_SIZE_16BIT,   // bit width of timer/counter
                TC_WAVE_GENERATION_NORMAL_PWM // frequency or PWM mode
                );

  zt3.setCompare(0, 0xFFFF/4);
  zt3.setCallback(true, TC_CALLBACK_CC_CHANNEL0, isr_psd);
  zt3.enable(true);
  */
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
        # endif
      } else {
        // -------------------
        //  Not running
        // -------------------
        strCmd = sc_data.getCmd();

        if (strcmpi(strCmd, "id?") == 0) {
          // Reply identity string
          Ser_data.println("Arduino lock-in amp");

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
          Ser_data.print(ref_V_center);
          Ser_data.print('\t');
          Ser_data.print(ref_V_p2p);
          Ser_data.print('\t');
          Ser_data.print(ref_freq);
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
          Ser_data.println(ref_freq);

        } else if (strncmpi(strCmd, "ref_V_center", 12) == 0) {
          // Set center voltage of cosine reference signal [V]
          ref_V_center = parseFloatInString(strCmd, 12);
          ref_V_center = max(ref_V_center, 0.0);
          ref_V_center = min(ref_V_center, A_REF);
          noInterrupts();
          create_LUT();
          interrupts();
          Ser_data.println(ref_V_center);

        } else if (strncmpi(strCmd, "ref_V_p2p", 9) == 0) {
          // Set peak-to-peak voltage of cosine reference signal [V]
          ref_V_p2p = parseFloatInString(strCmd, 9);
          ref_V_p2p = max(ref_V_p2p, 0.0);
          ref_V_p2p = min(ref_V_p2p, A_REF);
          noInterrupts();
          create_LUT();
          interrupts();
          Ser_data.println(ref_V_p2p);
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
