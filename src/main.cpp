/*------------------------------------------------------------------------------
Arduino lock-in amplifier

Dennis van Gils
12-11-2018
------------------------------------------------------------------------------*/

#include <Arduino.h>
#include "DvG_SerialCommand.h"
#include "ZeroTimer.h"

// Define for writing debugging info to the terminal: Slow!
//#define DEBUG

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static void syncDAC() {while (DAC->STATUS.bit.SYNCBUSY == 1);}

// Wait for synchronization of registers between the clock domains
static __inline__ void syncADC() __attribute__((always_inline, unused));
static void syncADC() {while (ADC->STATUS.bit.SYNCBUSY == 1);}

// Serial   : Programming USB port
// SerialUSB: Native USB port. Baudrate setting gets ignored and is always as
//            fast as possible.
/* NOTE: Simply connecting a USB cable from the PC to a second serial port on
   the Arduino already reduces the timing accuracy of 'isr_psd()' by several
   microsec. Hence, use only one serial port for best performance.
*/
#define Ser_data    Serial      // Data channel
#ifdef DEBUG
  #define Ser_debug SerialUSB   // Debug channel
#endif

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);

// Interrupt service routine clock
// ISR_CLOCK: minimum 40 usec for only writing A0, no serial
//            minimum 50 usec for writing A0 and reading A1 combined, no serial
#define ISR_CLOCK 200     // 200 [usec]

// Buffers
// The buffer that will be send each transmission is BUFFER_SIZE samples long
// for each variable. Double the amount of memory is reserved to employ a double
// buffer technique, where alternatingly the first buffer half (buffer A) is
// being written to and the second buffer half (buffer B) is being sent.
#define BUFFER_SIZE 200   // [samples] 200
const uint16_t DOUBLE_BUFFER_SIZE = 2 * BUFFER_SIZE;

volatile uint32_t buffer_time       [DOUBLE_BUFFER_SIZE] = {0};
volatile uint16_t buffer_ref_X_phase[DOUBLE_BUFFER_SIZE] = {0};
volatile uint16_t buffer_sig_I      [DOUBLE_BUFFER_SIZE] = {0};

const uint16_t N_BYTES_TIME        = BUFFER_SIZE*sizeof(buffer_time[0]);
const uint16_t N_BYTES_REF_X_PHASE = BUFFER_SIZE*sizeof(buffer_ref_X_phase[0]);
const uint16_t N_BYTES_SIG_I       = BUFFER_SIZE*sizeof(buffer_sig_I[0]);

volatile bool fSend_buffer_A = false;
volatile bool fSend_buffer_B = false;

// Serial transmission start and end messages
const char SOM[] = {0x00, 0x00, 0x00, 0x00, 0xee}; // Start of message
const char EOM[] = {0x00, 0x00, 0x00, 0x00, 0xff}; // End of message
const uint8_t N_BYTES_SOM = sizeof(SOM);
const uint8_t N_BYTES_EOM = sizeof(EOM);

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
#define V_out_center 2.0      // [V] Center
#define V_out_p2p    2.0      // [V] Peak to peak

#define N_LUT 12288  // Number of samples for one full period.
volatile double LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
volatile double T_period_micros_dbl = 1.0 / ref_freq * 1e6;
uint16_t LUT_cos[N_LUT] = {0};
//uint16_t LUT_sin[N_LUT] = {0};

void create_LUT() {
  double offset = V_out_center / A_REF;
  double amplitude = 0.5 / A_REF * V_out_p2p;

  for (uint16_t i = 0; i < N_LUT; i++) {
    LUT_cos[i] = (uint16_t) round((pow(2, ANALOG_WRITE_RESOLUTION) - 1) * \
                 (offset + amplitude * cos(2*PI*i/N_LUT)));
  }

  /*
  for (uint16_t i = 0; i < N_LUT; i++) {
    LUT_sin[i] = LUT_cos[(i + N_LUT/4*3) % N_LUT];
  }
  */
}

/*------------------------------------------------------------------------------
    Interrupt service routine (isr) for phase-sentive detection (psd) 
------------------------------------------------------------------------------*/
volatile bool fRunning = false;

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
      DAC->DATA.reg = 0 & 0x3FF;    // Set output voltage to 0
    }
  }
  if (!fRunning) {return;}

  // Generate reference signals
  uint32_t now = micros();
  uint16_t LUT_idx = round(fmod(now, T_period_micros_dbl) * \
                           LUT_micros2idx_factor);
  uint16_t ref_X = LUT_cos[LUT_idx];    // aka v_RX

  // Output reference signal
  syncDAC();
  DAC->DATA.reg = ref_X & 0x3FF;

  // Read input signal
  syncADC();
  ADC->SWTRIG.bit.START = 1;
  while (ADC->INTFLAG.bit.RESRDY == 0); // Wait for conversion to complete
  uint16_t sig_I = ADC->RESULT.reg;     // aka V_I without amplification (g = 1)

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
  
  if (write_idx1 == BUFFER_SIZE) {
    fSend_buffer_A = true;
  } else if (write_idx1 == DOUBLE_BUFFER_SIZE) {
    fSend_buffer_B = true;
    write_idx1 = 0;
  }
  
  if (write_idx2 == DOUBLE_BUFFER_SIZE) {write_idx2 = 0;}
}

/*------------------------------------------------------------------------------
    setup
------------------------------------------------------------------------------*/

void setup() {
  #ifdef DEBUG
    Ser_debug.begin(9600);
  #endif

  #if Ser_data == Serial
    Ser_data.begin(1500000);
  #else
    Ser_data.begin(9600);
  #endif
  
  create_LUT();

  // Use built-in LED to signal running state of lock-in amp
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, fRunning);

  // DAC
  analogWriteResolution(ANALOG_WRITE_RESOLUTION);
  analogWrite(A0, 0);

  // ADC
  // Increase the ADC clock by setting the divisor from default DIV128 to DIV16.
  // Setting smaller divisors than DIV16 results in ADC errors.
  ADC->CTRLB.bit.PRESCALER = ADC_CTRLB_PRESCALER_DIV16_Val;
  analogReadResolution(ANALOG_READ_RESOLUTION);
  analogRead(A1);

  syncADC();
  ADC->CTRLA.bit.ENABLE = 0x01;
  syncADC();
  ADC->SWTRIG.bit.START = 1;
  ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY;

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
        Ser_data.flush();

        // Flush out and ignore the command
        sc_data.getCmd();
      } else {
        // -------------------
        //  Not running
        // -------------------
        strCmd = sc_data.getCmd();
        
        if (strcmpi(strCmd, "id?") == 0) {
          // Reply identity string
          Ser_data.println("Arduino lock-in amp");

        } else if (strcmpi(strCmd, "off") == 0) {
          // Lock-in amp is already off and we reply with an acknowledgement
          Ser_data.print("already_off\n");

        } else if (strcmpi(strCmd, "on") == 0) {
          // Start lock-in amp
          noInterrupts();
          fRunning = true;
          fSend_buffer_A = false;
          fSend_buffer_B = false;
          interrupts();
        
        } else if (strcmpi(strCmd, "ref?") == 0) {
          // Reply frequency of the output reference signal [Hz]
          Ser_data.println(ref_freq);
        
        } else if (strncmpi(strCmd, "ref", 3) == 0) {
          // Set frequency of the output reference signal [Hz]
          ref_freq = parseFloatInString(strCmd, 3);
          noInterrupts();
          LUT_micros2idx_factor = 1e-6 * ref_freq * (N_LUT - 1);
          T_period_micros_dbl = 1.0 / ref_freq * 1e6;
          interrupts();
          Ser_data.println(ref_freq);
        }
      }
    }
  }

  // Send buffers over the data channel
  if (fSend_buffer_A || fSend_buffer_B) {
    uint16_t bytes_sent = 0;
    uint16_t idx;

    if (fSend_buffer_A) {
      idx = 0;
      #ifdef DEBUG
        Ser_debug.print("A: ");
      #endif
    } else {
      idx = BUFFER_SIZE;
      #ifdef DEBUG
        Ser_debug.print("B: ");
      #endif
    }

    // Uncomment 'noInterrupts()' and 'interrupts()' only for debugging purposes
    // to get the execution time of a complete buffer transmission. Will suspend
    // 'isr_psd()'.
    //noInterrupts(); // Uncomment only for debugging purposes
    bytes_sent += Ser_data.write((uint8_t *) &SOM, \
                                 N_BYTES_SOM);
    bytes_sent += Ser_data.write((uint8_t *) &buffer_time [idx], \
                                 N_BYTES_TIME);
    bytes_sent += Ser_data.write((uint8_t *) &buffer_ref_X_phase[idx], \
                                 N_BYTES_REF_X_PHASE);
    bytes_sent += Ser_data.write((uint8_t *) &buffer_sig_I[idx], \
                                 N_BYTES_SIG_I);
    bytes_sent += Ser_data.write((uint8_t *) &EOM, \
                                 N_BYTES_EOM);
    //interrupts();   // Uncomment only for debugging purposes
    if (fSend_buffer_A) {fSend_buffer_A = false;}
    if (fSend_buffer_B) {fSend_buffer_B = false;}

    #ifdef DEBUG
      Ser_debug.println(bytes_sent);
    #endif
  }
}