/*------------------------------------------------------------------------------
Arduino lock-in amplifier

Dennis van Gils
10-11-2018
------------------------------------------------------------------------------*/

#include <Arduino.h>
#include "DvG_SerialCommand.h"
#include "ZeroTimer.h"

// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static void syncDAC() {while (DAC->STATUS.bit.SYNCBUSY == 1);}

// Wait for synchronization of registers between the clock domains
static __inline__ void syncADC() __attribute__((always_inline, unused));
static void syncADC() {while (ADC->STATUS.bit.SYNCBUSY == 1);}

// Serial   : Programming USB port
// SerialUSB: Native USB port. Baudrate setting gets ignored and is always as
//            fast as possible.
/* NOTE: simply connecting a USB cable from the PC to the other serial port 
   already reduces the timing of the ISR by several microsec.
*/
#define Ser_data Serial     // Data channel
//#define Ser_ctrl SerialUSB  // Control channel

// Instantiate serial command listeners
DvG_SerialCommand sc_data(Ser_data);
//DvG_SerialCommand sc_ctrl(Ser_ctrl);

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

volatile uint32_t buffer_time[DOUBLE_BUFFER_SIZE] = {0};
volatile uint16_t buffer_ref_X[DOUBLE_BUFFER_SIZE] = {0};
volatile uint16_t buffer_sig_I[DOUBLE_BUFFER_SIZE] = {0};

const uint16_t N_BYTES_TIME  = BUFFER_SIZE * sizeof(buffer_time[0]);
const uint16_t N_BYTES_REF_X = BUFFER_SIZE * sizeof(buffer_ref_X[0]);
const uint16_t N_BYTES_SIG_I = BUFFER_SIZE * sizeof(buffer_sig_I[0]);

//volatile bool fSend_buffer_A = false;
//volatile bool fSend_buffer_B = false;
bool fSend_buffer_A = false;
bool fSend_buffer_B = false;

// Serial transmission start and end messages
const char SOM[] = {0x00, 0x00, 0x00, 0x00, 0xee}; // Start of message
const char EOM[] = {0x00, 0x00, 0x00, 0x00, 0xff}; // End of message
const uint8_t N_BYTES_SOM = sizeof(SOM);
const uint8_t N_BYTES_EOM = sizeof(EOM);

// Analog port resolutions
#define ANALOG_WRITE_RESOLUTION 10      // [bits] Fixed to 10 on M0 Pro
#define ANALOG_READ_RESOLUTION  12      // [bits] 10 or 12 on M0 Pro

// Define for writing debugging info to the terminal: Slow!
//#define DEBUG

/*------------------------------------------------------------------------------
    Cosine wave look-up table (LUT)
------------------------------------------------------------------------------*/

double ref_freq = 137.0;      // [Hz], aka f_R

// Tip: limiting the output voltage range to slighty above 0.0 V will improve
// the shape of the sine wave at it's minimum. Apparently, the analog out port
// has difficulty in cleanly dropping the output voltage completely to 0.0 V.
#define A_REF        3.300    // [V] Analog voltage reference Arduino
#define V_out_center 2.0      // [V] Center
#define V_out_p2p    2.0      // [V] Peak to peak

#define N_LUT 2048  // Number of samples for one full period. Use power of 2.
volatile double LUT_micros2idx_factor = 1e-6 * ref_freq * N_LUT;
uint16_t LUT_cos[N_LUT] = {0};
uint16_t LUT_sin[N_LUT] = {0};

void create_LUT() {
  double offset = V_out_center / A_REF;
  double amplitude = 0.5 / A_REF * V_out_p2p;

  for (uint16_t i = 0; i < N_LUT; i++) {
    LUT_cos[i] = (uint16_t) round((pow(2, ANALOG_WRITE_RESOLUTION) - 1) * \
                 (offset + amplitude * cos(2*PI*i/N_LUT)));
  }

  for (uint16_t i = 0; i < N_LUT; i++) {
    LUT_sin[i] = LUT_cos[(i + N_LUT/4*3) % N_LUT];
  }
}

/*------------------------------------------------------------------------------
    Interrupt service routine (isr) for phase-sentive detection (psd) 
------------------------------------------------------------------------------*/
//volatile bool fRunning = false;
bool fRunning = false;

void isr_psd() {
  static uint16_t write_idx = 0;  // Current write index in double buffer
  
  if (!fRunning) {
    write_idx = 0;
    return;
  }

  // Generate reference signals
  uint32_t now = micros();
  uint16_t LUT_idx = ((uint16_t) round(now * LUT_micros2idx_factor)) % N_LUT;
  uint16_t ref_X = LUT_cos[LUT_idx];    // aka v_RX
  //uint16_t ref_Y = LUT_sin[LUT_idx];  // aka v_RY

  // Output reference signal
  syncDAC();
  DAC->DATA.reg = ref_X & 0x3FF;

  // Read input signal
  syncADC();
  ADC->SWTRIG.bit.START = 1;
  while (ADC->INTFLAG.bit.RESRDY == 0); // Wait for conversion to complete
  uint16_t sig_I = ADC->RESULT.reg;     // aka V_I without amplification (g = 1)

  // Store in buffers
  buffer_time[write_idx]  = now;
  buffer_ref_X[write_idx] = ref_X;
  buffer_sig_I[write_idx] = sig_I;
  write_idx++;

  if (write_idx == BUFFER_SIZE) {
      fSend_buffer_A = true;
  } else if (write_idx == DOUBLE_BUFFER_SIZE) {
      fSend_buffer_B = true;
      write_idx = 0;
  }
}

/*------------------------------------------------------------------------------
    setup
------------------------------------------------------------------------------*/

void setup() {
  #if Ser_data == Serial
    Ser_data.begin(1500000);
    //Ser_ctrl.begin(9600);
  #else
    Ser_data.begin(9600);
    //Ser_ctrl.begin(1500000);
  #endif
  
  create_LUT();

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

  // Process commands on the data channel.
  /* NOTE: The lock-in amp should not be runnning in order to correctly reply
     to serial ASCII commands. When the amp would be running, the ASCII reply
     will likely be intermixed wih binary data. Normally, one would flush the
     serial output buffer, but we can't do that here because the serial binary
     write would then be blocking and interfere with the ISR clock timing. 
  */
  if ((millis() - prev_millis) > 1) {
    prev_millis = millis();

    if (sc_data.available()) {
      strCmd = sc_data.getCmd();

      if (strcmpi(strCmd, "id?") == 0) {
        // Reply identity string
        Ser_data.print("Lock-in amp: data ok\r\n");
        
      } else if (strcmpi(strCmd, "on") == 0) {
        // Start lock-in amp
        fRunning = true;
      
      } else if (strcmpi(strCmd, "off") == 0) {
        // Stop lock-in amp
        //Ser_data.flush();
        fRunning = false;
        fSend_buffer_A = false;
        fSend_buffer_B = false;
        Ser_data.flush();
        delay(200);
        Ser_data.flush();
        Ser_data.print("ok\n");
        //Ser_data.flush();
        //delay(150);
      
      } else if (strcmpi(strCmd, "ref?") == 0) {
        // Reply frequency of the output reference signal [Hz]
        Ser_data.println(ref_freq);
      }
    }
  }

  // Process commands on the control channel every 1 ms. Deliberately slowed
  // down to improve timing stability of the interrupt clock.
  /*
  if ((millis() - prev_millis) > 1) {
    prev_millis = millis();

    if (sc_ctrl.available()) {
      strCmd = sc_ctrl.getCmd();

      if (strcmpi(strCmd, "id?") == 0) {
        // Identity string
        Ser_ctrl.println("Lock-in amp: control");
      
      } else if (strcmpi(strCmd, "on") == 0) {
        // Start lock-in amplifier
        fRunning = true;
        Ser_ctrl.println("Lock-in amp: on");
      
      } else if (strcmpi(strCmd, "off") == 0) {
        // Stop lock-in amplifier
        fRunning = false;
        Ser_ctrl.println("Lock-in amp: off");
      
      } else if (strcmpi(strCmd, "ref?") == 0) {
        // Get frequency of the output reference signal [Hz]
        Ser_ctrl.print("ref = ");
        Ser_ctrl.print(ref_freq);
        Ser_ctrl.println(" Hz");
      
      } else if (strncmpi(strCmd, "ref", 3) == 0) {
        // Set frequency of the output reference signal [Hz]
        ref_freq = parseFloatInString(strCmd, 3);
        LUT_micros2idx_factor = 1e-6 * ref_freq * N_LUT;
        Ser_ctrl.print("ref = ");
        Ser_ctrl.print(ref_freq);
        Ser_ctrl.println(" Hz");
      }
    }
  }
  */

  // Send buffers over the data channel
  if (fSend_buffer_A || fSend_buffer_B) {
    uint16_t bytes_sent = 0;
    uint16_t idx;

    if (fSend_buffer_A) {
      idx = 0;
      #ifdef DEBUG
        Ser_ctrl.print("A: ");
      #endif
    } else {
      idx = BUFFER_SIZE;
      #ifdef DEBUG
        Ser_ctrl.print("B: ");
      #endif
    }

    // Uncomment 'noInterrupts()' and 'interrupts()' only for debugging purposes
    // to get the execution time of a complete buffer transmission. Will suspend
    // the interrupt timer.
    //noInterrupts(); // Uncomment only for debugging purposes
    if (fRunning) {
      bytes_sent += Ser_data.write((uint8_t *) &SOM              , N_BYTES_SOM);
      bytes_sent += Ser_data.write((uint8_t *) &buffer_time[idx] , N_BYTES_TIME);
      bytes_sent += Ser_data.write((uint8_t *) &buffer_ref_X[idx], N_BYTES_REF_X);
      bytes_sent += Ser_data.write((uint8_t *) &buffer_sig_I[idx], N_BYTES_SIG_I);
      bytes_sent += Ser_data.write((uint8_t *) &EOM              , N_BYTES_EOM);
    }
    //interrupts();   // Uncomment only for debugging purposes
    if (fSend_buffer_A) {fSend_buffer_A = false;}
    if (fSend_buffer_B) {fSend_buffer_B = false;}

    #ifdef DEBUG
      Ser_ctrl.println(bytes_sent);
    #endif
  }
}