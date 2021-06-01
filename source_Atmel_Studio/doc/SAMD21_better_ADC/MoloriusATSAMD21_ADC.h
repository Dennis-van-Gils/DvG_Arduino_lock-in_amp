// Blake Felt
// ATSAMD21_ADC.h
// Adds some extra functions to the ADC, such as
// 16 bit and differential mode

#ifndef ATSAMD21_ADC_H
#define ATSAMD21_ADC_H

#ifdef __cplusplus
  extern "C" {
#endif // ifdef __cplusplus

#ifndef ADC_CTRLB_RESSEL_12BIT_Val 
#define ADC_CTRLB_RESSEL_8BIT_Val   0x03
#define ADC_CTRLB_RESSEL_10BIT_Val  0x02 // default by Arduino
#define ADC_CTRLB_RESSEL_12BIT_Val  0x00
#endif
#define ADC_CTRLB_RESSEL_16BIT_Val  0x01 // used for averaging mode output

#define ADC_PIN_TEMP                0x18 // positive mux, pg 870
#define ADC_PIN_BANDGAP             0x19
#define ADC_PIN_SCALEDCOREVCC       0x1A
#define ADC_PIN_SCALEDIOVCC         0x1B
#define ADC_PIN_DAC                 0x1C

#define ADC_PIN_GND                 0x18 // negative mux, pg 869
#define ADC_PIN_IOGND               0x19

#define ADC_GAIN_1                  0x00 // pg 868
#define ADC_GAIN_2                  0x01
#define ADC_GAIN_4                  0x02
#define ADC_GAIN_8                  0x03
#define ADC_GAIN_16                 0x04
#define ADC_GAIN1_DIV2              0x0F // default by Arduino

#define ADC_REF_INT1V               0x00 // 1.0V reference, pg 861
#define ADC_REF_INTVCC0             0x01 // 1/1.48 VDDANA
#define ADC_REF_INTVCC1             0x02 // 1/2 VDDANA (only for VDDANA > 2.0V) // default
#define ADC_REF_VREFA               0x03 // external reference
#define ADC_REF_VREFB               0x04 // external reference

#define ADC_PRESCALER_DIV4          0x00 // pg 864
#define ADC_PRESCALER_DIV8          0x01
#define ADC_PRESCALER_DIV16         0x02
#define ADC_PRESCALER_DIV32         0x03
#define ADC_PRESCALER_DIV64         0x04
#define ADC_PRESCALER_DIV128        0x05
#define ADC_PRESCALER_DIV256        0x06
#define ADC_PRESCALER_DIV512        0x07 // Arduino default

// NVM Software Calibration Area Mapping, pg 32. Address starting at NVMCTRL_OTP4. 
// NVM register access code modified from https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/USB/samd21_host.c
// ADC Linearity Calibration value. Should be written to the CALIB register.
#define NVM_ADC_LINEARITY_POS         27
#define NVM_ADC_LINEARITY_SIZE         8
// ADC Bias Calibration value. Should be written to the CALIB register.
#define NVM_ADC_BIASCAL_POS           35
#define NVM_ADC_BIASCAL_SIZE           3

// Taken from Arduino IDE:
// Wait for synchronization of registers between the clock domains
static __inline__ void syncADC() __attribute__((always_inline, unused));
static void syncADC() {
  while(ADC->STATUS.bit.SYNCBUSY == 1);
}

// Taken from Arduino IDE:
// Wait for synchronization of registers between the clock domains
static __inline__ void syncDAC() __attribute__((always_inline, unused));
static void syncDAC() {
  while (DAC->STATUS.bit.SYNCBUSY == 1);
}

// taken from Arduino IDE, changes the pin to an input:
int pinPeripheral( uint32_t ulPin, EPioType ulPeripheral );

uint8_t analogReadExtended(uint8_t bits) {
/*
 * Allows for adc to read 8, 10, or 12 bits normally or 13-16 bits using oversampling and decimation.
 * See pages 853 & 862
 * 8,10,12 bit = 1 sample ~ 436 microseconds
 * 13 bit = 4 samples ~ 1668 microseconds
 * 14 bit = 16 samples ~ 6595 microseconds
 * 15 bit = 64 samples ~ 26308 microseconds
 * 16 bit = 256 samples ~ 105156 microseconds
 */
  switch(bits) {
    case 8:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_8BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x0;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x0;
      return 0;
    break;

    case 10:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_10BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x0;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x0;
      return 0;
    break;
    
    case 12:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_12BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x0;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x0;
      return 0;
    break;
      
    case 13:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x1;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x2;
      return 0;
    break;
    
    case 14:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x2;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x4;
      return 0;
    break;
    
    case 15:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x1;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x6;
      return 0;
    break;
    
    case 16:
      ADC->CTRLB.bit.RESSEL = ADC_CTRLB_RESSEL_16BIT_Val;
      ADC->AVGCTRL.bit.ADJRES = 0x0;
      ADC->AVGCTRL.bit.SAMPLENUM = 0x8;
      return 0;
    break;
    
    default:
      return -1;
    break;
   }
}

// returns the internal pin value of the specified pin, useful
// for analogDifferentialRaw function
uint8_t internalPinValue(uint8_t pin) {
  return g_APinDescription[pin].ulADCChannelNumber;
}

// modified from Arduino analogRead, can be used in conjunction with analogRead:
int16_t analogDifferential(uint8_t pin_pos,uint8_t pin_neg) {
  if(pin_pos<A0) pin_pos += A0;
  if(pin_neg<A0) pin_neg += A0;
  
  if((g_APinDescription[pin_neg].ulADCChannelNumber>0x07) && (pin_neg<ADC_PIN_GND)) { // if the negative pin is out of bounds
    return 0;
  }  
  
  uint32_t value_read = 0;
  
  pinPeripheral(pin_pos,PIO_ANALOG); // set pins to analog mode
  pinPeripheral(pin_neg,PIO_ANALOG);

  if((pin_pos == A0) || (pin_neg == A0)) { // Disable DAC
    syncDAC();
    DAC->CTRLA.bit.ENABLE = 0x00; // Disable DAC
    syncDAC();
  }

  syncADC();
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[pin_pos].ulADCChannelNumber; // Selection for the positive ADC input
  ADC->INPUTCTRL.bit.MUXNEG = g_APinDescription[pin_neg].ulADCChannelNumber; // negative ADC input

  syncADC();
  ADC->CTRLA.bit.ENABLE = 0x01; // enable adc
  ADC->CTRLB.bit.DIFFMODE = 1; // set to differential mode

  syncADC();
  ADC->SWTRIG.bit.START = 1; // start conversion

  ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY; // clear the data ready flag
  syncADC();

  ADC->SWTRIG.bit.START = 1; // restart conversion, as changing inputs messes up first conversion
  
  while(ADC->INTFLAG.bit.RESRDY == 0);   // Wait for conversion to complete
  value_read = ADC->RESULT.reg; // read the value

  syncADC();
  ADC->CTRLA.bit.ENABLE = 0x00; // disable adc
  ADC->CTRLB.bit.DIFFMODE = 0; // put back into single-ended mode
  ADC->INPUTCTRL.bit.MUXNEG = ADC_PIN_GND; // set back muxneg to internal ground
  syncADC();

  return value_read;
}

// same as the above function, but no error checking, no pin types are changed, and the positive and negative
// inputs are the raw values being input. The DAC is not automatically shut off either. See datasheet page
int16_t analogDifferentialRaw(uint8_t mux_pos,uint8_t mux_neg) {
  
  uint32_t value_read = 0;

  syncADC();
  ADC->INPUTCTRL.bit.MUXPOS = mux_pos; // Selection for the positive ADC input
  ADC->INPUTCTRL.bit.MUXNEG = mux_neg; // negative ADC input

  syncADC();
  ADC->CTRLA.bit.ENABLE = 0x01; // enable adc
  ADC->CTRLB.bit.DIFFMODE = 1; // set to differential mode

  syncADC();
  ADC->SWTRIG.bit.START = 1; // start conversion

  ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY; // clear the data ready flag
  syncADC();

  ADC->SWTRIG.bit.START = 1; // restart conversion, as changing inputs messes up first conversion
  
  while(ADC->INTFLAG.bit.RESRDY == 0);   // Wait for conversion to complete
  value_read = ADC->RESULT.reg; // read the value

  syncADC();
  ADC->CTRLA.bit.ENABLE = 0x00; // disable adc
  ADC->CTRLB.bit.DIFFMODE = 0; // put back into single-ended mode
  ADC->INPUTCTRL.bit.MUXNEG = ADC_PIN_GND; // set back muxneg to internal ground
  syncADC();

  return value_read;
}

// sets the gain of the ADC. See page 868. All values defined above. 
void analogGain(uint8_t gain) {
  syncADC();
  ADC->INPUTCTRL.bit.GAIN = gain;
  syncADC();
}

// calibrates the bias and linearity based on the nvm register. 
// NVM register access code modified from https://github.com/arduino/ArduinoCore-samd/blob/master/cores/arduino/USB/samd21_host.c
// datasheet pages 32 and 882
void analogCalibrate() { 
  syncADC();
  // read NVM register
  uint32_t adc_linearity = (*((uint32_t *)(NVMCTRL_OTP4) // original position
          + (NVM_ADC_LINEARITY_POS / 32)) // move to the correct 32 bit window, read value
          >> (NVM_ADC_LINEARITY_POS % 32)) // shift value to match the desired position
          & ((1 << NVM_ADC_LINEARITY_SIZE) - 1); // apply a bitmask for the desired size

  uint32_t adc_biascal = (*((uint32_t *)(NVMCTRL_OTP4)
          + (NVM_ADC_BIASCAL_POS / 32))
          >> (NVM_ADC_BIASCAL_POS % 32))
          & ((1 << NVM_ADC_BIASCAL_SIZE) - 1);
  
  // write values to CALIB register
  ADC->CALIB.bit.LINEARITY_CAL = adc_linearity;
  ADC->CALIB.bit.BIAS_CAL = adc_biascal;
  syncADC();
}

// set the analog reference voltage, but with all available options
// (the Arduino IDE neglects some). The Arduino IDE also changes
// the gain when analogReference() is used, but this won't. pg 861
void analogReference2(uint8_t ref) {
  syncADC();
  ADC->REFCTRL.bit.REFSEL = ref;
  syncADC();
}

// increases accuracy of gain stage by enabling the reference buffer
// offset compensation. Takes longer to start. pg 861
void analogReferenceCompensation(uint8_t val) {
  if(val>0) val = 1; 
  syncADC();
  ADC->REFCTRL.bit.REFCOMP = val;
  syncADC();
}

// sets the ADC clock relative to the peripheral clock. pg 864
void analogPrescaler(uint8_t val) {
  syncADC();
  ADC->CTRLB.bit.PRESCALER = val;
  syncADC();
}

// resets the ADC. pg 860
// note that this doesn't put back the default values set by the 
// Arduino IDE. 
void analogReset() {
  syncADC();
  ADC->CTRLA.bit.SWRST = 1; // set reset bit
  while(ADC->CTRLA.bit.SWRST==1); // wait until it's finished
  syncADC();
}

#ifdef __cplusplus
  }
#endif // ifdef __cplusplus

#endif // ifndef ATSAMD21_ADC_H