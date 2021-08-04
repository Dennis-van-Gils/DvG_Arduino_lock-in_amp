/*------------------------------------------------------------------------------
  adc_dac_functions.cpp

  Dennis van Gils, 04-08-2021
------------------------------------------------------------------------------*/
// clang-format off
#include "Arduino.h"
#include "adc_dac_init.h"
#include "adc_dac_functions.h"
// clang-format on

void DAC_set_output(int16_t value) {
#ifdef __SAMD21__
  // DAC conversion time is approximately 2.85 μs, see
  // https://microchipdeveloper.com/32arm:samd21-dac-overview
  // Because there is no bit we can pol for knowing when the conversion has
  // stabilized, we simply wait several clock cycles.
  DAC->DATA.reg = value;
  for (uint8_t i = 0; i < 144; i++) { // 144 cycles = 3.00 μs @ 48 MHz
    __asm__("nop\n\t");
  }
#elif defined __SAMD51__
  while (!DAC->STATUS.bit.READY0) {} // Wait for DAC to become ready
  DAC->DATA[0].reg = value;
  while (!DAC->STATUS.bit.EOC0) {} // Wait for DAC output to stabilize
#endif
}

void DAC_set_output_nosync(int16_t value) {
#ifdef __SAMD21__
  DAC->DATA.reg = value;
#elif defined __SAMD51__
  DAC->DATA[0].reg = value;
#endif
}

int16_t ADC_read_signal() {
#ifdef __SAMD21__
  ADC->SWTRIG.bit.START = 1; // Request start conversion
  syncADC();                 // Make sure conversion has been initiated
  __asm__("nop\n\t");        // Tiny delay before we can poll RESRDY
  __asm__("nop\n\t");
  __asm__("nop\n\t");
  while (!ADC->INTFLAG.bit.RESRDY) {} // Wait for result ready to be read
  return ADC->RESULT.reg;             // Read result
#elif defined __SAMD51__
  ADC0->SWTRIG.bit.START = 1;       // Request start conversion
  syncADC(ADC0, ADC_SYNCBUSY_MASK); // Make sure conversion has been initiated
  __asm__("nop\n\t");               // Tiny delay before we can poll RESRDY
  __asm__("nop\n\t");
  __asm__("nop\n\t");
  while (!ADC0->INTFLAG.bit.RESRDY) {} // Wait for result ready to be read
  return ADC0->RESULT.reg;             // Read result
#endif
}

Calibration ADC_autocalibrate() {
#if (ADC_DIFFERENTIAL == 0)
  const uint16_t N_samples = 2048; // Number of ADC reads to average over
  double V_LO = A_REF * .1;        // Linear fit point, low voltage
  double V_HI = A_REF * .9;        // Linear fit point, high voltage
  double sig_LO = 0;               // Average read bitvalue, low voltage
  double sig_HI = 0;               // Average read bitvalue, high voltage
  double ideal_LO = (V_LO / A_REF * MAX_ADC_INPUT_BITVAL);
  double ideal_HI = (V_HI / A_REF * MAX_ADC_INPUT_BITVAL);

  // Internally route the DAC output to the ADC input and disable the
  // calibration correction.
#  ifdef __SAMD21__
  ADC->INPUTCTRL.bit.MUXPOS = ADC_INPUTCTRL_MUXPOS_DAC_Val;
  syncADC();
  ADC->CTRLB.bit.CORREN = 0;
  syncADC();
#  elif defined __SAMD51__
  ADC0->INPUTCTRL.bit.MUXPOS = ADC_INPUTCTRL_MUXPOS_DAC_Val;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->CTRLB.bit.CORREN = 0;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
#  endif

  // clang-format off
  // Acquire LO voltage signal
  DAC_set_output(round(V_LO / A_REF * MAX_DAC_OUTPUT_BITVAL));
  for (uint16_t i = 0; i < N_samples; i++) {sig_LO += ADC_read_signal();}
  sig_LO /= N_samples;

  // Acquire HI voltage signal
  DAC_set_output(round(V_HI / A_REF * MAX_DAC_OUTPUT_BITVAL));
  for (uint16_t i = 0; i < N_samples; i++) {sig_HI += ADC_read_signal();}
  sig_HI /= N_samples;

  DAC_set_output(0);

  // Compute calibration corrections
  // See Microchip document TB3185 at
  // http://ww1.microchip.com/downloads/en/DeviceDoc/90003185A.pdf
  double  gain_error     = (sig_HI - sig_LO) / (ideal_HI - ideal_LO);
  int16_t gain_error_int = round(2048. / gain_error);
  int16_t offset_error   = round(sig_LO - (2048. / gain_error_int * ideal_LO));
  int16_t gaincorr   = ADC_GAINCORR_GAINCORR(gain_error_int);
  int16_t offsetcorr = ADC_OFFSETCORR_OFFSETCORR(offset_error);
  // clang-format on

  // Enable the calibration correction
  ADC_set_calibration_correction(gaincorr, offsetcorr);

  // Restore the ADC input pins
#  ifdef __SAMD21__
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC();
#  elif defined __SAMD51__
  ADC0->INPUTCTRL.bit.MUXPOS = g_APinDescription[A1].ulADCChannelNumber;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
#  endif

  Calibration result;
  result.is_valid = true;
  result.gaincorr = gaincorr;
  result.offsetcorr = offsetcorr;

  result.V_LO = V_LO;
  result.V_HI = V_HI;
  result.sig_LO = sig_LO;
  result.sig_HI = sig_HI;
  result.gain_error = gain_error;
  result.offset_error = offset_error;

  return result;
#endif
}

void ADC_set_calibration_correction(int16_t gaincorr, int16_t offsetcorr) {
#ifdef __SAMD21__
  ADC->GAINCORR.reg = gaincorr;
  ADC->OFFSETCORR.reg = offsetcorr;
  ADC->CTRLB.bit.CORREN = 1;
  syncADC();
#elif defined __SAMD51__
  ADC0->GAINCORR.reg = gaincorr;
  syncADC(ADC0, ADC_SYNCBUSY_GAINCORR);
  ADC0->OFFSETCORR.reg = offsetcorr;
  syncADC(ADC0, ADC_SYNCBUSY_OFFSET);
  ADC0->CTRLB.bit.CORREN = 1;
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
#endif
}
