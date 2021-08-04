/*------------------------------------------------------------------------------
  Arduino lock-in amplifier
------------------------------------------------------------------------------*/
#include "setup_adc_dac.h"

// Set a single DAC output value with checking for synchronization
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

// Set a single DAC output value without checking for synchronization
void DAC_set_output_nosync(int16_t value) {
#ifdef __SAMD21__
  DAC->DATA.reg = value;
#elif defined __SAMD51__
  DAC->DATA[0].reg = value;
#endif
}

// Read and return a single ADC sample, software triggered
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

/*------------------------------------------------------------------------------
  ADC_autocalibrate
------------------------------------------------------------------------------*/

Calibration ADC_autocalibrate() {
  /* Autocalibration routine for the ADC in single-ended mode.

  The DAC voltage output will be internally routed to the ADC input, in addition
  to the analog output pin [A0]. During calibration the analog output will first
  output a low voltage, followed by a high voltage for each around 75 ms.

  - It is advised to first disconnect pins [A0] and [A1].
  - Only implemented for single-ended mode, not differential.
  - Make sure ADC and DAC are already enabled, otherwise the mcu will hang.
  */
#if (ADC_DIFFERENTIAL == 0)
  const uint16_t N_samples = 2048; // Number of ADC reads to average over
  double V_LO = A_REF * .1;        // Linear fit point, low voltage
  double V_HI = A_REF * .9;        // Linear fit point, high voltage
  double sig_LO = 0;               // Average read bitvalue, low voltage
  double sig_HI = 0;               // Average read bitvalue, high voltage
  double ideal_LO = (V_LO / A_REF * ((1 << ADC_INPUT_BITS) - 1));
  double ideal_HI = (V_HI / A_REF * ((1 << ADC_INPUT_BITS) - 1));

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

// Set ADC calibration
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

void ADC_init() {
#ifdef __SAMD21__
  // ADC source clock is default at 48 MHz (Generic Clock Generator 0)
  // DAC source clock is default at 48 MHz (Generic Clock Generator 0)
  // See
  // \.platformio\packages\framework-arduino-samd\cores\arduino\wiring.c
  //
  // Handy calculator:
  // https://blog.thea.codes/getting-the-most-out-of-the-samd21-adc/

  // Turn off ADC
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
  // default 63, stable 32 @ DIV32 & SAMPLENUM_4
  ADC->SAMPCTRL.bit.SAMPLEN = 32;
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

  // Turn on ADC
  ADC->CTRLA.bit.ENABLE = 1;
  syncADC();

#elif defined __SAMD51__
  // ADC source clock is default at 48 MHz (Generic Clock Generator 1)
  // DAC source clock is default at 12 MHz (Generic Clock Generator 4)
  // See
  // \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\wiring.c

  // Turn off ADC
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
  // default 5, stable 14 @ DIV16 & SAMPLENUM_4
  ADC0->SAMPCTRL.bit.OFFCOMP = 0; // When set to 1, SAMPLEN must be set to 0
  syncADC(ADC0, ADC_SYNCBUSY_MASK);
  ADC0->SAMPCTRL.bit.SAMPLEN = 14;
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

  // Turn on ADC
  ADC0->CTRLA.bit.ENABLE = 1;
  syncADC(ADC0, ADC_SYNCBUSY_SWRST | ADC_SYNCBUSY_ENABLE);

#endif
}