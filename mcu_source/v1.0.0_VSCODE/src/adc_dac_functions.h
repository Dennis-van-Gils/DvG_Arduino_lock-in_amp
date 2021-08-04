/*------------------------------------------------------------------------------
  adc_dac_functions.h
  Collection of ADC & DAC functions for SAMD21 and SAMD51

  Dennis van Gils, 04-08-2021
------------------------------------------------------------------------------*/
#ifndef ADC_DAC_FUNCTIONS_H
#define ADC_DAC_FUNCTIONS_H

#include "Arduino.h"

// ADC correction parameters determined by the autocalibration routine
typedef struct Calibration {
  bool is_valid; // Flag for the flash storage to indicate an autocalibration
  int16_t gaincorr;
  int16_t offsetcorr;

  // Only used for reporting information
  double V_LO, V_HI;     // Voltages output by the autocalibration routine
  double sig_LO, sig_HI; // Average read bitvalues by the ADC
  double gain_error;     // Fraction
  int16_t offset_error;  // Bitvalue
} Calibration;

#ifdef __SAMD21__
static inline void syncADC() {
  // Taken from `hri_adc_d21.h`
  while (ADC->STATUS.bit.SYNCBUSY == 1) {}
}
#elif defined __SAMD51__
static inline void syncADC(const Adc *hw, uint32_t reg) {
  // Taken from `hri_adc_d51.h`
  while (((Adc *)hw)->SYNCBUSY.reg & reg) {}
}
#endif

/*------------------------------------------------------------------------------
  DAC
------------------------------------------------------------------------------*/

// Set a single DAC output value with synchronization checking
void DAC_set_output(int16_t);

// Set a single DAC output value without synchronization checking
void DAC_set_output_nosync(int16_t);

/*------------------------------------------------------------------------------
  ADC
------------------------------------------------------------------------------*/

// Read and return a single ADC sample with synchronization checking, software
// triggered
int16_t ADC_read_signal();

/* Autocalibration routine for the ADC in single-ended mode.

  The DAC voltage output will be internally routed to the ADC input, in addition
  to the analog output pin [A0]. During calibration the analog output will first
  output a low voltage, followed by a high voltage for each around 75 ms.

  - It is advised to first disconnect pins [A0] and [A1].
  - Only implemented for single-ended mode, not differential.
  - Make sure ADC and DAC are already initialized, otherwise the mcu will hang.
*/
Calibration ADC_autocalibrate();

// Set ADC calibration
void ADC_set_calibration_correction(int16_t gaincorr, int16_t offsetcorr);

#endif /* ADC_DAC_FUNCTIONS_H */