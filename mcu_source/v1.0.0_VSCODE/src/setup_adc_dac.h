#ifndef SETUP_ADC_DAC_H
#define SETUP_ADC_DAC_H

/*------------------------------------------------------------------------------
  Arduino lock-in amplifier
------------------------------------------------------------------------------*/
#include "Arduino.h"

// ADC correction parameters
typedef struct Calibration {
  bool is_valid;
  int16_t gaincorr;
  int16_t offsetcorr;
  double V_LO, V_HI;
  double sig_LO, sig_HI;
  double gain_error;
  int16_t offset_error;
} Calibration;

#ifdef __SAMD21__
static inline void syncADC() {
  // Taken from `hri_adc_d21.h`
  while (ADC->STATUS.bit.SYNCBUSY == 1) {}
}
#  define DAC_OUTPUT_BITS 10
#  define ADC_INPUT_BITS 12
#elif defined __SAMD51__
static inline void syncADC(const Adc *hw, uint32_t reg) {
  // Taken from `hri_adc_d51.h`
  while (((Adc *)hw)->SYNCBUSY.reg & reg) {}
}
#  define DAC_OUTPUT_BITS 12
#  define ADC_INPUT_BITS 12
#endif

// Analog port
#define A_REF 3.300 // [V] Analog voltage reference Arduino
#define MAX_DAC_OUTPUT_BITVAL ((1 << DAC_OUTPUT_BITS) - 1)

void DAC_set_output(int16_t);

void DAC_set_output_nosync(int16_t);

int16_t ADC_read_signal();

Calibration ADC_autocalibrate();

void ADC_set_calibration_correction(int16_t, int16_t);

void ADC_init();

#endif /* SETUP_ADC_DAC_H */