/*------------------------------------------------------------------------------
  adc_init.cpp

  Dennis van Gils, 04-08-2021
------------------------------------------------------------------------------*/
// clang-format off
#include "Arduino.h"
#include "adc_dac_init.h"
#include "adc_dac_functions.h"
// clang-format on

void ADC_init() {
  /* Increase the ADC clock by setting the PRESCALER from default DIV128 to a
  smaller divisor. This is needed for DAQ rates larger than ~20 kHz on SAMD51
  and DAQ rates larger than ~10 kHz on SAMD21. Setting too small divisors will
  result in ADC errors. Keep as large as possible to increase ADC accuracy.

  SAMD21:
    ADC input clock is default at 48 MHz (Generic Clock Generator 0).
    \.platformio\packages\framework-arduino-samd\cores\arduino\wiring.c

    Max ADC input clock frequency    :  48 MHz    (datasheet table 37-7)
    Max ADC internal clock frequency : 2.1 MHz    (datasheet table 37-24)

    Handy calculator:
    https://blog.thea.codes/getting-the-most-out-of-the-samd21-adc/

  SAMD51:
    ADC source clock is default at 48 MHz (Generic Clock Generator 1).
    \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\wiring.c

    Max ADC input clock frequency    : 100 MHz    (datasheet table 54-8)
    Max ADC internal clock frequency :  16 MHz    (datasheet table 54-24)
  */

#ifdef __SAMD21__

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

  // The ADC clock must remain below 2.1 MHz, see SAMD21 datasheet table 37-24.
  // Hence, don't go below DIV32 @ 48 MHz.
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

  // The ADC clock must remain below 16 MHz, see SAMD51 datasheet table 54-24.
  // Hence, don't go below DIV4 @ 48 MHz.
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

#endif /* #elif defined __SAMD51__ */
}

void DAC_init() {
  /* Simply rely on the built-in Arduino functions for initialization.

  SAMD21:
    DAC input clock is default at 48 MHz (Generic Clock Generator 0).
    \.platformio\packages\framework-arduino-samd\cores\arduino\wiring.c

    Max DAC input clock frequency    : 350 kHz    (*, datasheet table 37-7)
    Max DAC sample rate              : 350 ksps   (datasheet table 37-33)
    Note: There is no clock divider.
    (*): We still get away with the default 48 MHz?! Strange.

  SAMD51:
    DAC source clock is default at 12 MHz (Generic Clock Generator 4).
    \.platformio\packages\framework-arduino-samd-adafruit\cores\arduino\wiring.c

    Max DAC input clock frequency    : 100 MHz    (datasheet table 54-8)
    Max DAC internal clock frequency :  12 MHz    (datasheet table 54-28)
    Note: There is no clock divider.
  */
  analogWriteResolution(DAC_OUTPUT_BITS);
  analogWrite(A0, 0);
}