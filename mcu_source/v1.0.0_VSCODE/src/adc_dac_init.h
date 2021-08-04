/*------------------------------------------------------------------------------
  adc_init.h
  Sets up the ADC and DAC registers for SAMD21 and SAMD51, specific to the
  current project.

  Dennis van Gils, 04-08-2021
------------------------------------------------------------------------------*/
#ifndef ADC_DAC_INIT_H
#define ADC_DAC_INIT_H

// OBSERVATION: Single-ended has half the noise compared to differential
#define ADC_DIFFERENTIAL 0

// Microcontroller unit (mcu)
#if defined __SAMD21G18A__
#  ifndef __SAMD21__
#    define __SAMD21__
#  endif
#elif defined __SAMD21E18A__
#  ifndef __SAMD21__
#    define __SAMD21__
#  endif
#endif

#ifdef __SAMD21__
#  define ADC_INPUT_BITS 12
#  define DAC_OUTPUT_BITS 10
#elif defined __SAMD51__
#  define ADC_INPUT_BITS 12
#  define DAC_OUTPUT_BITS 12
#endif

#define A_REF 3.300 // [V] Analog voltage reference
#define MAX_ADC_INPUT_BITVAL ((1 << ADC_INPUT_BITS) - 1)
#define MAX_DAC_OUTPUT_BITVAL ((1 << DAC_OUTPUT_BITS) - 1)

// Set up the ADC registers specific to the current project
void ADC_init();

// Set up the DAC registers specific to the current project
void DAC_init();

#endif /* ADC_DAC_INIT_H */