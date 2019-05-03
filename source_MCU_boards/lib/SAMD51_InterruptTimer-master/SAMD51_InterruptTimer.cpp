/*
  SAMD51_InterruptTimer.h
  Atmel SAMD51 interrupt service routine timer library
  For e.g. Adafruit M4 Metro/Feather/ItsyBitsy Express

  Method names mimic the 'ZeroTimer' library by Tamasa (@EHbtj) for easily
  switching compilation between SAMD21 and SAMD51 microprocessor boards.
  See https://github.com/EHbtj/ZeroTimer for the SAMD21 library.

  Dennis van Gils
  02-05-2019
*/

#include "Arduino.h"
#include "SAMD51_InterruptTimer.h"

#define CPU_HZ 48000000
#define TIMER_PRESCALER_DIV 1024

void (*func1)();

static inline void TC3_wait_for_sync() {
  while (TC3->COUNT16.SYNCBUSY.reg != 0) {}
}

void TC_Timer::startTimer(unsigned long period, void (*f)()) {
  // Activate timer TC3
  // CLK_TC3_APB
  MCLK->APBBMASK.reg |= MCLK_APBBMASK_TC3;

  // Set up the generic clock
  GCLK->GENCTRL[1].reg =
      // Divide clock source by divisor 1
      GCLK_GENCTRL_DIV(1) |
      // Set the duty cycle to 50/50 HIGH/LOW
      GCLK_GENCTRL_IDC |
      // Enable GCLK7
      GCLK_GENCTRL_GENEN |
      // Select 48MHz DFLL clock source
      GCLK_GENCTRL_SRC_DFLL;
      // Select 100MHz DPLL clock source
      //GCLK_GENCTRL_SRC_DPLL1;
      // Select 120MHz DPLL clock source
      //GCLK_GENCTRL_SRC_DPLL0;
  // Wait for synchronization
  while (GCLK->SYNCBUSY.bit.GENCTRL1);

  // Enable the TC bus clock
  GCLK->PCHCTRL[TC3_GCLK_ID].reg =
      // Enable the TC3 peripheral channel
      GCLK_PCHCTRL_CHEN |
      // Connect generic clock to TC3
      GCLK_PCHCTRL_GEN_GCLK1;
  while (GCLK->SYNCBUSY.reg > 0);

  TC3->COUNT16.CTRLA.bit.ENABLE = 0;
  
  // Use match mode so that the timer counter resets when the count matches the
  // compare register
  TC3->COUNT16.WAVE.bit.WAVEGEN = TC_WAVE_WAVEGEN_MFRQ;
  TC3_wait_for_sync();
  
   // Enable the compare interrupt
  TC3->COUNT16.INTENSET.reg = 0;
  TC3->COUNT16.INTENSET.bit.MC0 = 1;

  // Enable IRQ
  NVIC_EnableIRQ(TC3_IRQn);

  func1 = f;

  setPeriod(period);
}

void TC_Timer::setPeriod(unsigned long period) {
  int prescaler;
  uint32_t TC_CTRLA_PRESCALER_DIVN;

  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_ENABLE;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV1024;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV256;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV64;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV16;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV4;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV2;
  TC3_wait_for_sync();
  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_PRESCALER_DIV1;
  TC3_wait_for_sync();

  if (period > 300000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV1024;
    prescaler = 1024;
  } else if (80000 < period && period <= 300000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV256;
    prescaler = 256;
  } else if (20000 < period && period <= 80000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV64;
    prescaler = 64;
  } else if (10000 < period && period <= 20000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV16;
    prescaler = 16;
  } else if (5000 < period && period <= 10000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV8;
    prescaler = 8;
  } else if (2500 < period && period <= 5000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV4;
    prescaler = 4;
  } else if (1000 < period && period <= 2500) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV2;
    prescaler = 2;
  } else if (period <= 1000) {
    TC_CTRLA_PRESCALER_DIVN = TC_CTRLA_PRESCALER_DIV1;
    prescaler = 1;
  }
  TC3->COUNT16.CTRLA.reg |= TC_CTRLA_PRESCALER_DIVN;
  TC3_wait_for_sync();

  int compareValue = (int)(CPU_HZ / (prescaler/((float)period / 1000000))) - 1;

  // Make sure the count is in a proportional position to where it was
  // to prevent any jitter or disconnect when changing the compare value.
  TC3->COUNT16.COUNT.reg = map(TC3->COUNT16.COUNT.reg, 0,
                               TC3->COUNT16.CC[0].reg, 0, compareValue);
  TC3->COUNT16.CC[0].reg = compareValue;
  TC3_wait_for_sync();

  TC3->COUNT16.CTRLA.bit.ENABLE = 1;
  TC3_wait_for_sync();
}

void TC3_Handler() {
  // If this interrupt is due to the compare register matching the timer count
  if (TC3->COUNT16.INTFLAG.bit.MC0 == 1) {
    TC3->COUNT16.INTFLAG.bit.MC0 = 1;
    (*func1)();
  }
}

TC_Timer TC;






static inline void TCC0_wait_for_sync() {
  while (TCC0->SYNCBUSY.reg != 0) {}
}

void TCC_Timer_Pulse_Train::startTimer(unsigned long period) {
  // Activate timer TCC0
  MCLK->APBBMASK.reg |= MCLK_APBBMASK_TCC0;

  // Set up the generic clock
  GCLK->GENCTRL[7].reg =
      // Divide clock source by divisor 1
      GCLK_GENCTRL_DIV(1) |
      // Set the duty cycle to 50/50 HIGH/LOW
      GCLK_GENCTRL_IDC |
      // Enable GCLK7
      GCLK_GENCTRL_GENEN |
      // Select 48MHz DFLL clock source
      GCLK_GENCTRL_SRC_DFLL;
      // Select 100MHz DPLL clock source
      //GCLK_GENCTRL_SRC_DPLL1;
      // Select 120MHz DPLL clock source
      //GCLK_GENCTRL_SRC_DPLL0;
  // Wait for synchronization
  while (GCLK->SYNCBUSY.bit.GENCTRL7);

  // Enable the TC bus clock
  GCLK->PCHCTRL[TCC0_GCLK_ID].reg =
      // Enable the peripheral channel
      GCLK_PCHCTRL_CHEN |
      // Connect generic clock
      GCLK_PCHCTRL_GEN_GCLK7;
  while (GCLK->SYNCBUSY.reg > 0);

  // Enable the peripheral multiplexer on the desired digital output pin
  PORT->Group[g_APinDescription[5].ulPort].
      PINCFG[g_APinDescription[5].ulPin].bit.PMUXEN = 1;

  // Set the peripheral multiplexer
  // See datasheet, page 32, 6. I/O Multiplexing and Considerations
  // (peripheral A=0, B=1, C=2, D=3, E=4, F=5, G=6, etc)
  // http://forum.arduino.cc/index.php?topic=589655.msg4064311#msg4064311
  // 
  // Case A, Feather and Itsy: pin [9] (PORT_PA19), PORT_PMUX_PMUXO(6), TCC0
  // Case B, Itsy            : pin [5] (PORT_PA15), PORT_PMUX_PMUXO(6), TCC1
  // Case C, Feather         : pin [5] (PORT_PA16), PORT_PMUX_PMUXE(6), TCC0
  PORT->Group[g_APinDescription[5].ulPort].
      PMUX[g_APinDescription[5].ulPin >> 1].reg |= PORT_PMUX_PMUXE(6);

  TCC0->CTRLA.bit.ENABLE = 0;
  
  // Use match mode so that the timer counter resets when the count matches the
  // compare register
  TCC0->WAVE.bit.WAVEGEN = TCC_WAVE_WAVEGEN_MFRQ;
  TCC0_wait_for_sync();
  
  setPeriod(period);
}

void TCC_Timer_Pulse_Train::setPeriod(unsigned long period) {
  int prescaler;
  uint32_t TCC_CTRLA_PRESCALER_DIVN;

  TCC0->CTRLA.reg &= ~TCC_CTRLA_ENABLE;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV1024;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV256;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV64;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV16;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV4;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV2;
  TCC0_wait_for_sync();
  TCC0->CTRLA.reg &= ~TCC_CTRLA_PRESCALER_DIV1;
  TCC0_wait_for_sync();

  if (period > 300000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV1024;
    prescaler = 1024;
  } else if (80000 < period && period <= 300000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV256;
    prescaler = 256;
  } else if (20000 < period && period <= 80000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV64;
    prescaler = 64;
  } else if (10000 < period && period <= 20000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV16;
    prescaler = 16;
  } else if (5000 < period && period <= 10000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV8;
    prescaler = 8;
  } else if (2500 < period && period <= 5000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV4;
    prescaler = 4;
  } else if (1000 < period && period <= 2500) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV2;
    prescaler = 2;
  } else if (period <= 1000) {
    TCC_CTRLA_PRESCALER_DIVN = TCC_CTRLA_PRESCALER_DIV1;
    prescaler = 1;
  }
  TCC0->CTRLA.reg |= TCC_CTRLA_PRESCALER_DIVN;
  TCC0_wait_for_sync();

  int compareValue = (int)(CPU_HZ / (prescaler/((float)period / 1000000))) - 1;

  // Make sure the count is in a proportional position to where it was
  // to prevent any jitter or disconnect when changing the compare value.
  TCC0->COUNT.reg = map(TCC0->COUNT.reg, 0,
                        TCC0->CC[0].reg, 0, compareValue);
  TCC0->CC[0].reg = compareValue;
  TCC0_wait_for_sync();

  TCC0->CTRLA.bit.ENABLE = 1;
  TCC0_wait_for_sync();
}

TCC_Timer_Pulse_Train TCC_pulse_train;