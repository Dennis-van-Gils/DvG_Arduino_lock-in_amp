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

#ifndef SAMD51_ISR_Timer_h
#define SAMD51_ISR_Timer_h

class TC_Timer {
  public:
    void startTimer(unsigned long period, void (*f)());
    void setPeriod(unsigned long period);
};

extern TC_Timer TC;


class TCC_Timer_Pulse_Train {
  public:
    void startTimer(unsigned long period);
    void setPeriod(unsigned long period);
};

extern TCC_Timer_Pulse_Train TCC_pulse_train;

#endif