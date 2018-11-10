/*
Copyright (C) 2012  Albert van Dalen http://www.avdweb.nl
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License at http://www.gnu.org/licenses .

These libraries should also be installed:
Adafruit_ZeroTimer https://github.com/adafruit/Adafruit_ZeroTimer   
Adafruit_ASFcore https://github.com/adafruit/Adafruit_ASFcore

AUTHOR: Albert van Dalen
WEBSITE: http://www.avdweb.nl/arduino/libraries/samd21-timer.html

HISTORY:
1.0.0 26-1-2017

16 bit timer: max period_us = 1398080us (0.7Hz), min period_us = 1us (1MHz)

note *1 SCK, MOSI, MISO are on Arduino Zero SPI header

Timer              Arduino Zero pins              SAM15x15 pins

SAMDtimer 3 16bit  D5 D12                         d[17] d[21]
SAMDtimer 4 16bit  D16/A2 D21/SCL                 d[3]  d[25] 
SAMDtimer 5 16bit  D24/SCK*1                      d[13] 
*/

#include "avdweb_SAMDtimer.h"

SAMDtimer::SAMDtimer(byte timerNr, tc_counter_size countersize, byte pin, unsigned period_us, int pulseWidth_us, bool timerEnable): 
Adafruit_ZeroTimer(timerNr), pin(pin), countersize(countersize), period_us(period_us)  
{ if(pulseWidth_us==-1) calc(period_us, period_us/2);
  else calc(period_us, pulseWidth_us);
  init(timerEnable);    
}

SAMDtimer::SAMDtimer(byte timerNr, tc_callback_t _ISR, unsigned period_us, bool ISRenable):
Adafruit_ZeroTimer(timerNr)  
{ ISR = _ISR;
  countersize = TC_COUNTER_SIZE_16BIT; 
  calc(period_us, period_us/2);
  init(1);
  setCallback(ISRenable, TC_CALLBACK_CC_CHANNEL1, ISR); 
}

void SAMDtimer::setPulseWidth(unsigned pulseWidth_us)
{ calc(period_us, pulseWidth_us);
  setPeriodMatch(periodCounter, PWcounter, 1); 
}

void SAMDtimer::attachInterrupt(tc_callback_t _ISR, bool interruptEnable)
{ ISR = _ISR;
  setCallback(interruptEnable, TC_CALLBACK_CC_CHANNEL1, ISR); 
}

void SAMDtimer::enableTimer(bool timerEnable)
{ enable(timerEnable);
}

void SAMDtimer::enableInterrupt(bool interruptEnable)
{ setCallback(interruptEnable, TC_CALLBACK_CC_CHANNEL1, ISR); 
}

void SAMDtimer::init(bool enabled)
{ configure(prescale, countersize, TC_WAVE_GENERATION_MATCH_PWM);
  PWMout(true, 1, pin); // must be ch1 for 16bit
  setPeriodMatch(periodCounter, PWcounter, 1);
  enable(enabled); 
}

void SAMDtimer::calc(unsigned period_us, unsigned pulseWidth_us)
{ periodCounter = (F_CPU * (signed)period_us) / 1000000; // why signed?
  PWcounter = (F_CPU * (signed)pulseWidth_us) / 1000000; 
  if(periodCounter < 65536) prescale = TC_CLOCK_PRESCALER_DIV1; 
  else if((PWcounter >>= 1, periodCounter >>= 1) < 65536) prescale = TC_CLOCK_PRESCALER_DIV2; // = 256
  else if((PWcounter >>= 1, periodCounter >>= 1) < 65536) prescale = TC_CLOCK_PRESCALER_DIV4; 
  else if((PWcounter >>= 1, periodCounter >>= 1) < 65536) prescale = TC_CLOCK_PRESCALER_DIV8; 
  else if((PWcounter >>= 1, periodCounter >>= 1) < 65536) prescale = TC_CLOCK_PRESCALER_DIV16; 
  else if((PWcounter >>= 2, periodCounter >>= 2) < 65536) prescale = TC_CLOCK_PRESCALER_DIV64; 
  else if((PWcounter >>= 2, periodCounter >>= 2) < 65536) prescale = TC_CLOCK_PRESCALER_DIV256; 
  else if((PWcounter >>= 2, periodCounter >>= 2) < 65536) prescale = TC_CLOCK_PRESCALER_DIV1024; 
}





