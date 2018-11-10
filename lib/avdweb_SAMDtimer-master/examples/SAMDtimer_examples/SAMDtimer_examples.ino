#include <Arduino.h>
#include "avdweb_SAMDtimer.h"

// Example timer library for the SAMD21 and Arduino Zero

const byte LED1   = 3;
const byte LED2   = 0;  
const byte LED3   = 2;
const byte LED4   = 5; // timer3 16bit has 2 pins: 5 12

void ISR_timer3_LED1(struct tc_module *const module_inst) 
{ static bool b;
  pinMode(LED1, OUTPUT);
  digitalWrite(LED1, b=!b);
}

void ISR_timer4_LED2(struct tc_module *const module_inst) 
{ static bool b;
  pinMode(LED2, OUTPUT);
  digitalWrite(LED2, b=!b);
}

void ISR_timer4_LED3(struct tc_module *const module_inst) 
{ static bool b;
  pinMode(LED3, OUTPUT);
  digitalWrite(LED3, b=!b);
}

// ------------- Timer3 with output, select 1 of 3 -------------
   SAMDtimer timer3_1Hz = SAMDtimer(3, TC_COUNTER_SIZE_16BIT, LED4, 1e6, 9e5); // LED4 1Hz, pulse width 0.9s (this is the constructor)
// SAMDtimer timer3_1Hz = SAMDtimer(3, TC_COUNTER_SIZE_16BIT, LED4, 1e6, 9e5, 0); // the same but timer is disabled *1)
// SAMDtimer timer3_1Hz = SAMDtimer(3, TC_COUNTER_SIZE_16BIT, LED4, 1e6); // default 50% duty cycle

// ------------- Timer4 with ISR, without output, select 1 of 2 -------------
   SAMDtimer timer4_2Hz = SAMDtimer(4, ISR_timer4_LED2, 5e5); // ISR LED2 1Hz (0.5s on, 0.5s off)
// SAMDtimer timer4_2Hz = SAMDtimer(4, ISR_timer4_LED2, 5e5, 0); // the same but ISR is disabled *2)

void setup() // test out the several functions:
{ // ------------- timer3 ------------- 
  // timer3_1Hz.enableTimer(1); // enable timer *1), LED4 blinks again
  // timer3_1Hz.setPulseWidth(1e5); // change pulse width to 0.1s
  
  // attach a new ISR to a timer with output, this can't be done in the constructor
     timer3_1Hz.attachInterrupt(ISR_timer3_LED1); // LED1 blinks
  // timer3_1Hz.attachInterrupt(ISR_timer3_LED1, 0); // the same but ISR is disabled, the timer-output is not disabled
  
  // ------------- timer4 -------------
  // timer4_2Hz.enableInterrupt(0); // disable ISR_timer4_LED2, LED2 stops blinking
  // timer4_2Hz.enableInterrupt(1); // enable ISR_timer4_LED2, LED2 blinks again *2)
  // timer4_2Hz.attachInterrupt(ISR_timer4_LED3); // exchange ISR_timer4_LED2 -> ISR_timer4_LED3, LED3 blinks
}

void loop() 
{   
}

