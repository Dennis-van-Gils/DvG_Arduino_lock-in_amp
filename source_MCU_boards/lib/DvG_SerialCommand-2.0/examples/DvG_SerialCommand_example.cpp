/*
Demonstrates how to listen to a serial port on an Arduino for commands.

Will turn the onboard LED at pin 13 on or off when the following commands are
received: 'toggle', 'on', 'off'.

Dennis van Gils, 14-08-2018
*/

#include <Arduino.h>
#include "DvG_SerialCommand.h"

#ifndef PIN_LED
#define PIN_LED 13
#endif

// Serial port to listen to
#define Ser Serial
//#define Ser Serial1
//#define Ser SerialUSB

// Instantiate serial command listener
DvG_SerialCommand sc(Ser);

bool fState = false;

/*------------------------------------------------------------------------------
    setup
------------------------------------------------------------------------------*/

void setup() {
  Ser.begin(9600);

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, fState);
}

/*------------------------------------------------------------------------------
    loop
------------------------------------------------------------------------------*/

void loop() {
  char* strCmd; // Incoming serial command string

  if (sc.available()) {
    strCmd = sc.getCmd();
    Ser.print("Received: "); Ser.println(strCmd);

    if (strcmp(strCmd, "toggle") == 0) {
      Ser.println(" -> Toggling LED");
      fState = not(fState);
      digitalWrite(PIN_LED, fState);

    } else if (strcmp(strCmd, "on") == 0) {
      Ser.println(" -> LED ON");
      fState = true;
      digitalWrite(PIN_LED, fState);

    } else if (strcmp(strCmd, "off") == 0) {
      Ser.println(" -> LED OFF");
      fState = false;
      digitalWrite(PIN_LED, fState);
    }
  }
}
