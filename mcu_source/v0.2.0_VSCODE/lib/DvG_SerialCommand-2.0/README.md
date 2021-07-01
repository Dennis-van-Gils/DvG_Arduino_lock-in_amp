# Serial command listener

This library allows listening to a serial port for incoming commands and act upon them. To keep the memory usage low, it uses a C-string (null-terminated character array) to store incoming characters received over the serial port, instead of using a memory hungry C++ string. Carriage return ('\r', ASCII 13) characters are ignored. Once a linefeed ('\n', ASCII 10) character is received, or whenever the incoming message length has exceeded the buffer of size STR_LEN (defined in DvG_SerialCommand.h), we speak of a received 'command'. It doesn't matter if the command is ASCII or binary encoded.

``available()`` should be called periodically to poll for incoming characters. It will return true when a new command is ready to be processed. Subsequently, the command string can be retrieved by calling ``getCmd()``.

Example usage on an Arduino:
```C
#include <Arduino.h>
#include "DvG_SerialCommand.h"

#define Ser Serial  // Listen on this port
DvG_SerialCommand sc(Ser);

void setup() {
  Ser.begin(115200);  // Open port
}

void loop() {
  char* strCmd;     // Incoming serial command string

  if (sc.available()) {
    strCmd = sc.getCmd();

    // Your command string comparison routines and actions here
    if (strcmp(strCmd, "id?") == 0) {
      Ser.println("Arduino, Blinker");
    }
  }
}
```
