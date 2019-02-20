/*
Listen to the serial port for commands.

This library uses a C-string (null-terminated character array) to store incoming
characters received over a serial port. Carriage return ('\r', ASCII 13)
characters are ignored. Once a linefeed ('\n', ASCII 10) character is received,
or whenever the incoming message length has exceeded the buffer of size STR_LEN,
we speak of a received 'command'.

'available()' should be called periodically to poll for incoming characters. It
will return true when a new command is ready to be processed. Subsequently, the
command string can be retrieved by calling 'getCmd()'.

A C-string is used to reduce memory overhead and prevent possible memory
fragmentation that otherwise could happen when using a C++ string instead.

Dennis van Gils, 14-08-2018
*/

#ifndef H_DvG_SerialCommand
#define H_DvG_SerialCommand

#include <Arduino.h>

// Buffer size for storing incoming characters. Includes the '\0' termination
// character. Change buffer size to your needs up to a maximum of 255.
#define STR_LEN 32

class DvG_SerialCommand {
 public:
  DvG_SerialCommand(Stream& mySerial);

  // Poll the serial port for characters and append to buffer. Return true if
  // a command is ready to be processed.
  bool available();

  // Return the incoming serial command only when it is ready, otherwise return
  // an empty C-string.
  char* getCmd();

 private:
  Stream& _port;              // Serial port reference
  char    _strIn[STR_LEN];    // Incoming serial command string
  bool    _fTerminated;       // Incoming serial command is/got terminated?
  uint8_t _iPos;              // Index within _strIn to insert new char
};

/*------------------------------------------------------------------------------
    Parse float value at end of string 'strIn' starting at position 'iPos'
------------------------------------------------------------------------------*/

float parseFloatInString(char* strIn, uint8_t iPos);

#endif
