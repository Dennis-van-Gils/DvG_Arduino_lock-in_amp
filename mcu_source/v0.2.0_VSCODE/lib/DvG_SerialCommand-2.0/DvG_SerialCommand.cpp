/*
Dennis van Gils, 11-03-2020
*/

#include "DvG_SerialCommand.h"

DvG_SerialCommand::DvG_SerialCommand(Stream& mySerial) :
_port(mySerial)   // Initialise reference before body
{
  _strIn[0] = '\0';
  _fTerminated = false;
  _iPos = 0;
}

bool DvG_SerialCommand::available() {
  char c;

  // Poll serial buffer
  if (_port.available()) {
    _fTerminated = false;
    while (_port.available()) {
      c = _port.peek();
      if (c == 13) {
        // Ignore ASCII 13 (carriage return)
        _port.read();             // Remove char from serial buffer
      } else if (c == 10) {
        // Found the proper termination character ASCII 10 (line feed)
        _port.read();             // Remove char from serial buffer
        _strIn[_iPos] = '\0';     // Terminate string
        _fTerminated = true;
        break;
      } else if (_iPos < STR_LEN - 1) {
        // Maximum length of incoming serial command is not yet reached. Append
        // characters to string.
        _port.read();             // Remove char from serial buffer
        _strIn[_iPos] = c;
        _iPos++;
      } else {
        // Maximum length of incoming serial command is reached. Forcefully
        // terminate string now. Leave the char in the serial buffer.
        _strIn[_iPos] = '\0';     // Terminate string
        _fTerminated = true;
        break;
      }
    }
  }
  return _fTerminated;
}

char* DvG_SerialCommand::getCmd() {
  if (_fTerminated) {
    _fTerminated = false;     // Reset incoming serial command char array
    _iPos = 0;                // Reset incoming serial command char array
    return (char*) _strIn;
  } else {
    return (char*) _empty;
  }
}

/*------------------------------------------------------------------------------
    Parse float value at end of string 'strIn' starting at position 'iPos'
------------------------------------------------------------------------------*/

float parseFloatInString(char* strIn, uint8_t iPos) {
  if (strlen(strIn) > iPos) {
    return (float) atof(&strIn[iPos]);
  } else {
    return 0.0f;
  }
}
