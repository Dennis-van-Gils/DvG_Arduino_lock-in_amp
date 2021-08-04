#ifndef SETUP_SERIAL_H
#define SETUP_SERIAL_H

/*------------------------------------------------------------------------------
  Serial
--------------------------------------------------------------------------------

  Arduino M0 Pro
    Serial    : UART , Programming USB port
    SerialUSB : USART, Native USB port

  Adafruit Feather M4 Express
    Serial    : USART
*/

#define SERIAL_DATA_BAUDRATE 1e6 // Only used when Serial is UART

#ifdef ARDUINO_SAMD_ZERO
#  define Ser Serial // Serial or SerialUSB
#else
#  define Ser Serial // Only Serial
#endif

#define MAXLEN_buf 100 // Outgoing string buffer
char buf[MAXLEN_buf];  // Outgoing string buffer

#endif /* SETUP_SERIAL_H */