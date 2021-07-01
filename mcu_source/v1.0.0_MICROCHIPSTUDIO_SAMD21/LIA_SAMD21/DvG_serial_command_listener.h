/*
Listen to the serial port for commands.
To be used in the Atmel Advanced Software Framework Version 4.

Dennis van Gils
08-06-2019

This library uses a C-string (null-terminated character array) to store incoming
characters received over a serial port. Carriage return ('\r', ASCII 13)
characters are ignored. Once a linefeed ('\n', ASCII 10) character is received,
or whenever the incoming message length has exceeded the buffer of size
SCL_STR_LEN, we speak of a received 'command'.

'scl_configure()' should be called once to link the io-stream to the serial
command listener.

Case: ASF4 USART_Async driver
    'scl_available()' should be called after a USART_ASYNC_RXC_CB callback has
    been received to retrieve the incoming character buffer.
Case: ASF4 USART_Sync driver
    'scl_available()' should be called periodically to poll for incoming
    characters.

'scl_available()' will return true when a new command is ready to be processed.
The command can be retrieved by calling 'scl_get_command()'.

Example usage in ASF4:
    struct io_descriptor* io;

    volatile static bool data_arrived = false;

    static void cb_USART_0_rxc(const struct usart_async_descriptor *const io_descr) {
        data_arrived = true;
    }

    int main(void) {
        usart_async_get_io_descriptor(&USART_0, &io);
        usart_async_register_callback(&USART_0, USART_ASYNC_RXC_CB, cb_USART_0_rxc);
        usart_async_enable(&USART_0);

        DvG_scl scl_1;
        scl_configure(&scl_1, io);

        char *str_cmd;
        char buffer[64];

        while (1) {
            if (data_arrived) {
                data_arrived = false;

                if (scl_available(&scl_1)) {
                    str_cmd = scl_get_command(&scl_1);

                    // Echo command back
                    sprintf(buffer, "%s\n", str_cmd);    \\ #include <stdio.h>
                    io_write(io, (uint8_t *) buffer, strlen(buffer));
                }
            }
        }
    }
*/

#ifndef H_DvG_serial_command_listener
#define H_DvG_serial_command_listener

#include <stdint.h>
#include <stdbool.h>

// Buffer size for storing incoming characters. Includes the '\0' termination
// character. Change buffer size to your needs up to a maximum of 255.
#define SCL_STR_LEN 32

typedef struct DvG_scl {
    struct io_descriptor *io;
    char    str_in[SCL_STR_LEN];  // Incoming serial command string
    bool    is_terminated;        // Incoming serial command is/got terminated?
    uint8_t idx;                  // Index within str_in to insert new char
} DvG_scl;

void scl_configure(DvG_scl *scl, struct io_descriptor *const io_descr);

bool scl_available(DvG_scl *scl);

char * scl_get_command(DvG_scl *scl);

#endif