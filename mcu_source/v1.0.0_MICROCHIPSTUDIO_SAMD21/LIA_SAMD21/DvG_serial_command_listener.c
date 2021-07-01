/*
Dennis van Gils
08-06-2019
*/

#include "DvG_serial_command_listener.h"
#include <hal_io.h>
#include <string.h>
#include <stdio.h>

void scl_configure(DvG_scl *scl, struct io_descriptor *const io_descr) {
    scl->io = io_descr;
    memset(scl->str_in, '\0', sizeof(scl->str_in));
    scl->is_terminated = false;
    scl->idx = 0;
}



bool scl_available(DvG_scl *scl) {
    uint8_t c;

    scl->is_terminated = false;
    while (io_read(scl->io, &c, 1) == 1) {
        if (c == 13) {
            // Ignore ASCII 13 (carriage return)
        } else if (c == 10) {
            // Found the proper termination character ASCII 10 (line feed)
            scl->str_in[scl->idx] = '\0';           // Terminate string
            scl->is_terminated = true;
            break;
        } else if (scl->idx < SCL_STR_LEN - 1) {
            // Maximum length of incoming serial command is not yet reached.
            // Append characters to string.
            scl->str_in[scl->idx] = (char) c;
            scl->idx++;
        } else {
            // Maximum length of incoming serial command is reached. Forcefully
            // terminate string now and flush the remaining buffer contents.
            scl->str_in[scl->idx] = '\0';           // Terminate string
            scl->is_terminated = true;
            while (io_read(scl->io, &c, 1) == 1) {} // Flush buffer
            break;
        }
    }

    return scl->is_terminated;
}



char * scl_get_command(DvG_scl *scl) {
    if (scl->is_terminated) {
        // Reset incoming serial command char array
        scl->is_terminated = false;
        scl->idx = 0;
        return scl->str_in;
    } else {
        return '\0';
    }
}
