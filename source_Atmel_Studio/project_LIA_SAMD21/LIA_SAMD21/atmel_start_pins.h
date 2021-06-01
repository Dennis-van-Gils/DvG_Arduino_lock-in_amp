/*
 * Code generated from Atmel Start.
 *
 * This file will be overwritten when reconfiguring your Atmel Start project.
 * Please copy examples or other code you want to keep to a separate file
 * to avoid losing it when reconfiguring.
 */
#ifndef ATMEL_START_PINS_H_INCLUDED
#define ATMEL_START_PINS_H_INCLUDED

#include <hal_gpio.h>

// SAMD21 has 8 pin functions

#define GPIO_PIN_FUNCTION_A 0
#define GPIO_PIN_FUNCTION_B 1
#define GPIO_PIN_FUNCTION_C 2
#define GPIO_PIN_FUNCTION_D 3
#define GPIO_PIN_FUNCTION_E 4
#define GPIO_PIN_FUNCTION_F 5
#define GPIO_PIN_FUNCTION_G 6
#define GPIO_PIN_FUNCTION_H 7

#define PIN_A0__VOUT GPIO(GPIO_PORTA, 2)
#define PIN_D1__SAMPLING GPIO(GPIO_PORTA, 10)
#define PIN_D0__TRIG_OUT GPIO(GPIO_PORTA, 11)
#define PIN_D4__MCK GPIO(GPIO_PORTA, 14)
#define PIN_D13__LED GPIO(GPIO_PORTA, 17)
#define PIN_USB_D0 GPIO(GPIO_PORTA, 24)
#define PIN_USB_D1 GPIO(GPIO_PORTA, 25)
#define PIN_A1__AIN GPIO(GPIO_PORTB, 8)
#define PIN_USART_TX GPIO(GPIO_PORTB, 22)
#define PIN_USART_RX GPIO(GPIO_PORTB, 23)

#endif // ATMEL_START_PINS_H_INCLUDED
