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
#define PIN_D13__LED GPIO(GPIO_PORTA, 17)
#define PIN_D12__TRIG_OUT GPIO(GPIO_PORTA, 19)
#define PIN_A1__AIN GPIO(GPIO_PORTB, 8)
#define PIN_USART_TX GPIO(GPIO_PORTB, 22)
#define PIN_USART_RX GPIO(GPIO_PORTB, 23)

#endif // ATMEL_START_PINS_H_INCLUDED
