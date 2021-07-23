/*
 * Code generated from Atmel Start.
 *
 * This file will be overwritten when reconfiguring your Atmel Start project.
 * Please copy examples or other code you want to keep to a separate file
 * to avoid losing it when reconfiguring.
 */

#include "driver_init.h"
#include <peripheral_clk_config.h>
#include <utils.h>
#include <hal_init.h>
#include <hpl_gclk_base.h>
#include <hpl_pm_base.h>

#include "DvG_init_mcu_SAMD21_48MHz.h"

/*! The buffer size for USART */
#define USART_0_BUFFER_SIZE 16

struct adc_dma_descriptor     ADC_0;
struct usart_async_descriptor USART_0;

static uint8_t USART_0_buffer[USART_0_BUFFER_SIZE];

struct dac_async_descriptor DAC_0;

struct timer_descriptor TIMER_0;

/**
 * \brief ADC initialization function
 *
 * Enables ADC peripheral, clocks and initializes ADC driver
 */
static void ADC_0_init(void)
{
	_pm_enable_bus_clock(PM_BUS_APBC, ADC);
	_gclk_enable_channel(ADC_GCLK_ID, CONF_GCLK_ADC_SRC);

	adc_dma_init(&ADC_0, ADC);

	// Disable digital pin circuitry
	gpio_set_pin_direction(PIN_A1__AIN, GPIO_DIRECTION_OFF);

	gpio_set_pin_function(PIN_A1__AIN, PINMUX_PB08B_ADC_AIN2);
}

/**
 * \brief USART Clock initialization function
 *
 * Enables register interface and peripheral clock
 */
void USART_0_CLOCK_init()
{

	_pm_enable_bus_clock(PM_BUS_APBC, SERCOM5);
	_gclk_enable_channel(SERCOM5_GCLK_ID_CORE, CONF_GCLK_SERCOM5_CORE_SRC);
}

/**
 * \brief USART pinmux initialization function
 *
 * Set each required pin to USART functionality
 */
void USART_0_PORT_init()
{

	gpio_set_pin_function(PIN_USART_TX, PINMUX_PB22D_SERCOM5_PAD2);

	gpio_set_pin_function(PIN_USART_RX, PINMUX_PB23D_SERCOM5_PAD3);
}

/**
 * \brief USART initialization function
 *
 * Enables USART peripheral, clocks and initializes USART driver
 */
void USART_0_init(void)
{
	USART_0_CLOCK_init();
	usart_async_init(&USART_0, SERCOM5, USART_0_buffer, USART_0_BUFFER_SIZE, (void *)NULL);
	USART_0_PORT_init();
}

void DAC_0_PORT_init(void)
{

	// Disable digital pin circuitry
	gpio_set_pin_direction(PIN_A0__VOUT, GPIO_DIRECTION_OFF);

	gpio_set_pin_function(PIN_A0__VOUT, PINMUX_PA02B_DAC_VOUT);
}

void DAC_0_CLOCK_init(void)
{

	_pm_enable_bus_clock(PM_BUS_APBC, DAC);
	_gclk_enable_channel(DAC_GCLK_ID, CONF_GCLK_DAC_SRC);
}

void DAC_0_init(void)
{
	DAC_0_CLOCK_init();
	dac_async_init(&DAC_0, DAC);
	DAC_0_PORT_init();
}

void EVENT_SYSTEM_0_init(void)
{
	_gclk_enable_channel(EVSYS_GCLK_ID_0, CONF_GCLK_EVSYS_CHANNEL_0_SRC);

	_pm_enable_bus_clock(PM_BUS_APBC, EVSYS);

	event_system_init();
}

void TIMER_0_CLOCK_init(void)
{
	_pm_enable_bus_clock(PM_BUS_APBC, TCC0);
	_gclk_enable_channel(TCC0_GCLK_ID, CONF_GCLK_TCC0_SRC);
}

void TIMER_0_init(void)
{
	TIMER_0_CLOCK_init();
	timer_init(&TIMER_0, TCC0, _tcc_get_timer());
}

void system_init(void)
{
	//init_mcu();
	init_mcu_SAMD21_48MHz();

	// GPIO on PA14

	gpio_set_pin_direction(PIN_D4__MCK,
	                       // <y> Pin direction
	                       // <id> pad_direction
	                       // <GPIO_DIRECTION_OFF"> Off
	                       // <GPIO_DIRECTION_IN"> In
	                       // <GPIO_DIRECTION_OUT"> Out
	                       GPIO_DIRECTION_OUT);

	gpio_set_pin_level(PIN_D4__MCK,
	                   // <y> Initial level
	                   // <id> pad_initial_level
	                   // <false"> Low
	                   // <true"> High
	                   false);

	gpio_set_pin_pull_mode(PIN_D4__MCK,
	                       // <y> Pull configuration
	                       // <id> pad_pull_config
	                       // <GPIO_PULL_OFF"> Off
	                       // <GPIO_PULL_UP"> Pull-up
	                       // <GPIO_PULL_DOWN"> Pull-down
	                       GPIO_PULL_OFF);

	gpio_set_pin_function(PIN_D4__MCK,
	                      // <y> Pin function
	                      // <id> pad_function
	                      // <i> Auto : use driver pinmux if signal is imported by driver, else turn off function
	                      // <GPIO_PIN_FUNCTION_OFF"> Auto
	                      // <GPIO_PIN_FUNCTION_OFF"> Off
	                      // <GPIO_PIN_FUNCTION_A"> A
	                      // <GPIO_PIN_FUNCTION_B"> B
	                      // <GPIO_PIN_FUNCTION_C"> C
	                      // <GPIO_PIN_FUNCTION_D"> D
	                      // <GPIO_PIN_FUNCTION_E"> E
	                      // <GPIO_PIN_FUNCTION_F"> F
	                      // <GPIO_PIN_FUNCTION_G"> G
	                      // <GPIO_PIN_FUNCTION_H"> H
	                      GPIO_PIN_FUNCTION_H);

	// GPIO on PA17

	gpio_set_pin_level(PIN_D13__LED,
	                   // <y> Initial level
	                   // <id> pad_initial_level
	                   // <false"> Low
	                   // <true"> High
	                   false);

	// Set pin direction to output
	gpio_set_pin_direction(PIN_D13__LED, GPIO_DIRECTION_OUT);

	gpio_set_pin_function(PIN_D13__LED, GPIO_PIN_FUNCTION_OFF);

	// GPIO on PA19

	gpio_set_pin_level(PIN_D12__TRIG_OUT,
	                   // <y> Initial level
	                   // <id> pad_initial_level
	                   // <false"> Low
	                   // <true"> High
	                   false);

	// Set pin direction to output
	gpio_set_pin_direction(PIN_D12__TRIG_OUT, GPIO_DIRECTION_OUT);

	gpio_set_pin_function(PIN_D12__TRIG_OUT, GPIO_PIN_FUNCTION_OFF);

	ADC_0_init();

	USART_0_init();

	DAC_0_init();

	EVENT_SYSTEM_0_init();

	TIMER_0_init();
}
