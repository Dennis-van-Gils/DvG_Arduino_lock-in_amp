/*
 * Code generated from Atmel Start.
 *
 * This file will be overwritten when reconfiguring your Atmel Start project.
 * Please copy examples or other code you want to keep to a separate file
 * to avoid losing it when reconfiguring.
 */

#include "driver_examples.h"
#include "driver_init.h"
#include "utils.h"

/*! The buffer size for ADC */
#define ADC_0_BUFFER_SIZE 16
static uint8_t ADC_0_buffer[ADC_0_BUFFER_SIZE];

static void convert_cb_ADC_0(const struct adc_dma_descriptor *const descr)
{
}

/**
 * Example of using ADC_0 to generate waveform.
 */
void ADC_0_example(void)
{
	/* Enable ADC freerun mode in order to make example work */
	adc_dma_register_callback(&ADC_0, ADC_DMA_COMPLETE_CB, convert_cb_ADC_0);
	adc_dma_enable_channel(&ADC_0, 0);
	adc_dma_read(&ADC_0, ADC_0_buffer, ADC_0_BUFFER_SIZE);
}

/**
 * Example of using USART_0 to write "Hello World" using the IO abstraction.
 *
 * Since the driver is asynchronous we need to use statically allocated memory for string
 * because driver initiates transfer and then returns before the transmission is completed.
 *
 * Once transfer has been completed the tx_cb function will be called.
 */

static uint8_t example_USART_0[12] = "Hello World!";

static void tx_cb_USART_0(const struct usart_async_descriptor *const io_descr)
{
	/* Transfer completed */
}

void USART_0_example(void)
{
	struct io_descriptor *io;

	usart_async_register_callback(&USART_0, USART_ASYNC_TXC_CB, tx_cb_USART_0);
	/*usart_async_register_callback(&USART_0, USART_ASYNC_RXC_CB, rx_cb);
	usart_async_register_callback(&USART_0, USART_ASYNC_ERROR_CB, err_cb);*/
	usart_async_get_io_descriptor(&USART_0, &io);
	usart_async_enable(&USART_0);

	io_write(io, example_USART_0, 12);
}

static uint16_t example_DAC_0[10] = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900};

static void tx_cb_DAC_0(struct dac_async_descriptor *const descr, const uint8_t ch)
{
	dac_async_write(descr, 0, example_DAC_0, 10);
}

/**
 * Example of using DAC_0 to generate waveform.
 */
void DAC_0_example(void)
{
	dac_async_enable_channel(&DAC_0, 0);
	dac_async_register_callback(&DAC_0, DAC_ASYNC_CONVERSION_DONE_CB, tx_cb_DAC_0);
	dac_async_write(&DAC_0, 0, example_DAC_0, 10);
}

/**
 * Example of using TIMER_0.
 */
struct timer_task TIMER_0_task1, TIMER_0_task2;

static void TIMER_0_task1_cb(const struct timer_task *const timer_task)
{
}

static void TIMER_0_task2_cb(const struct timer_task *const timer_task)
{
}

void TIMER_0_example(void)
{
	TIMER_0_task1.interval = 100;
	TIMER_0_task1.cb       = TIMER_0_task1_cb;
	TIMER_0_task1.mode     = TIMER_TASK_REPEAT;
	TIMER_0_task2.interval = 200;
	TIMER_0_task2.cb       = TIMER_0_task2_cb;
	TIMER_0_task2.mode     = TIMER_TASK_REPEAT;

	timer_add_task(&TIMER_0, &TIMER_0_task1);
	timer_add_task(&TIMER_0, &TIMER_0_task2);
	timer_start(&TIMER_0);
}
