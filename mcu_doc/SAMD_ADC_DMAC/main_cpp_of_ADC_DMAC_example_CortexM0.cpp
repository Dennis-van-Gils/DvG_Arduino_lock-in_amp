/**
 * \file
 *
 * \brief Application implement
 *
 * Copyright (c) 2018 Microchip Technology Inc. and its subsidiaries.
 *
 * \asf_license_start
 *
 * \page License
 *
 * Subject to your compliance with these terms, you may use Microchip
 * software and any derivatives exclusively with Microchip products.
 * It is your responsibility to comply with third party license terms applicable
 * to your use of third party software (including open source software) that
 * may accompany Microchip software.
 *
 * THIS SOFTWARE IS SUPPLIED BY MICROCHIP "AS IS".  NO WARRANTIES,
 * WHETHER EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS SOFTWARE,
 * INCLUDING ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY,
 * AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT WILL MICROCHIP BE
 * LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE, INCIDENTAL OR CONSEQUENTIAL
 * LOSS, DAMAGE, COST OR EXPENSE OF ANY KIND WHATSOEVER RELATED TO THE
 * SOFTWARE, HOWEVER CAUSED, EVEN IF MICROCHIP HAS BEEN ADVISED OF THE
 * POSSIBILITY OR THE DAMAGES ARE FORESEEABLE.  TO THE FULLEST EXTENT
 * ALLOWED BY LAW, MICROCHIP'S TOTAL LIABILITY ON ALL CLAIMS IN ANY WAY
 * RELATED TO THIS SOFTWARE WILL NOT EXCEED THE AMOUNT OF FEES, IF ANY,
 * THAT YOU HAVE PAID DIRECTLY TO MICROCHIP FOR THIS SOFTWARE.
 *
 * \asf_license_stop
 *
 */
/*
 * Support and FAQ: visit <a href="https://www.microchip.com/support/">Microchip Support</a>
 */

#include <atmel_start.h>
#include <hpl_dma.h>
#include <hpl_adc_config.h>

/*
 * Enable port toggle to calculate idle time.
 * Use Oscilloscope to probe the pins.
 */
#define ENABLE_PORT_TOGGLE

/*! Maximum reload value that can be loaded to SysTick */
#define SYSTICK_MAX_VALUE (SysTick_LOAD_RELOAD_Msk - 1)

/*! Number of ADC samples to be taken and transferred (for with DMA and without DMA case) */
#define BLOCK_COUNT 1024

/*! SysTick variables to take time stamp at different instance */
uint32_t static time_stamp1 = 0;

uint32_t static volatile time_stamp2 = 0;

/*! brief Contains the number of cycles taken from SysTick time stamp */
uint32_t static cycles_taken = 0;

/*! brief Counter representing number of times CPU enters into idle loop */
uint32_t static idle_loop_count = 0;

/*! brief Initialize SysTick reload variable */
uint32_t static systick_reload = 0;

/*! brief Initialize SysTick overflow counter variable */
uint32_t static volatile systick_counter = 0;

/*! brief Flag to indicate DMA transfer done */
bool static volatile adc_dma_transfer_is_done = false;

/*! brief  buffer to store ADC results */
uint8_t static adc_result[BLOCK_COUNT];

/*! brief  buffer to store copy of ADC results */
uint8_t static adc_result_copy[BLOCK_COUNT];

/**
 * \brief ADC DMA complete callback called on DMA transfer complete event
 */
void adc_dma_complete_callback(const struct adc_dma_descriptor *const descr)
{
#if defined(ENABLE_PORT_TOGGLE)
	/* 	 Use oscilloscope to probe the pin. */
	gpio_toggle_pin_level(PA06_INT_TOGGLE);
#endif
	/* Enable DMA CH-1 (Software trigger) */
	_dma_enable_transaction(1, true);
}

/**
 * \brief DMA channel 1 callback
 */
void dmac_channel_1_callback(struct _dma_resource *resource)
{
#if defined(ENABLE_PORT_TOGGLE)
	/* 	 Use oscilloscope to probe the pin. */
	gpio_toggle_pin_level(PA06_INT_TOGGLE);
#endif

	/* Enable DMA CH-2 */
	_dma_enable_transaction(2, false);
}

/**
 * \brief DMA channel 2 callback
 */
void dmac_channel_2_callback(struct _dma_resource *resource)
{
#if defined(ENABLE_PORT_TOGGLE)
	/* 	 Use oscilloscope to probe the pin. */
	gpio_toggle_pin_level(PA06_INT_TOGGLE);
#endif

	/* Get time stamp */
	time_stamp2 = SysTick->VAL;

	/* Indicate DMA transfer has been completed */
	adc_dma_transfer_is_done = true;
}

/**
 * \brief SysTick interrupt handler
 */
void SysTick_Handler(void)
{
	/* Increment the software counter */
	systick_counter++;
}

/**
 * \brief Initialize the SysTick timer
 */
void systick_init()
{
	/* Calculate the reload value */
	systick_reload = SYSTICK_MAX_VALUE;

	/* Initialize software counter */
	systick_counter = 0;

	/* Disable the SYSTICK Counter */
	SysTick->CTRL &= (~SysTick_CTRL_ENABLE_Msk);

	/* set reload register */
	SysTick->LOAD = systick_reload;

	/* set Priority for Cortex-M0 System Interrupts */
	NVIC_SetPriority(SysTick_IRQn, (1 << __NVIC_PRIO_BITS) - 1);

	/* Load the SysTick Counter Value */
	SysTick->VAL  = systick_reload;
	SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;

	/* Enable SysTick interrupt */
	NVIC_EnableIRQ(SysTick_IRQn);
}

/**
 * \brief Calculate number of cycles taken to execute certain number of
 * instructions from the time stamp taken with system timer (SysTick)
 */
uint32_t calculate_cycles_taken(uint32_t start_cycle, uint32_t end_cycle)
{

	uint32_t total_cycles = 0;

	/* Check if counter flow occurs */
	if (systick_counter == 0) {
		/* Ensure Start cycle is greater than end cycle */
		if (start_cycle > end_cycle) {
			total_cycles = start_cycle - end_cycle;
		}
	} else if (systick_counter > 0) {
		total_cycles = start_cycle + ((systick_counter - 1) * SYSTICK_MAX_VALUE) + (SYSTICK_MAX_VALUE - end_cycle);
	}

	return total_cycles;
}

/**
 * \brief Configure DMA channel 2
 */
void config_dma_channel_2(void)
{
	struct _dma_resource *dma_res;

	/* Set DMA CH-2 source address (SRAM: adc_result_copy)*/
	_dma_set_source_address(2, (void *)adc_result_copy);

	/* Set DMA CH-2 destination address (USART DATA REG)*/
	_dma_set_destination_address(2, (void *)&(((Sercom *)(USART_0.device.hw))->USART.DATA.reg));

	/* Set DMA CH-2 block length */
	_dma_set_data_amount(2, (uint32_t)BLOCK_COUNT);

	/* Get DMA CH-2 resource to set the application callback */
	_dma_get_channel_resource(&dma_res, 2);

	/* Set application callback to handle the DMA CH-2 transfer done */
	dma_res->dma_cb.transfer_done = dmac_channel_2_callback;

	/* Enable DMA CH-2 transfer complete interrupt */
	_dma_set_irq_state(2, DMA_TRANSFER_COMPLETE_CB, true);
}

/**
 * \brief Configure DMA channel 1
 */
void config_dma_channel_1(void)
{
	struct _dma_resource *dma_res;

	/* Set DMA CH-1 source address (SRAM: adc_result)*/
	_dma_set_source_address(1, (void *)adc_result);

	/* Set DMA CH-1 destination address (SRAM: adc_result_copy)*/
	_dma_set_destination_address(1, (void *)adc_result_copy);

	/* Set DMA CH-1 block length */
	_dma_set_data_amount(1, (uint32_t)BLOCK_COUNT);

	/* Get DMA CH-1 resource to set the application callback */
	_dma_get_channel_resource(&dma_res, 1);

	/* Set application callback to handle the DMA CH-0 transfer done */
	dma_res->dma_cb.transfer_done = dmac_channel_1_callback;

	/* Enable DMA CH-1 transfer complete interrupt */
	_dma_set_irq_state(1, DMA_TRANSFER_COMPLETE_CB, true);
}

/**
 * \brief Register application callback and enable ADC module
 * Set up the DMA destination address and number of bytes to read
 */
void configure_adc(void)
{
	/* Set destination address of DMA and the number of bytes to read */
	adc_dma_read(&ADC_0, (uint8_t *)adc_result, BLOCK_COUNT);

	/* Register an application callback on DMA transfer complete event */
	adc_dma_register_callback(&ADC_0, ADC_DMA_COMPLETE_CB, adc_dma_complete_callback);

	/* Enable ADC module */
	adc_dma_enable_channel(&ADC_0, 0);
}

/**
 * \brief Enable SERCOM - USART
 */
void configure_usart(void)
{
	/* Enable USART module */
	usart_sync_enable(&USART_0);
}

int main(void)
{
	/* Initializes MCU, drivers and middleware */
	atmel_start_init();

	/* Start SysTick Timer */
	systick_init();

	/* Register callback and enable ADC */
	configure_adc();

	/* Configure DMA channels 1 */
	config_dma_channel_1();

	/* Configure DMA channels 2 */
	config_dma_channel_2();

	/* Enable SERCOM - USART */
	configure_usart();

	/* Get the time stamp 1 before starting ADC transfer */
	time_stamp1 = SysTick->VAL;

	/*
	 * Trigger first ADC conversion through software.
	 * Further conversions are triggered through event generated when previous
	 * ADC result is transferred to destination (RAM buffer) by the DMA.
	 */
	adc_dma_start_conversion(&ADC_0);

	while (1) {
		/* Increment idle count whenever application reached while(1) loop */
		idle_loop_count++;

#if defined(ENABLE_PORT_TOGGLE)
		/* Use oscilloscope to probe the pin */
		gpio_toggle_pin_level(CPU_IDLE_TIME_CNTR_INC_TIME);
#endif

		if (true == adc_dma_transfer_is_done) {
			struct io_descriptor *io;
			/*
			 * Calculate number of cycles taken from the time stamp
			 * taken before start of the conversion and after 1024 transfer
			 * is completed.
			 * NOTE: This value in relation to the idle_loop_count is
			 * used in calculating CPU usage.
			 */
			cycles_taken = calculate_cycles_taken(time_stamp1, time_stamp2);

			usart_sync_get_io_descriptor(&USART_0, &io);

			/* Write the CPU cycles taken on USART */
			io_write(io, (uint8_t *)&cycles_taken, sizeof(cycles_taken));

			/* Print idle loop count on USART */
			io_write(io, (uint8_t *)&idle_loop_count, sizeof(idle_loop_count));

			/* Enter into forever loop as all transfers are completed */

			while (1)
				;
		}
	}
}

