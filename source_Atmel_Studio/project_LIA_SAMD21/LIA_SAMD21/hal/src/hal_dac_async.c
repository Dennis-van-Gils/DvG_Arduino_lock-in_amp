/**
 * \file
 *
 * \brief DAC functionality implementation.
 *
 * Copyright (c) 2014-2018 Microchip Technology Inc. and its subsidiaries.
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
 * THIS SOFTWARE IS SUPPLIED BY MICROCHIP "AS IS". NO WARRANTIES,
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

#include "hal_dac_async.h"
#include <utils_assert.h>
#include <utils.h>
#include <hal_atomic.h>

/**
 * \brief Driver version
 */
#define DRIVER_VERSION 0x00000001u

static void dac_tx_ready(struct _dac_async_device *device, const uint8_t ch);
static void dac_tx_error(struct _dac_async_device *device, const uint8_t ch);

/**
 * \brief Initialize the DAC HAL instance and hardware.
 */
int32_t dac_async_init(struct dac_async_descriptor *const descr, void *const hw)
{
	struct _dac_async_device *device;
	uint8_t                   i;
	int32_t                   rc;

	ASSERT(descr && hw);

	device = &descr->device;

	rc = _dac_async_init(device, hw);
	if (rc) {
		return rc;
	}

	device->dac_cb.tx_ready_cb = dac_tx_ready;
	device->dac_cb.tx_error_cb = dac_tx_error;

	for (i = 0; i < CHANNEL_NUM; i++) {
		descr->sel_ch[i].buffer = NULL;
		descr->sel_ch[i].length = 0;
	}

	return ERR_NONE;
}

/**
 * \brief Deinitialize the DAC HAL instance and hardware.
 */
int32_t dac_async_deinit(struct dac_async_descriptor *const descr)
{
	ASSERT(descr);

	_dac_async_deinit(&descr->device);

	return ERR_NONE;
}

/**
 * \brief Enable DAC channel.
 */
int32_t dac_async_enable_channel(struct dac_async_descriptor *const descr, const uint8_t ch)
{
	ASSERT(descr && (ch < CHANNEL_NUM));

	_dac_async_enable_channel(&descr->device, ch);

	return ERR_NONE;
}

/**
 * \brief Disable DAC channel.
 */
int32_t dac_async_disable_channel(struct dac_async_descriptor *const descr, const uint8_t ch)
{
	ASSERT(descr && (ch < CHANNEL_NUM));

	_dac_async_disable_channel(&descr->device, ch);

	return ERR_NONE;
}

/**
 * \brief Register DAC callback.
 */
int32_t dac_async_register_callback(struct dac_async_descriptor *const descr, const enum dac_async_callback_type type,
                                    dac_async_cb_t cb)
{
	ASSERT(descr);

	switch (type) {
	case DAC_ASYNC_CONVERSION_DONE_CB:
		descr->dac_cb.conversion_done = cb;
		break;

	case DAC_ASYNC_ERROR_CB:
		descr->dac_cb.error = cb;
		break;

	default:
		return ERR_INVALID_ARG;
	}

	_dac_async_set_irq_state(&descr->device, (enum _dac_callback_type)type, NULL != cb);

	return ERR_NONE;
}

/**
 * \brief DAC convert digital data to analog output.
 */
int32_t dac_async_write(struct dac_async_descriptor *const descr, const uint8_t ch, uint16_t *buffer, uint32_t length)
{
	ASSERT(descr && (ch < CHANNEL_NUM) && buffer && length);

	/* check whether channel is enable */
	if (!_dac_async_is_channel_enable(&descr->device, ch)) {
		return ERR_INVALID_ARG;
	}

	descr->sel_ch[ch].buffer = buffer;
	descr->sel_ch[ch].length = length;

	_dac_async_write_data(&descr->device, *(descr->sel_ch[ch].buffer), ch);

	return ERR_NONE;
}

/**
 * \brief Get DAC driver version.
 */
uint32_t dac_async_get_version(void)
{
	return DRIVER_VERSION;
}

/**
 * \internal Process transfer completion
 *
 * \param[in] device The pointer to DAC device structure
 */
static void dac_tx_ready(struct _dac_async_device *device, const uint8_t ch)
{
	struct dac_async_descriptor *const descr = CONTAINER_OF(device, struct dac_async_descriptor, device);

	if (descr->sel_ch[ch].length) {
		descr->sel_ch[ch].length--;
		if (descr->sel_ch[ch].length) {
			descr->sel_ch[ch].buffer++;
			_dac_async_write_data(&descr->device, *(descr->sel_ch[ch].buffer), ch);
		} else {
			if (descr->dac_cb.conversion_done) {
				descr->dac_cb.conversion_done(descr, ch);
			}
		}
	}
}

/**
 * \internal Error occurs in transfer process
 *
 * \param[in] device The pointer to DAC device structure
 */
static void dac_tx_error(struct _dac_async_device *device, const uint8_t ch)
{
	struct dac_async_descriptor *const descr = CONTAINER_OF(device, struct dac_async_descriptor, device);

	if (descr->dac_cb.error) {
		descr->dac_cb.error(descr, ch);
	}
}
//@}
