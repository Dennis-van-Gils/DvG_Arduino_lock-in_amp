/**
 * \file
 *
 * \brief EVSYS related functionality implementation.
 *
 * Copyright (c) 2016-2018 Microchip Technology Inc. and its subsidiaries.
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

#include <err_codes.h>
#include <hpl_evsys_config.h>
#include <hpl_init.h>
#include <utils_repeat_macro.h>

/* Stub macros for reserved channel */
#ifndef CONF_CHANNEL_29
#define CONF_CHANNEL_29 0
#define CONF_ONDEMAND_29 0
#define CONF_RUNSTDBY_29 0
#define CONF_EDGSEL_29 0
#define CONF_PATH_29 0
#define CONF_EVGEN_29 0
#endif
#ifndef CONF_CHANNEL_30
#define CONF_CHANNEL_30 0
#define CONF_ONDEMAND_30 0
#define CONF_RUNSTDBY_30 0
#define CONF_EDGSEL_30 0
#define CONF_PATH_30 0
#define CONF_EVGEN_30 0
#endif

/* Event user configuration */
#define USER_MUX_CONF(i, n) EVSYS_USER_CHANNEL(CONF_CHANNEL_##n) | EVSYS_USER_USER(n),

/* This macro is used for repeat macro: i - unused, n - amount of channels.
 * It contains channel configuration. */
#define CHANNEL_CONF(i, n)                                                                                             \
	EVSYS_CHANNEL_CHANNEL(n) | EVSYS_CHANNEL_EDGSEL(CONF_EDGSEL_##n) | EVSYS_CHANNEL_PATH(CONF_PATH_##n)               \
	    | EVSYS_CHANNEL_EVGEN(CONF_EVGEN_##n),

/* This macro is used for repeat macro: i - unused, n - amount of channels
 * It contains interrupts configuration. */
#define INT_CFG(i, n)                                                                                                  \
	(CONF_OVR_##n << (n + EVSYS_INTENSET_OVR0_Pos)) | (CONF_EVD_##n << (n + EVSYS_INTENSET_EVD0_Pos)) |

static const uint16_t user_mux_confs[] = {REPEAT_MACRO(USER_MUX_CONF, i, EVSYS_USERS)};

static const uint32_t channel_confs[] = {REPEAT_MACRO(CHANNEL_CONF, i, EVSYS_CHANNELS)};

static const uint32_t interrupt_cfg = REPEAT_MACRO(INT_CFG, i, EVSYS_CHANNELS) 0;

/**
 * \brief Initialize event system
 */
int32_t _event_system_init(void)
{
	uint8_t i;

	/* configure user multiplexers */
	for (i = 0; i < EVSYS_USERS; i++) {
		hri_evsys_write_USER_reg(EVSYS, user_mux_confs[i]);
	}

	/* configure channels */
	for (i = 0; i < EVSYS_CHANNELS; i++) {
		hri_evsys_write_CHANNEL_reg(EVSYS, channel_confs[i]);
	}

	hri_evsys_write_INTEN_reg(EVSYS, interrupt_cfg);

	return ERR_NONE;
}

/**
 * \brief De-initialize event system.
 */
int32_t _event_system_deinit()
{
	hri_evsys_write_CTRL_reg(EVSYS, EVSYS_CTRL_SWRST);

	return ERR_NONE;
}

/**
 * \brief Enable/disable event reception by the given user from the given
 *        channel
 */
int32_t _event_system_enable_user(const uint16_t user, const uint16_t channel, const bool on)
{
	if (on) {
		hri_evsys_write_USER_reg(EVSYS, EVSYS_USER_CHANNEL(channel) | EVSYS_USER_USER(user));
	} else {
		hri_evsys_write_USER_reg(EVSYS, EVSYS_USER_USER(user));
	}

	return ERR_NONE;
}

/**
 * \brief Enable/disable event generation by the given generator for the given
 *        channel
 */
int32_t _event_system_enable_generator(const uint16_t generator, const uint16_t channel, const bool on)
{
	uint32_t cfg;
	if (channel >= EVSYS_CHANNELS) {
		return ERR_INVALID_ARG;
	}
	cfg = channel_confs[channel];
	cfg &= ~(EVSYS_CHANNEL_CHANNEL_Msk | EVSYS_CHANNEL_EVGEN_Msk);
	cfg |= EVSYS_CHANNEL_CHANNEL(channel);
	if (on) {
		cfg |= EVSYS_CHANNEL_EVGEN(generator);
		hri_evsys_write_CHANNEL_reg(EVSYS, cfg);
	} else {
		hri_evsys_write_CHANNEL_reg(EVSYS, cfg);
	}

	return ERR_NONE;
}
