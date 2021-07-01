/*
Dennis van Gils
12-07-2019
*/

#include "DvG_init_mcu_SAMD21_48MHz.h"
#include <hal_init.h>
#include <hpl_pm_base.h>

#include <hpl_dma.h>
#include <hpl_dmac_config.h>
#include <hpl_pm_config.h>

#define VARIANT_MAINOSC (32768ul)     // Frequency of the board main oscillator
#define VARIANT_MCK     (48000000ul)  // Master clock frequency

void init_mcu_SAMD21_48MHz(void) {
	/*
	Taken from Arduino IDE distribution and modified

	Configures the needed clocks and according Flash Read Wait States in order
	to have the CPU run at 48 MHz. Atmel Start can't do this for us (yet? 28-06-2019).

	At reset:
	- OSC8M clock source is enabled with a divider by 8 (1MHz).
	- Generic Clock Generator 0 (GCLKMAIN) is using OSC8M as source.
	We need to:
	1) Enable XOSC32K clock (External on-board 32.768Hz oscillator), will be used as DFLL48M reference.
	2) Put XOSC32K as source of Generic Clock Generator 1
	3) Put Generic Clock Generator 1 as source for Generic Clock Multiplexer 0 (DFLL48M reference)
	4) Enable DFLL48M clock
	5) Switch Generic Clock Generator 0 to DFLL48M. CPU will run at 48MHz.
	6) Modify PRESCaler value of OSCM to have 8MHz
	7) Put OSC8M as source for Generic Clock Generator 3
	...) and add more Clock and Generic Clock Generator setup to fit your needs
	*/

	// From init_mcu()
	// ---------------

	hri_nvmctrl_set_CTRLB_RWS_bf(NVMCTRL, CONF_NVM_WAIT_STATE);
	_pm_init();

	// From Arduino IDE distribution
	// -----------------------------

	// Set 1 Flash Wait State for 48MHz, cf tables 20.9 and 35.27 in SAMD21 Datasheet
	NVMCTRL->CTRLB.bit.RWS = NVMCTRL_CTRLB_RWS_HALF_Val ;

	// Turn on the digital interface clock
	PM->APBAMASK.reg |= PM_APBAMASK_GCLK ;

  #if defined(CRYSTALLESS)

	// 1)
	// Enable OSC32K clock (Internal 32.768Hz oscillator)
	// ---------------------------------------------------------------------------------------------

	uint32_t calib = (*((uint32_t *) FUSES_OSC32K_CAL_ADDR) & FUSES_OSC32K_CAL_Msk) >> FUSES_OSC32K_CAL_Pos;

	SYSCTRL->OSC32K.reg = SYSCTRL_OSC32K_CALIB(calib) |
                          SYSCTRL_OSC32K_STARTUP( 0x6u ) | // cf table 15.10 of product datasheet in chapter 15.8.6
                          SYSCTRL_OSC32K_EN32K |
                          SYSCTRL_OSC32K_ENABLE;

	// Wait for oscillator stabilization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_OSC32KRDY) == 0 );

  #else // has crystal

	// 1)
	// Enable XOSC32K clock (External on-board 32.768Hz oscillator)
	// ---------------------------------------------------------------------------------------------

	SYSCTRL->XOSC32K.reg = SYSCTRL_XOSC32K_STARTUP( 0x6u ) | // cf table 15.10 of product data sheet in chapter 15.8.6
                           SYSCTRL_XOSC32K_XTALEN | SYSCTRL_XOSC32K_EN32K ;
	SYSCTRL->XOSC32K.bit.ENABLE = 1 ; // separate call, as described in chapter 15.6.3

	// Wait for oscillator stabilization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_XOSC32KRDY) == 0 ) {}

  #endif

	// Software reset the module to ensure it is re-initialized correctly
	// Note: Due to synchronization, there is a delay from writing CTRL.SWRST until the reset is complete.
	// CTRL.SWRST and STATUS.SYNCBUSY will both be cleared when the reset is complete, as described in chapter 13.8.1
	GCLK->CTRL.reg = GCLK_CTRL_SWRST ;

	// Wait for reset to complete
	while ( (GCLK->CTRL.reg & GCLK_CTRL_SWRST) && (GCLK->STATUS.reg & GCLK_STATUS_SYNCBUSY) ) {}

	// 2)
	// Generic Clock Generator 1: sourced by (X)OSC32K
	// ---------------------------------------------------------------------------------------------

	GCLK->GENDIV.reg = GCLK_GENDIV_ID( 1 ) ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	GCLK->GENCTRL.reg = GCLK_GENCTRL_ID( 1 ) |
  #if defined(CRYSTALLESS)
                        GCLK_GENCTRL_SRC_OSC32K |  // Internal 32KHz Oscillator
  #else
                        GCLK_GENCTRL_SRC_XOSC32K | // External 32KHz Oscillator
  #endif
                        //GCLK_GENCTRL_OE | // Output clock to a pin for tests
                        GCLK_GENCTRL_GENEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	// 3)
	// Generic Clock Multiplexer 0 (DFLL48M reference): sourced by Generic Clock Generator 1
	// ---------------------------------------------------------------------------------------------

	GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID( 0 ) |   // Generic Clock Multiplexer 0
                        GCLK_CLKCTRL_GEN_GCLK1 | // Generic Clock Generator 1 is source
                        GCLK_CLKCTRL_CLKEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	// 4)
	// Enable DFLL48M clock
	// ---------------------------------------------------------------------------------------------

	// DFLL Configuration in Closed Loop mode, cf product data sheet chapter 15.6.7.1 - Closed-Loop Operation

	// Remove the OnDemand mode, Bug http://avr32.icgroup.norway.atmel.com/bugzilla/show_bug.cgi?id=9905
	SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE;

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

	SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_CSTEP( 31 ) |  // Coarse step is 31, half of the max value
                           SYSCTRL_DFLLMUL_FSTEP( 511 ) | // Fine step is 511, half of the max value
                           SYSCTRL_DFLLMUL_MUL( (VARIANT_MCK + VARIANT_MAINOSC/2) / VARIANT_MAINOSC ) ; // External 32KHz is the reference

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

  #if defined(CRYSTALLESS)

	#define NVM_SW_CALIB_DFLL48M_COARSE_VAL 58

	// Turn on DFLL
	uint32_t coarse = ( *((uint32_t *)(NVMCTRL_OTP4) + (NVM_SW_CALIB_DFLL48M_COARSE_VAL / 32)) >> (NVM_SW_CALIB_DFLL48M_COARSE_VAL % 32) )
					  & ((1 << 6) - 1);
	if (coarse == 0x3f) { coarse = 0x1f; }
	// TODO(tannewt): Load this value from memory we've written previously. There
	// isn't a value from the Atmel factory.
	uint32_t fine = 0x1ff;

	SYSCTRL->DFLLVAL.bit.COARSE = coarse;
	SYSCTRL->DFLLVAL.bit.FINE = fine;
	// Write full configuration to DFLL control register
	SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_CSTEP( 0x1f / 4 ) | // Coarse step is 31, half of the max value
                           SYSCTRL_DFLLMUL_FSTEP( 10 ) |
                           SYSCTRL_DFLLMUL_MUL( (48000) ) ;

	SYSCTRL->DFLLCTRL.reg = 0;

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

	SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_MODE |
                            SYSCTRL_DFLLCTRL_CCDIS |
                            SYSCTRL_DFLLCTRL_USBCRM |	// USB correction
                            SYSCTRL_DFLLCTRL_BPLCKC;

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

	// Enable the DFLL
	SYSCTRL->DFLLCTRL.reg |= SYSCTRL_DFLLCTRL_ENABLE ;

  #else   // has crystal

	// Write full configuration to DFLL control register
	SYSCTRL->DFLLCTRL.reg |= SYSCTRL_DFLLCTRL_MODE |	// Enable the closed loop mode
                             SYSCTRL_DFLLCTRL_WAITLOCK |
                             SYSCTRL_DFLLCTRL_QLDIS ;	// Disable Quick lock

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

	// Enable the DFLL
	SYSCTRL->DFLLCTRL.reg |= SYSCTRL_DFLLCTRL_ENABLE ;

	// Wait for locks flags
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLLCKC) == 0 ||
			(SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLLCKF) == 0 ) {}

  #endif

	// Wait for synchronization
	while ( (SYSCTRL->PCLKSR.reg & SYSCTRL_PCLKSR_DFLLRDY) == 0 ) {}

	// 5)
	// Generic Clock Generator 0: sourced by DFLL48M
	// ---------------------------------------------------------------------------------------------
	// CPU

	GCLK->GENDIV.reg = GCLK_GENDIV_ID( 0 ) |
                       GCLK_GENDIV_DIV(1);  // Set to 1 for CPU @ 48 MHz

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	GCLK->GENCTRL.reg = GCLK_GENCTRL_ID( 0 ) |
                        GCLK_GENCTRL_SRC_DFLL48M |
                        //GCLK_GENCTRL_OE |   // Output clock to a pin for tests (PA14: PIN_D4__MCK)
                        GCLK_GENCTRL_IDC |  // Set 50/50 duty cycle
                        GCLK_GENCTRL_GENEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	// 6)
	// Modify PRESCaler value of OSC8M to have 8MHz
	// ---------------------------------------------------------------------------------------------

	SYSCTRL->OSC8M.bit.PRESC = SYSCTRL_OSC8M_PRESC_0_Val ;  // CMSIS 4.5 changed the prescaler defines
	SYSCTRL->OSC8M.bit.ONDEMAND = 0 ;


	// Other Generic Clock Generators to be setup as well
	// --------------------------------------------------

	// 7)
	// Generic Clock Generator 3: sourced by OSC8M
	// ---------------------------------------------------------------------------------------------
	// General purpose 8 MHz (not used)

	/*
	GCLK->GENDIV.reg = GCLK_GENDIV_ID( 3 ) |
                       GCLK_GENDIV_DIV(1);

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	GCLK->GENCTRL.reg = GCLK_GENCTRL_ID( 3 ) |
                        GCLK_GENCTRL_SRC_OSC8M |
                        GCLK_GENCTRL_GENEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}
	*/

	// 8)
	// Generic Clock Generator 4: sourced by DFLL48M
	// ---------------------------------------------------------------------------------------------
	// ADC, DAC

	GCLK->GENDIV.reg = GCLK_GENDIV_ID( 4 ) |
	                   GCLK_GENDIV_DIV(1);

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	GCLK->GENCTRL.reg = GCLK_GENCTRL_ID( 4 ) |
                        GCLK_GENCTRL_SRC_DFLL48M |
						//GCLK_GENCTRL_IDC |  // Set 50/50 duty cycle
                        GCLK_GENCTRL_GENEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	// 9)
	// Generic Clock Generator 5: sourced by DFLL48M
	// ---------------------------------------------------------------------------------------------
	// USB

	GCLK->GENDIV.reg = GCLK_GENDIV_ID( 5 ) |
                       GCLK_GENDIV_DIV(1);

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	GCLK->GENCTRL.reg = GCLK_GENCTRL_ID( 5 ) |
                        GCLK_GENCTRL_SRC_DFLL48M |
						//GCLK_GENCTRL_IDC |  // Set 50/50 duty cycle
                        GCLK_GENCTRL_GENEN ;

	while (GCLK->STATUS.bit.SYNCBUSY) {}

	// From Arduino IDE distribution
	// -----------------------------

	// Now that all system clocks are configured, we can set CPU and APBx BUS clocks.
	// There values are normally the one present after Reset.

	PM->CPUSEL.reg  = PM_CPUSEL_CPUDIV_DIV1 ;
	PM->APBASEL.reg = PM_APBASEL_APBADIV_DIV1_Val ;
	PM->APBBSEL.reg = PM_APBBSEL_APBBDIV_DIV1_Val ;
	PM->APBCSEL.reg = PM_APBCSEL_APBCDIV_DIV1_Val ;

	SystemCoreClock=VARIANT_MCK ;

	// From init_mcu()
	// ---------------

  #if CONF_DMAC_ENABLE
	_pm_enable_bus_clock(PM_BUS_AHB, DMAC);
	_pm_enable_bus_clock(PM_BUS_APBB, DMAC);
	_dma_init();
  #endif
}