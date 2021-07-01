/* Auto-generated config file hpl_evsys_config.h */
#ifndef HPL_EVSYS_CONFIG_H
#define HPL_EVSYS_CONFIG_H

// <<< Use Configuration Wizard in Context Menu >>>

// <e> Channel 0 settings
// <id> evsys_channel_setting_0
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_0
#define CONF_EVSYS_CHANNEL_SETTINGS_0 1
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_0
#ifndef CONF_EDGSEL_0
#define CONF_EDGSEL_0 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_0
#ifndef CONF_PATH_0
#define CONF_PATH_0 EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_0
#ifndef CONF_EVGEN_0
#define CONF_EVGEN_0 37
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_0
#ifndef CONF_OVR_0
#define CONF_OVR_0 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_0
#ifndef CONF_EVD_0
#define CONF_EVD_0 0
#endif

// </e>

// <e> Channel 1 settings
// <id> evsys_channel_setting_1
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_1
#define CONF_EVSYS_CHANNEL_SETTINGS_1 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_1
#ifndef CONF_EDGSEL_1
#define CONF_EDGSEL_1 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_1
#ifndef CONF_PATH_1
#define CONF_PATH_1 EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_1
#ifndef CONF_EVGEN_1
#define CONF_EVGEN_1 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_1
#ifndef CONF_OVR_1
#define CONF_OVR_1 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_1
#ifndef CONF_EVD_1
#define CONF_EVD_1 0
#endif

// </e>

// <e> Channel 2 settings
// <id> evsys_channel_setting_2
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_2
#define CONF_EVSYS_CHANNEL_SETTINGS_2 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_2
#ifndef CONF_EDGSEL_2
#define CONF_EDGSEL_2 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_2
#ifndef CONF_PATH_2
#define CONF_PATH_2 EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_2
#ifndef CONF_EVGEN_2
#define CONF_EVGEN_2 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_2
#ifndef CONF_OVR_2
#define CONF_OVR_2 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_2
#ifndef CONF_EVD_2
#define CONF_EVD_2 0
#endif

// </e>

// <e> Channel 3 settings
// <id> evsys_channel_setting_3
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_3
#define CONF_EVSYS_CHANNEL_SETTINGS_3 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_3
#ifndef CONF_EDGSEL_3
#define CONF_EDGSEL_3 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_3
#ifndef CONF_PATH_3
#define CONF_PATH_3 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_3
#ifndef CONF_EVGEN_3
#define CONF_EVGEN_3 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_3
#ifndef CONF_OVR_3
#define CONF_OVR_3 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_3
#ifndef CONF_EVD_3
#define CONF_EVD_3 0
#endif

// </e>

// <e> Channel 4 settings
// <id> evsys_channel_setting_4
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_4
#define CONF_EVSYS_CHANNEL_SETTINGS_4 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_4
#ifndef CONF_EDGSEL_4
#define CONF_EDGSEL_4 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_4
#ifndef CONF_PATH_4
#define CONF_PATH_4 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_4
#ifndef CONF_EVGEN_4
#define CONF_EVGEN_4 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_4
#ifndef CONF_OVR_4
#define CONF_OVR_4 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_4
#ifndef CONF_EVD_4
#define CONF_EVD_4 0
#endif

// </e>

// <e> Channel 5 settings
// <id> evsys_channel_setting_5
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_5
#define CONF_EVSYS_CHANNEL_SETTINGS_5 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_5
#ifndef CONF_EDGSEL_5
#define CONF_EDGSEL_5 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_5
#ifndef CONF_PATH_5
#define CONF_PATH_5 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_5
#ifndef CONF_EVGEN_5
#define CONF_EVGEN_5 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_5
#ifndef CONF_OVR_5
#define CONF_OVR_5 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_5
#ifndef CONF_EVD_5
#define CONF_EVD_5 0
#endif

// </e>

// <e> Channel 6 settings
// <id> evsys_channel_setting_6
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_6
#define CONF_EVSYS_CHANNEL_SETTINGS_6 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_6
#ifndef CONF_EDGSEL_6
#define CONF_EDGSEL_6 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_6
#ifndef CONF_PATH_6
#define CONF_PATH_6 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_6
#ifndef CONF_EVGEN_6
#define CONF_EVGEN_6 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_6
#ifndef CONF_OVR_6
#define CONF_OVR_6 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_6
#ifndef CONF_EVD_6
#define CONF_EVD_6 0
#endif

// </e>

// <e> Channel 7 settings
// <id> evsys_channel_setting_7
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_7
#define CONF_EVSYS_CHANNEL_SETTINGS_7 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_7
#ifndef CONF_EDGSEL_7
#define CONF_EDGSEL_7 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_7
#ifndef CONF_PATH_7
#define CONF_PATH_7 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_7
#ifndef CONF_EVGEN_7
#define CONF_EVGEN_7 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_7
#ifndef CONF_OVR_7
#define CONF_OVR_7 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_7
#ifndef CONF_EVD_7
#define CONF_EVD_7 0
#endif

// </e>

// <e> Channel 8 settings
// <id> evsys_channel_setting_8
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_8
#define CONF_EVSYS_CHANNEL_SETTINGS_8 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_8
#ifndef CONF_EDGSEL_8
#define CONF_EDGSEL_8 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_8
#ifndef CONF_PATH_8
#define CONF_PATH_8 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_8
#ifndef CONF_EVGEN_8
#define CONF_EVGEN_8 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_8
#ifndef CONF_OVR_8
#define CONF_OVR_8 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_8
#ifndef CONF_EVD_8
#define CONF_EVD_8 0
#endif

// </e>

// <e> Channel 9 settings
// <id> evsys_channel_setting_9
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_9
#define CONF_EVSYS_CHANNEL_SETTINGS_9 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_9
#ifndef CONF_EDGSEL_9
#define CONF_EDGSEL_9 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_9
#ifndef CONF_PATH_9
#define CONF_PATH_9 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_9
#ifndef CONF_EVGEN_9
#define CONF_EVGEN_9 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_9
#ifndef CONF_OVR_9
#define CONF_OVR_9 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_9
#ifndef CONF_EVD_9
#define CONF_EVD_9 0
#endif

// </e>

// <e> Channel 10 settings
// <id> evsys_channel_setting_10
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_10
#define CONF_EVSYS_CHANNEL_SETTINGS_10 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_10
#ifndef CONF_EDGSEL_10
#define CONF_EDGSEL_10 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_10
#ifndef CONF_PATH_10
#define CONF_PATH_10 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_10
#ifndef CONF_EVGEN_10
#define CONF_EVGEN_10 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_10
#ifndef CONF_OVR_10
#define CONF_OVR_10 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_10
#ifndef CONF_EVD_10
#define CONF_EVD_10 0
#endif

// </e>

// <e> Channel 11 settings
// <id> evsys_channel_setting_11
#ifndef CONF_EVSYS_CHANNEL_SETTINGS_11
#define CONF_EVSYS_CHANNEL_SETTINGS_11 0
#endif

// <y> Edge detection
// <i> Indicates whether channel is enabled in standby sleep mode
// <EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val"> No event output when using the resynchronized or synchronous path
// <EVSYS_CHANNEL_EDGSEL_RISING_EDGE_Val"> Event is detected on the rising edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_FALLING_EDGE_Val"> Event is detected on the falling edge of the signal from event generator
// <EVSYS_CHANNEL_EDGSEL_BOTH_EDGES_Val"> Event is detected on the rising and falling edge of the signal from event generator
// <id> evsys_edgsel_11
#ifndef CONF_EDGSEL_11
#define CONF_EDGSEL_11 EVSYS_CHANNEL_EDGSEL_NO_EVT_OUTPUT_Val
#endif

// <y> Path selection
// <i> Indicates which path for the event signal is chosen
// <EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val"> Synchronous path
// <EVSYS_CHANNEL_PATH_RESYNCHRONIZED_Val"> Resynchronized path
// <EVSYS_CHANNEL_PATH_ASYNCHRONOUS_Val"> Asynchronous path
// <id> evsys_path_11
#ifndef CONF_PATH_11
#define CONF_PATH_11 EVSYS_CHANNEL_PATH_SYNCHRONOUS_Val
#endif

// <o> Event generator
// <i> Determines event generator for channel
// <0x0=>No event generator
// <0x1=>RTC compare 0 or alarm 0
// <0x2=>RTC compare 1
// <0x3=>RTC overflow
// <0x4=>RTC period 0
// <0x5=>RTC period 1
// <0x6=>RTC period 2
// <0x7=>RTC period 3
// <0x8=>RTC period 4
// <0x9=>RTC period 5
// <0xA=>RTC period 6
// <0xB=>RTC period 7
// <0xC=>EIC external interrupt 0
// <0xD=>EIC external interrupt 1
// <0xE=>EIC external interrupt 2
// <0xF=>EIC external interrupt 3
// <0x10=>EIC external interrupt 4
// <0x11=>EIC external interrupt 5
// <0x12=>EIC external interrupt 6
// <0x13=>EIC external interrupt 7
// <0x14=>EIC external interrupt 8
// <0x15=>EIC external interrupt 9
// <0x16=>EIC external interrupt 10
// <0x17=>EIC external interrupt 11
// <0x18=>EIC external interrupt 12
// <0x19=>EIC external interrupt 13
// <0x1A=>EIC external interrupt 14
// <0x1B=>EIC external interrupt 15
// <0x1E=>DMAC channel 0
// <0x1F=>DMAC channel 1
// <0x20=>DMAC channel 2
// <0x21=>DMAC channel 3
// <0x22=>TCC0 overflow
// <0x23=>TCC0 trig
// <0x24=>TCC0 counter
// <0x25=>TCC0 match/capture 0
// <0x26=>TCC0 match/capture 1
// <0x27=>TCC0 match/capture 2
// <0x28=>TCC0 match/capture 3
// <0x29=>TCC1 overflow
// <0x2A=>TCC1 trig
// <0x2B=>TCC1 counter
// <0x2C=>TCC1 match/capture 0
// <0x2D=>TCC1 match/capture 1
// <0x2E=>TCC2 overflow
// <0x2F=>TCC2 trig
// <0x30=>TCC2 counter
// <0x31=>TCC2 match/capture 0
// <0x32=>TCC2 match/capture 1
// <0x33=>TC3 overflow
// <0x34=>TC3 match/capture 0
// <0x35=>TC3 match/capture 1
// <0x36=>TC4 overflow
// <0x37=>TC4 match/capture 0
// <0x38=>TC4 match/capture 1
// <0x39=>TC5 overflow
// <0x3A=>TC5 match/capture 0
// <0x3B=>TC5 match/capture 1
// <0x3C=>TC6 overflow
// <0x3D=>TC6 match/capture 0
// <0x3E=>TC6 match/capture 1
// <0x3F=>TC7 overflow
// <0x40=>TC7 match/capture 0
// <0x41=>TC7 match/capture 1
// <0x42=>ADC result ready
// <0x43=>ADC window monitor
// <0x44=>AC comparator 0
// <0x45=>AC comparator 1
// <0x46=>AC window 0
// <0x47=>DAC data buffer empty
// <0x48=>PTC end of conversion
// <0x49=>PTC window comparator
// <id> evsys_evgen_11
#ifndef CONF_EVGEN_11
#define CONF_EVGEN_11 0
#endif

// <q> Overrun channel interrupt
// <i> Indicates whether overrun channel interrupt is enabled or not
// <id> evsys_ovr_11
#ifndef CONF_OVR_11
#define CONF_OVR_11 0
#endif

// <q> Event detected interrupt
// <i> Indicates whether event detected interrupt is enabled or not
// <id> evsys_evd_11
#ifndef CONF_EVD_11
#define CONF_EVD_11 0
#endif

// </e>

// <h>  Event User Security Attribution Settings

//</h>

// <h> DMAC events
// <o> Channel selection for DMAC channel 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_0
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_0
#define CONF_CHANNEL_0 0
#endif

// <o> Channel selection for DMAC channel 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_1
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_1
#define CONF_CHANNEL_1 0
#endif

// <o> Channel selection for DMAC channel 2
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_2
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_2
#define CONF_CHANNEL_2 0
#endif

// <o> Channel selection for DMAC channel 3
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_3
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_3
#define CONF_CHANNEL_3 0
#endif
//</h>

// <h> TCC events
// <o> Channel selection for TCC0 Event 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_4
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_4
#define CONF_CHANNEL_4 0
#endif

// <o> Channel selection for TCC0 Event 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_5
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_5
#define CONF_CHANNEL_5 0
#endif

// <o> Channel selection for TCC0 match/capture 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_6
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_6
#define CONF_CHANNEL_6 0
#endif

// <o> Channel selection for TCC0 match/capture 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_7
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_7
#define CONF_CHANNEL_7 0
#endif

// <o> Channel selection for TCC0 match/capture 2
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_8
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_8
#define CONF_CHANNEL_8 0
#endif

// <o> Channel selection for TCC0 match/capture 3
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_9
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_9
#define CONF_CHANNEL_9 0
#endif

// <o> Channel selection for TCC1 Event 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_10
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_10
#define CONF_CHANNEL_10 0
#endif

// <o> Channel selection for TCC1 Event 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_11
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_11
#define CONF_CHANNEL_11 0
#endif

// <o> Channel selection for TCC1 match/capture 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_12
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_12
#define CONF_CHANNEL_12 0
#endif

// <o> Channel selection for TCC1 match/capture 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_13
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_13
#define CONF_CHANNEL_13 0
#endif

// <o> Channel selection for TCC2 Event 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_14
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_14
#define CONF_CHANNEL_14 0
#endif

// <o> Channel selection for TCC2 Event 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_15
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_15
#define CONF_CHANNEL_15 0
#endif

// <o> Channel selection for TCC2 match/capture 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_16
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_16
#define CONF_CHANNEL_16 0
#endif

// <o> Channel selection for TCC2 match/capture 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_17
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_17
#define CONF_CHANNEL_17 0
#endif
//</h>

// <h> TC events
// <o> Channel selection for TC3 event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_18
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_18
#define CONF_CHANNEL_18 0
#endif

// <o> Channel selection for TC4 event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_19
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_19
#define CONF_CHANNEL_19 0
#endif

// <o> Channel selection for TC5 event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_20
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_20
#define CONF_CHANNEL_20 0
#endif

// <o> Channel selection for TC6 event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_21
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_21
#define CONF_CHANNEL_21 0
#endif

// <o> Channel selection for TC7 event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_22
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_22
#define CONF_CHANNEL_22 0
#endif
//</h>

// <h> ADC events
// <o> Channel selection for ADC start conversion event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_23
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_23
#define CONF_CHANNEL_23 1
#endif

// <o> Channel selection for ADC flush event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_24
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_24
#define CONF_CHANNEL_24 0
#endif
//</h>

// <h> AC events
// <o> Channel selection for start comparator event 0
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_25
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_25
#define CONF_CHANNEL_25 0
#endif

// <o> Channel selection for start comparator event 1
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_26
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_26
#define CONF_CHANNEL_26 0
#endif
//</h>

// <h> DAC events
// <o> Channel selection for DAC start conversion event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_27
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_27
#define CONF_CHANNEL_27 1
#endif
//</h>

// <h> PTC events
// <o> Channel selection for PTC start conversion event
// <0x0=>No channel output selected
// <0x1=>Channel 0
// <0x2=>Channel 1
// <0x3=>Channel 2
// <0x4=>Channel 3
// <0x5=>Channel 4
// <0x6=>Channel 5
// <0x7=>Channel 6
// <0x8=>Channel 7
// <0x9=>Channel 8
// <0xA=>Channel 9
// <0xB=>Channel 10
// <0xC=>Channel 11
// <id> evsys_channel_28
// <i> Indicates which channel is chosen for user
#ifndef CONF_CHANNEL_28
#define CONF_CHANNEL_28 0
#endif
//</h>

// <<< end of configuration section >>>

#endif // HPL_EVSYS_CONFIG_H
