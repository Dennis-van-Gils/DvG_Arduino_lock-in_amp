#include "Arduino.h"

enum tc_clock_prescaler {
	/** Divide clock by 1 */
	TC_CLOCK_PRESCALER_DIV1             = TC_CTRLA_PRESCALER(0),
	/** Divide clock by 2 */
	TC_CLOCK_PRESCALER_DIV2             = TC_CTRLA_PRESCALER(1),
	/** Divide clock by 4 */
	TC_CLOCK_PRESCALER_DIV4             = TC_CTRLA_PRESCALER(2),
	/** Divide clock by 8 */
	TC_CLOCK_PRESCALER_DIV8             = TC_CTRLA_PRESCALER(3),
	/** Divide clock by 16 */
	TC_CLOCK_PRESCALER_DIV16            = TC_CTRLA_PRESCALER(4),
	/** Divide clock by 64 */
	TC_CLOCK_PRESCALER_DIV64            = TC_CTRLA_PRESCALER(5),
	/** Divide clock by 256 */
	TC_CLOCK_PRESCALER_DIV256           = TC_CTRLA_PRESCALER(6),
	/** Divide clock by 1024 */
	TC_CLOCK_PRESCALER_DIV1024          = TC_CTRLA_PRESCALER(7),
};

enum tc_counter_size {
	/** The counter's maximum value is 0xFF, the period register is
	 * available to be used as top value
	 */
	TC_COUNTER_SIZE_8BIT                = TC_CTRLA_MODE_COUNT8,

	/** The counter's maximum value is 0xFFFF. There is no separate
	 * period register, to modify top one of the capture compare
	 * registers has to be used. This limits the amount of
	 * available channels.
	 */
	TC_COUNTER_SIZE_16BIT               = TC_CTRLA_MODE_COUNT16,

	/** The counter's maximum value is 0xFFFFFFFF. There is no separate
	 * period register, to modify top one of the capture compare
	 * registers has to be used. This limits the amount of
	 * available channels.
	 */
	TC_COUNTER_SIZE_32BIT               = TC_CTRLA_MODE_COUNT32,
};

#if defined(__SAMD51__)
  enum tc_wave_generation {
    TC_WAVE_GENERATION_NORMAL_FREQ      = TC_WAVE_WAVEGEN_NFRQ,
    TC_WAVE_GENERATION_MATCH_FREQ       = TC_WAVE_WAVEGEN_MFRQ,
    TC_WAVE_GENERATION_NORMAL_PWM       = TC_WAVE_WAVEGEN_NPWM,
    TC_WAVE_GENERATION_MATCH_PWM        = TC_WAVE_WAVEGEN_MPWM,
  };
#else
enum tc_wave_generation {
	/** Top is maximum, except in 8-bit counter size where it is the PER
	 * register
	 */
	TC_WAVE_GENERATION_NORMAL_FREQ      = TC_CTRLA_WAVEGEN_NFRQ,

	/** Top is CC0, except in 8-bit counter size where it is the PER
	 * register
	 */
	TC_WAVE_GENERATION_MATCH_FREQ       = TC_CTRLA_WAVEGEN_MFRQ,

	/** Top is maximum, except in 8-bit counter size where it is the PER
	 * register
	 */
	TC_WAVE_GENERATION_NORMAL_PWM       = TC_CTRLA_WAVEGEN_NPWM,

	/** Top is CC0, except in 8-bit counter size where it is the PER
	 * register
	 */
	TC_WAVE_GENERATION_MATCH_PWM        = TC_CTRLA_WAVEGEN_MPWM,
};
#endif

enum tc_count_direction {
  TC_COUNT_DIRECTION_UP = 0,
  TC_COUNT_DIRECTION_DOWN
};

enum tc_callback {
  TC_CALLBACK_OVERFLOW = 0,
  TC_CALLBACK_ERROR, 
  TC_CALLBACK_CC_CHANNEL0,
  TC_CALLBACK_CC_CHANNEL1,
  TC_CALLBACK_N
};

#define NUM_PWM_CHANNELS 2
#define NUM_CC_CHANNELS 2

class Adafruit_ZeroTimer {
 public:
  Adafruit_ZeroTimer(uint8_t tn);

  boolean PWMout(boolean pwmout, uint8_t channum, uint8_t pin);
  void setPeriodMatch(uint32_t period, uint32_t match, uint8_t channum = 1);
  void enable(boolean en);

  void configure(tc_clock_prescaler prescale, tc_counter_size countersize, tc_wave_generation wavegen, tc_count_direction countdir = TC_COUNT_DIRECTION_UP);
  void setCompare(uint8_t channum, uint32_t compare);
  void setTop(uint32_t top);
  void invertWave(uint8_t invert);
  void setCallback(boolean enable, tc_callback cb_type, void(* callback_func) (void) = NULL);

  static void timerHandler(uint8_t timerNum);
 protected:
  uint8_t _timernum;

  Tc *_hw;

  tc_clock_prescaler _clock_prescaler;
  tc_counter_size _counter_size;
  tc_wave_generation _wave_generation;
  uint8_t _waveform_invert_output;
  tc_count_direction _count_direction;

  struct counter_8_bit {
    uint8_t compare_capture_channel[NUM_CC_CHANNELS];
    uint8_t period;
    uint8_t value;
  };
  counter_8_bit _counter_8_bit;

  struct counter_16_bit {
    uint16_t compare_capture_channel[NUM_CC_CHANNELS];
    uint16_t value;
  };
  counter_16_bit _counter_16_bit;

  struct counter_32_bit {
    uint32_t compare_capture_channel[NUM_CC_CHANNELS];
    uint32_t value;
  };
  counter_32_bit _counter_32_bit;

  struct pwm_channel {
    bool enabled;
    uint32_t pin_mux;
    uint32_t pin_out;
  };
  pwm_channel _pwm_channel[NUM_PWM_CHANNELS];

  bool tc_init();

};
