#include "Arduino.h"
#include "DvG_SerialCommand.h"
#include "Streaming.h"

#define Ser Serial
#define W 10   // String format width
char *str_cmd; // Incoming serial command string
DvG_SerialCommand sc(Ser);

/*------------------------------------------------------------------------------
  Multiplicative modulo algorithms `(a * b) mod n`.
  Safe against overflow of `a * b`.

  Results on SAMD51, Adafruit Feather M4 Express @ 120 MHz
    Fixed `a` and `N`, varying `b`.

    mulmod_float()
            small `b`: 28.3 us
            large `b`: 65.9 us

    mulmod_int()
            small `b`:  1.6 us
            large `b`:  4.0 us

    mulmod_int_pow2()
            small `b`:  1.2 us
            large `b`:  3.1 us
------------------------------------------------------------------------------*/

double mulmod_float(double a, uint32_t b, uint16_t n) {
  // Works on floats and returns fractional results. Very, very slow.
  double sum = 0.;
  a = fmod(a, n); // Strangely, this line speeds up the code though `a < n` ?!
  while (b) {
    if (b & 1)
      sum = fmod((sum + a), n);
    a = fmod((a * 2), n);
    b = b >> 1; // b /= 2;
  }

  return fmod(sum, n);
}

uint32_t mulmod_int(uint16_t a, uint32_t b, uint16_t n) {
  // Integers only, but no constraint on `n`.
  uint32_t sum = 0;
  while (b) {
    if (b & 1)
      sum = (sum + a) % n;
    a = (a << 1) % n;
    b = b >> 1;
  }
  return sum;
}

uint32_t mulmod_int_pow2(uint16_t a, uint32_t b, uint16_t n) {
  // Integers only and constraint on `n` being a power of 2.
  uint32_t sum = 0;
  while (b) {
    if (b & 1)
      sum = (sum + a) & (n - 1); // sum = (sum + a) % n;
    a = (a << 1) & (n - 1);      // a = (a << 1) % n;
    b = b >> 1;
  }
  return sum;
}

/*------------------------------------------------------------------------------
  setup
------------------------------------------------------------------------------*/

double freq_ref = 250.1;            // [Hz]
double DAQ_period = 50;             // [us]
double DAQ_rate = 1e6 / DAQ_period; // [Hz]
uint16_t N_LUT = 20000;

// Derived
double idx_per_iter = N_LUT / DAQ_rate * freq_ref;
uint16_t ideal_idx_per_iter = (uint16_t)round(idx_per_iter);
double ideal_freq_ref = DAQ_rate * ideal_idx_per_iter / N_LUT;

// Results
uint32_t r_int;
double r_dbl;

void wait_for_enter() {
  while (!sc.available()) {} // Wait for user to press enter
  str_cmd = sc.getCmd();     // Flush command
}

void set_title(const char *title) {
  Ser << _PAD(40, '-') << endl;
  Ser << title << endl;
  Ser.flush();
}

void setup() {
  Ser.begin(1e6);
  wait_for_enter();
}

/*------------------------------------------------------------------------------
  loop
------------------------------------------------------------------------------*/

void loop() {
  uint16_t N_iters = 3000;
  uint32_t iter;
  uint32_t iter_offset = 1 << 31;
  uint32_t tick;

  Ser << _PAD(40, '-') << endl;
  Ser << "  DAQ_period   " << _FLOATW(DAQ_period, 3, W) << "  us" << endl
      << "  DAQ_rate     " << _FLOATW(DAQ_rate, 3, W) << "  Hz" << endl
      << "  N_LUT        " << _WIDTH(N_LUT, W) << endl
      << endl
      << "  idx_per_iter " << _FLOATW(idx_per_iter, 3, W) << endl
      << "     ideal     " << _WIDTH(ideal_idx_per_iter, W) << endl
      << "  freq_ref     " << _FLOATW(freq_ref, 3, W) << "  Hz" << endl
      << "     ideal     " << _FLOATW(ideal_freq_ref, 3, W) << "  Hz" << endl;
  Ser << _PAD(40, '-') << endl;

  // ----------
  //  Config 1
  // ----------
  set_title("mulmod_float");

  tick = micros();
  for (iter = 0; iter < N_iters; iter++) {
    r_dbl = mulmod_float(idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per small iter"
      << ", iter " << iter - 1 << " = " << r_dbl << endl;

  tick = micros();
  for (iter = iter_offset; iter < iter_offset + N_iters; iter++) {
    r_dbl = mulmod_float(idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per large iter"
      << ", iter " << iter - 1 << " = " << r_dbl << endl;

  wait_for_enter();

  // ----------
  //  Config 2
  // ----------
  set_title("mulmod_int");

  tick = micros();
  for (iter = 0; iter < N_iters; iter++) {
    r_int = mulmod_int(ideal_idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per small iter"
      << ", iter " << iter - 1 << " = " << r_int << endl;

  tick = micros();
  for (iter = iter_offset; iter < iter_offset + N_iters; iter++) {
    r_int = mulmod_int(ideal_idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per large iter"
      << ", iter " << iter - 1 << " = " << r_int << endl;

  wait_for_enter();

  // ----------
  //  Config 3
  // ----------
  set_title("mulmod_int_pow2");

  tick = micros();
  for (iter = 0; iter < N_iters; iter++) {
    r_int = mulmod_int_pow2(ideal_idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per small iter"
      << ", iter " << iter - 1 << " = " << r_int << endl;

  tick = micros();
  for (iter = iter_offset; iter < iter_offset + N_iters; iter++) {
    r_int = mulmod_int_pow2(ideal_idx_per_iter, iter, N_LUT);
  }
  Ser << _FLOAT((double)(micros() - tick) / N_iters, 1) << " us per large iter"
      << ", iter " << iter - 1 << " = " << r_int << endl;

  wait_for_enter();
}