/*
Dennis van Gils
23-07-2019

Adapted from:
https://www.mathworks.com/matlabcentral/fileexchange/10858-ecg-simulation-using-matlab
*/

#ifndef H_DvG_ECG_simulation
#define H_DvG_ECG_simulation

/*
This library can operate in two modes depending on the available resources.

1) SLOW CALCULATION BUT USE LESS RAM
    When memory is limited, choose this by defining the preprocessor directive:
    #define DvG_ECG__CALC_SLOWER_BUT_USE_LESS_RAM

    The used algorithm will iterate towards convergence, one point of the
    waveform array at a time, without the use of an extra array defined
    internally. Hence, the C-compiler is limited in optimizing the algorithm.

2) FAST CALCULATION BUT USE MORE RAM
    Only use this when enough memory is available.
    Choose this by commenting out the preprocessor directive:
    //#define DvG_ECG__CALC_SLOWER_BUT_USE_LESS_RAM

    The used algorithm will iterate towards convergence, operating on all points
    of the waveform array at the same time, using an extra array that is defined
    internally. Hence, the C-compiler is able to drastically optimize the
    algorithm, speeding it up by a factor of ~2.4.

    You must make sure that ECG_MAX_N_LUT, defined inside the source file, is
    larger or equal to N_SMP as passed into 'generate_ECG()'.

*/
#define ECG_CALC_SLOWER_BUT_USE_LESS_RAM

// Number of iterations for convergence of the ECG waveform. Set to:
//  10   poor resemblance.
//  16   for coarse resemblance, noticeable ringing in intermediate parts.
//  36   for well recognizable as ECG with little ringing in intermediate parts.
//  100  for good ECG with flat intermediate parts.
#define ECG_N_ITER 16

#include <stdint.h>

void generate_ECG(float *ecg,            // Array to output ECG waveform to
                  const uint16_t N_SMP); // No. samples for one full period

#endif