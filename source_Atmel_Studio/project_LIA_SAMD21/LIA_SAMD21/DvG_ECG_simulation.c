/*
Dennis van Gils
23-07-2019

Adapted from:
https://www.mathworks.com/matlabcentral/fileexchange/10858-ecg-simulation-using-matlab
*/

#include "DvG_ECG_simulation.h"
#include <utils_assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ECG_params {
    double a_p, d_p, t_p;
    double a_q, d_q, t_q;
    double a_qrs, d_qrs, t_qrs;
    double a_s, d_s, t_s;
    double a_t, d_t, t_t;
    double a_u, d_u, t_u;
} ECG_params;

typedef struct ECG_derived {
    double x_shift_p, x_shift_q, x_shift_qrs, x_shift_s, x_shift_t, x_shift_u;
    double b_p, b_q, b_qrs, b_s, b_t, b_u;
    double M_PI_2b_p, M_PI_2b_q, M_PI_2b_qrs, M_PI_2b_s, M_PI_2b_t, M_PI_2b_u;
    double D_p, D_q, D_qrs, D_s, D_t, D_u;
    double p1_p, p1_q, p1_qrs, p1_s, p1_t, p1_u;
} ECG_derived;

struct ECG_params  ecg_prms;
struct ECG_derived ecg_drvd;
double M_TWOPI_NSMP;

static void init_ECG_params(void);
static void derive_ECG_params(const ECG_params  *prms,
                                    ECG_derived *drvd);

#ifdef ECG_CALC_SLOWER_BUT_USE_LESS_RAM
    static double iter_fun_1(uint16_t idx,
                             double b,
                             double x_shift,
                             double M_PI_2b);
    static double iter_fun_2(uint16_t idx,
                             double b,
                             double x_shift,
                             double D,
                             double d);
    static double iterate_ECG_wave(const uint16_t idx,
                                   const uint16_t N_SMP);
#else
    #define ECG_MAX_N_LUT 500
    double p2[ECG_MAX_N_LUT] = {0.};

    typedef enum ECG_wave_part {
        ECG_P,
        ECG_Q,
        ECG_QRS,
        ECG_S,
        ECG_T,
        ECG_U
    } ECG_wave_part;

    static void add_ECG_wave_part(ECG_wave_part ecg_part,
                                  float *wave,
                                  const uint16_t N_SMP);
#endif

/*------------------------------------------------------------------------------
    ECG_params
------------------------------------------------------------------------------*/

static void init_ECG_params(void) {
    // Standard ECG parameters
    ecg_prms.a_p = 0.25;
    ecg_prms.d_p = 0.09;
    ecg_prms.t_p = 0.16;

    ecg_prms.a_q = 0.025;
    ecg_prms.d_q = 0.066;
    ecg_prms.t_q = 0.166;

    ecg_prms.a_qrs = 1.6;
    ecg_prms.d_qrs = 0.11;
    ecg_prms.t_qrs = 0.;

    ecg_prms.a_s = 0.25;
    ecg_prms.d_s = 0.066;
    ecg_prms.t_s = 0.09;

    ecg_prms.a_t = 0.35;
    ecg_prms.d_t = 0.142;
    ecg_prms.t_t = 0.2;

    ecg_prms.a_u = 0.035;
    ecg_prms.d_u = 0.0476;
    ecg_prms.t_u = 0.433;

    derive_ECG_params(&ecg_prms, &ecg_drvd);
}

static void derive_ECG_params(const ECG_params  *prms,
                                    ECG_derived *drvd) {
    drvd->x_shift_p   =  M_TWOPI * prms->t_p;
    drvd->x_shift_q   =  M_TWOPI * prms->t_q;
    drvd->x_shift_qrs =  M_TWOPI * prms->t_qrs;
    drvd->x_shift_s   = -M_TWOPI * prms->t_s;
    drvd->x_shift_t   =  M_TWOPI * (prms->t_t - 0.045);
    drvd->x_shift_u   = -M_TWOPI * prms->t_u;

    drvd->b_p   = 1. / prms->d_p;
    drvd->b_q   = 1. / prms->d_q;
    drvd->b_qrs = 1. / prms->d_qrs;
    drvd->b_s   = 1. / prms->d_s;
    drvd->b_t   = 1. / prms->d_t;
    drvd->b_u   = 1. / prms->d_u;

    drvd->M_PI_2b_p   = M_PI_2 * prms->d_p;
    drvd->M_PI_2b_q   = M_PI_2 * prms->d_q;
    drvd->M_PI_2b_qrs = M_PI_2 * prms->d_qrs;
    drvd->M_PI_2b_s   = M_PI_2 * prms->d_s;
    drvd->M_PI_2b_t   = M_PI_2 * prms->d_t;
    drvd->M_PI_2b_u   = M_PI_2 * prms->d_u;

    drvd->D_p   = 2. * prms->a_p   / prms->d_p   / (M_PI * M_PI);
    drvd->D_q   = 2. * prms->a_q   / prms->d_q   / (M_PI * M_PI);
    drvd->D_qrs = 2. * prms->a_qrs / prms->d_qrs / (M_PI * M_PI);
    drvd->D_s   = 2. * prms->a_s   / prms->d_s   / (M_PI * M_PI);
    drvd->D_t   = 2. * prms->a_t   / prms->d_t   / (M_PI * M_PI);
    drvd->D_u   = 2. * prms->a_u   / prms->d_u   / (M_PI * M_PI);

    drvd->p1_p   = 2.;
    drvd->p1_q   = prms->a_q   / 2. * drvd->b_q   * (2. - drvd->b_q);
    drvd->p1_qrs = prms->a_qrs / 2. * drvd->b_qrs * (2. - drvd->b_qrs);
    drvd->p1_s   = prms->a_s   / 2. * drvd->b_s   * (2. - drvd->b_s);
    drvd->p1_t   = 2.;
    drvd->p1_u   = 2.;
}

#ifdef ECG_CALC_SLOWER_BUT_USE_LESS_RAM

/*------------------------------------------------------------------------------
    generate_ECG
------------------------------------------------------------------------------*/

void generate_ECG(float *ecg,
                  const uint16_t N_SMP) {
    init_ECG_params();
    M_TWOPI_NSMP = M_TWOPI / N_SMP;

    // Iterate ECG waveform
    uint16_t i;
    for (i = 0; i < N_SMP; i++) {
        ecg[i] = (float) iterate_ECG_wave(i, N_SMP);

        //sprintf(str_buffer, "\r  %3i%%", i * 100 / (N_SMP - 1));
        //io_print(str_buffer);
    }

    // Normalize ECG output to [0 ... 1]
    float ecg_min = 0.;
    float ecg_max = 0.;
    for (i = 0; i < N_SMP; i++) {
        if (i == 0) {
            ecg_min = ecg[i];
            ecg_max = ecg[i];
        } else {
            if (ecg_min > ecg[i]) {ecg_min = ecg[i];}
            if (ecg_max < ecg[i]) {ecg_max = ecg[i];}
        }
    }

    for (i = 0; i < N_SMP; i++) {
        ecg[i] = (ecg[i] - ecg_min) / (ecg_max - ecg_min);
    }
}

/*------------------------------------------------------------------------------
    iter_fun
------------------------------------------------------------------------------*/

static double iter_fun_1(uint16_t idx,
                         double b,
                         double x_shift,
                         double M_PI_2b) {
    double p2 = 0.;
    double C;

    for (uint16_t j = 1; j <= ECG_N_ITER; j++) {
        C = M_2_PI * (sin(M_PI_2b * (b - 2. * j)) / (b - 2. * j) +
                      sin(M_PI_2b * (b + 2. * j)) / (b + 2. * j));
        p2 += C * cos((M_TWOPI_NSMP * idx + x_shift) * j);
    }

    return p2;
}

static double iter_fun_2(uint16_t idx,
                         double b,
                         double x_shift,
                         double D,
                         double d) {
    double p2 = 0.;
    double C;

    for (uint16_t j = 1; j <= ECG_N_ITER; j++) {
        C = D / (j * j) * (1. - cos(M_PI * d * j));
        p2 += C * cos((M_TWOPI_NSMP * idx + x_shift) * j);
    }

    return p2;
}

/*------------------------------------------------------------------------------
    iterate_ECG_wave
------------------------------------------------------------------------------*/

static double iterate_ECG_wave(const uint16_t idx,
                               const uint16_t N_SMP) {
    double wave;

    wave = ecg_prms.a_p * (ecg_drvd.p1_p + iter_fun_1(idx,
                                                      ecg_drvd.b_p,
                                                      ecg_drvd.x_shift_p,
                                                      ecg_drvd.M_PI_2b_p))
         + ecg_prms.a_t * (ecg_drvd.p1_t + iter_fun_1(idx,
                                                      ecg_drvd.b_t,
                                                      ecg_drvd.x_shift_t,
                                                      ecg_drvd.M_PI_2b_t))
         + ecg_prms.a_u * (ecg_drvd.p1_u + iter_fun_1(idx,
                                                      ecg_drvd.b_u,
                                                      ecg_drvd.x_shift_u,
                                                      ecg_drvd.M_PI_2b_u))
         - ecg_drvd.p1_q   - iter_fun_2(idx,
                                        ecg_drvd.b_q,
                                        ecg_drvd.x_shift_q,
                                        ecg_drvd.D_q,
                                        ecg_prms.d_q)
         + ecg_drvd.p1_qrs + iter_fun_2(idx,
                                        ecg_drvd.b_qrs,
                                        ecg_drvd.x_shift_qrs,
                                        ecg_drvd.D_qrs,
                                        ecg_prms.d_qrs)
         - ecg_drvd.p1_s   - iter_fun_2(idx,
                                        ecg_drvd.b_s,
                                        ecg_drvd.x_shift_s,
                                        ecg_drvd.D_s,
                                        ecg_prms.d_s);

    return wave;
}

#else

/*------------------------------------------------------------------------------
    generate_ECG
------------------------------------------------------------------------------*/

void generate_ECG(float *ecg,
                  const uint16_t N_SMP) {
    ASSERT(ECG_MAX_N_LUT >= N_SMP);

    init_ECG_params();
    M_TWOPI_NSMP = M_TWOPI / N_SMP;

    // Clear ECG waveform before adding separate ECG parts
    uint16_t i;
    for (i = 0; i < N_SMP; i++) {ecg[i] = 0.;}

    // Iterate ECG waveform
    add_ECG_wave_part(ECG_P  , ecg, N_SMP);
    add_ECG_wave_part(ECG_Q  , ecg, N_SMP);
    add_ECG_wave_part(ECG_QRS, ecg, N_SMP);
    add_ECG_wave_part(ECG_S  , ecg, N_SMP);
    add_ECG_wave_part(ECG_T  , ecg, N_SMP);
    add_ECG_wave_part(ECG_U  , ecg, N_SMP);

    // Normalize ECG output to [0 ... 1]
    float ecg_min = 0.;
    float ecg_max = 0.;
    for (i = 0; i < N_SMP; i++) {
        if (i == 0) {
            ecg_min = ecg[i];
            ecg_max = ecg[i];
        } else {
            if (ecg_min > ecg[i]) {ecg_min = ecg[i];}
            if (ecg_max < ecg[i]) {ecg_max = ecg[i];}
        }
    }

    for (i = 0; i < N_SMP; i++) {
        ecg[i] = (ecg[i] - ecg_min) / (ecg_max - ecg_min);
    }
}

/*------------------------------------------------------------------------------
    add_ECG_wave_part
------------------------------------------------------------------------------*/

static void add_ECG_wave_part(ECG_wave_part ecg_part,
                              float *wave,            // Waveform array to add new part to
                              const uint16_t N_SMP) { // No. samples for one full period
    uint16_t i; // Iterator
    uint16_t j; // Iterator
    double a, d, x_shift, b, M_PI_2b, D, p1;

    switch (ecg_part) {
        default:
        case ECG_P:
            a       = ecg_prms.a_p;
            d       = ecg_prms.d_p;
            x_shift = ecg_drvd.x_shift_p;
            b       = ecg_drvd.b_p;
            M_PI_2b = ecg_drvd.M_PI_2b_p;
            D       = ecg_drvd.D_p;
            p1      = ecg_drvd.p1_p;
            break;
        case ECG_Q:
            a       = ecg_prms.a_q;
            d       = ecg_prms.d_q;
            x_shift = ecg_drvd.x_shift_q;
            b       = ecg_drvd.b_q;
            M_PI_2b = ecg_drvd.M_PI_2b_q;
            D       = ecg_drvd.D_q;
            p1      = ecg_drvd.p1_q;
            break;
        case ECG_QRS:
            a       = ecg_prms.a_qrs;
            d       = ecg_prms.d_qrs;
            x_shift = ecg_drvd.x_shift_qrs;
            b       = ecg_drvd.b_qrs;
            M_PI_2b = ecg_drvd.M_PI_2b_qrs;
            D       = ecg_drvd.D_qrs;
            p1      = ecg_drvd.p1_qrs;
            break;
        case ECG_S:
            a       = ecg_prms.a_s;
            d       = ecg_prms.d_s;
            x_shift = ecg_drvd.x_shift_s;
            b       = ecg_drvd.b_s;
            M_PI_2b = ecg_drvd.M_PI_2b_s;
            D       = ecg_drvd.D_s;
            p1      = ecg_drvd.p1_s;
            break;
        case ECG_T:
            a       = ecg_prms.a_t;
            d       = ecg_prms.d_t;
            x_shift = ecg_drvd.x_shift_t;
            b       = ecg_drvd.b_t;
            M_PI_2b = ecg_drvd.M_PI_2b_t;
            D       = ecg_drvd.D_t;
            p1      = ecg_drvd.p1_t;
            break;
        case ECG_U:
            a       = ecg_prms.a_u;
            d       = ecg_prms.d_u;
            x_shift = ecg_drvd.x_shift_u;
            b       = ecg_drvd.b_u;
            M_PI_2b = ecg_drvd.M_PI_2b_u;
            D       = ecg_drvd.D_u;
            p1      = ecg_drvd.p1_u;
            break;
    }

    // Clear array for upcoming iterations
    for (i = 0; i < N_SMP; i++) {p2[i] = 0.;}

    double C;
    switch (ecg_part) {
        case ECG_P:
        case ECG_T:
        case ECG_U:
            for (j = 1; j <= ECG_N_ITER; j++) {
                C = M_2_PI * (sin(M_PI_2b * (b - 2. * j)) / (b - 2. * j) +
                              sin(M_PI_2b * (b + 2. * j)) / (b + 2. * j));

                for (i = 0; i < N_SMP; i++) {
                    p2[i] += C * cos((M_TWOPI_NSMP * i + x_shift) * j);
                }
            }
            break;

        case ECG_Q:
        case ECG_QRS:
        case ECG_S:
            for (j = 1; j <= ECG_N_ITER; j++) {
                C = D / (j * j) * (1. - cos(M_PI * d * j));

                for (i = 0; i < N_SMP; i++) {
                    p2[i] += C * cos((M_TWOPI_NSMP * i + x_shift) * j);
                }
            }
            break;
    }

    switch (ecg_part) {
        case ECG_P:
        case ECG_T:
        case ECG_U:
            for (i = 0; i < N_SMP; i++) {wave[i] += a * (p1 + p2[i]);}
            break;

        case ECG_Q:
            for (i = 0; i < N_SMP; i++) {wave[i] += -p1 - p2[i];}
            break;

        case ECG_QRS:
            for (i = 0; i < N_SMP; i++) {wave[i] += p1 + p2[i];}
            break;

        case ECG_S:
            for (i = 0; i < N_SMP; i++) {wave[i] += -p1 - p2[i];}
            break;
    }
}

#endif

/*------------------------------------------------------------------------------
    Copy of source:
    https://www.mathworks.com/matlabcentral/fileexchange/10858-ecg-simulation-using-matlab
------------------------------------------------------------------------------*/

/*
function [pwav]=p_wav(x,a_pwav,d_pwav,t_pwav,li)
    l=li;
    a=a_pwav;
    x=x+t_pwav;
    b=(2*l)/d_pwav;
    n=100;
    p1=1/l;
    p2=0;
    for i = 1:n
        harm1=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/(b+(2*i)))*(2/pi))*cos((i*pi*x)/l);
        p2=p2+harm1;
    end
    pwav1=p1+p2;
    pwav=a*pwav1;
*/
/*
function [twav]=t_wav(x,a_twav,d_twav,t_twav,li)
    l=li;
    a=a_twav;
    x=x-t_twav-0.045;
    b=(2*l)/d_twav;
    n=100;
    t1=1/l;
    t2=0;
    for i = 1:n
        harm2=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/(b+(2*i)))*(2/pi))*cos((i*pi*x)/l);
        t2=t2+harm2;
    end
    twav1=t1+t2;
    twav=a*twav1;
*/
/*
function [uwav]=u_wav(x,a_uwav,d_uwav,t_uwav,li)
    l=li;
    a=a_uwav
    x=x-t_uwav;
    b=(2*l)/d_uwav;
    n=100;
    u1=1/l
    u2=0
    for i = 1:n
        harm4=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/(b+(2*i)))*(2/pi))*cos((i*pi*x)/l);
        u2=u2+harm4;
    end
    uwav1=u1+u2;
    uwav=a*uwav1;
*/
/*
function [qrswav]=qrs_wav(x,a_qrswav,d_qrswav,li)
    l=li;
    a=a_qrswav;
    b=(2*l)/d_qrswav;
    n=100;
    qrs1=(a/(2*b))*(2-b);
    qrs2=0;
    for i = 1:n
        harm=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
        qrs2=qrs2+harm;
    end
    qrswav=qrs1+qrs2;
*/
/*
function [qwav]=q_wav(x,a_qwav,d_qwav,t_qwav,li)
    l=li;
    x=x+t_qwav;
    a=a_qwav;
    b=(2*l)/d_qwav;
    n=100;
    q1=(a/(2*b))*(2-b);
    q2=0;
    for i = 1:n
        harm5=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
        q2=q2+harm5;
    end
    qwav=-1*(q1+q2);
*/
/*
function [swav]=s_wav(x,a_swav,d_swav,t_swav,li)
    l=li;
    x=x-t_swav;
    a=a_swav;
    b=(2*l)/d_swav;
    n=100;
    s1=(a/(2*b))*(2-b);
    s2=0;
    for i = 1:n
        harm3=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
        s2=s2+harm3;
    end
    swav=-1*(s1+s2);
*/
/*
x=0.01:0.01:2;
default=input('Press 1 if u want default ecg signal else press 2:\n');
if(default==1)
li=30/72;

a_pwav=0.25;
d_pwav=0.09;
t_pwav=0.16;

a_qwav=0.025;
d_qwav=0.066;
t_qwav=0.166;

a_qrswav=1.6;
d_qrswav=0.11;

a_swav=0.25;
d_swav=0.066;
t_swav=0.09;

a_twav=0.35;
d_twav=0.142;
t_twav=0.2;

a_uwav=0.035;
d_uwav=0.0476;
t_uwav=0.433;
else
rate=input('\n\nenter the heart beat rate :');
li=30/rate;

%p wave specifications
fprintf('\n\np wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_pwav=0.25;
d_pwav=0.09;
t_pwav=0.16;
else
a_pwav=input('amplitude = ');
d_pwav=input('duration = ');
t_pwav=input('p-r interval = ');
d=0;
end


%q wave specifications
fprintf('\n\nq wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_qwav=0.025;
d_qwav=0.066;
t_qwav=0.166;
else
a_qwav=input('amplitude = ');
d_qwav=input('duration = ');
t_qwav=0.166;
d=0;
end



%qrs wave specifications
fprintf('\n\nqrs wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_qrswav=1.6;
d_qrswav=0.11;
else
a_qrswav=input('amplitude = ');
d_qrswav=input('duration = ');
d=0;
end



%s wave specifications
fprintf('\n\ns wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_swav=0.25;
d_swav=0.066;
t_swav=0.09;
else
a_swav=input('amplitude = ');
d_swav=input('duration = ');
t_swav=0.09;
d=0;
end


%t wave specifications
fprintf('\n\nt wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_twav=0.35;
d_twav=0.142;
t_twav=0.2;
else
a_twav=input('amplitude = ');
d_twav=input('duration = ');
t_twav=input('s-t interval = ');
d=0;
end


%u wave specifications
fprintf('\n\nu wave specifications\n');
d=input('Enter 1 for default specification else press 2: \n');
if(d==1)
a_uwav=0.035;
d_uwav=0.0476;
t_uwav=0.433;
else
a_uwav=input('amplitude = ');
d_uwav=input('duration = ');
t_uwav=0.433;
d=0;
end



end
pwav=p_wav(x,a_pwav,d_pwav,t_pwav,li);

%qwav output
qwav=q_wav(x,a_qwav,d_qwav,t_qwav,li);

%qrswav output
qrswav=qrs_wav(x,a_qrswav,d_qrswav,li);
%swav output
swav=s_wav(x,a_swav,d_swav,t_swav,li);

%twav output
twav=t_wav(x,a_twav,d_twav,t_twav,li);

%uwav output
uwav=u_wav(x,a_uwav,d_uwav,t_uwav,li);
%ecg output
ecg=pwav+qrswav+twav+swav+qwav+uwav;
figure(1)
plot(x,ecg);
*/