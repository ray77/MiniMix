#include <string.h>
#include <math.h>
#include "tns.h"

#define TNS_LEVELS 16

/* reflection coefficient of a 4-bit index: k = sin(index * pi / TNS_LEVELS)-like
   companding around zero, symmetric, |k| <= 0.96 */
static double coef_value(unsigned char index)
{
    double u = ((double)index - 7.5) / 7.5;   /* -1 .. 1 */
    return 0.96 * sin(u * 3.14159265358979323846 / 2.0);
}

static unsigned char coef_index(double k)
{
    unsigned char best = 0;
    double bd = 1e9;
    unsigned int i;
    for (i = 0; i < TNS_LEVELS; i++)
    {
        double d = fabs(coef_value((unsigned char)i) - k);
        if (d < bd) { bd = d; best = (unsigned char)i; }
    }
    return best;
}

/* reflection coefficients -> direct form a[1..order] (a[0] = 1) */
static void step_up(const MMXTns *t, double *a)
{
    double tmp[MMX_TNS_MAX_ORDER + 1];
    unsigned int m, i;
    a[0] = 1.0;
    for (m = 1; m <= t->order; m++)
    {
        double k = coef_value(t->coef[m - 1]);
        for (i = 1; i < m; i++)
            tmp[i] = a[i] + k * a[m - i];
        for (i = 1; i < m; i++)
            a[i] = tmp[i];
        a[m] = k;
    }
}

double mmx_tns_analyze(const float *x, unsigned long k0, unsigned long k1, unsigned int max_order,
                       double min_gain, MMXTns *t)
{
    double r[MMX_TNS_MAX_ORDER + 1], a[MMX_TNS_MAX_ORDER + 1], tmp[MMX_TNS_MAX_ORDER + 1], err, gain = 1.0;
    unsigned int m, i, order = max_order > MMX_TNS_MAX_ORDER ? MMX_TNS_MAX_ORDER : max_order;
    unsigned long k, n = k1 > k0 ? k1 - k0 : 0;

    memset(t, 0, sizeof(*t));
    if (n < 64 || order == 0)
        return 1.0;
    for (m = 0; m <= order; m++)
    {
        double s = 0.0;
        for (k = k0 + m; k < k1; k++)
            s += (double)x[k] * x[k - m];
        r[m] = s;
    }
    if (r[0] <= 1e-20)
        return 1.0;
    r[0] *= 1.0 + 1e-6;   /* white-noise correction */

    /* Levinson-Durbin with reflection coefficients quantized on the fly so the
       transmitted filter is exactly the one whose gain we measure */
    err = r[0];
    a[0] = 1.0;
    for (m = 1; m <= order; m++)
    {
        double acc = r[m], k, e_new;
        for (i = 1; i < m; i++)
            acc += a[i] * r[m - i];
        k = -acc / err;
        if (k > 0.99) k = 0.99;
        if (k < -0.99) k = -0.99;
        t->coef[m - 1] = coef_index(k);
        k = coef_value(t->coef[m - 1]);
        for (i = 1; i < m; i++)
            tmp[i] = a[i] + k * a[m - i];
        for (i = 1; i < m; i++)
            a[i] = tmp[i];
        a[m] = k;
        e_new = err * (1.0 - k * k);
        if (e_new <= 1e-20 || e_new > err)
            break;
        err = e_new;
        t->order = m;
        gain = r[0] / err;
    }
    /* drop trailing near-zero coefficients (cost bits, no gain) */
    while (t->order > 0 && (t->coef[t->order - 1] == 7 || t->coef[t->order - 1] == 8))
        t->order--;
    t->active = t->order > 0 && gain >= min_gain;
    if (!t->active)
    {
        t->order = 0;
        return gain;
    }
    return gain;
}

void mmx_tns_filter(const MMXTns *t, float *x, unsigned long k0, unsigned long k1)
{
    double a[MMX_TNS_MAX_ORDER + 1], hist[MMX_TNS_MAX_ORDER];
    unsigned long k;
    unsigned int i;
    if (!t->active || t->order == 0)
        return;
    step_up(t, a);
    memset(hist, 0, sizeof(hist));
    for (k = k0; k < k1; k++)
    {
        double in = x[k], out = in;
        for (i = 0; i < t->order; i++)
            out += a[i + 1] * hist[i];
        for (i = t->order; i-- > 1;)
            hist[i] = hist[i - 1];
        hist[0] = in;
        x[k] = (float)out;
    }
}

void mmx_tns_inverse(const MMXTns *t, float *x, unsigned long k0, unsigned long k1)
{
    double a[MMX_TNS_MAX_ORDER + 1], hist[MMX_TNS_MAX_ORDER];
    unsigned long k;
    unsigned int i;
    if (!t->active || t->order == 0)
        return;
    step_up(t, a);
    memset(hist, 0, sizeof(hist));
    for (k = k0; k < k1; k++)
    {
        double out = x[k];
        for (i = 0; i < t->order; i++)
            out -= a[i + 1] * hist[i];
        for (i = t->order; i-- > 1;)
            hist[i] = hist[i - 1];
        hist[0] = out;
        x[k] = (float)out;
    }
}

void mmx_tns_time_envelope(const MMXTns *t, unsigned long n, double *env)
{
    double a[MMX_TNS_MAX_ORDER + 1];
    unsigned long i, m = n / 2;
    unsigned int k;
    if (!t->active || t->order == 0)
    {
        for (i = 0; i < n; i++) env[i] = 1.0;
        return;
    }
    step_up(t, a);
    for (i = 0; i < n; i++)
    {
        double theta = 3.14159265358979323846 * ((double)i + 0.5 + (double)m / 2.0) / (double)m;
        double re = 1.0, im = 0.0;
        for (k = 1; k <= t->order; k++)
        {
            re += a[k] * cos(theta * k);
            im -= a[k] * sin(theta * k);
        }
        env[i] = 1.0 / (re * re + im * im + 1e-12);
    }
}

double mmx_tns_noise_gain(const MMXTns *t)
{
    double a[MMX_TNS_MAX_ORDER + 1], hist[MMX_TNS_MAX_ORDER], energy = 0.0;
    unsigned int i, n;
    if (!t->active || t->order == 0)
        return 1.0;
    step_up(t, a);
    memset(hist, 0, sizeof(hist));
    for (n = 0; n < 256; n++)
    {
        double out = n == 0 ? 1.0 : 0.0;
        for (i = 0; i < t->order; i++)
            out -= a[i + 1] * hist[i];
        for (i = t->order; i-- > 1;)
            hist[i] = hist[i - 1];
        hist[0] = out;
        energy += out * out;
    }
    return energy < 1.0 ? 1.0 : energy;
}
