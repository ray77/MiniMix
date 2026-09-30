/* MDCT against the direct O(N^2) definition and TDAC perfect reconstruction. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mdct.h"

#define N 2048
#define M (N / 2)

int main(void)
{
    MMXMdct t;
    float x[N], out[M], y[N];
    double maxerr = 0.0, sig[N * 4], rec[N * 5];
    unsigned long n, k, f;

    if (mmx_mdct_init(&t, N) != 0) { printf("init failed\n"); return 1; }

    srand(7);
    for (n = 0; n < N; n++)
        x[n] = (float)((double)rand() / RAND_MAX - 0.5);
    mmx_mdct_forward(&t, x, out);
    for (k = 0; k < M; k += 37) /* spot check */
    {
        double ref = 0.0, e;
        for (n = 0; n < N; n++)
            ref += x[n] * t.window[n] * cos(2.0 * 3.14159265358979323846 / N * (n + 0.5 + N / 4.0) * (k + 0.5));
        e = fabs(ref - out[k]);
        if (e > maxerr) maxerr = e;
    }
    printf("mdct: max error vs direct definition %.3g\n", maxerr);
    if (maxerr > 1e-3) return 1;

    for (n = 0; n < N * 4; n++) sig[n] = (double)rand() / RAND_MAX - 0.5;
    for (n = 0; n < N * 5; n++) rec[n] = 0.0;
    for (f = 0; f + N <= N * 4; f += M)
    {
        for (n = 0; n < N; n++) x[n] = (float)sig[f + n];
        mmx_mdct_forward(&t, x, out);
        mmx_mdct_inverse(&t, out, y);
        for (n = 0; n < N; n++) rec[f + n] += y[n];
    }
    maxerr = 0.0;
    for (n = M; n < N * 4 - M; n++)
    {
        double e = fabs(rec[n] - sig[n]);
        if (e > maxerr) maxerr = e;
    }
    printf("mdct: TDAC reconstruction max error %.3g\n", maxerr);
    if (maxerr > 1e-5) return 1;

    /* block switching: LONG START SHORT(x8) STOP LONG must still reconstruct perfectly */
    {
        MMXMdct st;
        double wstart[N], wstop[N];
        float sx[256], sout[128], sy[256];
        unsigned long off, k, f2;
        const int seq[6] = {0, 1, 2, 3, 0, 0}; /* 0 long, 1 start, 2 short, 3 stop */
        if (mmx_mdct_init(&st, 256) != 0) { printf("short init failed\n"); return 1; }
        mmx_mdct_window_start(&t, &st, wstart);
        mmx_mdct_window_stop(&t, &st, wstop);
        off = mmx_mdct_short_offset(&t, &st);
        for (n = 0; n < N * 5; n++) rec[n] = 0.0;
        for (f2 = 0; f2 < 6; f2++)
        {
            f = f2 * M;
            for (n = 0; n < N; n++) x[n] = (float)sig[f + n];
            if (seq[f2] == 2)
            {
                for (k = 0; k < 8; k++)
                {
                    for (n = 0; n < 256; n++) sx[n] = x[off + k * 128 + n];
                    mmx_mdct_forward(&st, sx, sout);
                    mmx_mdct_inverse(&st, sout, sy);
                    for (n = 0; n < 256; n++) rec[f + off + k * 128 + n] += sy[n];
                }
            }
            else
            {
                const double *w = seq[f2] == 1 ? wstart : seq[f2] == 3 ? wstop : t.window;
                mmx_mdct_forward_w(&t, x, w, out);
                mmx_mdct_inverse_w(&t, out, w, y);
                for (n = 0; n < N; n++) rec[f + n] += y[n];
            }
        }
        maxerr = 0.0;
        for (n = M; n < 5 * M; n++)
        {
            double e = fabs(rec[n] - sig[n]);
            if (e > maxerr) maxerr = e;
        }
        printf("mdct: block switching (long start 8xshort stop long) reconstruction max error %.3g\n", maxerr);
        if (maxerr > 1e-5) return 1;
        mmx_mdct_free(&st);
    }

    mmx_mdct_free(&t);
    printf("test_mdct OK\n");
    return 0;
}
