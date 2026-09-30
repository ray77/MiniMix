/* Range coder round trip: adaptive bits, bypass bits, signed/unsigned EG values. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rangecoder.h"

#define COUNT 200000

int main(void)
{
    MMXRangeEncoder e;
    MMXRangeDecoder d;
    MMXProb ctx[4][32];
    unsigned int *bits = malloc(sizeof(unsigned int) * COUNT);
    long long *vals = malloc(sizeof(long long) * COUNT);
    unsigned long i, ones = 0;
    int c;

    srand(11);
    for (i = 0; i < COUNT; i++)
    {
        bits[i] = (rand() % 100) < 8; /* skewed source */
        vals[i] = (rand() % 7 == 0) ? (rand() % 4000) - 2000 : (rand() % 5) - 2;
        ones += bits[i];
    }
    for (c = 0; c < 4; c++)
        for (i = 0; i < 32; i++)
            ctx[c][i] = MMX_PROB_INIT;

    mmx_rc_enc_init(&e);
    for (i = 0; i < COUNT; i++)
    {
        mmx_rc_enc_bit(&e, &ctx[0][0], bits[i]);
        mmx_rc_enc_bypass(&e, (unsigned long)(i & 7), 3);
        mmx_rc_enc_seg(&e, ctx[1], vals[i]);
        mmx_rc_enc_ueg(&e, ctx[2], (unsigned long)(i % 300));
    }
    mmx_rc_enc_finish(&e);
    printf("rangecoder: %lu symbols -> %lu bytes (skewed bit p=%.3f, ideal %.0f bytes for the bits alone)\n",
           (unsigned long)COUNT, e.size, (double)ones / COUNT,
           COUNT * (-(0.08 * (-3.64)) + 0.92 * 0.1203) / 8.0);

    for (c = 0; c < 4; c++)
        for (i = 0; i < 32; i++)
            ctx[c][i] = MMX_PROB_INIT;
    mmx_rc_dec_init(&d, e.data, e.size);
    for (i = 0; i < COUNT; i++)
    {
        if (mmx_rc_dec_bit(&d, &ctx[0][0]) != bits[i]) { printf("bit mismatch at %lu\n", i); return 1; }
        if (mmx_rc_dec_bypass(&d, 3) != (i & 7)) { printf("bypass mismatch at %lu\n", i); return 1; }
        if (mmx_rc_dec_seg(&d, ctx[1]) != vals[i]) { printf("seg mismatch at %lu\n", i); return 1; }
        if (mmx_rc_dec_ueg(&d, ctx[2]) != i % 300) { printf("ueg mismatch at %lu\n", i); return 1; }
    }
    if (d.failed) { printf("decoder read past end\n"); return 1; }

    mmx_rc_enc_free(&e);
    free(bits); free(vals);
    printf("test_rangecoder OK\n");
    return 0;
}
