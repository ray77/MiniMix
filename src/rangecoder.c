#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rangecoder.h"

#define TOP (1UL << 24)
#define MOVE_BITS 5

static void enc_put(MMXRangeEncoder *e, unsigned char b)
{
    if (e->failed)
        return;
    if (e->size == e->capacity)
    {
        unsigned long nc = e->capacity ? e->capacity * 2 : 4096;
        unsigned char *g = (unsigned char *)realloc(e->data, nc);
        if (!g)
        {
            e->failed = 1;
            return;
        }
        e->data = g;
        e->capacity = nc;
    }
    e->data[e->size++] = b;
}

static void enc_shift_low(MMXRangeEncoder *e)
{
    if ((unsigned long)(e->low & 0xFFFFFFFFUL) < 0xFF000000UL || (e->low >> 32) != 0)
    {
        unsigned char carry = (unsigned char)(e->low >> 32);
        unsigned char temp = e->cache;
        do
        {
            enc_put(e, (unsigned char)(temp + carry));
            temp = 0xFF;
        } while (--e->cache_size != 0);
        e->cache = (unsigned char)((e->low >> 24) & 0xFF);
    }
    e->cache_size++;
    e->low = (e->low & 0x00FFFFFFULL) << 8;
}

void mmx_rc_enc_init(MMXRangeEncoder *e)
{
    memset(e, 0, sizeof(*e));
    e->range = 0xFFFFFFFFUL;
    e->cache_size = 1;
}

void mmx_rc_enc_free(MMXRangeEncoder *e)
{
    if (!e)
        return;
    free(e->data);
    memset(e, 0, sizeof(*e));
}

void mmx_rc_enc_bit(MMXRangeEncoder *e, MMXProb *p, unsigned int bit)
{
    unsigned long bound = (e->range >> MMX_PROB_BITS) * (*p);
    if (bit == 0)
    {
        e->range = bound;
        *p = (MMXProb)(*p + (((1U << MMX_PROB_BITS) - *p) >> MOVE_BITS));
    }
    else
    {
        e->low += bound;
        e->range -= bound;
        *p = (MMXProb)(*p - (*p >> MOVE_BITS));
    }
    while (e->range < TOP)
    {
        e->range <<= 8;
        enc_shift_low(e);
    }
}

void mmx_rc_enc_bypass(MMXRangeEncoder *e, unsigned long value, unsigned int bits)
{
    while (bits-- > 0)
    {
        e->range >>= 1;
        if ((value >> bits) & 1UL)
            e->low += e->range;
        while (e->range < TOP)
        {
            e->range <<= 8;
            enc_shift_low(e);
        }
    }
}

void mmx_rc_enc_finish(MMXRangeEncoder *e)
{
    int i;
    for (i = 0; i < 5; i++)
        enc_shift_low(e);
}

unsigned long mmx_rc_enc_bytes(const MMXRangeEncoder *e)
{
    return e->size + (unsigned long)e->cache_size;
}

static unsigned char dec_get(MMXRangeDecoder *d)
{
    if (d->pos < d->size)
        return d->data[d->pos++];
    d->failed = 1;
    return 0;
}

void mmx_rc_dec_init(MMXRangeDecoder *d, const unsigned char *data, unsigned long size)
{
    int i;
    memset(d, 0, sizeof(*d));
    d->data = data;
    d->size = size;
    d->range = 0xFFFFFFFFUL;
    for (i = 0; i < 5; i++)
        d->code = ((d->code << 8) | dec_get(d)) & 0xFFFFFFFFUL;
}

unsigned int mmx_rc_dec_bit(MMXRangeDecoder *d, MMXProb *p)
{
    unsigned long bound = (d->range >> MMX_PROB_BITS) * (*p);
    unsigned int bit;
    if (d->code < bound)
    {
        d->range = bound;
        *p = (MMXProb)(*p + (((1U << MMX_PROB_BITS) - *p) >> MOVE_BITS));
        bit = 0;
    }
    else
    {
        d->code -= bound;
        d->range -= bound;
        *p = (MMXProb)(*p - (*p >> MOVE_BITS));
        bit = 1;
    }
    while (d->range < TOP)
    {
        d->range <<= 8;
        d->code = ((d->code << 8) | dec_get(d)) & 0xFFFFFFFFUL;
    }
    return bit;
}

unsigned long mmx_rc_dec_bypass(MMXRangeDecoder *d, unsigned int bits)
{
    unsigned long v = 0;
    while (bits-- > 0)
    {
        d->range >>= 1;
        v <<= 1;
        if (d->code >= d->range)
        {
            d->code -= d->range;
            v |= 1;
        }
        while (d->range < TOP)
        {
            d->range <<= 8;
            d->code = ((d->code << 8) | dec_get(d)) & 0xFFFFFFFFUL;
        }
    }
    return v;
}

/* -------------------------------------------------------------- counters */

static unsigned short ctr_recip[MMX_CTR_CAP + 1];   /* 65536 / (n + 2) */
static int ctr_ready = 0;

static void ctr_tables(void)
{
    unsigned int i;
    if (ctr_ready)
        return;
    for (i = 0; i <= MMX_CTR_CAP; i++)
        ctr_recip[i] = (unsigned short)(65536U / (i + 2U));
    ctr_ready = 1;
}

void mmx_ctr_init(MMXCtr *c)
{
    ctr_tables();
    c->p = (unsigned short)(MMX_CTR_ONE >> 1);
    c->n = 0;
}

void mmx_ctr_init_array(MMXCtr *c, unsigned long n)
{
    unsigned long i;
    ctr_tables();
    for (i = 0; i < n; i++)
    {
        c[i].p = (unsigned short)(MMX_CTR_ONE >> 1);
        c[i].n = 0;
    }
}

/* p moves 1/(n+2) of the way to the observed bit; n stops at MMX_CTR_CAP.
   The rounding is symmetric so that a long run of zeros and a long run of
   ones converge to the same distance from the edge. */
static void ctr_update(MMXCtr *c, unsigned int bit)
{
    int target = bit ? 0 : (int)MMX_CTR_ONE;
    int d = target - (int)c->p;
    int p = (int)c->p + (((d * (int)ctr_recip[c->n]) + 32768) >> 16);
    if (p < 1)
        p = 1;
    if (p > (int)MMX_CTR_ONE - 1)
        p = (int)MMX_CTR_ONE - 1;
    c->p = (unsigned short)p;
    if (c->n < MMX_CTR_CAP)
        c->n++;
}

void mmx_rc_enc_ctr(MMXRangeEncoder *e, MMXCtr *c, unsigned int bit)
{
    unsigned long bound = (e->range >> MMX_CTR_BITS) * c->p;
    if (bit == 0)
        e->range = bound;
    else
    {
        e->low += bound;
        e->range -= bound;
    }
    ctr_update(c, bit);
    while (e->range < TOP)
    {
        e->range <<= 8;
        enc_shift_low(e);
    }
}

/* A bit with an explicit 15-bit probability p1 of a one (1..32767), for models that keep their own state
   (the LL2 bitplane coder). */
void mmx_rc_enc_p15(MMXRangeEncoder *e, unsigned int p1, unsigned int bit)
{
    unsigned long bound = (e->range >> 15) * (32768UL - p1);
    if (bit == 0)
        e->range = bound;
    else
    {
        e->low += bound;
        e->range -= bound;
    }
    while (e->range < TOP)
    {
        e->range <<= 8;
        enc_shift_low(e);
    }
}

unsigned int mmx_rc_dec_p15(MMXRangeDecoder *d, unsigned int p1)
{
    unsigned long bound = (d->range >> 15) * (32768UL - p1);
    unsigned int bit;
    if (d->code < bound)
    {
        d->range = bound;
        bit = 0;
    }
    else
    {
        d->code -= bound;
        d->range -= bound;
        bit = 1;
    }
    while (d->range < TOP)
    {
        d->range <<= 8;
        d->code = ((d->code << 8) | dec_get(d)) & 0xFFFFFFFFUL;
    }
    return bit;
}

unsigned int mmx_rc_dec_ctr(MMXRangeDecoder *d, MMXCtr *c)
{
    unsigned long bound = (d->range >> MMX_CTR_BITS) * c->p;
    unsigned int bit;
    if (d->code < bound)
    {
        d->range = bound;
        bit = 0;
    }
    else
    {
        d->code -= bound;
        d->range -= bound;
        bit = 1;
    }
    ctr_update(c, bit);
    while (d->range < TOP)
    {
        d->range <<= 8;
        d->code = ((d->code << 8) | dec_get(d)) & 0xFFFFFFFFUL;
    }
    return bit;
}

/* Exp-Golomb (k=0) with adaptive prefix bits: value+1 has L bits; prefix = L-1 ones
   (contexts by position) and a zero; suffix = low L-1 bits bypass. */
void mmx_rc_enc_ueg(MMXRangeEncoder *e, MMXProb *ctx, unsigned long value)
{
    unsigned long v = value + 1;
    unsigned int len = 0, i;
    while ((v >> len) > 1)
        len++;
    for (i = 0; i < len; i++)
        mmx_rc_enc_bit(e, &ctx[i < 19 ? i : 19], 1);
    mmx_rc_enc_bit(e, &ctx[len < 19 ? len : 19], 0);
    if (len)
        mmx_rc_enc_bypass(e, v & ((1UL << len) - 1UL), len);
}

unsigned long mmx_rc_dec_ueg(MMXRangeDecoder *d, MMXProb *ctx)
{
    unsigned int len = 0;
    unsigned long v;
    while (len < 31 && mmx_rc_dec_bit(d, &ctx[len < 19 ? len : 19]))
        len++;
    v = (1UL << len);
    if (len)
        v |= mmx_rc_dec_bypass(d, len);
    return v - 1;
}

void mmx_rc_enc_seg(MMXRangeEncoder *e, MMXProb *ctx, long long value)
{
    if (value == 0)
    {
        mmx_rc_enc_bit(e, &ctx[0], 0);
        return;
    }
    mmx_rc_enc_bit(e, &ctx[0], 1);
    mmx_rc_enc_bit(e, &ctx[1], value < 0);
    mmx_rc_enc_ueg(e, ctx + 2, (unsigned long)(value < 0 ? -value : value) - 1);
}

long long mmx_rc_dec_seg(MMXRangeDecoder *d, MMXProb *ctx)
{
    long long mag;
    unsigned int neg;
    if (!mmx_rc_dec_bit(d, &ctx[0]))
        return 0;
    neg = mmx_rc_dec_bit(d, &ctx[1]);
    mag = (long long)mmx_rc_dec_ueg(d, ctx + 2) + 1;
    return neg ? -mag : mag;
}

double mmx_rc_enc_bits(const MMXRangeEncoder *e)
{
    return 8.0 * (double)(e->size + e->cache_size) + 32.0 - log((double)e->range) / log(2.0);
}
