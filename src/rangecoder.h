#ifndef MMX_RANGECODER_H
#define MMX_RANGECODER_H

/* Adaptive binary range coder (LZMA style, 11-bit probabilities). All entropy
   coding in the MiniMix native codec goes through this. */

typedef unsigned short MMXProb;
#define MMX_PROB_INIT 1024
#define MMX_PROB_BITS 11

typedef struct
{
    unsigned char *data;
    unsigned long size;
    unsigned long capacity;
    unsigned long long low;
    unsigned long range;
    unsigned char cache;
    unsigned long long cache_size;
    int failed;
} MMXRangeEncoder;

typedef struct
{
    const unsigned char *data;
    unsigned long size;
    unsigned long pos;
    unsigned long range;
    unsigned long code;
    int failed;
} MMXRangeDecoder;

void mmx_rc_enc_init(MMXRangeEncoder *e);
void mmx_rc_enc_free(MMXRangeEncoder *e);
void mmx_rc_enc_bit(MMXRangeEncoder *e, MMXProb *p, unsigned int bit);
void mmx_rc_enc_bypass(MMXRangeEncoder *e, unsigned long value, unsigned int bits);
/* Finishes the stream. Data stays owned by the encoder until free. */
void mmx_rc_enc_finish(MMXRangeEncoder *e);
/* Approximate bytes produced so far (for cost decisions during encoding). */
unsigned long mmx_rc_enc_bytes(const MMXRangeEncoder *e);
/* Exact position in the arithmetic code in bits (bytes out including the
   pending cache run, plus the part of the 32-bit range already consumed);
   the difference of two positions is the cost of the symbols in between. */
double mmx_rc_enc_bits(const MMXRangeEncoder *e);

void mmx_rc_dec_init(MMXRangeDecoder *d, const unsigned char *data, unsigned long size);
unsigned int mmx_rc_dec_bit(MMXRangeDecoder *d, MMXProb *p);
unsigned long mmx_rc_dec_bypass(MMXRangeDecoder *d, unsigned int bits);

/* -------------------------------------------------------------- counters

   Second, finer probability representation for the lossless residual coder.
   The 11-bit MMXProb with its fixed 1/32 update cannot get closer to 0 or 1
   than 1/64 (p stops moving once p >> 5 == 0), which costs 0.023 bit on every
   near-certain bit; the unary exponent of a residual spends about nine of
   those per sample. MMXCtr keeps a 15-bit probability and a symbol count, and
   updates with the rate 1/(n + 2) until n reaches MMX_CTR_CAP - fast while the
   context is new, then slow enough to represent a probability of 1/32768.
   Measured on the residual stream of four tracks: -0.7 to -0.8 % alone.
   The decoder cost is one multiply and one increment per bit. */

#define MMX_CTR_BITS 15
#define MMX_CTR_ONE (1U << MMX_CTR_BITS)
#define MMX_CTR_CAP 224                 /* final adaptation rate 1/226 */

typedef struct
{
    unsigned short p;                   /* P(bit = 0), 15 bit, 1 .. 32767 */
    unsigned short n;                   /* symbols seen, capped at MMX_CTR_CAP */
} MMXCtr;

/* Sets p to 1/2, n to 0, and builds the reciprocal table on first use. */
void mmx_ctr_init(MMXCtr *c);
void mmx_ctr_init_array(MMXCtr *c, unsigned long n);
void mmx_rc_enc_ctr(MMXRangeEncoder *e, MMXCtr *c, unsigned int bit);
unsigned int mmx_rc_dec_ctr(MMXRangeDecoder *d, MMXCtr *c);
/* explicit 15-bit probability of a one, 1..32767 */
void mmx_rc_enc_p15(MMXRangeEncoder *e, unsigned int p1, unsigned int bit);
unsigned int mmx_rc_dec_p15(MMXRangeDecoder *d, unsigned int p1);

/* Adaptive Exp-Golomb style coding of unsigned values with a context array of
   `nctx` prefix probabilities (nctx >= 20). */
void mmx_rc_enc_ueg(MMXRangeEncoder *e, MMXProb *ctx, unsigned long value);
unsigned long mmx_rc_dec_ueg(MMXRangeDecoder *d, MMXProb *ctx);

/* Signed variant: zero flag + sign + magnitude. ctx needs 2 + 20 entries. */
void mmx_rc_enc_seg(MMXRangeEncoder *e, MMXProb *ctx, long long value);
long long mmx_rc_dec_seg(MMXRangeDecoder *d, MMXProb *ctx);

#endif
