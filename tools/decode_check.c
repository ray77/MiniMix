/* Decodes an .mmx file with the step decoder from a memory buffer and checks the PCM against a
   reference WAV: a lossless file must match sample for sample, a lossy file within an SNR bound
   (its MDCT math is floating point and differs by a few ulp between compilers). Needs only the
   decoder subset of the core, so it builds wherever the foobar2000 component builds — CI runs it on
   Windows to prove that build decodes right.
     decode_check file.mmx reference.wav exact|snr */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minimix/mmx_reader.h"
#include "minimix/decoder.h"

static unsigned char *slurp(const char *path, size_t *size)
{
    FILE *fp = fopen(path, "rb");
    unsigned char *buf;
    long n;
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); n = ftell(fp); fseek(fp, 0, SEEK_SET);
    buf = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    *size = (size_t)n;
    return buf;
}

static unsigned long u32(const unsigned char *p) { return p[0] | (p[1] << 8) | ((unsigned long)p[2] << 16) | ((unsigned long)p[3] << 24); }

/* 16- or 24-bit PCM RIFF/WAVE: returns the interleaved sample bytes, their count and the bits per sample */
static const unsigned char *wav_samples(const unsigned char *buf, size_t size, unsigned *channels, unsigned long *rate,
                                        size_t *count, unsigned *bits)
{
    size_t pos = 12;
    const unsigned char *data = NULL;
    *bits = 0;
    if (size < 12 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) return NULL;
    *count = 0;
    while (pos + 8 <= size)
    {
        unsigned long len = u32(buf + pos + 4);
        if (memcmp(buf + pos, "fmt ", 4) == 0 && len >= 16)
        {
            *bits = buf[pos + 22] | (buf[pos + 23] << 8);
            if (buf[pos + 8] != 1 || (*bits != 16 && *bits != 24 && *bits != 32)) return NULL;
            *channels = buf[pos + 10] | (buf[pos + 11] << 8);
            *rate = u32(buf + pos + 12);
        }
        else if (memcmp(buf + pos, "data", 4) == 0)
        {
            data = buf + pos + 8;
            *count = (size_t)len;                 /* bytes here; samples below, once the bits are known */
            if (pos + 8 + len > size) *count = size - pos - 8;
        }
        pos += 8 + len + (len & 1);
    }
    if (!*bits) return NULL;
    *count /= *bits / 8;
    return data;
}

static long ref_sample(const unsigned char *d, size_t i, unsigned bits)
{
    if (bits == 16) return (long)(short)(d[2 * i] | (d[2 * i + 1] << 8));
    if (bits == 32)
    {
        unsigned long u = d[4 * i] | (d[4 * i + 1] << 8) | ((unsigned long)d[4 * i + 2] << 16) | ((unsigned long)d[4 * i + 3] << 24);
        return u >= 0x80000000UL ? (long)((long long)u - 0x100000000LL) : (long)u;
    }
    {
        long v = d[3 * i] | (d[3 * i + 1] << 8) | ((long)d[3 * i + 2] << 16);
        return v >= (1L << 23) ? v - (1L << 24) : v;
    }
}

int main(int argc, char **argv)
{
    size_t msize, wsize, count, i, n;
    unsigned char *mbuf, *wbuf;
    const unsigned char *ref;
    unsigned channels, bits;
    unsigned long rate;
    MMXMemorySource mem;
    MMXByteSource src;
    MMXFile f;
    MMXAudioBuffer pcm;
    MMXDecoder *d;
    int rc, exact;
    double sig = 0.0, noise = 0.0, worst = 0.0;
    unsigned long long mismatches = 0;

    if (argc != 4) { fprintf(stderr, "usage: decode_check file.mmx reference.wav exact|snr\n"); return 2; }
    exact = strcmp(argv[3], "exact") == 0;
    mbuf = slurp(argv[1], &msize);
    wbuf = slurp(argv[2], &wsize);
    if (!mbuf || !wbuf) { fprintf(stderr, "cannot read the inputs\n"); return 2; }
    ref = wav_samples(wbuf, wsize, &channels, &rate, &count, &bits);
    if (!ref) { fprintf(stderr, "reference must be 16-, 24- or 32-bit PCM WAV\n"); return 2; }

    mmx_byte_source_memory(&src, &mem, mbuf, msize);
    if (mmx_reader_read_source(&src, argv[1], &f, 1) != 0) { fprintf(stderr, "read failed\n"); return 1; }
    if (f.channels != channels || f.sample_rate != rate) { fprintf(stderr, "format differs from the reference\n"); return 1; }
    if (mmx_decoder_open(&f, &pcm, &d) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    while ((rc = mmx_decoder_step(d, 7)) > 0)            /* odd step sizes cross block boundaries mid-step */
        ;
    if (rc != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    if (mmx_decoder_valid_frames(d) != f.frame_count) { fprintf(stderr, "decoder finished early\n"); return 1; }
    mmx_decoder_close(d);

    n = (size_t)f.frame_count * f.channels;
    if (n != count) { fprintf(stderr, "length differs: %lu vs %lu samples\n", (unsigned long)n, (unsigned long)count); return 1; }
    for (i = 0; i < n; i++)
    {
        const double full = bits == 32 ? 2147483648.0 : bits == 24 ? 8388608.0 : 32768.0;
        const long rv = ref_sample(ref, i, bits);
        double v = pcm.samples[i] * full, r = (double)rv, e;
        long long q = (long long)floor(v + 0.5);
        if (q > (long long)full - 1) q = (long long)full - 1; if (q < -(long long)full) q = -(long long)full;
        if (bits == 32 && pcm.isamples) q = pcm.isamples[i];          /* the float cannot hold 32 bits: the exact ints */
        if (q != rv) mismatches++;
        e = v - r;
        sig += r * r; noise += e * e;
        if (fabs(e) > worst) worst = fabs(e);
    }
    if (exact)
    {
        printf("%s: %llu of %lu samples differ from the reference\n", argv[1], mismatches, (unsigned long)n);
        return mismatches == 0 ? 0 : 1;
    }
    else
    {
        double snr = noise > 0.0 ? 10.0 * log10(sig / noise) : 200.0;
        printf("%s: SNR %.1f dB against the reference decode, worst sample %.3f LSB\n", argv[1], snr, worst);
        return snr >= 60.0 ? 0 : 1;
    }
}
