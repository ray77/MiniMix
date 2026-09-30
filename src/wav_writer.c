#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wav_writer.h"
#include "log.h"

static void wr_u32(unsigned char *p, unsigned long v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

static void wr_u16(unsigned char *p, unsigned int v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static long long clamp_round(double v, double lo, double hi)
{
    double r = floor(v + 0.5);
    if (r < lo) r = lo;
    if (r > hi) r = hi;
    return (long long)r;
}

int mmx_wav_write(const char *path, const MMXAudioBuffer *buf, unsigned int bits)
{
    FILE *fp;
    unsigned char hdr[44];
    unsigned int bytes_per_sample;
    unsigned long block_align, data_size;
    unsigned char *raw;
    unsigned long long i;
    unsigned int c;

    if (bits != 16 && bits != 24 && bits != 32)
        bits = 16;
    bytes_per_sample = bits / 8;
    block_align = (unsigned long)buf->channels * bytes_per_sample;
    data_size = (unsigned long)(buf->frame_count * block_align);

    fp = fopen(path, "wb");
    if (!fp)
    {
        mmx_error("Cannot create WAV file: %s", path);
        return -1;
    }

    memcpy(hdr, "RIFF", 4);
    wr_u32(hdr + 4, 36 + data_size);
    memcpy(hdr + 8, "WAVE", 4);
    memcpy(hdr + 12, "fmt ", 4);
    wr_u32(hdr + 16, 16);
    wr_u16(hdr + 20, 1);
    wr_u16(hdr + 22, buf->channels);
    wr_u32(hdr + 24, buf->sample_rate);
    wr_u32(hdr + 28, buf->sample_rate * block_align);
    wr_u16(hdr + 32, (unsigned int)block_align);
    wr_u16(hdr + 34, bits);
    memcpy(hdr + 36, "data", 4);
    wr_u32(hdr + 40, data_size);

    if (fwrite(hdr, 1, 44, fp) != 44)
    {
        fclose(fp);
        return -1;
    }

    raw = (unsigned char *)malloc(data_size ? data_size : 1);
    if (!raw)
    {
        fclose(fp);
        return -1;
    }

    for (i = 0; i < buf->frame_count; i++)
    {
        const float *src = buf->samples + (size_t)i * buf->channels;
        unsigned char *dst = raw + (size_t)i * block_align;
        for (c = 0; c < buf->channels; c++)
        {
            long long v;
            unsigned char *p = dst + c * bytes_per_sample;
            switch (bits)
            {
            case 16:
                v = clamp_round((double)src[c] * 32768.0, -32768.0, 32767.0);
                wr_u16(p, (unsigned int)(v & 0xFFFF));
                break;
            case 24:
                v = clamp_round((double)src[c] * 8388608.0, -8388608.0, 8388607.0);
                p[0] = (unsigned char)(v & 0xFF);
                p[1] = (unsigned char)((v >> 8) & 0xFF);
                p[2] = (unsigned char)((v >> 16) & 0xFF);
                break;
            default:
                v = buf->isamples ? (long long)buf->isamples[(size_t)i * buf->channels + c]      /* exact 32-bit source */
                                  : clamp_round((double)src[c] * 2147483648.0, -2147483648.0, 2147483647.0);
                wr_u32(p, (unsigned long)(v & 0xFFFFFFFFLL));
                break;
            }
        }
    }

    if (fwrite(raw, 1, data_size, fp) != data_size)
    {
        free(raw);
        fclose(fp);
        return -1;
    }
    free(raw);
    fclose(fp);
    return 0;
}
