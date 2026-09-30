#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64                /* 64-bit file positions for RF64 / Wave64 beyond 4 GB */
#endif
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L             /* fseeko / ftello */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wav_reader.h"
#include "log.h"

#ifdef _WIN32
#define wav_seek(f, o) _fseeki64((f), (long long)(o), SEEK_SET)
#define wav_tell(f) ((long long)_ftelli64(f))
static long long wav_size(FILE *f) { _fseeki64(f, 0, SEEK_END); return (long long)_ftelli64(f); }
#else
#define wav_seek(f, o) fseeko((f), (off_t)(o), SEEK_SET)
#define wav_tell(f) ((long long)ftello(f))
static long long wav_size(FILE *f) { fseeko(f, 0, SEEK_END); return (long long)ftello(f); }
#endif

static unsigned long long rd_u64(const unsigned char *p)
{
    unsigned long long v = 0;
    int i;
    for (i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

/* Sony Wave64: every chunk id is a GUID; the first four bytes are the RIFF name, the rest is this tail */
static const unsigned char w64_tail[12] = { 0xF3, 0xAC, 0xD3, 0x11, 0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A };
static const unsigned char w64_riff[16] = { 'r', 'i', 'f', 'f', 0x2E, 0x91, 0xCF, 0x11, 0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00 };

static unsigned long rd_u32(const unsigned char *p)
{
    return (unsigned long)p[0] | ((unsigned long)p[1] << 8) |
           ((unsigned long)p[2] << 16) | ((unsigned long)p[3] << 24);
}

static unsigned int rd_u16(const unsigned char *p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8);
}

static float sample_from_bytes(const unsigned char *p, unsigned int bits, unsigned int format)
{
    long long v;
    unsigned long u;
    float f;

    if (format == 3) /* IEEE float */
    {
        u = rd_u32(p);
        memcpy(&f, &u, sizeof(f));
        return f;
    }

    switch (bits)
    {
    case 8:
        return ((float)p[0] - 128.0f) / 128.0f;
    case 16:
        v = (long long)(short)rd_u16(p);
        return (float)v / 32768.0f;
    case 24:
        u = (unsigned long)p[0] | ((unsigned long)p[1] << 8) | ((unsigned long)p[2] << 16);
        if (u & 0x800000UL)
            u |= 0xFF000000UL;
        v = (long long)(int)u;
        return (float)v / 8388608.0f;
    case 32:
        u = rd_u32(p);
        v = (long long)(int)u;
        return (float)((double)v / 2147483648.0);
    }
    return 0.0f;
}

int mmx_wav_read(const char *path, MMXAudioBuffer *out)
{
    FILE *fp;
    unsigned char hdr[12];
    unsigned char chunk[8];
    unsigned int format = 0, channels = 0, bits = 0, block_align = 0;
    unsigned long sample_rate = 0;
    unsigned long long data_size = 0, ds64_data = 0;
    long long data_pos = -1;
    int have_fmt = 0, rf64 = 0, w64 = 0;
    long long file_size;
    unsigned long long frames;
    unsigned char *raw;
    unsigned long long i;
    unsigned int c;

    memset(out, 0, sizeof(*out));

    fp = fopen(path, "rb");
    if (!fp)
    {
        mmx_error("Cannot open WAV file: %s", path);
        return -1;
    }

    file_size = wav_size(fp);
    wav_seek(fp, 0);

    /* RIFF/WAVE, RF64 and BW64 (EBU: 64-bit sizes in a ds64 chunk) or Sony Wave64 (GUID chunks, 64-bit sizes) */
    if (fread(hdr, 1, 12, fp) != 12)
    {
        mmx_error("Not a WAV file: %s", path);
        fclose(fp);
        return -1;
    }
    if (memcmp(hdr, "riff", 4) == 0)
    {
        unsigned char g[40];
        wav_seek(fp, 0);
        if (fread(g, 1, 40, fp) != 40 || memcmp(g, w64_riff, 16) != 0 || memcmp(g + 24, "wave", 4) != 0 ||
            memcmp(g + 28, w64_tail, 12) != 0)
        {
            mmx_error("Not a Wave64 file: %s", path);
            fclose(fp);
            return -1;
        }
        w64 = 1;
    }
    else if ((memcmp(hdr, "RIFF", 4) != 0 && memcmp(hdr, "RF64", 4) != 0 && memcmp(hdr, "BW64", 4) != 0) ||
             memcmp(hdr + 8, "WAVE", 4) != 0)
    {
        mmx_error("Not a RIFF/WAVE file: %s", path);
        fclose(fp);
        return -1;
    }
    else
        rf64 = memcmp(hdr, "RIFF", 4) != 0;

    for (;;)
    {
        unsigned long long size;
        long long here;
        if (w64)
        {   /* GUID (16), size including this 24-byte header (8); chunks are 8-byte aligned */
            unsigned char g[24];
            if (fread(g, 1, 24, fp) != 24) break;
            if (memcmp(g + 4, w64_tail, 12) != 0) break;
            memcpy(chunk, g, 4);
            size = rd_u64(g + 16);
            if (size < 24) break;
            size -= 24;
        }
        else
        {
            if (fread(chunk, 1, 8, fp) != 8) break;
            size = rd_u32(chunk + 4);
        }
        here = wav_tell(fp);

        if (!w64 && memcmp(chunk, "ds64", 4) == 0 && size >= 16)
        {   /* RF64: riff size, data size, sample count (64 bit each) */
            unsigned char d[16];
            if (fread(d, 1, 16, fp) != 16) break;
            ds64_data = rd_u64(d + 8);
            wav_seek(fp, here + (long long)size + (long long)(size & 1));
        }
        else if (memcmp(chunk, "fmt ", 4) == 0)
        {
            unsigned char fmt[40];
            unsigned long want = size < 40 ? (unsigned long)size : 40;
            if (want < 16 || fread(fmt, 1, want, fp) != want)
            {
                mmx_error("Truncated fmt chunk");
                fclose(fp);
                return -1;
            }
            format = rd_u16(fmt);
            channels = rd_u16(fmt + 2);
            sample_rate = rd_u32(fmt + 4);
            block_align = rd_u16(fmt + 12);
            bits = rd_u16(fmt + 14);
            if (format == 0xFFFE && want >= 26) /* WAVE_FORMAT_EXTENSIBLE: sub-format GUID first 2 bytes */
                format = rd_u16(fmt + 24);
            have_fmt = 1;
            wav_seek(fp, w64 ? here + (long long)((size + 7) & ~7ULL) : here + (long long)size + (long long)(size & 1));
        }
        else if (memcmp(chunk, "data", 4) == 0)
        {
            data_pos = here;
            data_size = size;
            if (rf64 && size == 0xFFFFFFFFULL && ds64_data) data_size = ds64_data;
            /* Some writers leave size as 0 or 0xFFFFFFFF for streamed output. */
            if (data_size == 0 || (!w64 && data_size == 0xFFFFFFFFULL) || here + (long long)data_size > file_size)
                data_size = (unsigned long long)(file_size - here);
            break;
        }
        else
            wav_seek(fp, w64 ? here + (long long)((size + 7) & ~7ULL) : here + (long long)size + (long long)(size & 1));
    }

    if (!have_fmt || data_pos < 0)
    {
        mmx_error("WAV file is missing fmt or data chunk");
        fclose(fp);
        return -1;
    }
    if (format != 1 && format != 3)
    {
        mmx_error("Unsupported WAV format tag %u (only PCM and IEEE float)", format);
        fclose(fp);
        return -1;
    }
    if (format == 3 && bits != 32)
    {
        mmx_error("Only 32-bit IEEE float WAV is supported");
        fclose(fp);
        return -1;
    }
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32)
    {
        mmx_error("Unsupported bit depth: %u", bits);
        fclose(fp);
        return -1;
    }
    if (channels == 0 || channels > 32)
    {
        mmx_error("Unsupported channel count: %u", channels);
        fclose(fp);
        return -1;
    }
    if (block_align == 0)
        block_align = channels * (bits / 8);

    frames = data_size / block_align;

    if (mmx_audio_buffer_init(out, sample_rate, (unsigned short)channels, frames) != 0)
    {
        mmx_error("Out of memory allocating audio buffer");
        fclose(fp);
        return -1;
    }
    out->source_bits = (unsigned short)(format == 3 ? 24 : bits);

    if ((unsigned long long)(size_t)data_size != data_size)
    {
        mmx_error("WAV data too large for this build (%llu bytes)", data_size);
        mmx_audio_buffer_free(out);
        fclose(fp);
        return -1;
    }
    raw = (unsigned char *)malloc(data_size ? (size_t)data_size : 1);
    if (!raw)
    {
        mmx_error("Out of memory reading WAV data");
        mmx_audio_buffer_free(out);
        fclose(fp);
        return -1;
    }

    wav_seek(fp, data_pos);
    if (fread(raw, 1, (size_t)data_size, fp) != (size_t)data_size)
    {
        mmx_error("Short read on WAV data");
        free(raw);
        mmx_audio_buffer_free(out);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    /* Whether the float buffer still holds every input sample exactly. Up to 24-bit integers always
       fit the 24-bit significand. A 32-bit integer only fits when its low bits are clear, a float only
       when it sits on the 24-bit grid the lossless coder quantizes to. Measured: a 32-bit
       file with real content in the low byte came back from a "lossless" round trip with 278,260 of
       1,058,400 bytes changed, a float file came back as 24 bit - both silently. Decided here, while
       the integer is still in hand; after the conversion the information is gone. */
    out->exact = 1;
    if (format == 1 && bits == 32 && mmx_audio_buffer_alloc_int(out) != 0)   /* 32-bit integers kept exactly */
    {
        free(raw);
        mmx_audio_buffer_free(out);
        return -1;
    }
    for (i = 0; i < frames; i++)
    {
        const unsigned char *fr = raw + (size_t)i * block_align;
        float *dst = out->samples + (size_t)i * channels;
        for (c = 0; c < channels; c++)
        {
            const unsigned char *p = fr + c * (bits / 8);
            dst[c] = sample_from_bytes(p, bits, format);
            if (out->exact)
            {
                if (format == 3)
                {
                    double g = (double)dst[c] * 8388608.0;
                    if (g != (double)(long long)g || g > 8388607.0 || g < -8388608.0)
                        out->exact = 0;
                }
                else if (bits == 32)
                    out->isamples[(size_t)i * channels + c] = (int)rd_u32(p);   /* exact: the LL2 core codes these */
            }
        }
    }
    if (!out->exact)
        mmx_warning("Input carries more than the 24 bits a lossless encode can keep (%s): a lossless request will be refused",
                 format == 3 ? "32-bit float off the 24-bit grid" : "32-bit integer with bits below the 24-bit significand");

    free(raw);
    mmx_debug("WAV loaded: %lu Hz, %u ch, %u bit, %llu frames", sample_rate, channels, bits, frames);
    return 0;
}
