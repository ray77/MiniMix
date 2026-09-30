/* The resumable decoder and the memory byte source: decoding one hop at a time from a memory
   buffer must give byte for byte what the one-call decoder gives from the file, and the valid
   frame count must only ever grow and never claim a sample that still changes. Lossless and
   lossy files alike. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minimix/encoder.h"
#include "minimix/decoder.h"
#include "minimix/mmx_writer.h"
#include "minimix/mmx_reader.h"
#include "log.h"

#define PI 3.14159265358979323846

static unsigned long rng = 4242;
static double frand(void)
{
    rng = rng * 1103515245UL + 12345UL;
    return ((rng >> 8) & 0xFFFF) / 65536.0 - 0.5;
}

static int check(const char *label, const MMXAudioBuffer *audio, unsigned int quality)
{
    MMXEncoderParams params;
    MMXFile file, from_file, from_memory;
    MMXStatistics stats;
    MMXAudioBuffer whole, stepped;
    MMXDecoder *d;
    MMXByteSource src;
    MMXMemorySource mem;
    const char *path = "bin/test_decoder_steps.mmx";
    unsigned long long written, valid = 0, checked = 0, size;
    unsigned char *buf;
    unsigned long steps = 0, i;
    size_t ch;
    FILE *fp;
    int rc;

    mmx_encoder_params_default(&params);
    params.quality = quality;
    params.analysis = 4;
    mmx_file_init(&file);
    if (mmx_encoder_encode(audio, &params, &file, &stats, NULL, NULL) != 0) { printf("%s: encode failed\n", label); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("%s: write failed\n", label); return 1; }
    mmx_file_free(&file);

    /* the reference: file -> one call */
    if (mmx_reader_read(path, &from_file, 1) != 0) { printf("%s: read failed\n", label); return 1; }
    if (mmx_decoder_decode(&from_file, &whole) != 0) { printf("%s: decode failed\n", label); return 1; }

    /* the candidate: memory buffer -> one hop at a time */
    fp = fopen(path, "rb");
    if (!fp) { printf("%s: cannot reopen\n", label); return 1; }
    fseek(fp, 0, SEEK_END); size = (unsigned long long)ftell(fp); fseek(fp, 0, SEEK_SET);
    buf = (unsigned char *)malloc((size_t)size);
    if (!buf || fread(buf, 1, (size_t)size, fp) != size) { printf("%s: cannot slurp\n", label); return 1; }
    fclose(fp);
    mmx_byte_source_memory(&src, &mem, buf, (size_t)size);
    if (mmx_reader_read_source(&src, "memory", &from_memory, 1) != 0) { printf("%s: memory read failed\n", label); return 1; }
    if (from_memory.block_count != from_file.block_count || from_memory.metadata_len != from_file.metadata_len)
    { printf("%s: memory read differs from file read\n", label); return 1; }

    if (mmx_decoder_open(&from_memory, &stepped, &d) != 0) { printf("%s: open failed\n", label); return 1; }
    ch = from_memory.channels;
    while ((rc = mmx_decoder_step(d, 1)) > 0)
    {
        unsigned long long v = mmx_decoder_valid_frames(d);
        steps++;
        if (v < valid) { printf("%s: valid frames went backwards (%llu -> %llu)\n", label, valid, v); return 1; }
        valid = v;
        /* everything declared final must already equal the finished decode */
        for (i = (unsigned long)checked * ch; i < (unsigned long)valid * ch; i++)
            if (stepped.samples[i] != whole.samples[i])
            {
                printf("%s: sample %lu differs after step %lu while declared final (valid %llu)\n", label, i, steps, valid);
                return 1;
            }
        checked = valid;
    }
    if (rc != 0) { printf("%s: step failed\n", label); return 1; }
    valid = mmx_decoder_valid_frames(d);
    mmx_decoder_close(d);
    if (valid != from_memory.frame_count) { printf("%s: finished with %llu of %llu frames valid\n", label, valid, from_memory.frame_count); return 1; }
    if (memcmp(stepped.samples, whole.samples, sizeof(float) * (size_t)whole.frame_count * ch) != 0)
    { printf("%s: stepped decode differs from the one-call decode\n", label); return 1; }
    printf("%s: %lu blocks, %lu steps of one hop, %llu frames, memory source == file, stepped == whole, valid never early\n",
           label, from_memory.block_count, steps, valid);

    mmx_audio_buffer_free(&whole);
    mmx_audio_buffer_free(&stepped);
    mmx_file_free(&from_file);
    mmx_file_free(&from_memory);
    free(buf);
    return 0;
}

int main(void)
{
    MMXAudioBuffer audio;
    unsigned long bar = 44100, bars = 6, total = bar * bars, i, b;
    float *bar_a;

    mmx_log_set_level(MMX_LOG_WARNING);
    mmx_audio_buffer_init(&audio, 44100, 2, total);
    audio.source_bits = 16;
    bar_a = malloc(sizeof(float) * bar * 2);
    for (i = 0; i < bar; i++)
    {
        double t = (double)i / 44100.0, env = exp(-3.0 * fmod(t, 0.5));
        double s = env * (0.5 * sin(2 * PI * 110 * t) + 0.2 * sin(2 * PI * 331 * t)) + 0.02 * frand();
        bar_a[2 * i] = (float)s;
        bar_a[2 * i + 1] = (float)(0.7 * s + 0.01 * frand());
    }
    /* A A B A A' A  (B = noise, A' = A at -6 dB): references and fresh audio both occur */
    for (b = 0; b < bars; b++)
    {
        float *dst = audio.samples + (size_t)b * bar * 2;
        if (b == 2)
            for (i = 0; i < bar * 2; i++) dst[i] = (float)(0.3 * frand());
        else if (b == 4)
            for (i = 0; i < bar * 2; i++) dst[i] = bar_a[i] * 0.5f;
        else
            memcpy(dst, bar_a, sizeof(float) * bar * 2);
    }
    for (i = 0; i < total * 2; i++)
        audio.samples[i] = (float)(floor(audio.samples[i] * 32768.0 + 0.5) / 32768.0);
    free(bar_a);

    if (check("lossless", &audio, 0) != 0) return 1;
    if (check("lossy q7", &audio, 7) != 0) return 1;
    mmx_audio_buffer_free(&audio);
    printf("test_decoder_steps OK\n");
    return 0;
}
