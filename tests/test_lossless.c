/* Lossless mode: encode -> write -> read -> decode must reproduce the 16-bit
   input bit-exactly, and the global references must save bytes on a track
   with exact repeats. First the old integer core (MMX_LL2=0: its baseline and
   predictor statistics), then the LL2 core that --lossless uses by default. */
/* setenv/unsetenv are POSIX, not C99: ask for them, and map them on Windows */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif
#include <stdlib.h>
#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#define unsetenv(name) _putenv_s((name), "")
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minimix/encoder.h"
#include "minimix/decoder.h"
#include "minimix/mmx_writer.h"
#include "minimix/mmx_reader.h"
#include "log.h"
#include "lossless.h"
#include "ll2codec.h"

#define PI 3.14159265358979323846

static unsigned long rng = 4242;
static double frand(void)
{
    rng = rng * 1103515245UL + 12345UL;
    return ((rng >> 8) & 0xFFFF) / 65536.0 - 0.5;
}

int main(void)
{
    MMXAudioBuffer audio, decoded;
    MMXEncoderParams params;
    MMXFile file, loaded;
    MMXStatistics stats;
    const char *path = "bin/test_lossless.mmx";
    unsigned long bar = 44100, bars = 8, total = bar * bars, i, b, mismatches = 0;
    float *bar_a;
    unsigned long long written;
    double total_payload;

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
    /* A A A' A B B A A  (A' = A at -6 dB, B = noise) */
    for (b = 0; b < bars; b++)
    {
        float *dst = audio.samples + (size_t)b * bar * 2;
        if (b == 4 || b == 5)
            for (i = 0; i < bar * 2; i++) dst[i] = (float)(0.3 * frand());
        else if (b == 2)
            for (i = 0; i < bar * 2; i++) dst[i] = bar_a[i] * 0.5f;
        else
            memcpy(dst, bar_a, sizeof(float) * bar * 2);
    }
    for (i = 0; i < total * 2; i++)
        audio.samples[i] = (float)(floor(audio.samples[i] * 32768.0 + 0.5) / 32768.0);

    setenv("MMX_LL2", "0", 1);
    mmx_encoder_params_default(&params);
    params.quality = 0;
    params.analysis = 4;
    mmx_file_init(&file);
    memset(&mmx_ll_stats, 0, sizeof(mmx_ll_stats));
    if (mmx_encoder_encode(&audio, &params, &file, &stats, NULL, NULL) != 0) { printf("encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("read failed\n"); return 1; }
    if (loaded.codec_id != MMX_CODEC_LOSSLESS || loaded.quality != 0) { printf("header not lossless\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &decoded) != 0) { printf("decode failed\n"); return 1; }

    for (i = 0; i < total * 2; i++)
        if (decoded.samples[i] != audio.samples[i])
            mismatches++;
    total_payload = (double)(stats.payload_bytes[0] + stats.payload_bytes[1] + stats.payload_bytes[2]);
    printf("lossless: %lu frames, %lu AUDIO, %lu REF, %lu blocks; %llu bytes = %.1f %% of PCM, baseline %llu -> global savings %.1f %%\n",
           stats.total_frames, stats.frames_by_sources[0], stats.frames_by_sources[1], stats.block_count, written,
           100.0 * written / stats.input_bytes, stats.baseline_bytes, 100.0 * (1.0 - total_payload / stats.baseline_bytes));
    printf("lossless: %lu sample mismatches\n", mismatches);
    printf("lossless: predictors: %lu channel frames, %lu LPC, %lu fixed, %lu through the filter cascade; LPC orders:",
           mmx_ll_stats.channel_frames, mmx_ll_stats.lpc_frames, mmx_ll_stats.fixed_frames, mmx_ll_stats.nlms_frames);
    for (i = 1; i <= MMX_LL_LPC_MAX_ORDER; i++)
        printf(" %lu", mmx_ll_stats.lpc_order_hist[i]);
    printf("\n");
    if (mismatches) { printf("NOT bit-exact\n"); return 1; }
    if (stats.frames_by_sources[1] == 0) { printf("no references used\n"); return 1; }
    if (total_payload >= stats.baseline_bytes) { printf("references did not save bytes\n"); return 1; }
    if (mmx_ll_stats.lpc_frames == 0) { printf("no frame chose the LPC predictor\n"); return 1; }
    if (mmx_ll_stats.nlms_frames == 0 || mmx_ll_stats.nlms_frames == mmx_ll_stats.channel_frames)
    { printf("filter cascade never chosen or never rejected\n"); return 1; }

    /* the LL2 core: bit-exact, references used, blocks of at most MMX_LL2_BLOCK_FRAMES frames */
    unsetenv("MMX_LL2");
    mmx_file_free(&file); mmx_file_free(&loaded); mmx_audio_buffer_free(&decoded);
    mmx_file_init(&file);
    memset(&stats, 0, sizeof(stats));
    mismatches = 0;
    if (mmx_encoder_encode(&audio, &params, &file, &stats, NULL, NULL) != 0) { printf("LL2 encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("LL2 write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("LL2 read failed\n"); return 1; }
    if (loaded.ll_core != MMX_LL2_CORE) { printf("LL2 not the default lossless core\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &decoded) != 0) { printf("LL2 decode failed\n"); return 1; }
    for (i = 0; i < total * 2; i++)
        if (decoded.samples[i] != audio.samples[i])
            mismatches++;
    printf("LL2: %lu frames, %lu AUDIO, %lu REF, %lu blocks; %llu bytes = %.1f %% of PCM; %lu sample mismatches\n",
           stats.total_frames, stats.frames_by_sources[0], stats.frames_by_sources[1], (unsigned long)loaded.block_count,
           written, 100.0 * written / stats.input_bytes, mismatches);
    if (mismatches) { printf("LL2 NOT bit-exact\n"); return 1; }
    if (stats.frames_by_sources[1] == 0) { printf("LL2: no references used\n"); return 1; }
    for (i = 0; i < loaded.block_count; i++)
        if (loaded.blocks[i].frame_count > MMX_LL2_BLOCK_FRAMES) { printf("LL2: block %lu too long\n", i); return 1; }

    free(bar_a);
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&audio); mmx_audio_buffer_free(&decoded);
    remove(path);
    printf("test_lossless OK\n");
    return 0;
}
