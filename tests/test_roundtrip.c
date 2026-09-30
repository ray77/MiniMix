/* Full pipeline on a synthetic track with exact and approximate repeats:
   analyze -> encode -> write -> read -> decode. Checks bit-exact agreement of
   decoder and encoder reconstruction, the perceptual quality barrier, and that
   global analysis actually saves bytes against the AUDIO-only baseline. */
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
#include "minimix/quality.h"
#include "log.h"
#include "wav_writer.h"

#define PI 3.14159265358979323846

static unsigned long rng = 777;
static double frand(void)
{
    rng = rng * 1103515245UL + 12345UL;
    return ((rng >> 8) & 0xFFFF) / 65536.0 - 0.5;
}

static void render_bar(float *dst, unsigned long frames, double f0, double noise)
{
    unsigned long i;
    for (i = 0; i < frames; i++)
    {
        double t = (double)i / 44100.0;
        double env = exp(-3.0 * fmod(t, 0.5));
        double s = env * (0.5 * sin(2 * PI * f0 * t) + 0.25 * sin(2 * PI * f0 * 2.01 * t) + 0.1 * sin(2 * PI * f0 * 3.0 * t));
        s += 0.4 * exp(-40.0 * fmod(t, 0.5)) * sin(2 * PI * 60.0 * fmod(t, 0.5)); /* kick */
        dst[2 * i] = (float)(s + noise * frand());
        dst[2 * i + 1] = (float)(0.8 * s + noise * frand());
    }
}

int main(void)
{
    MMXAudioBuffer audio, enc_decoded, dec_decoded;
    MMXEncoderParams params;
    MMXFile file, loaded;
    MMXStatistics stats;
    MMXQualityReport q;
    const char *path = "bin/test_roundtrip.mmx";
    unsigned long bar = 44100, bars = 12, total = bar * bars, i;
    float *bar_a, *bar_b;
    unsigned long long written, written5, mp3;
    double total_payload;

    mmx_log_set_level(MMX_LOG_WARNING);

    mmx_audio_buffer_init(&audio, 44100, 2, total);
    audio.source_bits = 16;
    bar_a = malloc(sizeof(float) * bar * 2);
    bar_b = malloc(sizeof(float) * bar * 2);
    render_bar(bar_a, bar, 110.0, 0.001);
    render_bar(bar_b, bar, 146.83, 0.003);

    /* A A B B  A' A' B' B'  A A silence silence  (A' = A at -3 dB, B' = B shifted 137 samples) */
    for (i = 0; i < bars; i++)
    {
        float *dst = audio.samples + (size_t)i * bar * 2;
        unsigned long k;
        if (i >= 10) continue;
        if (i == 2 || i == 3) memcpy(dst, bar_b, sizeof(float) * bar * 2);
        else if (i == 6 || i == 7) { memcpy(dst + 137 * 2, bar_b, sizeof(float) * (bar - 137) * 2); }
        else if (i == 4 || i == 5) for (k = 0; k < bar * 2; k++) dst[k] = bar_a[k] * 0.7071f;
        else memcpy(dst, bar_a, sizeof(float) * bar * 2);
    }
    for (i = 0; i < total * 2; i++)
        audio.samples[i] = (float)(floor(audio.samples[i] * 32768.0 + 0.5) / 32768.0);

    mmx_encoder_params_default(&params);
    params.quality = 7;
    params.analysis = 5;
    params.metadata = "title=Roundtrip\n";

    mmx_file_init(&file);
    if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("read failed\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("decode failed\n"); return 1; }
    written5 = written;

    total_payload = (double)(stats.payload_bytes[0] + stats.payload_bytes[1] + stats.payload_bytes[2]);
    mp3 = mmx_statistics_cbr_bytes(stats.duration_seconds, 320.0);
    printf("roundtrip: %lu frames: %lu AUDIO, %lu REF, %lu REF2 in %lu blocks; max depth %u\n", stats.total_frames,
           stats.frames_by_sources[0], stats.frames_by_sources[1], stats.frames_by_sources[2], stats.block_count, stats.max_depth_used);
    printf("roundtrip: mmx %llu bytes (%.1f kbit/s, %.1f %% of MP3 320), baseline %llu -> global savings %.1f %%\n", written,
           written * 8.0 / stats.duration_seconds / 1000.0, 100.0 * written / mp3, stats.baseline_bytes,
           100.0 * (1.0 - total_payload / stats.baseline_bytes));
    printf("roundtrip: analysis %.1f s, encoding %.1f s, worst NMR %.2f dB\n", stats.analysis_seconds, stats.encoding_seconds, stats.worst_nmr_db);

    for (i = 0; i < total * 2; i++)
        if (enc_decoded.samples[i] != dec_decoded.samples[i])
        { printf("decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }

    /* tolerance 8 dB: the sub-frame analysis adds the aliased noise of neighbouring
       frames (~3 dB) and block-boundary effects; the real judge is listening */
    if (mmx_quality_compare(&audio, &dec_decoded, 7, 0, 8.0, &q) != 0) { printf("compare failed\n"); return 1; }
    printf("roundtrip: perceptual check worst NMR %.2f dB (at %.2f s ch %u band %u), %.2f %% sub-frames over, %.2f %% bands over, SNR %.1f dB, passed %d\n",
           q.worst_nmr_db, (q.worst_frame - 2.0) * 512.0 / 44100.0, q.worst_channel, q.worst_band, q.frames_over_percent, q.bands_over_percent, q.snr_db, q.passed);
    printf("roundtrip: pre-echo: %lu attacks, %lu below 10 dB pre-attack SNR, worst %.1f dB at %.2f s\n",
           q.attacks, q.attacks_pre_echo, q.worst_pre_attack_snr_db, q.worst_pre_attack_seconds);
    mmx_wav_write("bin/test_roundtrip.wav", &audio, 16);
    mmx_wav_write("bin/test_roundtrip.dec.wav", &dec_decoded, 16);
    if (!q.passed) { printf("quality barrier violated\n"); return 1; }
    if (stats.frames_by_sources[1] + stats.frames_by_sources[2] == 0) { printf("no references on a repetitive signal\n"); return 1; }
    if (total_payload >= stats.baseline_bytes) { printf("global analysis did not save bytes\n"); return 1; }
    if (stats.max_depth_used > params.max_ref_depth) { printf("depth limit violated\n"); return 1; }

    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

    /* analysis level 7: Viterbi with tentative depths, the coder-mirroring
       estimate and the second pass against the decoded signal. The same
       barriers, except that the perceptual check allows 10 dB: the deeper
       plan plays the kick bars from other bars, and in the 23 ms cell right
       before the kick at 2.46 s (tonal 440 Hz next to a silent band) the
       closed-loop decisions by coded bits and the rate-distortion quantizer
       (noise up to 1.5 dB under the threshold, RDO_SLACK_DB) use up the margin
       the plain quantizer left: 9.0 dB here, 8.2-8.7 dB with either feature
       alone, 6.3 dB with a 3 dB slack (MMX_RDOQ_SLACK=3); the real judge is
       listening. The file must not be larger than level 5's by more than 1 %. */
    params.analysis = 7;
    mmx_file_init(&file);
    if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("level 7 encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("level 7 write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("level 7 read failed\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("level 7 decode failed\n"); return 1; }
    total_payload = (double)(stats.payload_bytes[0] + stats.payload_bytes[1] + stats.payload_bytes[2]);
    printf("roundtrip: level 7: %lu frames: %lu AUDIO, %lu REF, %lu REF2 in %lu blocks; max depth %u; mmx %llu bytes (level 5: %llu), analysis %.1f s, encoding %.1f s\n",
           stats.total_frames, stats.frames_by_sources[0], stats.frames_by_sources[1], stats.frames_by_sources[2], stats.block_count,
           stats.max_depth_used, written, written5, stats.analysis_seconds, stats.encoding_seconds);
    for (i = 0; i < total * 2; i++)
        if (enc_decoded.samples[i] != dec_decoded.samples[i])
        { printf("level 7 decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
    if (mmx_quality_compare(&audio, &dec_decoded, 7, 0, 10.0, &q) != 0) { printf("level 7 compare failed\n"); return 1; }
    printf("roundtrip: level 7: perceptual check worst NMR %.2f dB (at %.2f s ch %u band %u), %.2f %% bands over, passed %d\n",
           q.worst_nmr_db, (q.worst_frame - 2.0) * 512.0 / 44100.0, q.worst_channel, q.worst_band, q.bands_over_percent, q.passed);
    if (!q.passed) { printf("level 7 quality barrier violated\n"); return 1; }
    if (stats.frames_by_sources[1] + stats.frames_by_sources[2] == 0) { printf("level 7: no references on a repetitive signal\n"); return 1; }
    if (total_payload >= stats.baseline_bytes) { printf("level 7: global analysis did not save bytes\n"); return 1; }
    if (stats.max_depth_used > params.max_ref_depth) { printf("level 7: depth limit violated\n"); return 1; }
    if (written > written5 + written5 / 100) { printf("level 7 larger than level 5 (%llu > %llu bytes)\n", written, written5); return 1; }
    params.analysis = 5;
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

    /* noise substitution from 2 kHz on: the substituted bands are noise the
       decoder generates itself, so the closed loop must reproduce them exactly */
    params.pns_hz = 2000;
    mmx_file_init(&file);
    if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("pns encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("pns write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("pns read failed\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("pns decode failed\n"); return 1; }
    printf("roundtrip: noise substitution: %lu of %lu coded channel bands above 2 kHz play as noise, %llu bytes\n",
           stats.pns_bands, stats.pns_candidates, written);
    for (i = 0; i < total * 2; i++)
        if (enc_decoded.samples[i] != dec_decoded.samples[i])
        { printf("pns decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
    if (stats.pns_bands == 0) { printf("no noise bands on a signal with a noise floor\n"); return 1; }
    params.pns_hz = 0;
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

    /* band replication from 6 kHz on: the bands above the crossover carry only
       their level and a mix index, and the decoder rebuilds them by copying the
       octave below the crossover and mixing in its own noise. Encoder and
       decoder must land on the same samples, the file must be smaller than the
       same encode without replication, and the crossover must survive
       write -> read (header byte 36). */
    {
        double level_no_bwe, level_bwe;
        params.target_kbps = 24;   /* the replication is a low-rate tool: at a fixed quality the
                                      masking model already zeroes the highs and a level per band
                                      and frame costs more than the zero flags it replaces */
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("bwe reference encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("bwe reference write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("bwe reference read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("bwe reference decode failed\n"); return 1; }
        if (mmx_quality_compare(&audio, &dec_decoded, 7, 0, 100.0, &q) != 0) { printf("bwe reference compare failed\n"); return 1; }
        level_no_bwe = q.eq_level_abs_db[6];   /* 8-12 kHz: energy match of the region the replication owns */
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

        params.bwe_hz = 6000;
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("bwe encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("bwe write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("bwe read failed\n"); return 1; }
        if (loaded.bwe_hz != 6000) { printf("bwe crossover lost in the header: %u\n", loaded.bwe_hz); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("bwe decode failed\n"); return 1; }
        if (mmx_quality_compare(&audio, &dec_decoded, 7, 0, 100.0, &q) != 0) { printf("bwe compare failed\n"); return 1; }
        level_bwe = q.eq_level_abs_db[6];
        printf("roundtrip: band replication at 24 kbit/s: %lu channel bands above 6 kHz regenerated (%lu silent), %llu bytes, 8-12 kHz level error %.2f dB (without replication %.2f dB)\n",
               stats.bwe_bands, stats.bwe_bands_zero, written, level_bwe, level_no_bwe);
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("bwe decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (stats.bwe_bands == 0) { printf("no replicated bands above the crossover\n"); return 1; }
        if (level_bwe > level_no_bwe)
        { printf("band replication did not improve the energy match above the crossover (%.2f dB > %.2f dB)\n", level_bwe, level_no_bwe); return 1; }
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

        /* the same with the linear patch mapping (MMX_BWE_MODE=1) instead of
           the octave one: a different reconstruction, the same closed loop */
        setenv("MMX_BWE_MODE", "1", 1);
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("bwe rate encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("bwe rate write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("bwe rate read failed\n"); return 1; }
        if (loaded.bwe_mode != 1) { printf("bwe patch mode lost in the header\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("bwe rate decode failed\n"); return 1; }
        printf("roundtrip: band replication at 24 kbit/s, linear patch: %lu bands, %llu bytes\n", stats.bwe_bands, written);
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("bwe rate decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        unsetenv("MMX_BWE_MODE");
        params.target_kbps = 0;
        params.bwe_hz = 0;
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);
    }

    /* noise filling at every rate (MMX_NF_ON) with the thresholds 9 dB above the
       model (a rate-matched file's offset): the zeros of the coded regions and
       the holes carry noise the decoder generates itself, so the closed loop
       must reproduce it exactly */
    params.noise_fill = MMX_NF_ON;
    params.quality = 3;
    mmx_file_init(&file);
    if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("nf encode failed\n"); return 1; }
    if (mmx_writer_write(path, &file, &written) != 0) { printf("nf write failed\n"); return 1; }
    if (mmx_reader_read(path, &loaded, 1) != 0) { printf("nf read failed\n"); return 1; }
    if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("nf decode failed\n"); return 1; }
    printf("roundtrip: noise filling: %lu of %lu channel regions filled, %lu holes kept as noise bands, %llu bytes\n",
           stats.nf_regions, stats.nf_regions_total, stats.nf_hole_bands, written);
    for (i = 0; i < total * 2; i++)
        if (enc_decoded.samples[i] != dec_decoded.samples[i])
        { printf("nf decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
    if (stats.nf_regions == 0) { printf("no noise filling on a signal with a noise floor\n"); return 1; }
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

    /* the same with the zero-distortion weight of the filled regions below 1
       (MMX_NF_ZERO): the quantizer then counts a zero the decoder will fill as
       half its energy and moves bits out of those regions, so the quantization
       changes - the closed loop must still reproduce the decoder exactly */
    {
        unsigned long long written_alpha1 = written;
        setenv("MMX_NF_ZERO", "0.5", 1);
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("nf alpha encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("nf alpha write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("nf alpha read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("nf alpha decode failed\n"); return 1; }
        printf("roundtrip: noise filling, zero weight 0.5: %lu of %lu channel regions filled, %lu holes, %llu bytes (weight 1: %llu)\n",
               stats.nf_regions, stats.nf_regions_total, stats.nf_hole_bands, written, written_alpha1);
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("nf alpha decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (stats.nf_regions == 0) { printf("no noise filling with a zero weight below 1\n"); return 1; }
        if (written == written_alpha1) { printf("zero weight 0.5 changed nothing (%llu bytes either way)\n", written); return 1; }
        unsetenv("MMX_NF_ZERO");
    }
    params.noise_fill = MMX_NF_DEFAULT;
    params.quality = 7;
    params.pns_hz = 0;
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);

    /* energy preservation (the gurgling rules): a rate-matched file, where the
       threshold offset would otherwise zero the bands near the lifted
       threshold in one frame and code them in the next. The restored bands are
       coefficients below the crossover and noise the decoder generates itself
       above it, so the closed loop must reproduce them exactly. */
    {
        unsigned long long written_keep;
        unsigned long keep_bands, keep_noise;
        params.target_kbps = 16;   /* the track codes at 25 kbit/s at quality 7: this needs a real offset */
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("keep encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("keep write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("keep read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("keep decode failed\n"); return 1; }
        written_keep = written;
        keep_bands = stats.keep_bands;
        keep_noise = stats.keep_noise_bands;
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("keep decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (keep_bands == 0 || keep_noise == 0)
        { printf("energy preservation did not fire on a rate-matched file (%lu coefficients, %lu noise bands)\n", keep_bands, keep_noise); return 1; }
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);
        /* MMX_KEEP=0 is the coder before the rules: it must still work and must
           leave holes where the rules fill them (a different, smaller file) */
        setenv("MMX_KEEP", "0", 1);
        mmx_file_init(&file);
        if (mmx_encoder_encode(&audio, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("keep-off encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("keep-off write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("keep-off read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("keep-off decode failed\n"); return 1; }
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("keep-off decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (stats.keep_bands || stats.keep_noise_bands) { printf("MMX_KEEP=0 still restored bands\n"); return 1; }
        unsetenv("MMX_KEEP");
        printf("roundtrip: energy preservation: %lu channel bands restored as coefficients, %lu as noise bands, %llu bytes (rules off: %llu)\n",
               keep_bands, keep_noise, written_keep, written);
        params.target_kbps = 0;
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);
    }

    /* intensity stereo from 2 kHz on, on the same track with the right channel
       the left at -3 dB (every band coherent, a source panned right of
       centre): the decoder scales the coded mid by the position gains, so the
       closed loop must reproduce that exactly, and the file must be smaller
       than the same signal coded without */
    {
        MMXAudioBuffer panned;
        unsigned long long written_no_is;
        mmx_audio_buffer_init(&panned, 44100, 2, total);
        panned.source_bits = 16;
        for (i = 0; i < total; i++)
        {
            panned.samples[2 * i] = audio.samples[2 * i];
            panned.samples[2 * i + 1] = (float)(floor(audio.samples[2 * i] * 0.7071 * 32768.0 + 0.5) / 32768.0);
        }
        params.is_hz = 0;
        mmx_file_init(&file);
        if (mmx_encoder_encode(&panned, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("panned encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written_no_is) != 0) { printf("panned write failed\n"); return 1; }
        mmx_file_free(&file);
        mmx_audio_buffer_free(&enc_decoded);
        params.is_hz = 2000;
        mmx_file_init(&file);
        if (mmx_encoder_encode(&panned, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("is encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("is write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("is read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("is decode failed\n"); return 1; }
        printf("roundtrip: intensity stereo: %lu of %lu coded bands above 2 kHz carry the mid once, %llu bytes (without %llu)\n",
               stats.is_bands, stats.is_candidates, written, written_no_is);
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("is decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (stats.is_bands == 0) { printf("no intensity bands on a coherent stereo signal\n"); return 1; }
        if (written >= written_no_is) { printf("intensity stereo did not save bytes (%llu >= %llu)\n", written, written_no_is); return 1; }

        /* intensity stereo and noise filling together: the intensity bands are
           left alone by the filling, everything else is filled - the closed
           loop must still be bit-exact */
        mmx_file_free(&file); mmx_file_free(&loaded);
        mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);
        params.noise_fill = MMX_NF_ON;
        params.quality = 3;
        mmx_file_init(&file);
        if (mmx_encoder_encode(&panned, &params, &file, &stats, &enc_decoded, NULL) != 0) { printf("is+nf encode failed\n"); return 1; }
        if (mmx_writer_write(path, &file, &written) != 0) { printf("is+nf write failed\n"); return 1; }
        if (mmx_reader_read(path, &loaded, 1) != 0) { printf("is+nf read failed\n"); return 1; }
        if (mmx_decoder_decode(&loaded, &dec_decoded) != 0) { printf("is+nf decode failed\n"); return 1; }
        printf("roundtrip: intensity stereo + noise filling: %lu intensity bands, %lu of %lu channel regions filled, %llu bytes\n",
               stats.is_bands, stats.nf_regions, stats.nf_regions_total, written);
        for (i = 0; i < total * 2; i++)
            if (enc_decoded.samples[i] != dec_decoded.samples[i])
            { printf("is+nf decoder/encoder mismatch at sample %lu: %g vs %g\n", i, enc_decoded.samples[i], dec_decoded.samples[i]); return 1; }
        if (stats.is_bands == 0 || stats.nf_regions == 0) { printf("is+nf: one of the two tools did nothing\n"); return 1; }
        params.noise_fill = MMX_NF_DEFAULT;
        params.quality = 7;
        params.is_hz = 0;
        mmx_audio_buffer_free(&panned);
    }

    free(bar_a); free(bar_b);
    mmx_file_free(&file); mmx_file_free(&loaded);
    mmx_audio_buffer_free(&audio); mmx_audio_buffer_free(&enc_decoded); mmx_audio_buffer_free(&dec_decoded);
    remove(path);
    printf("test_roundtrip OK\n");
    return 0;
}
