#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "commands.h"
#include "wav_reader.h"
#include "wav_writer.h"
#include "minimix/encoder.h"
#include "minimix/decoder.h"
#include "minimix/analyzer.h"
#include "minimix/mmx_writer.h"
#include "minimix/mmx_reader.h"
#include "minimix/quality.h"
#include "statistics.h"
#include "percept.h"
#include "log.h"

static const char *basename_of(const char *path)
{
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

static int has_suffix(const char *path, const char *suffix)
{
    size_t n = strlen(path), m = strlen(suffix);
    return n >= m && strcmp(path + n - m, suffix) == 0;
}

static void print_input_banner(const char *tool, const char *path, const MMXAudioBuffer *a)
{
    char dur[32];
    mmx_format_duration(mmx_audio_buffer_duration_seconds(a), dur, sizeof(dur));
    mmx_info("%s %s", tool, MMX_VERSION_STRING);
    mmx_info("Input: %s", path);
    mmx_info("Length: %s", dur);
    mmx_info("Sample Rate: %lu Hz", a->sample_rate);
    mmx_info("Channels: %u", a->channels);
    mmx_info("Source depth: %u bit", a->source_bits);
}

static const char *mime_for(const char *path)
{
    if (has_suffix(path, ".png")) return "image/png";
    if (has_suffix(path, ".webp")) return "image/webp";
    if (has_suffix(path, ".avif")) return "image/avif";
    return "image/jpeg";
}

static unsigned char *read_file(const char *path, unsigned long *len)
{
    FILE *fp = fopen(path, "rb");
    unsigned char *data;
    long long size;
    if (!fp)
        return NULL;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    data = (unsigned char *)malloc(size > 0 ? (size_t)size : 1);
    if (!data || (size > 0 && fread(data, 1, (size_t)size, fp) != (size_t)size))
    {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *len = (unsigned long)size;
    return data;
}

/* ASCII timeline: one character per column over the whole track. */
static void print_timeline(const MMXAnalysis *a, unsigned long sample_rate, int columns)
{
    unsigned long nf = a->frame_count, f;
    int col;
    printf("\nTimeline (%d columns, %.1f s each): A = new audio, r = reference, R = reference (2 sources), . = silence\n",
           columns, (double)nf * MMX_HOP / sample_rate / columns);
    for (col = 0; col < columns; col++)
    {
        unsigned long from = nf * col / columns, to = nf * (col + 1) / columns, n = 0, r = 0, r2 = 0, s = 0;
        for (f = from; f < to; f++)
        {
            n++;
            if (a->silent[f]) s++;
            else if (a->plan[f].n_sources == 1) r++;
            else if (a->plan[f].n_sources == 2) r2++;
        }
        if (!n) continue;
        if (s * 2 > n) putchar('.');
        else if (r2 * 2 > n) putchar('R');
        else if ((r + r2) * 2 > n) putchar('r');
        else putchar('A');
    }
    putchar('\n');
}

static void meta_append(char *buf, size_t size, const char *line)
{
    size_t n = strlen(buf);
    if (n + strlen(line) + 2 < size)
    {
        strcat(buf, line);
        strcat(buf, "\n");
    }
}

/* MOD-style sequence: which parts of the song are played from which earlier
   parts. Blocks shorter than half a second are only counted. */
static void print_patterns(const MMXFile *f, double sr, unsigned long max_lines)
{
    unsigned long i, shown = 0, blocks = 0;
    double pattern_s = 0.0, total_s = (double)f->frame_count / sr;
    printf("\nPattern sequence (later part <- earlier part, blocks >= 0.5 s)\n");
    for (i = 0; i < f->block_count; i++)
    {
        const MMXBlockEntry *b = &f->blocks[i];
        double t0, len;
        if (!b->n_sources)
            continue;
        t0 = (double)((long long)b->start_frame * (long long)f->hop - (long long)f->hop) / sr;
        if (t0 < 0.0) t0 = 0.0;
        len = (double)b->frame_count * (double)f->hop / sr;
        pattern_s += len;
        blocks++;
        if (len < 0.5)
            continue;
        if (shown < max_lines)
            printf("  %7.2f - %7.2f s  <-  %7.2f s   (lag %6.2f s%s)\n", t0, t0 + len, (double)b->src_start[0] / sr,
                   t0 - (double)b->src_start[0] / sr, b->n_sources > 1 ? ", mixed with a second source" : "");
        shown++;
    }
    if (shown > max_lines)
        printf("  ... %lu more\n", shown - max_lines);
    printf("  %.1f %% of the song plays from earlier parts (%lu pattern blocks)\n", total_s > 0 ? 100.0 * pattern_s / total_s : 0.0, blocks);
}

int mmx_cmd_encode(const MMXOptions *opts)
{
    MMXAudioBuffer audio;
    MMXEncoderParams params;
    MMXFile file;
    MMXStatistics stats;
    MMXAnalysis analysis;
    char metadata[MMX_MAX_META * 512 + 256];
    char line[600];
    unsigned char *cover = NULL;
    unsigned long cover_len = 0, written_table;
    unsigned long long written = 0;
    unsigned int i;
    int rc;

#if !MMX_LOSSY
    /* the lossy core is switched off for now (cli.h): lossless only */
    if (opts->lossy_opt)
    {
        mmx_error("%s: the lossy core is switched off for now - mmx encode is lossless (make MMX_LOSSY=1 brings it back)",
                  opts->lossy_opt);
        return 1;
    }
    if (opts->bitrate && !opts->lossless_given)
    {
        mmx_error("--bitrate without --lossless is a lossy target; the lossy core is switched off for now (with --lossless it "
                  "bounds the near-lossless error)");
        return 1;
    }
#endif
    if (mmx_wav_read(opts->input, &audio) != 0)
        return 1;
    print_input_banner("MiniMix Encoder", opts->input, &audio);

    mmx_encoder_params_default(&params);
    params.quality = MMX_LOSSY ? opts->quality : 0;
    params.drop_bits = opts->drop_bits;
    params.drop_step = opts->drop_step;
    params.pld_strict = opts->pld_strict;
    if (params.pld_strict && params.quality != 0)
    {
        mmx_error("--pld belongs to --lossless");
        return 1;
    }
    if (params.drop_step > 1 && params.quality != 0)
    {
        mmx_error("--drop-step belongs to --lossless");
        return 1;
    }
    if (params.drop_bits && params.quality != 0)
    {
        mmx_error("--drop-bits belongs to --lossless: it rounds the input for the lossless coder (a lossy encode has its own quality)");
        mmx_audio_buffer_free(&audio);
        return 1;
    }
    params.analysis = opts->analysis;
    params.target_kbps = opts->bitrate;
    params.tns = opts->tns;
    {
        /* without --pns / --no-pns the bitrate decides: at and below
           MMX_PNS_DEFAULT_MAX_KBPS noise substitution is what keeps the ear's
           region clean (see encoder.h). A fixed quality never gets it. */
        const char *e = getenv("MMX_PNS_DEFAULT_MAX_KBPS");
        unsigned int pns_max = e ? (unsigned int)atoi(e) : MMX_PNS_DEFAULT_MAX_KBPS;
        params.pns_hz = opts->pns_given ? opts->pns
                      : (opts->bitrate && opts->bitrate <= pns_max ? MMX_PNS_DEFAULT_HZ : 0);
    }
    {
        /* without --is / --no-is the bitrate decides, and the limit is 0 by
           default: intensity stereo is opt-in. MMX_IS_DEFAULT_MAX_KBPS raises
           the limit for listening experiments (the knob the measurements used). */
        const char *e = getenv("MMX_IS_DEFAULT_MAX_KBPS");
        unsigned int is_max = e ? (unsigned int)atoi(e) : MMX_IS_DEFAULT_MAX_KBPS;
        params.is_hz = opts->is_given ? opts->is_hz : (opts->bitrate && opts->bitrate <= is_max ? MMX_IS_DEFAULT_HZ : 0);
    }
    {
        /* without --bwe / --no-bwe the bitrate decides (see encoder.h):
           MMX_BWE_HZ overrides the crossover for experiments. */
        const char *e = getenv("MMX_BWE_HZ");
        params.bwe_hz = opts->bwe_given ? opts->bwe_hz : mmx_bwe_default_hz(opts->bitrate);
        if (e) params.bwe_hz = (unsigned int)atoi(e);
        if (params.quality == 0) params.bwe_hz = 0;
    }
    params.noise_fill = opts->nf;
    params.lowrate = getenv("MMX_LOWRATE") ? (unsigned int)(atoi(getenv("MMX_LOWRATE")) != 0) : 0;   /* experiment: EPB on */
    if (params.lowrate)
    {
        /* EPB replaces replication and noise filling; the PNS decision drops the band instead (MMX_EPB_PNS=0 switches it off).
           Fill policy knobs (go into the header, the decoder follows the header): MMX_EPB_FOLD_HZ (default 4000, 0 = never),
           MMX_EPB_NOISE (1), MMX_EPB_CODED (0), MMX_EPB_REF (0), MMX_EPB_FOLD_DB (0) */
        const char *e;
        double fold_hz = (e = getenv("MMX_EPB_FOLD_HZ")) ? atof(e) : 4000.0;
        params.bwe_hz = 0;
        params.noise_fill = MMX_NF_OFF;
        if ((e = getenv("MMX_EPB_PNS")) && atoi(e) == 0) params.pns_hz = 0;
        params.epb_fold = fold_hz <= 0.0 || fold_hz >= 63000.0 ? 255u : (unsigned int)(fold_hz / 250.0 + 0.5);
        params.epb_flags = 0;
        if ((e = getenv("MMX_EPB_NOISE")) ? atoi(e) != 0 : 1) params.epb_flags |= 1u;
        if ((e = getenv("MMX_EPB_CODED")) ? atoi(e) != 0 : 0) params.epb_flags |= 2u;
        if ((e = getenv("MMX_EPB_REF")) ? atoi(e) != 0 : 0) params.epb_flags |= 4u;
        params.epb_fill_db = (e = getenv("MMX_EPB_FOLD_DB")) ? atoi(e) : 0;
    }
    params.reuse = opts->reuse;
    params.mode = opts->mode;
    if (opts->mode == MMX_MODE_FUTURE && !opts->quality_given && params.quality != 0)
        params.quality = 4;   /* future: the unique material is coded at quality 4 */
    params.max_ref_depth = (unsigned char)opts->max_depth;
    params.verbose = opts->verbose;

    metadata[0] = '\0';
    snprintf(line, sizeof(line), "encoder=MiniMix %s", MMX_VERSION_STRING);
    meta_append(metadata, sizeof(metadata), line);
    snprintf(line, sizeof(line), "source=%s", basename_of(opts->input));
    meta_append(metadata, sizeof(metadata), line);
    if (opts->reuse)
    {
        snprintf(line, sizeof(line), "reuse=%u", opts->reuse);
        meta_append(metadata, sizeof(metadata), line);
    }
    if (opts->mode)
        meta_append(metadata, sizeof(metadata), opts->mode == MMX_MODE_FUTURE ? "mode=future" : "mode=tracker");
    if (params.pns_hz && params.quality != 0)
    {
        snprintf(line, sizeof(line), "pns=%u", params.pns_hz);
        meta_append(metadata, sizeof(metadata), line);
    }
    if (params.is_hz && params.quality != 0)
    {
        snprintf(line, sizeof(line), "is=%u", params.is_hz);
        meta_append(metadata, sizeof(metadata), line);
    }
    if (params.bwe_hz && params.quality != 0)
    {
        snprintf(line, sizeof(line), "bwe=%u", params.bwe_hz);
        meta_append(metadata, sizeof(metadata), line);
    }
    if (params.quality != 0 && (opts->nf == MMX_NF_ON || (opts->nf == MMX_NF_AUTO && opts->bitrate)))
        meta_append(metadata, sizeof(metadata), "nf=on");
    for (i = 0; i < opts->meta_count; i++)
        meta_append(metadata, sizeof(metadata), opts->meta[i]);
    params.metadata = metadata;

    if (opts->cover)
    {
        cover = read_file(opts->cover, &cover_len);
        if (!cover)
        {
            mmx_error("Cannot read cover image: %s", opts->cover);
            mmx_audio_buffer_free(&audio);
            return 1;
        }
        params.cover = cover;
        params.cover_len = cover_len;
        params.cover_mime = mime_for(opts->cover);
    }

    if (params.quality == 0)
        mmx_info("Lossless, analysis level %u/9, max reference depth %u", params.analysis, params.max_ref_depth);
    else
        mmx_info("Quality %u%s, analysis level %u/9, max reference depth %u%s", params.quality,
                 params.target_kbps ? " (bitrate target)" : "", params.analysis, params.max_ref_depth,
                 params.mode == MMX_MODE_FUTURE ? ", FUTURE mode" : params.mode ? ", TRACKER mode" : "");

    /* A lossless file must reproduce the input bit for bit, and the float buffer cannot hold more
       than 24 bits of it. Refusing is what FLAC does with float input; storing it anyway and calling
       the file lossless is what early versions of this encoder did. */
    if (params.quality == 0 && !audio.exact)
    {
        mmx_error("Lossless refused: the input carries more than the 24 bits this encoder can keep exactly "
                  "(32-bit integer with live low bits, or float off the 24-bit grid). "
                  "Convert it to 24 bit first, or encode lossy.");
        free(cover);
        mmx_audio_buffer_free(&audio);
        return 1;
    }
    mmx_file_init(&file);
    memset(&analysis, 0, sizeof(analysis));
    rc = mmx_encoder_encode(&audio, &params, &file, &stats, NULL, &analysis);
    if (rc != 0)
    {
        mmx_error("Encoding failed");
        mmx_file_free(&file);
        mmx_audio_buffer_free(&audio);
        free(cover);
        return 1;
    }

    mmx_info("Writing MMX container...");
    if (mmx_writer_write(opts->output, &file, &written) != 0)
    {
        mmx_file_free(&file);
        mmx_audio_buffer_free(&audio);
        free(cover);
        return 1;
    }
    stats.output_bytes = written;
    written_table = (unsigned long)(written - stats.payload_bytes[0] - stats.payload_bytes[1] - stats.payload_bytes[2]);
    stats.table_bytes = written_table;

    if (params.quality == 0)
        mmx_info("Done. %s: %llu bytes = %.1f kbit/s (%.1f %% of the PCM), %.1f %% frames referenced",
                 opts->output, written, (double)written * 8.0 / stats.duration_seconds / 1000.0,
                 stats.input_bytes ? 100.0 * (double)written / (double)stats.input_bytes : 0.0,
                 stats.total_frames ? 100.0 * (double)(stats.frames_by_sources[1] + stats.frames_by_sources[2]) / stats.total_frames : 0.0);
    else
        mmx_info("Done. %s: %llu bytes = %.1f kbit/s (%.1f %% of MP3 320), %.1f %% frames referenced, savings from global analysis %.1f %%, worst NMR %.2f dB",
                 opts->output, written, (double)written * 8.0 / stats.duration_seconds / 1000.0,
                 100.0 * (double)written / (double)mmx_statistics_cbr_bytes(stats.duration_seconds, 320.0),
                 stats.total_frames ? 100.0 * (double)(stats.frames_by_sources[1] + stats.frames_by_sources[2]) / stats.total_frames : 0.0,
                 stats.baseline_bytes ? 100.0 * (1.0 - (double)(stats.payload_bytes[0] + stats.payload_bytes[1] + stats.payload_bytes[2]) / (double)stats.baseline_bytes) : 0.0,
                 stats.worst_nmr_db);

    if (opts->json)
        mmx_statistics_print_json(&stats);
    else if (opts->stats || opts->verbose)
    {
        mmx_statistics_print(&stats);
        print_timeline(&analysis, audio.sample_rate, 100);
        print_patterns(&file, audio.sample_rate, opts->verbose ? 400 : 40);
    }

    mmx_analysis_free(&analysis);
    mmx_file_free(&file);
    mmx_audio_buffer_free(&audio);
    free(cover);
    return 0;
}

/* The float reconstruction is unbounded; the WAV writer rounds and clamps it
   to the output bit depth. Says how often that happens (every such sample is
   an error the encoder's model did not allow for; the encoder's clip guard
   keeps the audible ones down). MMX_DEBUG_CLIP=1 lists the worst seconds. */
static void report_clipping(const MMXAudioBuffer *pcm, unsigned int bits)
{
    double scale = pow(2.0, (double)bits - 1.0), lim_pos = (scale - 0.5) / scale, lim_neg = -(scale + 0.5) / scale;
    double peak = 0.0, clip_e = 0.0;
    unsigned long long i, n = pcm->frame_count * pcm->channels, over = 0, nsec = pcm->frame_count / pcm->sample_rate + 1;
    unsigned long *per_sec = getenv("MMX_DEBUG_CLIP") ? (unsigned long *)calloc((size_t)nsec, sizeof(unsigned long)) : NULL;
    for (i = 0; i < n; i++)
    {
        double x = pcm->samples[i];
        if (x > lim_pos || x < lim_neg)
        {
            double cl = x > lim_pos ? lim_pos : lim_neg;
            over++;
            clip_e += (x - cl) * (x - cl);
            if (per_sec) per_sec[i / pcm->channels / pcm->sample_rate]++;
        }
        if (fabs(x) > peak) peak = fabs(x);
    }
    if (over)
        mmx_info("Reconstruction leaves the %u-bit range in %llu samples (%.4f %%, peak %+.2f dBFS, clamp error %.1f dBFS rms over the file); they are clamped",
                 bits, over, 100.0 * (double)over / (double)(n ? n : 1), 20.0 * log10(peak), 10.0 * log10(clip_e / (double)(n ? n : 1) + 1e-30));
    if (per_sec)
    {
        int top;
        fprintf(stderr, "clipping per second (worst first):");
        for (top = 0; top < 10; top++)
        {
            unsigned long long s, best = 0;
            for (s = 1; s < nsec; s++) if (per_sec[s] > per_sec[best]) best = s;
            if (!per_sec[best]) break;
            fprintf(stderr, " %llu s: %lu", best, per_sec[best]);
            per_sec[best] = 0;
        }
        fprintf(stderr, "\n");
        free(per_sec);
    }
}

int mmx_cmd_decode(const MMXOptions *opts)
{
    MMXFile file;
    MMXAudioBuffer pcm;
    unsigned int bits;
    clock_t t0 = clock();

    if (mmx_reader_read(opts->input, &file, 1) != 0)
        return 1;

    mmx_info("MiniMix Decoder %s", MMX_VERSION_STRING);
    mmx_info("Input: %s (%lu blocks, %lu Hz, %u ch, quality %u)", opts->input, file.block_count, file.sample_rate, file.channels, file.quality);

    if (mmx_decoder_decode(&file, &pcm) != 0)
    {
        mmx_error("Decoding failed");
        mmx_file_free(&file);
        return 1;
    }
    bits = opts->out_bits ? opts->out_bits : (file.source_bits == 24 || file.source_bits == 32 ? file.source_bits : 16);
    report_clipping(&pcm, bits);
    if (mmx_wav_write(opts->output, &pcm, bits) != 0)
    {
        mmx_audio_buffer_free(&pcm);
        mmx_file_free(&file);
        return 1;
    }
    mmx_info("Done. %s written (%u bit PCM) in %.2f s", opts->output, bits, (double)(clock() - t0) / CLOCKS_PER_SEC);
    mmx_audio_buffer_free(&pcm);
    mmx_file_free(&file);
    return 0;
}

int mmx_cmd_analyze(const MMXOptions *opts)
{
    MMXAudioBuffer audio;
    MMXCodec codec;
    MMXAnalysisParams params;
    MMXAnalysis a;
    char dur[32];
    double seconds, est_saving;
    unsigned long f;

    if (has_suffix(opts->input, ".mmx"))
    {
        if (opts->percept) { mmx_error("--percept needs a WAV: the map is computed from the audio, not from the bitstream"); return 1; }
        return mmx_cmd_info(opts);
    }

    if (mmx_wav_read(opts->input, &audio) != 0)
        return 1;
    if (!opts->json)
        print_input_banner("MiniMix Analyzer", opts->input, &audio);

    if (opts->percept)
    {
        /* the perceptual similarity map (src/percept.c): descriptors and the all-lag search only */
        MMXPercept P;
        int rc = mmx_percept_analyze(&audio, opts->quality ? opts->quality : 7, opts->verbose, &P);
        if (rc == 0)
            rc = mmx_percept_plan(&audio, &P, opts->percept_db, MMX_PERCEPT_MIN_LAG_S);
        if (rc != 0)
        {
            mmx_error("Perceptual map failed");
            mmx_audio_buffer_free(&audio);
            return 1;
        }
        if (opts->json)
        {
            printf("{\n  \"input\": \"%s\",\n  \"duration\": %.3f,\n", basename_of(opts->input), mmx_audio_buffer_duration_seconds(&audio));
            mmx_percept_print(&P, opts->percept_db, 1);
            printf("}\n");
        }
        else
            mmx_percept_print(&P, opts->percept_db, 0);
        mmx_percept_free(&P);
        mmx_audio_buffer_free(&audio);
        return 0;
    }
    if (mmx_codec_init(&codec, audio.sample_rate, audio.channels, opts->quality) != 0)
    {
        mmx_audio_buffer_free(&audio);
        return 1;
    }
    mmx_analysis_params_default(&params);
    params.level = opts->analysis;
    params.quality = opts->quality;
    params.max_depth = (unsigned char)opts->max_depth;
    params.verbose = opts->verbose;

    if (mmx_analyzer_run(&audio, &codec, &params, &a) != 0)
    {
        mmx_error("Analysis failed");
        mmx_codec_free(&codec);
        mmx_audio_buffer_free(&audio);
        return 1;
    }
    seconds = mmx_audio_buffer_duration_seconds(&audio);
    mmx_format_duration(seconds, dur, sizeof(dur));
    est_saving = a.est_audio_bits > 0 ? 100.0 * (1.0 - a.est_plan_bits / a.est_audio_bits) : 0.0;

    if (opts->json)
    {
        printf("{\n");
        printf("  \"input\": \"%s\",\n", basename_of(opts->input));
        printf("  \"duration\": %.3f,\n", seconds);
        printf("  \"sample_rate\": %lu,\n", audio.sample_rate);
        printf("  \"channels\": %u,\n", audio.channels);
        printf("  \"quality\": %u,\n", opts->quality);
        printf("  \"analysis_level\": %u,\n", params.level);
        printf("  \"frames\": %lu,\n", a.frame_count);
        printf("  \"sections\": %lu,\n", a.sections);
        printf("  \"candidates_tested\": %lu,\n", a.candidates_tested);
        printf("  \"candidates_aligned\": %lu,\n", a.candidates_aligned);
        printf("  \"planned_ref_frames\": %lu,\n", a.planned_ref_frames);
        printf("  \"planned_ref2_frames\": %lu,\n", a.planned_ref2_frames);
        printf("  \"planned_blocks\": %lu,\n", a.planned_blocks);
        printf("  \"silent_frames\": %lu,\n", a.silent_frames);
        printf("  \"transient_frames\": %lu,\n", a.transient_frames);
        printf("  \"estimated_audio_kbps\": %.2f,\n", a.est_audio_bits / seconds / 1000.0);
        printf("  \"estimated_mmx_kbps\": %.2f,\n", a.est_plan_bits / seconds / 1000.0);
        printf("  \"estimated_saving_percent\": %.2f,\n", est_saving);
        printf("  \"analysis_seconds\": %.2f\n", a.seconds);
        printf("}\n");
    }
    else
    {
        printf("\nMiniMix Analyzer\n\n");
        printf("Input:                          %s\n", opts->input);
        printf("Duration:                       %s\n", dur);
        printf("Sample Rate:                    %lu Hz\n", audio.sample_rate);
        printf("Channels:                       %u\n", audio.channels);
        printf("Quality / analysis level:       %u / %u\n", opts->quality, params.level);
        printf("Frames (1024 samples):          %lu\n", a.frame_count);
        printf("Detected sections:              %lu\n", a.sections);
        printf("Candidates tested / aligned:    %lu / %lu (max %lu per frame)\n", a.candidates_tested, a.candidates_aligned, a.cand_per_frame);
        printf("Planned reference frames:       %lu (%.1f %%), %lu with two sources\n", a.planned_ref_frames + a.planned_ref2_frames,
               100.0 * (a.planned_ref_frames + a.planned_ref2_frames) / a.frame_count, a.planned_ref2_frames);
        printf("Planned blocks:                 %lu\n", a.planned_blocks);
        printf("Silent frames:                  %lu (%.1f %%)\n", a.silent_frames, 100.0 * a.silent_frames / a.frame_count);
        printf("Transient frames:               %lu\n", a.transient_frames);
        printf("Estimated bitrate as audio:     %.1f kbit/s\n", a.est_audio_bits / seconds / 1000.0);
        printf("Estimated bitrate with plan:    %.1f kbit/s\n", a.est_plan_bits / seconds / 1000.0);
        printf("Estimated saving:               %.1f %%\n", est_saving);
        printf("Achievable with this search:    %.1f kbit/s (cheapest candidate per frame, %.1f %% saving)\n",
               a.est_best_bits / seconds / 1000.0, a.est_audio_bits > 0 ? 100.0 * (1.0 - a.est_best_bits / a.est_audio_bits) : 0.0);
        printf("Analysis time:                  %.1f s\n", a.seconds);
        print_timeline(&a, audio.sample_rate, 100);

        /* --- song decomposition --- */
        printf("\nSong decomposition\n");
        printf("  Repeated sections (0.75 s fingerprint windows, later occurrence <- source):\n");
        if (a.repeat_count == 0)
            printf("    none found\n");
        for (f = 0; f < a.repeat_count; f++)
        {
            const MMXRepeat *r = &a.repeats[f];
            printf("    %6.1f - %6.1f s  <-  %6.1f s  (lag %6.1f s, spectral distance %.2f)\n",
                   r->start_s, r->end_s, r->source_s, r->start_s - r->source_s, r->distance);
        }
        {
            /* how predictable is each frame from earlier material? histogram of the
               achievable prediction gain (audio bits / cheapest candidate bits) */
            unsigned long hist[6] = {0, 0, 0, 0, 0, 0}, counted = 0;
            double band_sum[MMX_EQ_BANDS], band_n = 0.0;
            unsigned int e;
            static const char *labels[6] = {"< 1.5x (unique)", "1.5 - 2x", "2 - 4x", "4 - 8x", "8 - 16x", "> 16x (near copy)"};
            for (e = 0; e < MMX_EQ_BANDS; e++) band_sum[e] = 0.0;
            for (f = 0; f < a.frame_count; f++)
            {
                const MMXFramePlan *p = &a.plan[f];
                double ratio;
                if (a.silent[f] || p->best_bits <= 0.0f) continue;
                ratio = p->audio_bits / p->best_bits;
                hist[ratio < 1.5 ? 0 : ratio < 2 ? 1 : ratio < 4 ? 2 : ratio < 8 ? 3 : ratio < 16 ? 4 : 5]++;
                counted++;
                if (p->n_sources)
                {
                    for (e = 0; e < MMX_EQ_BANDS; e++) band_sum[e] += p->band_gain_db[e];
                    band_n += 1.0;
                }
            }
            printf("  Predictability of frames from earlier material (bits as audio / bits with best reference):\n");
            for (e = 0; e < 6; e++)
                printf("    %-18s %6.1f %%\n", labels[e], counted ? 100.0 * hist[e] / counted : 0.0);
            printf("  Mean prediction gain of referenced frames per band (dB):\n    ");
            {
                static const char *bands[MMX_EQ_BANDS] = {"<200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k Hz"};
                for (e = 0; e < MMX_EQ_BANDS; e++)
                    printf("%s: %.1f  ", bands[e], band_n > 0 ? band_sum[e] / band_n : 0.0);
                printf("\n");
            }
        }

        if (opts->verbose)
        {
            unsigned long start = 0;
            printf("\nPlanned blocks:\n%10s %8s %6s %14s %14s %6s\n", "start s", "len s", "type", "source s", "source2 s", "depth");
            for (f = 1; f <= a.frame_count; f++)
            {
                const MMXFramePlan *p = &a.plan[start];
                int boundary = f == a.frame_count || a.plan[f].n_sources != p->n_sources ||
                               (p->n_sources && a.plan[f].src_start[0] != p->src_start[0] + (long long)(f - start) * MMX_HOP) ||
                               (p->n_sources > 1 && a.plan[f].src_start[1] != p->src_start[1] + (long long)(f - start) * MMX_HOP);
                if (!boundary)
                    continue;
                printf("%10.2f %8.2f %6s ", (double)start * MMX_HOP / audio.sample_rate, (double)(f - start) * MMX_HOP / audio.sample_rate,
                       p->n_sources == 0 ? "AUDIO" : p->n_sources == 1 ? "REF" : "REF2");
                if (p->n_sources) printf("%14.3f ", (double)(p->src_start[0] + MMX_HOP) / audio.sample_rate); else printf("%14s ", "-");
                if (p->n_sources > 1) printf("%14.3f ", (double)(p->src_start[1] + MMX_HOP) / audio.sample_rate); else printf("%14s ", "-");
                printf("%6u\n", p->depth);
                start = f;
            }
        }
    }

    mmx_analysis_free(&a);
    mmx_codec_free(&codec);
    mmx_audio_buffer_free(&audio);
    return 0;
}

typedef struct { unsigned long long start; unsigned long count; } RegionUse;

int mmx_cmd_info(const MMXOptions *opts)
{
    MMXFile file;
    unsigned long i, blocks_by[3] = {0, 0, 0}, frames_by[3] = {0, 0, 0};
    unsigned long long bytes_by[3] = {0, 0, 0}, total_payload = 0, file_size = 0, mp3;
    unsigned long nf, max_depth = 0;
    char dur[32];
    double seconds;
    FILE *fp;

    if (mmx_reader_read(opts->input, &file, 0) != 0)
        return 1;
    fp = fopen(opts->input, "rb");
    if (fp) { fseek(fp, 0, SEEK_END); file_size = (unsigned long long)ftell(fp); fclose(fp); }

    for (i = 0; i < file.block_count; i++)
    {
        const MMXBlockEntry *b = &file.blocks[i];
        unsigned int t = b->n_sources <= 2 ? b->n_sources : 2;
        blocks_by[t]++;
        frames_by[t] += b->frame_count;
        bytes_by[t] += b->payload_size;
        total_payload += b->payload_size;
        if (b->depth > max_depth) max_depth = b->depth;
    }
    nf = frames_by[0] + frames_by[1] + frames_by[2];
    seconds = file.sample_rate ? (double)file.frame_count / file.sample_rate : 0.0;
    mmx_format_duration(seconds, dur, sizeof(dur));
    mp3 = mmx_statistics_cbr_bytes(seconds, 320.0);

    if (opts->json)
    {
        printf("{\n");
        printf("  \"file\": \"%s\",\n", basename_of(opts->input));
        printf("  \"format_version\": %d,\n", MMX_FORMAT_VERSION);
        printf("  \"file_bytes\": %llu,\n", file_size);
        printf("  \"kbps\": %.2f,\n", seconds > 0 ? file_size * 8.0 / seconds / 1000.0 : 0.0);
        printf("  \"sample_rate\": %lu,\n", file.sample_rate);
        printf("  \"channels\": %u,\n", file.channels);
        printf("  \"source_bits\": %u,\n", file.source_bits);
        printf("  \"samples\": %llu,\n", file.frame_count);
        printf("  \"duration\": %.3f,\n", seconds);
        printf("  \"quality\": %u,\n", file.quality);
        printf("  \"lossless\": %s,\n", file.codec_id == MMX_CODEC_LOSSLESS ? "true" : "false");
        printf("  \"analysis_level\": %u,\n", file.analysis_level);
        printf("  \"max_ref_depth\": %u,\n", file.max_ref_depth);
        printf("  \"blocks\": %lu,\n", file.block_count);
        printf("  \"frames\": %lu,\n", nf);
        printf("  \"audio_frames\": %lu,\n", frames_by[0]);
        printf("  \"ref_frames\": %lu,\n", frames_by[1]);
        printf("  \"ref2_frames\": %lu,\n", frames_by[2]);
        printf("  \"audio_bytes\": %llu,\n", bytes_by[0]);
        printf("  \"residual_bytes\": %llu,\n", bytes_by[1] + bytes_by[2]);
        printf("  \"table_bytes\": %llu,\n", file_size - total_payload);
        printf("  \"referenced_percent\": %.2f,\n", nf ? 100.0 * (frames_by[1] + frames_by[2]) / nf : 0.0);
        printf("  \"mp3_320_bytes\": %llu,\n", mp3);
        printf("  \"reduction_vs_mp3_percent\": %.2f,\n", mp3 ? 100.0 * (1.0 - (double)file_size / mp3) : 0.0);
        printf("  \"cover_bytes\": %lu\n", file.cover_len);
        printf("}\n");
    }
    else
    {
        char *title = mmx_file_metadata_get(&file, "title"), *artist = mmx_file_metadata_get(&file, "artist");
        printf("MiniMix Inspector\n\n");
        printf("File:            %s (%llu bytes, %.1f kbit/s)\n", opts->input, file_size, seconds > 0 ? file_size * 8.0 / seconds / 1000.0 : 0.0);
        printf("Format:          MMX version %d, codec %u (%s), bitstream revision %u, hop %lu\n", MMX_FORMAT_VERSION, file.codec_id,
               file.codec_id == MMX_CODEC_LOSSLESS ? "lossless integer" : "native MDCT", file.bitstream_rev, file.hop);
        if (file.bwe_hz)
            printf("Band replication: from %u Hz, patch mapping %u (%s)\n", file.bwe_hz, file.bwe_mode,
                   file.bwe_mode ? "linear shift" : "octave transposition");
        if (title || artist)
            printf("Track:           %s - %s\n", artist ? artist : "?", title ? title : "?");
        printf("Duration:        %s (%llu samples)\n", dur, file.frame_count);
        printf("Sample Rate:     %lu Hz, %u channels, source %u bit\n", file.sample_rate, file.channels, file.source_bits);
        if (file.quality == 0)
        {
            int shift = file.ll_shift, drop = file.ll_dropped, qd = file.ll_qdrop > 1 ? file.ll_qdrop : 1;
            if (qd > 1)
                printf("Quality:         NEAR-LOSSLESS - input rounded to steps of %d = 2^%.2f LSB (%d bits + x%d; %d of %u bits kept), analysis level %u, max reference depth %u (used %lu)\n",
                       (1 << drop) * qd, log((double)(1 << drop) * qd) / log(2.0), drop, qd, (int)file.source_bits - shift, file.source_bits, file.analysis_level, file.max_ref_depth, max_depth);
            else if (drop > 0)
                printf("Quality:         NEAR-LOSSLESS - input rounded by %d bit (max error +-%d LSB, %d of %u bits kept), analysis level %u, max reference depth %u (used %lu)\n",
                       drop, 1 << (drop - 1), (int)file.source_bits - shift, file.source_bits, file.analysis_level, file.max_ref_depth, max_depth);
            else if (shift > 0)
                printf("Quality:         lossless (bit-exact; %d of %u bits carry signal, the rest were zero), analysis level %u, max reference depth %u (used %lu)\n",
                       (int)file.source_bits - shift, file.source_bits, file.analysis_level, file.max_ref_depth, max_depth);
            else
                printf("Quality:         lossless (bit-exact), analysis level %u, max reference depth %u (used %lu)\n", file.analysis_level, file.max_ref_depth, max_depth);
        }
        else
            printf("Quality:         %u, analysis level %u, max reference depth %u (used %lu)\n", file.quality, file.analysis_level, file.max_ref_depth, max_depth);
        printf("Blocks:          %lu (%lu frames)\n", file.block_count, nf);
        printf("  %-12s %8lu blocks %8lu frames %12llu bytes\n", "AUDIO", blocks_by[0], frames_by[0], bytes_by[0]);
        printf("  %-12s %8lu blocks %8lu frames %12llu bytes\n", "REF", blocks_by[1], frames_by[1], bytes_by[1]);
        printf("  %-12s %8lu blocks %8lu frames %12llu bytes\n", "REF2", blocks_by[2], frames_by[2], bytes_by[2]);
        printf("  %-12s %8s %8s %12llu bytes (header, tables, metadata%s)\n", "Overhead", "", "", file_size - total_payload, file.cover_len ? ", cover" : "");
        printf("Referenced:      %.1f %% of frames\n", nf ? 100.0 * (frames_by[1] + frames_by[2]) / nf : 0.0);
        if (file.quality == 0)
        {
            unsigned long long pcm = file.frame_count * file.channels * (((file.source_bits ? file.source_bits : 16) + 7) / 8);
            printf("Size:            %.1f %% of the PCM (%llu bytes)\n", pcm ? 100.0 * file_size / pcm : 0.0, pcm);
        }
        else
            printf("Size comparison: MP3 320 equivalent %llu bytes -> MMX is %.1f %% of it (reduction %.1f %%)\n", mp3,
                   mp3 ? 100.0 * file_size / mp3 : 0.0, mp3 ? 100.0 * (1.0 - (double)file_size / mp3) : 0.0);
        printf("\nSavings visualization (share of file)\n");
        {
            const char *labels[4] = {"New audio", "Residuals", "Residuals (2)", "Overhead"};
            unsigned long long parts[4];
            int t, n, j;
            parts[0] = bytes_by[0]; parts[1] = bytes_by[1]; parts[2] = bytes_by[2]; parts[3] = file_size - total_payload;
            for (t = 0; t < 4; t++)
            {
                n = file_size ? (int)(40.0 * parts[t] / file_size + 0.5) : 0;
                printf("  %-14s ", labels[t]);
                for (j = 0; j < n; j++) fputs("\xe2\x96\x88", stdout);
                printf(" %5.1f %%\n", file_size ? 100.0 * parts[t] / file_size : 0.0);
            }
        }
        if (file.metadata)
            printf("\nMetadata:\n%s", file.metadata);
        if (file.cover_len)
            printf("Cover:           %s, %lu bytes\n", file.cover_mime, file.cover_len);
        free(title); free(artist);

        if (opts->verbose)
        {
            RegionUse *use = (RegionUse *)calloc(file.block_count ? file.block_count : 1, sizeof(RegionUse));
            unsigned long nuse = 0, j;
            printf("\nTimeline:\n%6s %10s %8s %6s %5s %12s %12s %10s\n", "block", "start s", "len s", "type", "depth", "source s", "source2 s", "bytes");
            for (i = 0; i < file.block_count; i++)
            {
                const MMXBlockEntry *b = &file.blocks[i];
                printf("%6lu %10.2f %8.2f %6s %5u ", i, (double)b->start_frame * file.hop / file.sample_rate,
                       (double)b->frame_count * file.hop / file.sample_rate, mmx_block_type_name(b), b->depth);
                if (b->n_sources) printf("%12.3f ", (double)(b->src_start[0] + (long long)file.hop) / file.sample_rate); else printf("%12s ", "-");
                if (b->n_sources > 1) printf("%12.3f ", (double)(b->src_start[1] + (long long)file.hop) / file.sample_rate); else printf("%12s ", "-");
                printf("%10lu\n", b->payload_size);
                /* most referenced regions: cluster source starts to 1 s */
                if (b->n_sources && use)
                {
                    unsigned long long key = (unsigned long long)(b->src_start[0] + (long long)file.hop) / file.sample_rate;
                    for (j = 0; j < nuse; j++)
                        if (use[j].start == key) { use[j].count += b->frame_count; break; }
                    if (j == nuse) { use[nuse].start = key; use[nuse].count = b->frame_count; nuse++; }
                }
            }
            if (use && nuse)
            {
                unsigned long top;
                printf("\nMost referenced regions (source second -> referencing frames):\n");
                for (top = 0; top < 8 && top < nuse; top++)
                {
                    unsigned long best = top;
                    RegionUse tmp;
                    for (j = top + 1; j < nuse; j++) if (use[j].count > use[best].count) best = j;
                    tmp = use[top]; use[top] = use[best]; use[best] = tmp;
                    printf("  %6llu s  %8lu frames (%.1f s)\n", use[top].start, use[top].count, (double)use[top].count * file.hop / file.sample_rate);
                }
            }
            free(use);
        }
    }

    mmx_file_free(&file);
    return 0;
}

/* The EQ band regions of the gain bands (psymodel.c), as the compare report names them. */
static const char *eq_region_name[MMX_EQ_BANDS] = { "<200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k" };

int mmx_cmd_compare(const MMXOptions *opts)
{
    MMXAudioBuffer a, b;
    MMXQualityReport r;

    if (mmx_wav_read(opts->input, &a) != 0)
        return 1;
    if (mmx_wav_read(opts->output, &b) != 0)
    {
        mmx_audio_buffer_free(&a);
        return 1;
    }
    if (mmx_quality_compare(&a, &b, opts->quality ? opts->quality : MMX_QUALITY_MAX, opts->maxfreq, 6.0, &r) != 0)
    {
        mmx_error("Format mismatch: %lu Hz/%u ch/%llu frames vs %lu Hz/%u ch/%llu frames",
                  a.sample_rate, a.channels, a.frame_count, b.sample_rate, b.channels, b.frame_count);
        mmx_audio_buffer_free(&a);
        mmx_audio_buffer_free(&b);
        return 1;
    }
    if (opts->json)
    {
        unsigned int e;
        printf("{ \"frames\": %lu, \"worst_nmr_db\": %.2f, \"mean_nmr_db\": %.2f, \"bands_over_percent\": %.3f, \"frames_over_percent\": %.3f, \"attacks\": %lu, \"attacks_pre_echo\": %lu, \"worst_pre_attack_snr_db\": %.1f, \"snr_db\": %.2f, \"max_diff_lsb\": %.1f, \"passed\": %s,\n  \"eq\": [",
               r.frames, r.worst_nmr_db, r.mean_nmr_db, r.bands_over_percent, r.frames_over_percent, r.attacks, r.attacks_pre_echo,
               r.worst_pre_attack_snr_db, r.snr_db, r.max_diff_lsb, r.passed ? "true" : "false");
        for (e = 0; e < MMX_EQ_BANDS; e++)
            printf("%s\n    { \"hz\": \"%s\", \"cells\": %lu, \"over_percent\": %.3f, \"worst_nmr_db\": %.2f, \"mean_nmr_db\": %.2f, \"level_abs_db\": %.2f, \"level_bias_db\": %.2f }",
                   e ? "," : "", eq_region_name[e], r.eq_cells[e], r.eq_over_percent[e], r.eq_worst_nmr_db[e], r.eq_mean_nmr_db[e],
                   r.eq_level_abs_db[e], r.eq_level_bias_db[e]);
        printf(" ],\n  \"weighted\": { \"crossover_hz\": %u, \"over_percent\": %.3f, \"mean_nmr_db\": %.2f, \"noise_loudness_percent\": %.2f,\n"
               "    \"hi_energy\": { \"cells\": %lu, \"err_db\": %.2f, \"bias_db\": %.2f, \"over_3db_percent\": %.2f, \"worst_db\": %.1f },\n    \"eq\": [",
               r.w_crossover_hz, r.w_over_percent, r.w_mean_nmr_db, r.w_noise_loudness_percent,
               r.hi_energy_cells, r.hi_energy_err_db, r.hi_energy_bias_db, r.hi_energy_over3_percent, r.hi_energy_worst_db);
        for (e = 0; e < MMX_EQ_BANDS; e++)
            printf("%s { \"hz\": \"%s\", \"over_percent\": %.3f, \"weight_percent\": %.2f }",
                   e ? "," : "", eq_region_name[e], r.eq_w_over_percent[e], r.eq_weight_percent[e]);
        printf(" ] },\n  \"stereo\": { \"ild_err_db\": %.3f, \"coh_err\": %.4f, \"hi_ild_err_db\": %.3f, \"hi_coh_err\": %.4f, \"eq\": [",
               r.stereo_ild_err_db_all, r.stereo_coh_err_all, r.stereo_ild_err_db_hi, r.stereo_coh_err_hi);
        for (e = 0; e < MMX_EQ_BANDS; e++)
            printf("%s { \"hz\": \"%s\", \"cells\": %lu, \"ild_err_db\": %.3f, \"ild_over_2db_percent\": %.2f, \"coh_err\": %.4f, \"coh_over_0.3_percent\": %.2f }",
                   e ? "," : "", eq_region_name[e], r.stereo_cells[e], r.stereo_ild_err_db[e], r.stereo_ild_over_percent[e], r.stereo_coh_err[e], r.stereo_coh_over_percent[e]);
        printf(" ] },\n  \"warble\": { \"warble_db\": %.2f, \"hole_percent\": %.2f, \"toggles_per_second\": %.2f, \"warble_worst_db\": %.2f, \"warble_worst_seconds\": %.0f, \"toggles_worst\": %.0f, \"toggles_worst_seconds\": %.0f },\n"
               "  \"beyond_model\": { \"clipped_decoded\": %lu, \"clipped_original\": %lu, \"clipped_new\": %lu, \"attacks_ghost\": %lu, \"worst_ghost_db\": %.1f, \"worst_ghost_seconds\": %.3f, \"dc_worst_dbfs\": %.1f, \"border_1024_db\": %.2f, \"border_128_db\": %.2f, \"level_over_1db_percent\": %.2f, \"level_over_3db_percent\": %.2f, \"level_worst_db\": %.2f, \"level_worst_seconds\": %.0f, \"level_worst_region\": \"%s\" } }\n",
               r.warble_db, r.hole_percent, r.toggles_per_second, r.warble_worst_db, r.warble_worst_seconds, r.toggles_worst, r.toggles_worst_seconds,
               r.clipped_decoded, r.clipped_original, r.clipped_new, r.attacks_ghost, r.worst_ghost_db, r.worst_ghost_seconds, r.dc_worst_db,
               r.border_1024_db, r.border_128_db, r.level_over_1db_percent, r.level_over_3db_percent, r.level_worst_db, r.level_worst_seconds,
               eq_region_name[r.level_worst_band]);
    }
    else
    {
        unsigned int e;
        printf("compare (quality %u masking model):\n", opts->quality);
        printf("  worst band NMR:      %.2f dB   (sub-frame analysis, 11.6 ms; 0 dB = masking threshold of the model)\n", r.worst_nmr_db);
        printf("  mean band NMR:       %.2f dB\n", r.mean_nmr_db);
        printf("  bands above 0 dB:    %.3f %%   (worst at %.2f s, channel %u, band %u)\n", r.bands_over_percent,
               ((double)r.worst_frame - 2.0) * 512 / a.sample_rate, r.worst_channel, r.worst_band);
        printf("  sub-frames above 0 dB: %.3f %%\n", r.frames_over_percent);
        printf("  weighted (ERB x loudness, band energy above %u Hz):\n", r.w_crossover_hz);
        printf("    weighted error share: %.2f %%   (the same cells and the same 0 dB criterion, each cell weighted\n", r.w_over_percent);
        printf("                                     by its ERB width times its Zwicker specific loudness)\n");
        printf("    weighted mean NMR:   %.2f dB\n", r.w_mean_nmr_db);
        printf("    noise loudness:      %.2f %% of the loudness of the music (unmasked noise, Zwicker 0.23)\n", r.w_noise_loudness_percent);
        printf("    high band energy:    %.2f dB mean deviation above %u Hz (bias %+.2f dB, %.1f %% of the cells off by > 3 dB, worst %+.1f dB)\n",
               r.hi_energy_err_db, r.w_crossover_hz, r.hi_energy_bias_db, r.hi_energy_over3_percent, r.hi_energy_worst_db);
        printf("  pre-echo:            %lu attacks, %lu with pre-attack SNR < 10 dB, worst %.1f dB at %.2f s\n",
               r.attacks, r.attacks_pre_echo, r.worst_pre_attack_snr_db, r.worst_pre_attack_seconds);
        printf("  sample SNR:          %.2f dB, max diff %.1f LSB\n", r.snr_db, r.max_diff_lsb);
        if (opts->regions)
        {
            printf("  regions              ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%9s", eq_region_name[e]);
            printf("\n    over %% unweighted: ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.eq_over_percent[e]);
            printf("\n    over %% weighted:   ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.eq_w_over_percent[e]);
            printf("\n    perceptual weight: ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.eq_weight_percent[e]);
            printf("\n    mean band NMR:     ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%7.1f dB", r.eq_mean_nmr_db[e]);
            printf("\n");
        }
        printf("  per EQ region:       ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%9s", eq_region_name[e]);
        printf("\n    bands above 0 dB:  ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.eq_over_percent[e]);
        printf("\n    worst band NMR:    ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%7.1f dB", r.eq_worst_nmr_db[e]);
        printf("\n    level error:       ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%7.2f dB", r.eq_level_abs_db[e]);
        printf("\n    level bias:        ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%+7.2f dB", r.eq_level_bias_db[e]);
        printf("\n    mean band NMR:     ");
        for (e = 0; e < MMX_EQ_BANDS; e++) printf("%7.1f dB", r.eq_mean_nmr_db[e]);
        if (a.channels == 2)
        {
            printf("\n    stereo: ILD error  ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%7.2f dB", r.stereo_ild_err_db[e]);
            printf("\n      ... > 2 dB       ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.stereo_ild_over_percent[e]);
            printf("\n    coherence error    ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%10.3f", r.stereo_coh_err[e]);
            printf("\n      ... > 0.3        ");
            for (e = 0; e < MMX_EQ_BANDS; e++) printf("%8.2f%%", r.stereo_coh_over_percent[e]);
            printf("\n  stereo image:        L/R level ratio off by %.2f dB on average (%.2f dB from 4 kHz up), coherence off by %.3f (%.3f from 4 kHz up); cells with signal in both channels",
                   r.stereo_ild_err_db_all, r.stereo_ild_err_db_hi, r.stereo_coh_err_all, r.stereo_coh_err_hi);
        }
        printf("\n  beyond the model:    full scale %lu samples decoded (%lu original, %lu new flat tops); %lu attacks with a 5.8 ms block before them where the error exceeds the signal (worst %+.1f dB at %.3f s)\n",
               r.clipped_decoded, r.clipped_original, r.clipped_new, r.attacks_ghost, r.worst_ghost_db, r.worst_ghost_seconds);
        printf("                       DC of the error per second worst %.1f dBFS; error steps at the 1024 / 128 grid %+.2f / %+.2f dB; band level per second off by > 1 dB in %.2f %% of the cells, > 3 dB in %.2f %% (worst %+.1f dB at %.0f s, %s Hz)\n",
               r.dc_worst_db, r.border_1024_db, r.border_128_db, r.level_over_1db_percent, r.level_over_3db_percent, r.level_worst_db,
               r.level_worst_seconds, eq_region_name[r.level_worst_band]);
        printf("  gurgling:            warble index %.2f dB (worst second %.2f dB at %.0f s), holes %.2f %% of the cells, %.2f band toggles per second (worst %.0f at %.0f s)\n",
               r.warble_db, r.warble_worst_db, r.warble_worst_seconds, r.hole_percent, r.toggles_per_second, r.toggles_worst, r.toggles_worst_seconds);
        printf("  verdict:             %s\n", r.passed ? "PASS" : "FAIL (above tolerance)");
    }
    mmx_audio_buffer_free(&a);
    mmx_audio_buffer_free(&b);
    return r.passed ? 0 : 3;
}

/* Adds or replaces metadata lines in an existing file without touching the audio. */
int mmx_cmd_tag(const MMXOptions *opts)
{
    MMXFile file;
    char *text;
    size_t cap, len = 0;
    unsigned int i;
    unsigned long long written;
    const char *p;

    if (opts->meta_count == 0 && !opts->cover)
    {
        mmx_error("tag needs --meta key=value (or --title/--artist/...) or --cover");
        return 2;
    }
    if (mmx_reader_read(opts->input, &file, 1) != 0)
        return 1;

    cap = (file.metadata ? strlen(file.metadata) : 0) + (size_t)opts->meta_count * 520 + 16;
    text = (char *)malloc(cap);
    if (!text)
    {
        mmx_file_free(&file);
        return 1;
    }
    text[0] = '\0';
    /* keep old lines whose key is not replaced */
    p = file.metadata ? file.metadata : "";
    while (*p)
    {
        const char *eol = strchr(p, '\n');
        size_t line_len = eol ? (size_t)(eol - p) : strlen(p);
        const char *eq = memchr(p, '=', line_len);
        int replaced = 0;
        if (eq)
            for (i = 0; i < opts->meta_count; i++)
            {
                const char *meq = strchr(opts->meta[i], '=');
                if (meq && (size_t)(meq - opts->meta[i]) == (size_t)(eq - p) && strncmp(opts->meta[i], p, (size_t)(eq - p)) == 0)
                    replaced = 1;
            }
        if (!replaced && line_len)
        {
            memcpy(text + len, p, line_len);
            len += line_len;
            text[len++] = '\n';
            text[len] = '\0';
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    for (i = 0; i < opts->meta_count; i++)
    {
        size_t l = strlen(opts->meta[i]);
        memcpy(text + len, opts->meta[i], l);
        len += l;
        text[len++] = '\n';
        text[len] = '\0';
    }
    mmx_file_set_metadata(&file, text);
    free(text);
    if (opts->cover)
    {
        unsigned long cover_len = 0;
        unsigned char *cover = read_file(opts->cover, &cover_len);
        if (!cover)
        {
            mmx_error("Cannot read cover image: %s", opts->cover);
            mmx_file_free(&file);
            return 1;
        }
        mmx_file_set_cover(&file, mime_for(opts->cover), cover, cover_len);
        free(cover);
    }
    if (mmx_writer_write(opts->input, &file, &written) != 0)
    {
        mmx_file_free(&file);
        return 1;
    }
    mmx_info("Tagged %s (%llu bytes)", opts->input, written);
    mmx_file_free(&file);
    return 0;
}
