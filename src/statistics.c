#include <stdio.h>
#include <string.h>
#include <math.h>
#include "statistics.h"

void mmx_statistics_init(MMXStatistics *s)
{
    memset(s, 0, sizeof(*s));
    s->worst_nmr_db = -200.0;
}

unsigned long long mmx_statistics_cbr_bytes(double seconds, double kbps)
{
    return (unsigned long long)(seconds * kbps * 1000.0 / 8.0);
}

static double pct(unsigned long long part, unsigned long long whole)
{
    return whole ? 100.0 * (double)part / (double)whole : 0.0;
}

static double kbps_of(unsigned long long bytes, double seconds)
{
    return seconds > 0.0 ? (double)bytes * 8.0 / seconds / 1000.0 : 0.0;
}

static void bar(const char *label, unsigned long long part, unsigned long long whole)
{
    int n = whole ? (int)(40.0 * (double)part / (double)whole + 0.5) : 0, i;
    printf("  %-18s ", label);
    for (i = 0; i < n; i++) fputs("\xe2\x96\x88", stdout);
    printf(" %5.1f %%  %llu bytes\n", pct(part, whole), part);
}

void mmx_statistics_print(const MMXStatistics *s)
{
    unsigned long long mp3 = mmx_statistics_cbr_bytes(s->duration_seconds, 320.0);
    unsigned long long total_payload = s->payload_bytes[0] + s->payload_bytes[1] + s->payload_bytes[2];
    int d;

    printf("\nCompression statistics\n");
    printf("  Input PCM:         %12llu bytes\n", s->input_bytes);
    printf("  Output MMX:        %12llu bytes  (%.1f %% of PCM, %.1f kbit/s)\n",
           s->output_bytes, pct(s->output_bytes, s->input_bytes), kbps_of(s->output_bytes, s->duration_seconds));
    if (s->worst_nmr_db > -199.0)       /* lossy only: lossless has no MP3 counterpart and no baseline pass */
    {
        printf("  MP3 320 equivalent:%12llu bytes  -> MMX is %.1f %% of MP3 320 (reduction %.1f %%)\n",
               mp3, pct(s->output_bytes, mp3), 100.0 - pct(s->output_bytes, mp3));
        printf("  AUDIO-only baseline (same codec, no references): %llu bytes  -> savings from global analysis %.1f %%\n",
               s->baseline_bytes, s->baseline_bytes ? 100.0 * (1.0 - (double)total_payload / (double)s->baseline_bytes) : 0.0);
    }
    printf("\n  Savings visualization (share of output)\n");
    bar("New audio", s->payload_bytes[0], s->output_bytes);
    bar("Residuals (1 src)", s->payload_bytes[1], s->output_bytes);
    bar("Residuals (2 src)", s->payload_bytes[2], s->output_bytes);
    bar("Tables/metadata", s->table_bytes, s->output_bytes);
    printf("\n  Frames:            %lu total, %lu AUDIO, %lu REF, %lu REF2 (%.1f %% referenced), %lu inside REF blocks coded without prediction\n",
           s->total_frames, s->frames_by_sources[0], s->frames_by_sources[1], s->frames_by_sources[2],
           pct(s->frames_by_sources[1] + s->frames_by_sources[2], s->total_frames), s->ref_frames_gains_off);
    printf("  Blocks:            %lu (avg %.1f frames)\n", s->block_count,
           s->block_count ? (double)s->total_frames / s->block_count : 0.0);
    printf("  Reference depth:   max %u, histogram", s->max_depth_used);
    for (d = 0; d < 8 && (unsigned)d <= s->max_depth_used; d++)
        printf(" d%d=%lu", d, s->depth_histogram[d]);
    printf("\n  Stereo:            %lu frames M/S, %lu transient frames, %lu frames with TNS, %lu pure reference frames (reuse), %lu short-block frames\n", s->stereo_ms_frames, s->transient_frames, s->tns_frames, s->pure_ref_frames, s->short_frames);
    if (s->coded_bands)
        printf("  Tracker:           %lu of %lu channel bands of referenced frames played from the source without residual (%.1f %%), %lu frames completely, %lu EQ bands vetoed by the sub-window check\n",
               s->pure_bands, s->coded_bands, 100.0 * (double)s->pure_bands / (double)s->coded_bands, s->pure_ref_frames, s->pure_vetoed);
    if (s->bwe_bands)
        printf("  Replicated bands:  %lu channel bands above the replication crossover carry only their level and mix index, %lu of them silent\n",
               s->bwe_bands, s->bwe_bands_zero);
    if (s->pns_candidates)
        printf("  Noise bands:       %lu of %lu coded channel bands in the substitutable range play as noise of the same energy (%.1f %%, %.1f %% of all %lu coded bands)\n",
               s->pns_bands, s->pns_candidates, 100.0 * (double)s->pns_bands / (double)s->pns_candidates,
               s->coded_bands_total ? 100.0 * (double)s->pns_bands / (double)s->coded_bands_total : 0.0, s->coded_bands_total);
    if (s->is_candidates)
        printf("  Intensity bands:   %lu of %lu coded bands in the intensity range carry the mid once with a position (%.1f %%)\n",
               s->is_bands, s->is_candidates, 100.0 * (double)s->is_bands / (double)s->is_candidates);
    if (s->nf_regions_total)
        printf("  Noise filling:     %lu of %lu channel regions of long frames fill their zeros with noise (%.1f %%), %lu zeroed channel bands keep their energy as noise bands\n",
               s->nf_regions, s->nf_regions_total, 100.0 * (double)s->nf_regions / (double)s->nf_regions_total, s->nf_hole_bands);
    if (s->keep_bands || s->keep_noise_bands)
        printf("  Energy kept:       %lu channel bands the quantizer had zeroed above the model's own threshold carry a coefficient again, %lu carry their energy as a noise band (%.1f %% of the coded channel bands)\n",
               s->keep_bands, s->keep_noise_bands,
               s->coded_bands_total ? 100.0 * (double)(s->keep_bands + s->keep_noise_bands) / (double)s->coded_bands_total : 0.0);
    if (s->clip_samples_raw || s->clip_samples || s->clip_frames)
        printf("  Full scale:        %lu samples of the reconstruction outside the range of the sink (%lu before the clip guard, %lu of them inaudible by the model, peak %+.2f dBFS), %lu frames quantized finer, %lu still audible\n",
               s->clip_samples, s->clip_samples_raw, s->clip_samples_quiet, s->clip_peak > 0.0 ? 20.0 * log10(s->clip_peak) : -200.0, s->clip_frames, s->clip_frames_left);
    if (s->worst_nmr_db <= -199.0)
        printf("  Quality:           lossless (bit-exact)\n");
    else
        printf("  Quality:           worst band NMR %.2f dB, mean %.2f dB, %lu frames above 0 dB\n",
               s->worst_nmr_db, s->mean_nmr_db, s->frames_over_threshold);
    printf("  Time:              analysis %.1f s, encoding %.1f s (%.2fx realtime)\n", s->analysis_seconds, s->encoding_seconds,
           s->duration_seconds > 0.0 ? s->duration_seconds / (s->analysis_seconds + s->encoding_seconds + 1e-9) : 0.0);
}

void mmx_statistics_print_json(const MMXStatistics *s)
{
    unsigned long long total_payload = s->payload_bytes[0] + s->payload_bytes[1] + s->payload_bytes[2];
    printf("{\n");
    printf("  \"input_bytes\": %llu,\n", s->input_bytes);
    printf("  \"output_bytes\": %llu,\n", s->output_bytes);
    printf("  \"baseline_bytes\": %llu,\n", s->baseline_bytes);
    printf("  \"global_savings_percent\": %.3f,\n", s->baseline_bytes ? 100.0 * (1.0 - (double)total_payload / (double)s->baseline_bytes) : 0.0);
    printf("  \"mp3_320_bytes\": %llu,\n", mmx_statistics_cbr_bytes(s->duration_seconds, 320.0));
    printf("  \"kbps\": %.2f,\n", kbps_of(s->output_bytes, s->duration_seconds));
    printf("  \"duration\": %.3f,\n", s->duration_seconds);
    printf("  \"frames\": %lu,\n", s->total_frames);
    printf("  \"audio_frames\": %lu,\n", s->frames_by_sources[0]);
    printf("  \"ref_frames\": %lu,\n", s->frames_by_sources[1]);
    printf("  \"ref2_frames\": %lu,\n", s->frames_by_sources[2]);
    printf("  \"blocks\": %lu,\n", s->block_count);
    printf("  \"max_depth_used\": %u,\n", s->max_depth_used);
    printf("  \"worst_nmr_db\": %.3f,\n", s->worst_nmr_db);
    printf("  \"mean_nmr_db\": %.3f,\n", s->mean_nmr_db);
    printf("  \"frames_over_threshold\": %lu,\n", s->frames_over_threshold);
    printf("  \"keep_bands\": %lu,\n", s->keep_bands);
    printf("  \"keep_noise_bands\": %lu,\n", s->keep_noise_bands);
    printf("  \"coded_bands_total\": %lu,\n", s->coded_bands_total);
    printf("  \"clip_samples\": %lu,\n", s->clip_samples);
    printf("  \"clip_samples_raw\": %lu,\n", s->clip_samples_raw);
    printf("  \"clip_samples_quiet\": %lu,\n", s->clip_samples_quiet);
    printf("  \"clip_frames\": %lu,\n", s->clip_frames);
    printf("  \"clip_frames_left\": %lu,\n", s->clip_frames_left);
    printf("  \"clip_peak_dbfs\": %.3f,\n", s->clip_peak > 0.0 ? 20.0 * log10(s->clip_peak) : -200.0);
    printf("  \"analysis_seconds\": %.3f,\n", s->analysis_seconds);
    printf("  \"encoding_seconds\": %.3f\n", s->encoding_seconds);
    printf("}\n");
}
