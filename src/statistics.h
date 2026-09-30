#ifndef MMX_STATISTICS_H
#define MMX_STATISTICS_H

typedef struct
{
    unsigned long long input_bytes;        /* PCM size at source bit depth */
    unsigned long long output_bytes;       /* final .mmx size */
    unsigned long long baseline_bytes;     /* same codec, AUDIO-only */
    unsigned long long table_bytes;        /* container overhead */
    unsigned long long payload_bytes[3];   /* by number of sources 0/1/2 */
    unsigned long frames_by_sources[3];
    unsigned long total_frames;
    unsigned long block_count;
    unsigned long ref_frames_gains_off;    /* frames inside REF blocks coded without prediction */
    unsigned int max_depth_used;
    unsigned long depth_histogram[8];
    double analysis_seconds;
    double encoding_seconds;
    double worst_nmr_db;
    double mean_nmr_db;
    unsigned long frames_over_threshold;
    double duration_seconds;
    unsigned long stereo_ms_frames;
    unsigned long transient_frames;
    unsigned long tns_frames;
    unsigned long pure_ref_frames;   /* reuse: played from the reference without residual */
    unsigned long short_frames;      /* block switching: frames coded with eight short transforms */
    unsigned long pure_bands;        /* tracker: channel bands played from the source without residual */
    unsigned long coded_bands;       /* tracker: channel bands of referenced frames (denominator) */
    unsigned long pure_vetoed;       /* tracker: pure candidates rejected by the sub-window (pre-echo) check */
    unsigned long pns_bands;         /* noise substitution: channel bands carrying only their level */
    unsigned long pns_candidates;    /* ... of the coded channel bands in the substitutable range */
    unsigned long bwe_bands;         /* band replication: channel bands regenerated from the octave below the crossover */
    unsigned long bwe_bands_zero;    /* ... of them written as zero bands (no energy) */
    unsigned long coded_bands_total; /* all coded channel bands (long frames) */
    unsigned long is_bands;          /* intensity stereo: bands coded once with a position */
    unsigned long is_candidates;     /* ... of the bands in the intensity range coded in at least one channel */
    unsigned long nf_regions;        /* noise filling: channel regions of long frames carrying a fill level */
    unsigned long nf_regions_total;  /* ... of the channel regions in the fillable range */
    unsigned long nf_hole_bands;     /* noise filling: zeroed channel bands that keep their energy as a noise band */
    unsigned long keep_bands;        /* energy preservation: zeroed channel bands of long frames given a coefficient
                                        back (see encoder.c; the short frames' groups are not counted) */
    unsigned long keep_noise_bands;  /* ... given their energy as a noise band */
    /* clip guard: the float reconstruction against the full-scale range of the sink */
    unsigned long clip_samples_raw;  /* samples outside the range after the first quantization */
    unsigned long clip_samples;      /* ... in the final reconstruction */
    unsigned long clip_samples_quiet;/* ... outside the range but the clamp stays under the masking model (left alone) */
    unsigned long clip_frames;       /* frames quantized again with tightened thresholds */
    unsigned long clip_frames_left;  /* ... whose clamp the model still hears after the last step */
    double clip_peak;                /* largest |sample| of the final reconstruction (1.0 = full scale) */
} MMXStatistics;

void mmx_statistics_init(MMXStatistics *s);
void mmx_statistics_print(const MMXStatistics *s);
void mmx_statistics_print_json(const MMXStatistics *s);

/* Equivalent size of a CBR file at `kbps` for the track duration. */
unsigned long long mmx_statistics_cbr_bytes(double seconds, double kbps);

#endif
