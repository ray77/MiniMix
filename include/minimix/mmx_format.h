#ifndef MMX_FORMAT_H
#define MMX_FORMAT_H

/* On-disk MMX container structures, format version 2 ("MMX2").
   See docs/MMX_Format_Specification.md. */

#define MMX_MAGIC "MMX2"
#define MMX_FORMAT_VERSION 2
#define MMX_BITSTREAM_REVISION 10  /* 10 = lossless core LL2 (header byte 87 = 2, src/ll2codec.c): OLS + sign-LMS bank + mixer in the signal domain, source stage on the whitened reference, bitplane residual coder; files with byte 87 = 0 decode with the revision 8 lossless core. 9 = low-rate energy-preserving bands (EPB, header byte 82): every band of a long frame carries its energy index, the decoder normalises the band to it and folds the uncoded remainder from the spectrum below; revisions 6-8 decode unchanged (byte 82 = 0). 8 = lossless: (a) the filter cascade adapts with the error-normalised step (nlms.h: tap step scaled by |err| / mean|err|, capped at 2; step shift 8/7/7), (b) coded channel 1 runs through the joint-stereo integer least-squares stage (own past, channel 0 around t) in front of its cascade; revisions 6-7 still decode — the decoder switches the cascade back to sign-sign and the stage off for them (mmx_ll_state_set_revision) — and the lossy syntax is unchanged. 7 = lossless frames of a two-source block carry a second gain and side stage; revision 6 files decode unchanged (they never had two-source lossless blocks, and the lossy syntax is the same). Before that: bumped whenever the entropy coder or frame syntax changes: 6 = band replication (BWE) above the crossover in header byte 36, and the lossless residual coder (15-bit count-adaptive counters, half-octave class ladder carried across blocks, bypass sign) */

#define MMX_DEFAULT_MAX_REF_DEPTH 5
#define MMX_CODEC_MDCT 2
#define MMX_CODEC_LOSSLESS 3
#define MMX_MAX_BLOCK_SOURCES 2

/* A block is a run of consecutive frames sharing the same source lineage.
   n_sources == 0: self-contained AUDIO frames. Otherwise every frame i of the
   block predicts from decoded windows starting at src_start[s] + i*HOP. */
typedef struct
{
    unsigned long start_frame;
    unsigned long frame_count;
    unsigned char n_sources;
    unsigned char depth;                 /* reference chain depth, 0 for AUDIO */
    unsigned short flags;                /* reserved */
    long long src_start[MMX_MAX_BLOCK_SOURCES];
    unsigned long payload_size;
    unsigned long crc32;
    unsigned long long payload_offset;   /* absolute file offset, set by reader/writer */
    unsigned char *payload;              /* owned; NULL when not loaded */
} MMXBlockEntry;

typedef struct
{
    unsigned long sample_rate;
    unsigned short channels;
    unsigned short source_bits;
    unsigned long long frame_count;      /* samples per channel */
    unsigned long hop;                   /* 1024 */
    unsigned char quality;
    unsigned char analysis_level;
    unsigned char max_ref_depth;
    unsigned char codec_id;
    unsigned char bitstream_rev;
    unsigned int bwe_hz;                 /* band replication crossover in Hz (0 = off); stored as byte 36 in units of 250 Hz */
    unsigned char bwe_mode;              /* patch mapping of the replication (byte 39): 0 = octave transposition, 1 = linear shift */
    unsigned char ll_shift;              /* byte 80, lossless codec: every sample was coded as x >> ll_shift (trailing bits
                                            that were zero in all of them, plus ll_dropped); the decoder shifts back */
    unsigned char lowrate;               /* byte 82: 1 = energy-preserving bands (EPB, revision 9); 0 = the revision 6-8 lossy syntax */
    unsigned char epb_fold;              /* byte 83: EPB fill: lowest band folded from the octave below, 250 Hz units (255 = never) */
    unsigned char epb_flags;             /* byte 84: EPB fill: bit0 noise under the fold floor, bit1 fill inside coded bands, bit2 fill kept out of the prediction reference */
    signed char epb_fill_db;             /* byte 85: EPB fill level relative to the missing energy, dB */
    unsigned char ll_core;               /* byte 87: lossless core, 0 = revision 6-8 coder, 2 = LL2 (revision 10) */
    unsigned char ll2_decay;             /* byte 88: LL2 OLS memory, R -= R >> decay; 0 = the default 9 */
    unsigned char ll2_bank;              /* byte 89: LL2 filter bank, 0 = four sign-LMS stages, 1 = six mixed stages */
    unsigned char ll2_xols[2];           /* bytes 90, 91: LL2 extra OLS memories (decay 6..14) beside byte 88, 0 = none */
    unsigned char ll2_mix;               /* byte 92: LL2 mixer, 0 = two experts (sign, error), 1 = four Huber experts */
    unsigned char ll2_profile;           /* byte 93: LL2 format profile, 0 = CD (frozen), 1 = hi-res (src/ll2codec.h) */
    unsigned char ll2_own;               /* byte 94: LL2 OLS taps on the channel's own past, 0 = the profile's */
    unsigned char ll2_pld;               /* byte 95: LL2 entry points every this many seconds (patchwork landscape
                                            decoding: segments decode independently), 0 = none */
    unsigned char ll_qdrop;              /* byte 86: near-lossless step multiplier M (0/1 = none): the coder works on round(x / (2^ll_shift * M)),
                                            the decoder multiplies back; fractional drop bits log2 M for the rate loop */
    unsigned char ll_dropped;            /* byte 81: of those, the bits --drop-bits rounded away. 0 = bit-exact; otherwise
                                            NEAR-lossless with max error +-2^(ll_dropped-1) LSB, and labelled so */
    char *metadata;                      /* UTF-8 "key=value\n" lines, may be NULL */
    unsigned long metadata_len;
    char cover_mime[16];                 /* e.g. "image/jpeg", empty if none */
    unsigned char *cover;                /* owned */
    unsigned long cover_len;
    unsigned long block_count;
    unsigned long block_capacity;
    MMXBlockEntry *blocks;
} MMXFile;

void mmx_file_init(MMXFile *f);
void mmx_file_free(MMXFile *f);

/* Appends a block entry (takes ownership of payload). Returns index or -1. */
long long mmx_file_add_block(MMXFile *f, const MMXBlockEntry *entry);

/* Sets metadata text and cover (copied). */
int mmx_file_set_metadata(MMXFile *f, const char *text);
int mmx_file_set_cover(MMXFile *f, const char *mime, const unsigned char *data, unsigned long len);

/* Looks up a metadata value by key; returns malloc'd string or NULL. */
char *mmx_file_metadata_get(const MMXFile *f, const char *key);

const char *mmx_block_type_name(const MMXBlockEntry *b);

#endif
