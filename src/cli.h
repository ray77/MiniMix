#ifndef MMX_CLI_H
#define MMX_CLI_H

#define MMX_VERSION_STRING "0.3.0"
#define MMX_MAX_META 256   /* metadata lines per file */
#define MMX_MAX_FILES 1024 /* mmx play: files per call */

/* The lossy MDCT core is switched off for now (lossless first; a new lossy class built on the lossless core comes
   later): `mmx encode` always codes lossless and refuses the lossy options. Decoding lossy files, analyze and
   compare stay. `make clean; make MMX_LOSSY=1` builds the lossy encoder back in. */
#ifndef MMX_LOSSY
#define MMX_LOSSY 0
#endif

typedef enum
{
    MMX_CMD_NONE = 0,
    MMX_CMD_ENCODE,
    MMX_CMD_DECODE,
    MMX_CMD_ANALYZE,
    MMX_CMD_INFO,
    MMX_CMD_COMPARE,
    MMX_CMD_TAG,
    MMX_CMD_PLAY,
    MMX_CMD_HELP,
    MMX_CMD_VERSION
} MMXCommand;

typedef struct
{
    MMXCommand command;
    const char *input;
    const char *output;
    unsigned int quality;        /* 1..10, 0 = lossless */
    unsigned int bitrate;        /* kbit/s target, 0 = quality fixed */
    unsigned int drop_bits;      /* --drop-bits N: near-lossless (lossless coder on input rounded by N bits) */
    unsigned int drop_step;      /* --drop-step M: near-lossless, extra integer step multiplier (1..255) */
    int pld_strict;              /* --pld: lossless segments fully independent (no reference crosses an entry point) */
    unsigned int analysis;       /* 1..9 */
    unsigned int max_depth;
    unsigned int out_bits;       /* 0 = same as source */
    int verbose;
    int debug;
    int json;
    int stats;
    int tns;
    unsigned int pns;            /* noise substitution from this frequency on, 0 = off */
    int pns_given;               /* --pns / --no-pns was on the command line (otherwise the bitrate decides) */
    unsigned int bwe_hz;         /* band replication from this frequency on, 0 = off */
    int bwe_given;               /* --bwe / --no-bwe was on the command line (otherwise the bitrate decides) */
    unsigned int is_hz;          /* intensity stereo from this frequency on, 0 = off */
    int is_given;                /* --is / --no-is was on the command line (otherwise the bitrate decides) */
    unsigned int nf;             /* noise filling: MMX_NF_OFF / MMX_NF_AUTO / MMX_NF_ON */
    unsigned int reuse;
    unsigned int mode;           /* 0 normal, 1 tracker, 2 future */
    int percept;                 /* analyze --percept: print the perceptual similarity map instead of the repeat analysis */
    double percept_db;           /* --percept DB: a section matches an earlier one under this distance (dB) */
    int quality_given;           /* --quality was on the command line */
    int lossless_given;          /* --lossless was on the command line */
    const char *lossy_opt;       /* the first option that asks for the lossy encoder (refused while MMX_LOSSY is 0) */
    unsigned int maxfreq;        /* compare: highest frequency evaluated, 0 = codec cutoff */
    int regions;                 /* compare: print the per-region table as compact one-line summaries */
    const char *meta[MMX_MAX_META]; /* "key=value" */
    unsigned int meta_count;
    const char *cover;
    const char *files[MMX_MAX_FILES]; /* mmx play */
    unsigned int file_count;
} MMXOptions;

/* Parses argv into opts. Returns 0 on success, -1 on error (message already printed). */
int mmx_cli_parse(int argc, char **argv, MMXOptions *opts);

void mmx_cli_usage(void);

#endif
