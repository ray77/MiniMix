#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cli.h"
#include "minimix/mmx_format.h"
#include "minimix/encoder.h"
#include "percept.h"
#include "log.h"

void mmx_cli_usage(void)
{
    printf("MiniMix %s - globally analyzed music codec (reference implementation)\n\n", MMX_VERSION_STRING);
    printf("Usage:\n");
    printf("  mmx encode  input.wav output.mmx [options]\n");
    printf("  mmx decode  input.mmx output.wav [--bits 16|24|32]\n");
    printf("  mmx play    input.mmx [more.mmx ...]         play in the terminal: space pause, arrows seek, n next, q quit\n");
    printf("  mmx analyze input.wav|input.mmx [options]      whole-file analysis report, no output file\n");
    printf("  mmx info    input.mmx [--verbose] [--json]     inspector\n");
    printf("  mmx compare original.wav decoded.wav [--quality N] [--json] [--regions]   perceptual comparison\n");
    printf("  mmx tag     file.mmx --meta key=value ...       add or replace metadata without re-encoding\n\n");
    printf("Options:\n");
    if (MMX_LOSSY)
        printf("  --quality N      1..10 or low|medium|high|max (default 7 ~ MP3 320 target, 10 = practically transparent)\n");
    printf("  --lossless       bit-exact mode (integer residual coder with the same global references)\n");
    if (!MMX_LOSSY)
        printf("                   - the only encode mode for now: the lossy core is switched off (make MMX_LOSSY=1)\n");
    printf("  --pld            lossless: every 30-s segment decodes on its own - a player jumps anywhere in about a second.\n");
    printf("                   References stay inside their segment, which costs songs built from loops size (about +5 %%,\n");
    printf("                   up to +11 %% seen); without it (the default) they may reach back into earlier segments\n");
    printf("  --drop-bits N    near-lossless: round the input by N bits first (1..8), then code it losslessly - a hard\n");
    printf("                   bound of +-2^(N-1) LSB, bit-exact to the rounded signal. N=2/3/4 gave 79/69/59 %% of the\n");
    printf("                   lossless size at 76/71/65 dB SNR on a loop-built title. Labelled as such, never as lossless\n");
    printf("  --drop-step M    near-lossless: multiply the rounding step by M (1..255) on top of --drop-bits (log2 M\n");
    printf("                   fractional bits)\n");
    printf("  --bitrate N      with --lossless: near-lossless with a size target of N kbit/s (16..2000); bits and step\n");
    printf("                   are searched to meet it\n");
    if (MMX_LOSSY)
    {
        printf("  --bitrate N      target bitrate in kbit/s: thresholds are scaled until the size matches (analysis once);\n");
        printf("                   the offset is tilted over frequency (bass and mids protected, the highs pay), at\n");
        printf("                   %u kbit/s and below the highs are replicated instead of coded (--no-bwe keeps it off) and at\n", MMX_BWE_DEFAULT_MAX_KBPS);
        printf("                   %u kbit/s and below noise substitution switches on by itself (--no-pns keeps it off)\n", MMX_PNS_DEFAULT_MAX_KBPS);
        printf("  --tns            experimental temporal noise shaping\n");
        printf("  --pns [HZ]       perceptual noise substitution: noise-like, stationary bands above HZ (default %u)\n", MMX_PNS_DEFAULT_HZ);
        printf("                   carry only their energy and play as noise (the model counts them as error, judge by ear)\n");
        printf("  --no-pns         switch it off (also at a bitrate target of %u kbit/s or less, where it is automatic)\n", MMX_PNS_DEFAULT_MAX_KBPS);
        printf("  --bwe [HZ]       band replication: above HZ no coefficients are coded, the decoder regenerates the band\n");
        printf("                   from the octave below it at the transmitted energy. On by itself at a bitrate target of\n");
        printf("                   %u kbit/s and below (crossover falls with the rate), --no-bwe keeps it off\n", MMX_BWE_DEFAULT_MAX_KBPS);
        printf("  --no-bwe         switch it off\n");
        printf("  --is [HZ]        intensity stereo (off by default): coherent bands above HZ (default %u) are coded once with\n", MMX_IS_DEFAULT_HZ);
        printf("                   a position when that is cheaper; the difference of the band is dropped, so the model counts\n");
        printf("                   it as error - judge by ear\n");
        printf("  --no-is          switch it off\n");
        printf("  --nf             noise filling (off by default): the zeros of the coded bands above %u Hz and the bands the\n", MMX_NF_DEFAULT_HZ);
        printf("                   rate loop zeroed keep their energy as noise instead of leaving holes; judge by ear\n");
        printf("  --no-nf          switch it off\n");
        printf("  --tracker        tracker mode: parts that repeat are played from the earlier part like in a MOD file\n");
        printf("                   (per band: 90 %% the same = no residual, the rest with a 6 dB more generous threshold)\n");
        printf("  --future         maximum summarization: pure playback from 6 dB prediction gain on, residual 12 dB more\n");
        printf("                   generous, unique material at quality 4 (unless --quality is given). Smallest file.\n");
        printf("  --reuse N        legacy experiment 0..3: frame-level version of --tracker (3/6/9 dB, pure from 10/7 dB)\n");
    }
    printf("  --percept [DB]   analyze only: the perceptual similarity map. Sections that SOUND like an earlier section\n");
    printf("                   (spectral and 23 ms level envelope within DB, default %.0f, after a level fit per band)\n", MMX_PERCEPT_DEFAULT_DB);
    printf("                   are reported with their lag. Analysis tool, no encode mode.\n");
    printf("  --maxfreq HZ     compare: evaluate bands only up to this frequency (fair across codecs with different lowpass)\n");
    printf("  --regions        compare: one line per quantity across the 8 EQ regions (unweighted and weighted error share,\n");
    printf("                   perceptual weight, band energy error) instead of reading the JSON\n");
    printf("  --analysis N     1..9 or best (default 5). Higher = more candidates, longer search. Always the whole file.\n");
    printf("  --depth N        maximum reference chain depth (default %d)\n", MMX_DEFAULT_MAX_REF_DEPTH);
    printf("  --title/--artist/--album/--year/--track/--genre/--comment TEXT   metadata\n");
    printf("  --meta key=value store any metadata line (repeatable)\n");
    printf("  --cover FILE     embed a cover image (jpeg, png, webp, avif)\n");
    printf("  --stats          print compression statistics after encoding\n");
    printf("  --json           machine-readable output\n");
    printf("  --verbose        debug log level and detailed listings\n");
    printf("  --debug          trace level\n");
    printf("  --quiet          errors only\n");
    printf("  --version, --help\n");
}

static int parse_quality(const char *s, unsigned int *out)
{
    if (strcmp(s, "lossless") == 0) { *out = 0; return 0; }
    if (strcmp(s, "low") == 0) { *out = 4; return 0; }
    if (strcmp(s, "medium") == 0) { *out = 6; return 0; }
    if (strcmp(s, "high") == 0) { *out = 8; return 0; }
    if (strcmp(s, "max") == 0) { *out = 10; return 0; }
    {
        char *end;
        long long v = strtol(s, &end, 10);
        if (*end != '\0' || v < 0 || v > 10)
            return -1;
        *out = (unsigned int)v;
    }
    return 0;
}

static int parse_analysis(const char *s, unsigned int *out)
{
    if (strcmp(s, "best") == 0 || strcmp(s, "insane") == 0 || strcmp(s, "max") == 0) { *out = 9; return 0; }
    {
        char *end;
        long long v = strtol(s, &end, 10);
        if (*end != '\0' || v < 1 || v > 9)
            return -1;
        *out = (unsigned int)v;
    }
    return 0;
}

static int parse_ulong(const char *s, unsigned long lo, unsigned long hi, unsigned long *out)
{
    char *end;
    unsigned long v = strtoul(s, &end, 10);
    if (*end != '\0' || v < lo || v > hi)
        return -1;
    *out = v;
    return 0;
}

static int add_meta(MMXOptions *opts, const char *key, const char *value)
{
    static char storage[MMX_MAX_META][512];
    if (opts->meta_count >= MMX_MAX_META)
    {
        mmx_error("Too many metadata entries");
        return -1;
    }
    snprintf(storage[opts->meta_count], sizeof(storage[0]), "%s=%s", key, value);
    opts->meta[opts->meta_count] = storage[opts->meta_count];
    opts->meta_count++;
    return 0;
}

int mmx_cli_parse(int argc, char **argv, MMXOptions *opts)
{
    int i, positional = 0, quiet = 0;

    memset(opts, 0, sizeof(*opts));
    opts->quality = 7;
    opts->analysis = 5;
    opts->pns = MMX_PNS_DEFAULT_ON ? MMX_PNS_DEFAULT_HZ : 0;
    opts->nf = MMX_NF_DEFAULT;
    opts->max_depth = MMX_DEFAULT_MAX_REF_DEPTH;
    opts->percept_db = MMX_PERCEPT_DEFAULT_DB;

    if (argc < 2)
    {
        opts->command = MMX_CMD_HELP;
        return 0;
    }

    for (i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        unsigned long v;

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) { opts->command = MMX_CMD_HELP; return 0; }
        if (strcmp(a, "--version") == 0) { opts->command = MMX_CMD_VERSION; return 0; }
        if (strcmp(a, "--verbose") == 0 || strcmp(a, "-v") == 0) { opts->verbose = 1; continue; }
        if (strcmp(a, "--debug") == 0) { opts->debug = 1; opts->verbose = 1; continue; }
        if (strcmp(a, "--quiet") == 0) { quiet = 1; continue; }
        if (strcmp(a, "--json") == 0) { opts->json = 1; continue; }
        if (strcmp(a, "--regions") == 0) { opts->regions = 1; continue; }
        if (strcmp(a, "--lossless") == 0) { opts->quality = 0; opts->lossless_given = 1; continue; }
        if (strcmp(a, "--tns") == 0) { opts->tns = 1; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--no-pns") == 0) { opts->pns = 0; opts->pns_given = 1; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--no-is") == 0) { opts->is_hz = 0; opts->is_given = 1; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--no-bwe") == 0) { opts->bwe_hz = 0; opts->bwe_given = 1; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--bwe") == 0)
        {
            opts->bwe_hz = MMX_BWE_DEFAULT_HZ;
            opts->bwe_given = 1;
            opts->lossy_opt = a;
            if (i + 1 < argc && parse_ulong(argv[i + 1], 2000, 24000, &v) == 0) { opts->bwe_hz = (unsigned int)v; i++; }
            continue;
        }
        if (strcmp(a, "--is") == 0)
        {
            opts->is_hz = MMX_IS_DEFAULT_HZ;
            opts->is_given = 1;
            opts->lossy_opt = a;
            if (i + 1 < argc && parse_ulong(argv[i + 1], MMX_IS_MIN_HZ, 24000, &v) == 0) { opts->is_hz = (unsigned int)v; i++; }
            continue;
        }
        if (strcmp(a, "--nf") == 0) { opts->nf = MMX_NF_ON; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--no-nf") == 0) { opts->nf = MMX_NF_OFF; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--pns") == 0)
        {
            opts->pns = MMX_PNS_DEFAULT_HZ;
            opts->pns_given = 1;
            opts->lossy_opt = a;
            if (i + 1 < argc && parse_ulong(argv[i + 1], 1000, 24000, &v) == 0) { opts->pns = (unsigned int)v; i++; }
            continue;
        }
        if (strcmp(a, "--tracker") == 0) { opts->mode = 1; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--future") == 0) { opts->mode = 2; opts->lossy_opt = a; continue; }
        if (strcmp(a, "--percept") == 0)
        {
            opts->percept = 1;
            if (i + 1 < argc)
            {
                char *end;
                double dv = strtod(argv[i + 1], &end);
                if (end != argv[i + 1] && *end == '\0' && dv > 0.0) { opts->percept_db = dv; i++; }
            }
            continue;
        }
        if (strcmp(a, "--stats") == 0) { opts->stats = 1; continue; }
        if (strcmp(a, "--pld") == 0) { opts->pld_strict = 1; continue; }

        if (a[0] == '-' && a[1] == '-')
        {
            const char *name = a + 2;
            const char *val;
            if (i + 1 >= argc)
            {
                mmx_error("Option %s needs a value", a);
                return -1;
            }
            val = argv[++i];
            if (strcmp(name, "quality") == 0)
            {
                if (parse_quality(val, &opts->quality) != 0) { mmx_error("Bad quality: %s", val); return -1; }
                opts->quality_given = 1;
                if (opts->quality != 0) opts->lossy_opt = "--quality";
            }
            else if (strcmp(name, "analysis") == 0)
            {
                if (parse_analysis(val, &opts->analysis) != 0) { mmx_error("Bad analysis level (1..9): %s", val); return -1; }
            }
            else if (strcmp(name, "reuse") == 0)
            {
                if (parse_ulong(val, 0, 3, &v) != 0) { mmx_error("Bad --reuse (0..3): %s", val); return -1; }
                opts->reuse = (unsigned int)v;
                opts->lossy_opt = "--reuse";
            }
            else if (strcmp(name, "maxfreq") == 0)
            {
                if (parse_ulong(val, 1000, 24000, &v) != 0) { mmx_error("Bad --maxfreq: %s", val); return -1; }
                opts->maxfreq = (unsigned int)v;
            }
            else if (strcmp(name, "drop-step") == 0)
            {
                if (parse_ulong(val, 1, 255, &v) != 0) { mmx_error("Bad --drop-step (1..255): %s", val); return -1; }
                opts->drop_step = (unsigned int)v;
            }
            else if (strcmp(name, "drop-bits") == 0)
            {
                if (parse_ulong(val, 0, 8, &v) != 0) { mmx_error("Bad --drop-bits (0..8): %s", val); return -1; }
                opts->drop_bits = (unsigned int)v;
            }
            else if (strcmp(name, "bitrate") == 0)
            {
                if (parse_ulong(val, 16, 2000, &v) != 0) { mmx_error("Bad bitrate: %s", val); return -1; }
                opts->bitrate = (unsigned int)v;
            }
            else if (strcmp(name, "depth") == 0)
            {
                if (parse_ulong(val, 0, 32, &v) != 0) { mmx_error("Bad depth: %s", val); return -1; }
                opts->max_depth = (unsigned int)v;
            }
            else if (strcmp(name, "bits") == 0)
            {
                if (parse_ulong(val, 16, 32, &v) != 0 || (v != 16 && v != 24 && v != 32)) { mmx_error("Bad bit depth: %s", val); return -1; }
                opts->out_bits = (unsigned int)v;
            }
            else if (strcmp(name, "meta") == 0)
            {
                const char *eq = strchr(val, '=');
                char key[128];
                if (!eq || eq == val) { mmx_error("--meta expects key=value"); return -1; }
                snprintf(key, sizeof(key), "%.*s", (int)(eq - val), val);
                if (add_meta(opts, key, eq + 1) != 0) return -1;
            }
            else if (strcmp(name, "cover") == 0)
                opts->cover = val;
            else if (strcmp(name, "title") == 0 || strcmp(name, "artist") == 0 || strcmp(name, "album") == 0 ||
                     strcmp(name, "year") == 0 || strcmp(name, "track") == 0 || strcmp(name, "genre") == 0 ||
                     strcmp(name, "comment") == 0)
            {
                if (add_meta(opts, name, val) != 0) return -1;
            }
            else
            {
                mmx_error("Unknown option: %s", a);
                return -1;
            }
            continue;
        }
        if (a[0] == '-' && a[1] != '\0')
        {
            mmx_error("Unknown option: %s", a);
            return -1;
        }

        if (positional == 0)
        {
            if (strcmp(a, "encode") == 0) opts->command = MMX_CMD_ENCODE;
            else if (strcmp(a, "decode") == 0) opts->command = MMX_CMD_DECODE;
            else if (strcmp(a, "analyze") == 0 || strcmp(a, "analyse") == 0) opts->command = MMX_CMD_ANALYZE;
            else if (strcmp(a, "info") == 0) opts->command = MMX_CMD_INFO;
            else if (strcmp(a, "compare") == 0) opts->command = MMX_CMD_COMPARE;
            else if (strcmp(a, "tag") == 0) opts->command = MMX_CMD_TAG;
            else if (strcmp(a, "play") == 0) opts->command = MMX_CMD_PLAY;
            else if (strcmp(a, "help") == 0) { opts->command = MMX_CMD_HELP; return 0; }
            else if (strcmp(a, "version") == 0) { opts->command = MMX_CMD_VERSION; return 0; }
            else { mmx_error("Unknown command: %s", a); return -1; }
        }
        else if (opts->command == MMX_CMD_PLAY)
        {
            if (opts->file_count >= MMX_MAX_FILES) { mmx_error("Too many files (at most %d)", MMX_MAX_FILES); return -1; }
            opts->files[opts->file_count++] = a;
            if (!opts->input) opts->input = a;
        }
        else if (positional == 1)
            opts->input = a;
        else if (positional == 2)
            opts->output = a;
        else
        {
            mmx_error("Too many arguments");
            return -1;
        }
        positional++;
    }

    if (quiet)
        mmx_log_set_level(MMX_LOG_ERROR);
    else if (opts->debug)
        mmx_log_set_level(MMX_LOG_TRACE);
    else if (opts->verbose)
        mmx_log_set_level(MMX_LOG_DEBUG);
    if (opts->json)
        mmx_log_set_quiet(1);

    if (opts->percept && opts->command != MMX_CMD_ANALYZE)
    {
        mmx_error("--percept is an analyze option (mmx analyze FILE --percept [DB]); there is no --percept encode mode");
        return -1;
    }
    switch (opts->command)
    {
    case MMX_CMD_ENCODE:
    case MMX_CMD_DECODE:
    case MMX_CMD_COMPARE:
        if (!opts->input || !opts->output)
        {
            mmx_error("%s needs two paths", opts->command == MMX_CMD_ENCODE ? "encode" :
                      opts->command == MMX_CMD_DECODE ? "decode" : "compare");
            return -1;
        }
        break;
    case MMX_CMD_ANALYZE:
    case MMX_CMD_INFO:
    case MMX_CMD_TAG:
    case MMX_CMD_PLAY:
        if (!opts->input)
        {
            mmx_error("Missing input path");
            return -1;
        }
        break;
    case MMX_CMD_NONE:
        mmx_error("No command given");
        return -1;
    default:
        break;
    }
    return 0;
}
