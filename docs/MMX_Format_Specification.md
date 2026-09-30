# MMX Container Format — Version 2 (`MMX2`)

Reference implementation: `src/mmx_writer.c`, `src/mmx_reader.c`,
`include/minimix/mmx_format.h`, bitstream in `src/framecodec.c`.

All integers little-endian. Sample positions are signed 64-bit.

## Layout

```text
[Header 96 bytes]
[Metadata  UTF-8 "key=value\n" lines]
[Cover     image bytes, mime in header]
[Block table  48 bytes * block_count]
[Payload area]
```

Everything needed to start playback (header, metadata, index) sits at the
front. References point strictly backwards, so a single
sequential pass decodes the file; the output buffer is the block cache.

## Frames

The timeline is a grid of frames of `hop` = 1024 new samples. Frame `f` is
computed from the 2048-sample window starting at sample `f*1024 - 1024`
(zero padded outside the signal) with a Kaiser-Bessel-derived (α = 4)
windowed MDCT and overlap-add reconstruction (TDAC). The window is part of
the format: decoders must use the same window. `frame_count_frames = ceil(samples / 1024) + 1`.

## Header

| Offset | Size | Field |
|-------:|-----:|-------|
| 0  | 4 | magic `"MMX2"` |
| 4  | 2 | version = 2 |
| 6  | 2 | header_bytes = 96 |
| 8  | 4 | sample_rate |
| 12 | 2 | channels |
| 14 | 2 | source_bits (informative) |
| 16 | 8 | samples per channel |
| 24 | 4 | hop (1024) |
| 28 | 4 | block_count |
| 32 | 1 | quality 1..10, 0 = lossless |
| 33 | 1 | analysis_level 1..9 (informative) |
| 34 | 1 | max_ref_depth |
| 35 | 1 | codec_id (2 = native MDCT, 3 = lossless integer) |
| 36 | 1 | bwe_crossover in units of 250 Hz (0 = band replication off) |
| 37 | 1 | bitstream_revision (readers refuse other revisions) |
| 38 | 1 | table_format: 0 = 48-byte rows, 1 = compact range-coded table (see below) |
| 39 | 1 | bwe_mode: 0 = octave transposition, 1 = linear shift (only with byte 36 != 0) |
| 40 | 4 | metadata_len |
| 44 | 4 | cover_len |
| 48 | 16 | cover_mime (NUL padded) |
| 64 | 8 | table_offset |
| 72 | 8 | payload_offset |
| 80 | u8 | `ll_shift` — lossless codec: every sample was coded as `x >> ll_shift` (trailing bits that were zero in all samples, plus `ll_dropped`); the decoder shifts back. 0 in older files. |
| 81 | u8 | `ll_dropped` — of those, the bits `--drop-bits` rounded away. 0 = bit-exact. Otherwise the file is NEAR-lossless with a hard bound of ±2^(ll_dropped−1) LSB and must be labelled so; it is never presented as lossless. |
| 82 | u8 | `lowrate` — lossy low-rate core flags (0 = off) |
| 83 | u8 | `epb_fold` — energy-preserving bands: fold setting |
| 84 | u8 | `epb_flags` — energy-preserving bands: flags (bit 2: fill-free reference buffer) |
| 85 | s8 | `epb_fill_db` — energy-preserving bands: fill level in dB |
| 86 | u8 | `ll_qdrop` — near-lossless drop-step multiplier (revision-8 core; 0/1 = none) |
| 87 | u8 | `ll_core` — lossless core: 0/1 = revision-8 core, 2 = LL2 (bitstream revision 10) |
| 88 | u8 | `ll2_decay` — LL2: OLS memory, R −= R >> decay (6..14 for profile 0, 6..16 otherwise) |
| 89 | u8 | `ll2_bank` — LL2: 0 = four sign-LMS stages, 1 = the mixed six-stage bank |
| 90, 91 | u8 ×2 | `ll2_xols` — LL2: memories of up to two extra OLS per channel (0 = none) |
| 92 | u8 | `ll2_mix` — LL2: 0 = two-expert mixer, 1 = four Huber experts |
| 93 | u8 | `ll2_profile` — LL2 format profile: 0 = CD (≤ 16 bit and ≤ 48 kHz; frozen: its files stay byte-identical), 1 = hi-res. Decoders refuse profiles they do not know. |
| 94 | u8 | `ll2_own` — LL2: OLS taps on a channel's own past, 0 = the profile's (16). Ignored for profile 0. |
| 95 | u8 | `ll2_pld` — LL2 entry points every this many seconds (patchwork landscape decoding, see below), 0 = none |

## Block table

Table format 1 (written by this version) is one adaptive range-coded stream
covering `table_offset .. payload_offset` (src/mmx_table.c): per block, in
order, start-frame delta to the end of the previous block (Exp-Golomb, adaptive
prefix), frame count, number of sources (2-bit tree), depth, flags, per source
the signed delta of its window start to a prediction — the previous block's
source window continued by its frame count when that block had this source,
otherwise one window (2048) before the block's own window — payload size, and
the CRC-32 as 32 raw bits. Payload offsets are cumulative. About 12 bytes per
block instead of 48 (a six-minute track has ~5000 blocks: 60 KB instead of
230 KB, 2–5 % of a lossy file). Readers still accept format 0:

## Block table entry (format 0, 48 bytes)

| Offset | Size | Field |
|-------:|-----:|-------|
| 0  | 4 | start_frame |
| 4  | 4 | frame_count |
| 8  | 1 | n_sources (0 = AUDIO, 1 = REF, 2 = REF2) |
| 9  | 1 | depth (0 for AUDIO) |
| 10 | 2 | flags (reserved: time-stretch/pitch modes) |
| 12 | 8 | src_start[0] — window start of the source for the block's first frame |
| 20 | 8 | src_start[1] |
| 28 | 4 | payload_size |
| 32 | 4 | crc32 (IEEE) of the payload |
| 36 | 8 | payload offset relative to payload_offset |

Frame `i` of a block uses source windows `src_start[s] + i*1024`. Rule:
`src_start[s] + 2048 <= f*1024 - 1024` (source samples are final when the
frame is decoded). Blocks cover the timeline in order without gaps.

## Payload: one range-coded stream per block

Adaptive binary range coder (LZMA style, 11-bit probabilities, shift 5).
Contexts are reset at block start. Per frame, in order:

1. `stereo_ms` flag (stereo only): coded domain is M/S (`M = (L+R)/2`, `S = (L-R)/2`).
2. For each source, each coded channel: polarity-flip flag, then for each of
   8 EQ bands (0–200, –500, –1k, –2k, –4k, –8k, –12k, –fs/2 Hz): `off` flag
   (context: previous frame's slot was off) and, if on, the signed delta of the
   gain index against the previous frame's slot (0.5 dB steps, −36…+12 dB).
3. Noise filling levels (long frames): for each coded channel, for each EQ
   band from the one containing the first band at or above 2000 Hz on (2–4k,
   4–8k, 8–12k, 12k–fs/2): `nf` flag (context: the region had a level in the
   previous frame; 0 after a short frame or a block start) and, if set, the
   signed Exp-Golomb delta of the level (1…15) against the region's previous
   level (9 when it had none). Level 0 = no filling.
4. For each coded channel, for each psychoacoustic band (≈45 half-bark bands).
   **Bands from the replication crossover on** (the first band whose center is
   at or above `bwe_crossover`, header byte 36; only when that byte is not 0,
   and only in long frames) use a syntax of their own and nothing else: the
   `zero_band` flag (same context as below) and, when the band is not zero, the
   signed Exp-Golomb delta of its level index against the level this band had
   in the previous long frame (against the channel's last replicated level when
   it had none, 48 at block start) and the signed Exp-Golomb delta of its
   noise-mix index (0…3) against the same band's mix in the previous long frame
   (0 when it had none). No intensity flag, no noise flag, no scalefactor and
   no coefficients are coded there. Below the crossover:
   stereo frames, channel 0, bands whose center is at or above 2000 Hz: an
   `intensity` flag first (context: previous band intensity, same band
   intensity in the previous frame; initialized near "no", an unused flag
   costs ~0.02 bit); if set, the signed position delta (Exp-Golomb) against
   the previous frame's position of the band when that band was intensity,
   else the last position of this frame (0 at the start); the band of channel 0
   then follows the normal syntax and carries the *mid* of the band, channel 1
   carries nothing for the band (its loop skips it). Then, for every band:
   `zero_band` flag; if coded and the band's center is at or above 2000 Hz:
   `noise_band` flag (context: previous band noise, same band noise in the
   previous frame). A noise band carries only a level index (signed
   Exp-Golomb delta against the channel's previous noise level, 48 at block
   start) and no coefficients. Otherwise: scalefactor delta (signed
   Exp-Golomb) against the last coded band (previous frame's last coded
   scalefactor at the start; noise levels do not enter this chain), then the
   coefficients: zero flag → sign (bypass) → >1 flag → >2 flag → Exp-Golomb
   remainder. Coefficient contexts are selected by frequency class and the
   magnitudes of the two previous coefficients; AUDIO and residual frames use
   separate context sets.

Quantizer step `Δ = 2^((sf − 48) / 4)` (1.5 dB steps). `sf` is chosen as the
largest step whose uniform noise `Δ²/12` stays below the masking threshold of
the band, per coefficient.

## Reconstruction

```text
pred[c][k] = Σ_s gain[s][c][eq(k)] · MDCT(decoded window src_start[s] + i·1024)[c][k]   (0 when n_sources = 0)
rec[c][k]  = pred[c][k] + q[c][k] · Δ(sf[c][band(k)])
           = pred[c][k] + fill(f, c, band(k))[k] where q[c][k] = 0 in a coded band at or above 2000 Hz
                                                whose EQ band has a noise filling level (rms = Δ · 2^((level − 16) / 4))
           = noise(f, c, band(k))[k] for a noise band (the prediction is not used there)
bwe band b (long frames, b >= crossover band, coded channel domain, after the
           prediction is added and before M/S → L/R):
           src(k)  = k >> j, smallest j with k >> j < X   (mode 0, octave transposition)
                   = k - m*(X - X0), smallest m with the result < X, floored at X0  (mode 1)
           X = first coefficient of the crossover band, X0 = max(X/2, 4)
           p[k]    = rec[c][src(k)], n[k] = noise(f, c, b)   (the noise-band generator)
           mix     = bwe_mix / 3
           v[k]    = sqrt((1-mix)*N/Σp²) * p[k] + sqrt(mix*N/Σn²) * n[k]      (N = band width)
           rec[c][k] = v[k] * Δ(level) * sqrt(N / Σv²)   -- the band carries exactly N*Δ(level)²
           (a band whose source is silent is pure noise; a zero band is silence)
L/R        = M/S → L/R if stereo_ms
intensity band b (long stereo frames): mid[k] = pred_mid[k] + q[0][k] · Δ(sf[0][b]),
           pred_mid = pred[0] if stereo_ms else (pred[0] + pred[1]) / 2,
           L[k] = gl(pos) · mid[k], R[k] = gr(pos) · mid[k]
           gl² = 2r / (1 + r), gr² = 2 / (1 + r), r = 10^(pos · 1.5 / 10), pos ∈ [−16, 16]
output    += IMDCT(rec) windowed, overlap-added at f·1024 − 1024
```

Intensity band: the encoder codes `(L + R) / 2` scaled so that `gl² + gr²
= 2` keeps the energy of both channels (the position is the quantized L/R
energy ratio); the difference `L − R` of the band is not transmitted. The
decoder's gains for all 33 positions are computed by the same expression on
both sides. Short frames, noise bands and TNS frames carry no intensity bands.

Noise band (perceptual noise substitution): `n` uniform values in [−1, 1)
from an xorshift32 generator seeded with
`((f+1)·0x9E3779B1 ^ (c+1)·0x85EBCA77 ^ (band+1)·0xC2B2AE3D) mod 2^32`
(`f` = frame index on the timeline, state 0 replaced by 0x1234567), each
value `((x >> 8) & 0xFFFFFF) / 2^23 − 1` after the three shifts (13, 17, 5),
then scaled so that the band's energy is exactly `n · Δ(level)²`
(`Δ` as for scalefactors, 1.5 dB steps). The generator and scaling are part
of the format: the encoder's closed loop reconstructs the same noise.

Noise filling: the same generator and seed, drawn only at the zero-quantized
positions of a coded band, scaled so that the filled coefficients carry
exactly `Δ(sf)² · 2^((level − 16) / 2)` energy each (level 1…15: 0.07…0.84
steps rms in 1.5 dB steps). The filled band keeps its prediction; noise bands
and short frames carry no filling, an intensity band's channel-0 band is
filled like any other coded band (the fill reaches both channels through the
gains) and its channel-1 band carries nothing. The encoder fills nothing by
default; `--nf` switches it on at every rate, `--no-nf` off. A band the rate
loop's offset zeroed although the quality's own model would have coded it is
written as a noise band of its energy instead (capped at the lifted
threshold); that uses no new syntax.

Band replication (`--bwe`, header byte 36): above the crossover the encoder
codes no coefficients at all. Every band of a long frame from the crossover
band on carries its energy on the noise-band level grid (1.5 dB steps) and a
two-bit noise share; the decoder rebuilds the band by copying the octave below
the crossover into it (mode 0 maps coefficient k to k >> j, i.e. a
transposition by whole octaves, so a harmonic series maps onto a subset of
itself), mixing in the deterministic noise of the noise-band generator
according to the share, and scaling the result to the transmitted energy. The
prediction is not used in a replicated band, and replicated bands carry no
noise filling and no intensity stereo. Short frames are never replicated. The
source coefficients all lie below the crossover and are final when the band is
rebuilt, so the operation is one pass over the frame.

The encoder runs exactly this code on its own decoded output (closed loop),
so encoder and decoder reconstructions are bit-identical on the same
platform.

## Psychoacoustic model (encoder side, informative)

Spectra for the model come from a Blackman-Harris windowed FFT (leak-free),
scaled to MDCT units. Half-bark bands (≥ 4 bins), Schroeder spreading,
tonality from spectral flatness (Johnston offsets 14.5 + z dB tonal, 5.5 dB
noise), Terhardt absolute threshold (0 dBFS sine = 96 dB SPL). Pre-echo
control: a frame's threshold is the minimum over the five 1024-sample
sub-windows overlapping its window (×2 for the MDCT size), each sub-window
containing an attack is scaled by 0.05 · (energy before the attack / attack
energy); temporal smoothing (min with the previous frame, +6 dB post-masking
against the next); thresholds may not drop more than 10 dB between
neighbouring bands (window main-lobe leakage); first and last frame 20 dB
stricter. Quality offset −6 … +12 dB and bandwidth 20.5 … 11 kHz for quality
10 … 1. Stereo M/S frames use `min(thr_L, thr_R) / 2` and additionally verify
the resulting L/R noise.

## Lossless codec payload (codec_id 3)

Same block table and frame grid. Frame `f` owns samples `[(f-1)*1024, f*1024)`
(frame 0 owns nothing and is skipped). For a referenced block, frame `i` predicts
sample `n` as `round(gain * decoded[src_start + i*1024 + n])` with a Q12 gain
per channel (`use_pred` flag + signed gain delta per channel). Joint stereo per
frame (2-bit tree): L/R, M/S (`mid = (L+R)>>1`, `side = L-R`), L/S or R/S.
Per coded channel a cascade flag (adaptive bit, context: the channel's
previous flag), then a predictor mode symbol (4-bit tree): fixed polynomial
order 0–3 or adaptive LPC of order 1–12 with a 4-bit coefficient shift and the
quantized coefficients (signed Exp-Golomb, one context set per coefficient
index); the prediction is evaluated in 64-bit integer arithmetic. With the
flag set the predictor works on the output of the channel's filter chain
instead of the coded signal: the first difference of the coded signal, minus
an adaptive two-sided 9-tap window (t−4..t+4, Q13 weights, taps past the
frame end read zero) on the first difference of the joint-domain source when
every contributing channel uses the reference gain, through three
backward-adaptive sign-sign LMS filters of 1024, 256 and 16 taps (weight
shifts 15, 13, 11; 16-bit saturated inputs and weights, dot products modulo
2^32, steps 8/16/32 by input magnitude against a running average, halved at
lags 1, 2 and 8 — Monkey's Audio's NN filter). The chain adapts on the actual
signal of every channel frame whatever the flag says; the source window
restarts at every block, the cascade is carried across blocks: the decoder
walks the blocks in file order. Residuals are signed values: zero flag, sign, unary exponent (capped
at 62 on both sides, no terminator at the cap), two adaptive mantissa bits
and bypass bits, with contexts by the magnitude class of a running mean of
|residual|. The predictor history of the first samples of a frame comes from
the last 12 samples of the previous frame of the same block (raw target,
source and chain-output samples are carried in the coder state and re-derived
in the current coded domain); the first frame of a block starts from zero
history. Samples are integers at `source_bits` (16 or 24); decoding is
bit-exact.

## LL2 entry points (patchwork landscape decoding)

With header byte 95 = `P` > 0, let `R = floor(P * sample_rate / 1024)` frames
(at least 1). The encoder starts a block at every frame that is a multiple of
`R`, and every block after the first whose `start_frame % R == 0` begins a new
**segment**. At a segment's first block the LL2 predictor (OLS, filter bank,
mixer, source stage) and the residual coder's model start from their initial
state exactly as at the start of the file, and the regressor taps (the other
channel's samples for channel 1) read nothing before the segment's first
sample. References may still read any earlier decoded audio, in any segment.
A decoder can so start at any segment once the samples its references read
are decoded, and decode segments side by side; the result is the same as the
sequential pass. The encoder writes `P = 30` by default (`MMX_PLD` overrides);
measured on full CD titles: +0.02..+0.05 % in size, about 1 kB per entry point. With `--pld` the encoder keeps
every reference, including the 256 samples around its source the LL2 source stage reads, inside its own
segment: each segment then decodes on its own (same format, no extra field).

## Bitstream revision history

| Revision | Change |
|---:|-------|
| 1 | inter-frame coefficient contexts, dead-zone rounding (lossy) |
| 2 | TNS syntax element per coded channel (lossy, off by default), lossless LPC / joint stereo / history |
| 3 | block type symbol per lossy frame (LONG / START / SHORT / STOP; short frames carry 8 groups of scalefactors and band flags). Tracker / future modes need no new syntax: pure bands are ordinary zero bands, the mode is metadata (`mode=tracker|future`). |
| 4 | lossy: noise-band flag and level per coded long-frame band from 2 kHz on (perceptual noise substitution, `--pns`, off by default); lossless: cascade flag per coded channel frame (integer NLMS chain carried across blocks), residual exponent capped at 62 on both sides; container: compact block table (header byte 38 = 1). |
| 5 | lossy, long frames only: intensity-stereo flag (and, when set, position delta) per band from 2 kHz on in channel 0 of a stereo frame (`--is`, off by default), and a noise-filling flag (and, when set, level delta) per coded channel and EQ region from 2 kHz on (`--nf`, off by default). Both flags are context-coded and initialized near "no": a file that uses neither pays about 0.02 bit per band flag and 0.05 bit per region flag, ~0.02-0.05 % of the payload. |
| 6 | lossy, long frames only: band replication (`--bwe`). Header byte 36 carries the crossover in units of 250 Hz (0 = off), byte 39 the patch mapping. Every band from the crossover band on carries a zero flag and, when it is not zero, a level index and a two-bit noise-mix index instead of a scalefactor and coefficients; the decoder regenerates it from the octave below the crossover. Files with byte 36 = 0 use the revision-5 syntax unchanged. |
| 7 | lossless: a frame of a two-source block carries a second gain (`gain2`) and a second side stage for the second source (`--lossless` with `MMX_LL_PAIRS=1`; off by default). Revision-6 files decode unchanged. Header bytes 80/81 = `ll_shift` / `ll_dropped` (near-lossless `--drop-bits`, the decoder shifts the output back). |
| 8 | lossless, two changes in the predictor, no new syntax elements: (a) the filter cascade adapts with an error-normalised step — each tap moves by round(d · (2·mq + 1) / 2^s) towards sign(err), mq = |err| / mean|err| in Q6 capped at 2, step shifts 8/7/7 for the 1024/256/16-tap stages, the mean of |err| a Q4 running average at rate 1/32 (an odd multiplier, so rounding never ties); (b) coded channel 1 of a stereo frame runs through a joint-stereo integer least-squares stage in front of its cascade — regressor = its own 8 past samples, channel 0's first difference at lags −7…+4 (channel 0 is decoded first, taps past the frame end read 0), Q16 weights from an exponentially weighted covariance (λ = 1 − 2⁻⁹) solved by a fixed-point LDLᵀ every 16 samples, state carried across blocks like the cascade. Decoders keep revisions 6–7 bit-exact by running the cascade sign-sign and the stage off for them. The lossy syntax is revision 7's. Measured on a set of CD tracks: −0.58 % against revision 7, decode ≈ 1.4× the time. |
