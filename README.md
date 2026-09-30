<p align="center"><img src="docs/images/mmx_logo.png" alt="MiniMix audio codec" width="360"></p>

# MiniMix `.mmx`

A lossless audio codec that analyses the whole piece before it codes it: repeated material anywhere in the track
becomes a reference, everything is predicted by an adaptive cascade (OLS, LMS filter bank, mixer) and the residual is
coded with context mixing. Bit-exact, C99 with POSIX threads and libm (the decoder also builds with MSVC). The
former lossy MDCT encoder is switched off for now (`make MMX_LOSSY=1` builds it back in; lossy files still decode); a
lossy class built on the lossless core is in preparation.

This public repository starts from a snapshot of the private development repository after 164 commits: the lossless
core, patchwork landscape decoding, the foobar2000 component and the former lossy codec were built there.

## Status (September 2026)

| | |
|---|---|
| CD audio, 82 titles (16 bit, 44.1 kHz) | −6.90 % against FLAC -8, −1.46 % against OptimFROG (max); 77 of 82 smaller |
| Hi-res (24 bit, 88.2 kHz) | −0.89 % against OptimFROG; 24- and 32-bit PCM, up to 384 kHz, RF64/Wave64 |
| Decoding (Apple M1) | about 10–25× realtime; 30-s segments decode side by side ([PLD](#patchwork-landscape-decoding-pld)) |

Measured on the author's own CD collection, which is not distributed. Mono and stereo.

## Build

```bash
make          # bin/mmx
make test
```

Ready-made builds of `mmx` (macOS universal, Windows x64) and of the foobar2000 component are on the
[releases page](https://github.com/ray77/MiniMix/releases). The component's sources are in
[components/foo_input_mmx](components/foo_input_mmx/README.md).

## Use

```bash
mmx encode song.wav song.mmx          # lossless (the only encode mode for now), smallest file
mmx encode song.wav song.mmx --pld    # every 30-s segment decodes on its own (see PLD below)
mmx play   song.mmx more.mmx          # play in the terminal while it decodes: space, arrows, 0-9, n, q
mmx decode song.mmx song.wav
mmx info   song.mmx
```

## Patchwork landscape decoding (PLD)

Every lossless file carries an entry point every 30 s (+0.02–0.05 % in size; near-lossless files have none). The decoder works on several segments at once, on
several cores, and a player starts decoding where the listener jumps; the gaps fill in while it plays. References may
still reach back into earlier segments - repeated material is what the codec lives on - so a jump into a song built
from loops waits for the segments its references read. `--pld` (off by default) makes every segment decode on its own:

```bash
mmx encode song.wav song.mmx --pld   # a jump anywhere plays after 0.6–1.5 s instead of up to ~20 s
```

What it costs, measured on the 81 CD titles: +0.60 % in total; most music +0.02–0.07 %, songs built from loops more
(one loop-built title +5.2 %, the most loop-heavy +11 %).

## Documentation

- [docs/MMX_Format_Specification.md](docs/MMX_Format_Specification.md) — container and bitstream

## What is ours, what is borrowed

| Feature | Origin |
|---|---|
| Whole-file analysis: repeated material anywhere becomes a reference that feeds the predictor | MiniMix |
| Predictor: OLS with stereo taps, LMS cascade, stage mixer, bias correction | design from sac; own deterministic integer/double implementation, mixer extended |
| Older lossless core (revision 8, still used for near-lossless): LPC with FLAC-style quantization, sign-sign LMS cascade | ideas from FLAC and Monkey's Audio, own implementation |
| Residual coder: bitplanes with context mixing and SSE | sac, integer port |
| Per-file parameter search in the encoder | idea from sac |
| Hi-res profile: adaptive stage scaling, raw low bitplanes, 24/32-bit, RF64/Wave64 | MiniMix (RF64/Wave64 are standard containers) |
| Patchwork landscape decoding: segments decoded side by side, seeks that start decoding at the target | MiniMix (seek points as such are standard, e.g. FLAC) |
| `--pld` independent segments | standard idea (independent frames), MiniMix implementation |
| Near-lossless `--drop-bits` | known idea (lossyWAV, WavPack hybrid), own implementation |
| `.mmx` container: metadata, cover, format profiles | MiniMix |
| Lossy MDCT encoder (switched off): psychoacoustics, TNS, PNS, band replication, intensity stereo, noise filling | textbook techniques (AAC, Opus); the reference-based tracker mode is MiniMix |
| foobar2000 component | MiniMix, on the foobar2000 SDK |

## License

MIT - Copyright (c) 2026 Domy Schmidt (domy@noisebay.org), see [LICENSE](LICENSE). The LL2 residual coder is an
integer port of the bitplane coder of [sac](https://github.com/slmdev/sac) (Sebastian Lehmann, MIT licence; its
notice is in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)).
