# foo_input_mmx — MiniMix for foobar2000

Plays `.mmx` files in foobar2000 2.x on Windows (x64) and macOS (Apple Silicon + Intel, macOS 11+).
Lossless, near-lossless and lossy files alike; tags and the codec profile show in the properties.

## How it plays

A lossless file with entry points (every 30 s, patchwork landscape decoding) decodes in segments side by side on
several cores; playback starts as soon as the audio at the play position is final. A seek decodes towards its target
and plays from there, or, when reaching the exact spot would take much longer (songs whose segments read many
earlier ones), from a nearby segment start. Older files and near-lossless or lossy files are resolved front to back.

The component compiles only the decoder subset of the core (22 C files, no encoder).

## Build

* **macOS** — no Xcode project needed, clang alone:
  `make -f Makefile.mac package` → `build/foo_input_mmx-mac.fb2k-component` (universal, ad-hoc signed).
* **Windows** — `msbuild foo_input_mmx.sln /p:Configuration=Release /p:Platform=x64` (VS 2022, v143)
  → `build/win/x64/Release/foo_input_mmx.dll`, packed under `x64/` in the `.fb2k-component`.
* **CI** — `.github/workflows/build.yml` builds both, checks the decoder against `testdata/` on each platform and
  merges everything into one `foo_input_mmx.fb2k-component`; a tag `v*` publishes it with the `mmx` builds.

Both builds expect the foobar2000 SDK 2026-09-17 under `tools/third_party/foobar2000_sdk/`:
download `https://www.foobar2000.org/downloads/SDK-2026-09-17.7z` (SHA-256
`24fa95ed66b42624f593fd4b94c5efb08ae81dfebb92b79fe69c8cd170961e48`) and
`7z x SDK-2026-09-17.7z -otools/third_party/foobar2000_sdk` from the repository root.

The test vectors in `testdata/` are synthetic signals made for these tests (tones, noise bursts, repeated loops; no
recordings), encoded by earlier versions of the encoder: they pin what the decoder must keep reading bit for bit.

## Install

Double-click the `.fb2k-component`, or Preferences → Components → Install. The file carries the
Windows DLL and the Mac bundle; foobar2000 picks its own.
