#include "stdafx.h"
#include "version.h"

DECLARE_COMPONENT_VERSION("MiniMix decoder", MMX_COMPONENT_VERSION,
    "Plays MiniMix (.mmx) files: lossless, near-lossless and lossy.\n"
    "Lossless files with entry points (patchwork landscape decoding) decode in 30-second segments on several cores; a\n"
    "seek decodes to its target, or lands on a nearby segment start when that is much cheaper. Older files are\n"
    "resolved front to back.\n"
    "https://github.com/ray77/MiniMix");

// Keeps the file name fixed, so the troubleshooter can tell versions apart (a no-op on macOS).
VALIDATE_COMPONENT_FILENAME("foo_input_mmx.dll");
