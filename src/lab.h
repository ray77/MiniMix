#ifndef MMX_LAB_H
#define MMX_LAB_H
#include <stdlib.h>
/* Lab switches: environment variables that change what the codec computes (prediction constants, format profiles,
   bit allocation, debug probes). A released build must produce the same bits whatever the environment holds - the
   foobar2000 component runs inside a host process whose environment nobody checks - so they are read only in a lab
   build (make MMX_LAB=1). Otherwise every one of them reads as unset and the frozen defaults apply. Variables that
   only change speed (MMX_THREADS, MMX_DECODE_THREADS, MMX_DECODE_LANES) and the encoder's own choices that the file
   records (MMX_PLD) are read with getenv directly. */
#ifndef MMX_LAB
#define MMX_LAB 0
#endif
#define mmx_lab_getenv(name) (MMX_LAB ? getenv(name) : (const char *)0)
#endif
