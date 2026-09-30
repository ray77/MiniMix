#!/usr/bin/env python3
"""Synthetic 31 s test song with song structure (intro, verse, chorus, verse,
chorus at -2 dB, bridge, chorus, chorus, silence). Standard library only.

    python3 tools/gen_test_song.py song.wav
"""
import math, random, struct, sys, wave

SR = 44100


def bar(seconds, notes, noise, seed):
    r = random.Random(seed)
    out = []
    for i in range(int(SR * seconds)):
        t = i / SR
        s = 0.0
        for k, f in enumerate(notes):
            env = math.exp(-2.0 * ((t * 2) % 1.0))
            s += env * 0.25 / (k + 1) * math.sin(2 * math.pi * f * t)
        ph = t % 0.5  # kick every half second
        s += 0.5 * math.exp(-30 * ph) * math.sin(2 * math.pi * 55 * ph)
        out.append((s + noise * (r.random() - 0.5), 0.85 * s + noise * (r.random() - 0.5)))
    return out


def main(path):
    verse = bar(4.0, [220, 277, 330], 0.004, 1)
    chorus = bar(4.0, [262, 330, 392, 523], 0.004, 2)
    bridge = bar(4.0, [196, 247, 294], 0.004, 3)
    intro = bar(2.0, [110], 0.002, 4)
    silence = [(0.0, 0.0)] * SR
    quiet_chorus = [(l * 0.79, r * 0.79) for l, r in chorus]
    song = intro + verse + chorus + verse + quiet_chorus + bridge + chorus + chorus + silence

    def c(x):
        return max(-32768, min(32767, int(round(x * 32767))))

    w = wave.open(path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(b"".join(struct.pack("<hh", c(l), c(r)) for l, r in song))
    w.close()
    print(f"{path}: {len(song) / SR:.1f} s, {len(song) * 4} bytes PCM")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "song.wav")
