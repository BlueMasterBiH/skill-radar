"""Original background music for a dental-gold reel, mixed under the voice with ducking.

usage: music.py <workdir> [--track song.mp3] [--volume 0.2] [--seed 1]
Reads <workdir>/dental-gold.mp4 and writes <workdir>/dental-gold_music.mp4.
Without --track it synthesises a calm, warm track (A minor, 92 bpm: pad, plucked arpeggio, soft kick and
shaker) of the video's length. It is generated here, so there is no licence to clear, also for paid ads.
With --track, the given song is looped or trimmed to the video and mixed the same way.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env(n, a, r):
    e = np.ones(n)
    ai, ri = min(n, int(a * SR)), min(n, int(r * SR))
    e[:ai] = np.linspace(0, 1, ai)
    if ri:
        e[-ri:] *= np.linspace(1, 0, ri)
    return e


def pad_voice(f, n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.12, 0.0, 0.11):           # detuned, soft-saw (few harmonics, 1/h^1.6)
        ff = f * 2 ** (det / 12)
        ph = rng.uniform(0, 2 * np.pi)
        for h in range(1, 7):
            out += np.sin(2 * np.pi * ff * h * t + ph * h) / h ** 1.6
    return out / 3


def pluck(f, n):
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)) \
        * np.exp(-t * 6.0) * env(n, 0.004, 0.05)


def kick(n):
    t = np.arange(n) / SR
    f = 45 + 75 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def synth(dur, seed=1):
    rng = np.random.default_rng(seed)
    bpm = 92
    beat = 60 / bpm
    bar = 4 * beat
    # Am  F  C  G  (two rounds), Am  F  C  E(sus) turnaround
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    roots = [45, 41, 48, 43]
    n = int((dur + 2) * SR)
    pad, arp, low, drums = (np.zeros(n) for _ in range(4))
    nb = int(np.ceil(dur / bar)) + 1
    for b in range(nb):
        i0 = int(b * bar * SR)
        ch, rt = chords[b % 4], roots[b % 4]
        ln = int(bar * SR * 1.25)
        seg = sum(pad_voice(midi(m), ln, rng) for m in ch) * env(ln, 0.6, 0.9)
        j1 = min(n, i0 + ln)
        pad[i0:j1] += seg[:j1 - i0]
        bl = int(bar * SR)
        tt = np.arange(bl) / SR
        bass = np.sin(2 * np.pi * midi(rt) * tt) * env(bl, 0.05, 0.3)
        j1 = min(n, i0 + bl)
        low[i0:j1] += bass[:j1 - i0]
        pattern = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12, ch[0] + 24, ch[2] + 12, ch[1] + 12, ch[2] + 12]
        build = b >= 2                       # the arpeggio enters after two bars
        for k in range(8):
            s0 = i0 + int(k * beat / 2 * SR)
            if build and s0 < n:
                pl = pluck(midi(pattern[k]), int(0.9 * SR)) * (0.9 if k % 2 == 0 else 0.6)
                j1 = min(n, s0 + len(pl))
                arp[s0:j1] += pl[:j1 - s0]
        if b >= 4:                            # soft kick on 1 and 3, shaker on the off-beats
            for k in (0, 2):
                s0 = i0 + int(k * beat * SR)
                kk = kick(int(0.4 * SR))
                j1 = min(n, s0 + len(kk))
                drums[s0:j1] += kk[:j1 - s0] * 0.9
            for k in range(8):
                if k % 2 == 1:
                    s0 = i0 + int(k * beat / 2 * SR)
                    ln2 = int(0.07 * SR)
                    nz = np.diff(rng.standard_normal(ln2 + 1)) * np.exp(-np.arange(ln2) / SR * 60)
                    j1 = min(n, s0 + ln2)
                    drums[s0:j1] += nz[:j1 - s0] * 0.08
    mix = 0.20 * pad + 0.22 * arp + 0.30 * low + 0.35 * drums
    # simple stereo reverb: a few feedback delays per side
    L, R = mix.copy(), mix.copy()
    for d, g, side in ((0.031, 0.35, L), (0.037, 0.33, R), (0.047, 0.3, L), (0.053, 0.28, R), (0.29, 0.18, L), (0.33, 0.16, R)):
        k = int(d * SR)
        y = side.copy()
        for _ in range(4):
            y[k:] += g * y[:-k]
        side[:] = 0.6 * side + 0.4 * y / (1 + 4 * g)
    st = np.stack([L, R], 1)[: int(dur * SR)]
    fi, fo = int(0.8 * SR), int(2.5 * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    st[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return st / (np.abs(st).max() + 1e-9) * 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--track")
    ap.add_argument("--volume", type=float, default=0.2, help="music level under the voice (0.1 quiet - 0.35 loud)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    wd = Path(a.workdir)
    vid = wd / "dental-gold.mp4"
    if not vid.exists():
        sys.exit("render dental-gold.mp4 first")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vid)],
                               capture_output=True, text=True, check=True).stdout)
    if a.track:
        src = ["-stream_loop", "-1", "-i", a.track]
    else:
        wav = wd / "music.wav"
        st = (synth(dur, a.seed) * 32767).astype("<i2")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-", str(wav)],
                       input=st.tobytes(), check=True)
        src = ["-i", str(wav)]
    fc = (f"[1:a]atrim=0:{dur:.2f},afade=t=out:st={max(0, dur - 2.5):.2f}:d=2.5,volume={a.volume}[m];"
          "[0:a]asplit=2[v][sc];"
          "[m][sc]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[md];"
          "[v][md]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    out = wd / "dental-gold_music.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), *src, "-filter_complex", fc, "-map", "0:v",
                    "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    print("OUTPUT", out)


if __name__ == "__main__":
    main()
