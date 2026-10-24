"""Lip sync: move the duck's jaw with its voice.

The robot's speech (text-to-speech output or a recording) goes to the speaker; the same samples go
through `MouthSync`, which turns loudness into a jaw angle at the 50 Hz control rate:

    sync = MouthSync(sample_rate=22050)
    for chunk in audio_chunks:            # any size; 20 ms (441 samples) lines up with the control loop
        jaw = sync.feed(chunk)            # radians, 0 = closed … MAX_OPEN
        robot.set_target("jaw", jaw)

Loud syllables open the mouth wide, quiet ones a little, silence closes it. The mouth opens fast and
closes a bit slower, like a real one, and never moves faster than the servo can follow, so it doesn't
buzz. Gain adapts to the voice, so a quiet or loud speaker both use the full range.

    python body/mouth.py              # demo on a synthetic "voice", prints the jaw track
"""
import numpy as np

MAX_OPEN = 0.61          # rad, the jaw's range (35°)
CTRL_HZ = 50
OPEN_TAU, CLOSE_TAU = 0.03, 0.08     # s: opens quickly, closes a bit slower
MAX_SPEED = 8.0          # rad/s the small jaw servo can manage
GATE = 0.08              # below this fraction of the recent peak loudness counts as silence


class MouthSync:
    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.hop = sample_rate // CTRL_HZ
        self.buf = np.zeros(0)
        self.peak = 1e-3
        self.level = 0.0
        self.jaw = 0.0

    def _step(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(float) ** 2)))
        self.peak = max(rms, self.peak * 0.995)                       # slowly forget loud moments
        x = rms / self.peak
        x = 0.0 if x < GATE else (x - GATE) / (1 - GATE)
        target = MAX_OPEN * min(1.0, x ** 0.7)                         # quiet sounds still show
        dt = 1 / CTRL_HZ
        tau = OPEN_TAU if target > self.level else CLOSE_TAU
        self.level += (target - self.level) * (1 - np.exp(-dt / tau))
        self.jaw += np.clip(self.level - self.jaw, -MAX_SPEED * dt, MAX_SPEED * dt)
        return self.jaw

    def feed(self, samples):
        """Add audio samples; returns the jaw angle after the newest complete 20 ms frame."""
        self.buf = np.concatenate([self.buf, np.asarray(samples, float).ravel()])
        while len(self.buf) >= self.hop:
            self._step(self.buf[:self.hop])
            self.buf = self.buf[self.hop:]
        return self.jaw

    def silence(self):
        """Speech stopped: close the mouth over the next frames."""
        return self.feed(np.zeros(self.hop))


def demo_voice(seconds=2.0, sr=22050, seed=0):
    """A voice-like test signal: syllables of a buzzy tone at 4-6 per second, with pauses."""
    r = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = np.zeros_like(t)
    s = 0.1
    while s < seconds - 0.2:
        n = int(r.integers(2, 6))                                      # a word of 2-5 syllables
        for _ in range(n):
            d = r.uniform(0.12, 0.22)
            mask = (t >= s) & (t < s + d)
            env[mask] = r.uniform(0.3, 1.0) * np.sin(np.pi * (t[mask] - s) / d)
            s += d + 0.03
        s += r.uniform(0.15, 0.4)                                      # pause between words
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    tone = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / sr)) * 0.5 + 0.3 * r.normal(size=t.size)
    return (env * tone).astype(np.float32), env


if __name__ == "__main__":
    sr = 22050
    audio, env = demo_voice(sr=sr)
    sync = MouthSync(sr)
    track = [sync.feed(audio[i:i + sync.hop]) for i in range(0, len(audio), sync.hop)]
    for i, j in enumerate(track):
        print(f"{i / CTRL_HZ:5.2f} s  {'#' * int(40 * j / MAX_OPEN):40s} {np.degrees(j):4.1f}°")
