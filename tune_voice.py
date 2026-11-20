#!/usr/bin/env python3
"""
CHIMERA Voice Preset Tuner (tune_voice.py)
------------------------------------------
Analyzes a target vocal sample (WAV/MP3, e.g. a Demucs-separated vocals.wav or a
10-30s clean voice recording) and computes the closest CHIMERA physical-modeling
parameters:
  - Continuous Body blend (male / female / child)
  - Pitch register (median F0 Hz & MIDI)
  - Spectral tilt & brightness -> pressed, twang, larynx
  - High-frequency noise & pitch jitter -> breathy, rough, fry
  - Sustained-note F0 modulation -> vibrato rate (Hz) & depth (cents)

Optionally compares the target sample's MFCC timbre against CHIMERA's own
`chimera.exe render vowels` outputs to verify the best base body preset!

Usage:
  python tune_voice.py --audio vocals.wav
  python tune_voice.py --audio vocals.wav --out my_preset.json --compare-chimera
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def find_chimera_exe(repo_root: Path) -> Optional[Path]:
    for c in [
        repo_root / "target" / "release" / "chimera.exe",
        repo_root / "target" / "release" / "chimera",
    ]:
        if c.exists():
            return c
    return None


def analyze_target_voice(audio_path: Path, max_sec: float = 45.0) -> Dict:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if max_sec > 0 and len(y) > int(max_sec * sr):
        y = y[: int(max_sec * sr)]

    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_thresh = max(0.012, float(np.percentile(rms, 60)) * 0.45)

    # 1. Pitch (F0) via pYIN
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop,
    )
    valid_mask = (~np.isnan(f0)) & (rms > rms_thresh)
    valid_f0 = f0[valid_mask]
    if len(valid_f0) < 10:
        raise RuntimeError("Could not detect enough voiced frames in the audio sample.")

    median_f0 = float(np.median(valid_f0))
    p10_f0 = float(np.percentile(valid_f0, 10))
    p90_f0 = float(np.percentile(valid_f0, 90))
    median_midi = float(librosa.hz_to_midi(median_f0))

    # 2. Spectral band energy balance (matching CHIMERA's CSD diagnostic bands)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    S_voiced = S[:, valid_mask]

    def band_db(f_lo: float, f_hi: float) -> float:
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return -80.0
        val = float(np.mean(S_voiced[m, :]))
        return 10.0 * np.log10(val + 1e-12)

    low_db = band_db(80, 500)
    mid_db = band_db(500, 2000)
    pres_db = band_db(2000, 4000)
    air_db = band_db(4000, 8000)

    # Spectral tilt & brightness
    tilt_pres = pres_db - low_db      # higher => brighter / more twang & pressed
    tilt_air = air_db - mid_db        # higher => more breathy / airy

    # 3. Spectral flatness & pitch jitter -> breathiness & roughness
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S + 1e-12))[0][valid_mask]
    mean_flatness = float(np.median(flatness))

    # Frame-to-frame F0 jitter in semitones on contiguous voiced frames
    midi_track = np.where(valid_mask, librosa.hz_to_midi(np.nan_to_num(f0, nan=median_f0)), np.nan)
    diffs = np.abs(np.diff(midi_track))
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs < 2.0]  # ignore octave jumps
    jitter_semitones = float(np.median(diffs)) if len(diffs) > 5 else 0.05

    # 4. Vibrato estimation on sustained voiced runs (>= 0.45s)
    frame_rate = sr / hop
    vib_rates = []
    vib_depths_cents = []
    run_start = None
    for i, ok in enumerate(valid_mask):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start >= int(0.45 * frame_rate):
                seg = midi_track[run_start:i]
                seg = seg - np.linspace(seg[0], seg[-1], len(seg))  # detrend
                spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                f_axis = np.fft.rfftfreq(len(seg), d=1.0 / frame_rate)
                v_mask = (f_axis >= 4.0) & (f_axis <= 8.0)
                if np.any(v_mask):
                    idx = int(np.argmax(spec[v_mask]))
                    vib_rates.append(float(f_axis[v_mask][idx]))
                    vib_depths_cents.append(float(np.std(seg) * 141.4))  # RMS->peak cents
            run_start = None

    vib_rate_hz = round(float(np.median(vib_rates)), 2) if vib_rates else 5.5
    vib_depth_cents = int(round(clamp(float(np.median(vib_depths_cents)) if vib_depths_cents else 25.0, 0, 80)))

    # 5. Map acoustic features to CHIMERA body blend & encoder parameters (0..100 scale)
    # Body blend: male (~120 Hz), female (~230 Hz), child (~330+ Hz)
    if median_f0 <= 165.0:
        female_w = clamp((median_f0 - 115.0) / 110.0, 0.0, 0.45)
        child_w = 0.0
    elif median_f0 <= 290.0:
        female_w = clamp((median_f0 - 140.0) / 120.0, 0.45, 1.0)
        child_w = clamp((median_f0 - 250.0) / 140.0, 0.0, 0.35)
    else:
        child_w = clamp((median_f0 - 260.0) / 140.0, 0.35, 1.0)
        female_w = clamp(1.0 - child_w * 0.7, 0.15, 0.7)

    # Base CLI preset recommendation
    if child_w >= 0.45:
        cli_body = "child"
    elif female_w >= 0.45:
        cli_body = "female"
    else:
        cli_body = "male"

    # CHIMERA encoder estimates (0-100%)
    breathy_pct = int(round(clamp((tilt_air + 34.0) * 2.2 + mean_flatness * 180.0, 5, 85)))
    pressed_pct = int(round(clamp((tilt_pres + 26.0) * 2.4, 10, 90)))
    twang_pct = int(round(clamp((pres_db - mid_db + 22.0) * 2.8, 5, 90)))
    rough_pct = int(round(clamp((jitter_semitones - 0.03) * 320.0, 0, 75)))
    fry_pct = int(round(clamp((130.0 - p10_f0) * 0.9, 0, 45))) if p10_f0 < 130 else 0
    larynx_pct = int(round(clamp(50.0 + (median_f0 - 210.0) * 0.15 + (tilt_pres + 15.0) * 0.6, 15, 85)))
    ring_pct = int(round(clamp((pres_db - low_db + 20.0) * 2.5, 0, 85)))

    # MFCC signature for optional CHIMERA candidate comparison
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[:, valid_mask]
    mfcc_mean = np.mean(mfcc[1:], axis=1)  # drop c0 energy

    return {
        "source_file": str(audio_path),
        "acoustics": {
            "median_f0_hz": round(median_f0, 1),
            "median_midi": round(median_midi, 1),
            "f0_range_hz": [round(p10_f0, 1), round(p90_f0, 1)],
            "bands_db": {
                "80_500hz": round(low_db, 1),
                "500_2000hz": round(mid_db, 1),
                "2000_4000hz": round(pres_db, 1),
                "4000_8000hz": round(air_db, 1),
            },
            "jitter_semitones": round(jitter_semitones, 4),
            "spectral_flatness": round(mean_flatness, 4),
        },
        "chimera_preset": {
            "cli_body": cli_body,
            "body_blend": {
                "male": round(max(0.0, 1.0 - max(female_w, child_w)), 2),
                "female": round(female_w, 2),
                "child": round(child_w, 2),
            },
            "play_voice": {
                "pitch_hz": round(median_f0, 1),
                "pressed_pct": pressed_pct,
                "vibrato_cents": vib_depth_cents,
            },
            "play_tone": {
                "breathy_pct": breathy_pct,
                "rough_pct": rough_pct,
                "fry_pct": fry_pct,
            },
            "play_throat_larynx": {
                "larynx_pct": larynx_pct,
                "twang_pct": twang_pct,
            },
            "sing_encoders": {
                "vibrato_depth_cents": vib_depth_cents,
                "vibrato_rate_hz": vib_rate_hz,
                "breathy_pct": breathy_pct,
                "ring_pct": ring_pct,
            },
        },
        "_mfcc": mfcc_mean.tolist(),
    }


def compare_against_chimera_bodies(chimera_exe: Path, target_mfcc: np.ndarray) -> Dict[str, float]:
    """Renders CHIMERA's 3 base bodies (`render vowels`) and computes MFCC Euclidean distance."""
    scores: Dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        for body in ("male", "female", "child"):
            out_stem = Path(td) / f"vowels_{body}"
            res = subprocess.run(
                [str(chimera_exe), "render", "vowels", body, str(out_stem)],
                capture_output=True,
                text=True,
            )
            wav_path = out_stem.with_suffix(".wav")
            if res.returncode == 0 and wav_path.exists():
                y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                cand = np.mean(mfcc[1:], axis=1)
                dist = float(np.linalg.norm(target_mfcc - cand))
                scores[body] = round(dist, 2)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a target voice and recommend CHIMERA synth preset values.")
    parser.add_argument("--audio", required=True, help="Path to clean vocal audio file (WAV/MP3)")
    parser.add_argument("--max-sec", type=float, default=45.0, help="Max seconds to analyze (default: 45)")
    parser.add_argument("--out", default="", help="Optional path to save preset JSON (e.g. my_voice_preset.json)")
    parser.add_argument("--compare-chimera", action="store_true", help="Also compare MFCC distance against chimera.exe bodies")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = analyze_target_voice(audio_path, max_sec=args.max_sec)
    target_mfcc = np.array(result.pop("_mfcc"), dtype=np.float32)

    if args.compare_chimera:
        exe = find_chimera_exe(Path.cwd())
        if exe:
            print("Comparing timbre against CHIMERA male / female / child bodies...")
            scores = compare_against_chimera_bodies(exe, target_mfcc)
            if scores:
                result["chimera_mfcc_distance"] = scores
                best = min(scores, key=scores.get)
                result["chimera_preset"]["cli_body_by_timbre"] = best

    p = result["chimera_preset"]
    ac = result["acoustics"]

    print("\n================ CHIMERA VOICE PRESET ================")
    print(f"Target Audio   : {audio_path.name}")
    print(f"Median Pitch   : {ac['median_f0_hz']} Hz (MIDI {ac['median_midi']}, range {ac['f0_range_hz'][0]}~{ac['f0_range_hz'][1]} Hz)")
    print(f"CLI Base Body  : {p['cli_body']}  |  BodyPad Blend: {p['body_blend']}")
    if "chimera_mfcc_distance" in result:
        print(f"MFCC Distance  : {result['chimera_mfcc_distance']} (lower = closer timbre)")
    print("\n[Play Mode Encoders]")
    print(f"  VOICE        : pitch={p['play_voice']['pitch_hz']} Hz, pressed={p['play_voice']['pressed_pct']}%, vibrato={p['play_voice']['vibrato_cents']} cents")
    print(f"  TONE         : breathy={p['play_tone']['breathy_pct']}%, rough={p['play_tone']['rough_pct']}%, fry={p['play_tone']['fry_pct']}%")
    print(f"  THROAT/LARYNX: larynx={p['play_throat_larynx']['larynx_pct']}%, twang={p['play_throat_larynx']['twang_pct']}%")
    print("\n[Sing Mode Encoders]")
    print(f"  SONG         : vibrato={p['sing_encoders']['vibrato_depth_cents']} cents @ {p['sing_encoders']['vibrato_rate_hz']} Hz, breathy={p['sing_encoders']['breathy_pct']}%, ring={p['sing_encoders']['ring_pct']}%")
    print("======================================================\n")

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved preset JSON -> {args.out}")


if __name__ == "__main__":
    main()