#!/usr/bin/env python3
"""
CHIMERA Voice Preset Tuner v2 (tune_voice.py)
---------------------------------------------
Analyzes a target vocal sample (WAV/MP3) using:
  - pYIN pitch tracking (up to C7 for bright/anime/character voices)
  - LPC Formant estimation (F1, F2, F3) to measure true Vocal Tract Length (VTL)
  - Spectral band balance, flatness, and F0 jitter for breathiness, tension, roughness, twang, pressure
  - Sustained-note vibrato rate & depth estimation
  - Optional MFCC distance comparison against CHIMERA's own `render vowels`
Outputs exact CHIMERA Web UI dial values AND a `source` block directly readable by `chimera.exe render notes`.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def estimate_lpc_formants(y: np.ndarray, sr: int, valid_frames: np.ndarray, hop: int) -> Tuple[float, float, float]:
    """Estimates median F1, F2, F3 (Hz) via LPC roots on voiced frames."""
    # Downsample to 10 kHz for clean vocal-tract LPC (order 12)
    sr_lpc = 10000
    y_ds = librosa.resample(y, orig_sr=sr, target_sr=sr_lpc)
    # Pre-emphasis filter
    y_pe = np.append(y_ds[0], y_ds[1:] - 0.97 * y_ds[:-1])
    hop_ds = int(round(hop * sr_lpc / sr))
    win = int(0.032 * sr_lpc)

    f1s, f2s, f3s = [], [], []
    voiced_indices = np.where(valid_frames)[0]
    step = max(1, len(voiced_indices) // 60)  # sample up to ~60 voiced frames

    for idx in voiced_indices[::step]:
        c = idx * hop_ds
        if c + win > len(y_pe):
            break
        frame = y_pe[c : c + win] * np.hamming(win)
        if np.max(np.abs(frame)) < 1e-4:
            continue
        try:
            a = librosa.lpc(frame, order=12)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0.01]
            angs = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angs * (sr_lpc / (2.0 * np.pi))
            bws = -0.5 * (sr_lpc / (2.0 * np.pi)) * np.log(np.abs(roots) + 1e-12)
            formants = sorted(f for f, bw in zip(freqs, bws) if 220 < f < 4200 and bw < 500)
            if len(formants) >= 3:
                f1s.append(formants[0])
                f2s.append(formants[1])
                f3s.append(formants[2])
        except Exception:
            continue

    f1 = float(np.median(f1s)) if f1s else 550.0
    f2 = float(np.median(f2s)) if f2s else 1650.0
    f3 = float(np.median(f3s)) if f3s else 2750.0
    return f1, f2, f3


def analyze_target_voice(audio_path: Path, max_sec: float = 45.0) -> Dict:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if max_sec > 0 and len(y) > int(max_sec * sr):
        y = y[: int(max_sec * sr)]

    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_thresh = max(0.012, float(np.percentile(rms, 60)) * 0.45)

    # 1. Pitch (F0) via pYIN up to C7
    f0, _, _ = librosa.pyin(
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

    # 2. LPC Formants -> Vocal Tract Length (VTL) scale factor
    f1, f2, f3 = estimate_lpc_formants(y, sr, valid_mask, hop)
    # Reference adult female average formants: F1~550, F2~1650, F3~2750
    vtl_ratio = ((f1 / 550.0) * 0.25 + (f2 / 1650.0) * 0.35 + (f3 / 2750.0) * 0.40)

    # 3. Spectral band energy balance
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    S_voiced = S[:, valid_mask]

    def band_db(f_lo: float, f_hi: float) -> float:
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return -80.0
        return 10.0 * np.log10(float(np.mean(S_voiced[m, :])) + 1e-12)

    low_db = band_db(80, 500)
    mid_db = band_db(500, 2000)
    pres_db = band_db(2000, 4000)
    air_db = band_db(4000, 8000)

    tilt_pres = pres_db - low_db
    tilt_air = air_db - mid_db

    # 4. Flatness & jitter -> breathiness & roughness
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S + 1e-12))[0][valid_mask]
    mean_flatness = float(np.median(flatness))

    midi_track = np.where(valid_mask, librosa.hz_to_midi(np.nan_to_num(f0, nan=median_f0)), np.nan)
    diffs = np.abs(np.diff(midi_track))
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs < 2.0]
    jitter_semitones = float(np.median(diffs)) if len(diffs) > 5 else 0.05

    # 5. Vibrato estimation
    frame_rate = sr / hop
    vib_rates, vib_depths_cents = [], []
    run_start = None
    for i, ok in enumerate(valid_mask):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start >= int(0.45 * frame_rate):
                seg = midi_track[run_start:i]
                seg = seg - np.linspace(seg[0], seg[-1], len(seg))
                spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                f_axis = np.fft.rfftfreq(len(seg), d=1.0 / frame_rate)
                v_mask = (f_axis >= 4.0) & (f_axis <= 8.0)
                if np.any(v_mask):
                    idx = int(np.argmax(spec[v_mask]))
                    vib_rates.append(float(f_axis[v_mask][idx]))
                    vib_depths_cents.append(float(np.std(seg) * 141.4))
            run_start = None

    vib_rate_hz = round(float(np.median(vib_rates)), 2) if vib_rates else 5.5
    vib_depth_cents = int(round(clamp(float(np.median(vib_depths_cents)) if vib_depths_cents else 20.0, 0, 80)))

    # 6. Formant + Pitch combined Body Blend (accurate for bright/cute/character voices!)
    # Combine F0 and VTL formant ratio so high-formant voices move cleanly toward child/bright female
    pitch_score = clamp((median_f0 - 110.0) / 240.0, 0.0, 1.3)
    formant_score = clamp((vtl_ratio - 0.80) / 0.35, 0.0, 1.3)
    combined = 0.55 * pitch_score + 0.45 * formant_score

    if combined < 0.40:
        female_w = clamp(combined * 1.1, 0.0, 0.40)
        child_w = 0.0
    elif combined < 0.82:
        female_w = clamp(0.45 + (combined - 0.40) * 1.3, 0.45, 1.0)
        child_w = clamp((combined - 0.55) * 1.1, 0.0, 0.45)
    else:
        child_w = clamp(0.45 + (combined - 0.82) * 1.4, 0.45, 1.0)
        female_w = clamp(1.0 - child_w * 0.65, 0.15, 0.70)

    if child_w >= 0.45:
        cli_body = "child"
    elif female_w >= 0.45:
        cli_body = "female"
    else:
        cli_body = "male"

    # Expressive timbre parameters (wider dynamic range so custom synths sound noticeably distinct!)
    energy_rms = float(np.percentile(rms[valid_mask], 80))
    pressure_val = round(clamp(7.5 + energy_rms * 32.0 + (tilt_pres + 18.0) * 0.12, 6.0, 16.5), 2)
    breathy_pct = int(round(clamp((tilt_air + 32.0) * 1.9 + mean_flatness * 160.0, 5, 85)))
    pressed_pct = int(round(clamp((tilt_pres + 22.0) * 2.5, 15, 92)))
    twang_pct = int(round(clamp((pres_db - mid_db + 20.0) * 3.0 + (vtl_ratio - 1.0) * 80.0, 10, 95)))
    rough_pct = int(round(clamp((jitter_semitones - 0.035) * 300.0, 0, 75)))
    fry_pct = int(round(clamp((130.0 - p10_f0) * 0.9, 0, 45))) if p10_f0 < 130 else 0
    larynx_pct = int(round(clamp(50.0 + (vtl_ratio - 1.0) * 110.0 + (median_f0 - 210.0) * 0.08, 10, 95)))
    ring_pct = int(round(clamp((pres_db - low_db + 18.0) * 2.6, 5, 90)))

    # Native CHIMERA SourcePatch dict (used directly by patched render.rs!)
    source_patch = {
        "pressure": pressure_val,
        "tension": round(pressed_pct / 100.0, 3),
        "breathiness": round(breathy_pct / 100.0, 3),
        "roughness": round(rough_pct / 100.0, 3),
        "fry": round(fry_pct / 100.0, 3),
        "vibrato_rate": vib_rate_hz,
        "glide": 0.045,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[:, valid_mask]
    mfcc_mean = np.mean(mfcc[1:], axis=1)

    return {
        "source_file": str(audio_path),
        "acoustics": {
            "median_f0_hz": round(median_f0, 1),
            "median_midi": round(median_midi, 1),
            "f0_range_hz": [round(p10_f0, 1), round(p90_f0, 1)],
            "formants_hz": {"F1": round(f1, 0), "F2": round(f2, 0), "F3": round(f3, 0)},
            "vtl_ratio": round(vtl_ratio, 3),
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
            "source": source_patch,
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
                scores[body] = round(float(np.linalg.norm(target_mfcc - cand)), 2)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a target voice and compute CHIMERA synth preset values.")
    parser.add_argument("--audio", required=True, help="Path to clean vocal audio file (WAV/MP3)")
    parser.add_argument("--max-sec", type=float, default=45.0, help="Max seconds to analyze (default: 45)")
    parser.add_argument("--out", default="", help="Optional path to save preset JSON")
    parser.add_argument("--compare-chimera", action="store_true", help="Compare MFCC distance against chimera.exe")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = analyze_target_voice(audio_path, max_sec=args.max_sec)
    target_mfcc = np.array(result.pop("_mfcc"), dtype=np.float32)

    if args.compare_chimera:
        exe = find_chimera_exe(Path.cwd())
        if exe:
            scores = compare_against_chimera_bodies(exe, target_mfcc)
            if scores:
                result["chimera_mfcc_distance"] = scores
                result["chimera_preset"]["cli_body_by_timbre"] = min(scores, key=scores.get)

    p = result["chimera_preset"]
    ac = result["acoustics"]
    print("\n================ CHIMERA VOICE PRESET v2 ================")
    print(f"Target Audio : {audio_path.name}")
    print(f"Pitch / VTL  : {ac['median_f0_hz']} Hz (MIDI {ac['median_midi']}) | Formants {ac['formants_hz']} (VTL={ac['vtl_ratio']})")
    print(f"Body Blend   : {p['body_blend']} (CLI: {p['cli_body']})")
    print(f"SourcePatch  : {p['source']}")
    print("=========================================================\n")

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
