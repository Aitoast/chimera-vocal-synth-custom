#!/usr/bin/env python3
"""
CHIMERA Auto-Show Pipeline v2 (auto_show.py)
--------------------------------------------
1. Takes a YouTube URL (--url) or local audio file (--audio)
   - Lyrics (--lyrics) are OPTIONAL! When omitted, automatically transcribes lyrics &
     word-level timestamps from the separated vocal stem using `faster-whisper`, and
     fills un-transcribed vocal/intro scat sections (e.g., "bum bum bum") automatically!
2. Separates vocals & instrumental MR with Demucs
3. Extracts BPM and vocal pitch contour with librosa (pYIN)
4. Matches exact CHIMERA syllable counts via `chimera.exe syllables`
5. Automatically assigns phrases across multiple singers (male / female / child)
6. Renders each voice's WAV + 3D animation telemetry (.tele) with `chimera.exe`
7. Mixes the extracted MR + all stereo-panned CHIMERA vocals into `mix.wav`
8. Writes `public/shows/<id>/show.json` for instant 3D playback in the CHIMERA Show UI!

Install:
  pip install yt-dlp demucs librosa soundfile numpy faster-whisper
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


CHIMERA_T0 = 0.3  # Default onset time (seconds) used by `chimera render notes`


def run_cmd(cmd: List[str], desc: str) -> None:
    print(f"\n[STEP] {desc}", flush=True)
    print("  >", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {desc}")


def find_chimera_exe(repo_root: Path) -> Path:
    candidates = [
        repo_root / "target" / "release" / "chimera.exe",
        repo_root / "target" / "release" / "chimera",
        repo_root / "target" / "debug" / "chimera.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "chimera.exe not found. Run `cargo build --release -p chimera-cli` first."
    )


def find_ffmpeg_dir() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)
    local_app = os.environ.get("LOCALAPPDATA", "")
    candidates: List[Path] = []
    if local_app:
        candidates.append(Path(local_app) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
        pkg_root = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if pkg_root.exists():
            candidates.extend(pkg_root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
    candidates.append(Path("C:/ffmpeg/bin/ffmpeg.exe"))
    for c in candidates:
        if c.exists():
            d = str(c.parent)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            print(f"  Found ffmpeg automatically: {d}", flush=True)
            return d
    return None


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    out_tpl = str(work_dir / "source.%(ext)s")
    ff_dir = find_ffmpeg_dir()
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--newline",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        out_tpl,
    ]
    if ff_dir:
        cmd.extend(["--ffmpeg-location", ff_dir])
    cmd.append(url)
    run_cmd(cmd, "Downloading audio with yt-dlp")
    wavs = list(work_dir.glob("source*.wav"))
    if not wavs:
        raise FileNotFoundError("yt-dlp did not produce a .wav file (check ffmpeg/yt-dlp installation).")
    return wavs[0]


def separate_stems_demucs(audio_path: Path, work_dir: Path) -> Tuple[Path, Path]:
    sep_dir = work_dir / "separated"
    run_cmd(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            "-o",
            str(sep_dir),
            str(audio_path),
        ],
        "Separating vocals & MR with Demucs",
    )
    stem_folder = sep_dir / "htdemucs" / audio_path.stem
    vocals = stem_folder / "vocals.wav"
    no_vocals = stem_folder / "no_vocals.wav"
    if not vocals.exists() or not no_vocals.exists():
        vocals = next(sep_dir.rglob("vocals.wav"))
        no_vocals = next(sep_dir.rglob("no_vocals.wav"))
    return vocals, no_vocals


def sanitize_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    text = re.sub(r"[^\w\s가-힣']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_chimera_syllables(chimera_exe: Path, text: str) -> int:
    clean = text.strip()
    if not clean:
        return 0
    res = subprocess.run(
        [str(chimera_exe), "syllables", clean],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            if isinstance(data, list) and len(data) > 0:
                return len(data)
        except json.JSONDecodeError:
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            if lines:
                return len(lines)
    hangul = len(re.findall(r"[가-힣]", clean))
    latin_words = re.findall(r"[A-Za-z']+", clean)
    latin_syls = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in latin_words)
    return max(1, hangul + latin_syls)


def sample_pitch_midi(times: np.ndarray, f0: np.ndarray, s0: float, s1: float, fallback: int = 65) -> int:
    i0 = int(np.searchsorted(times, s0))
    i1 = max(i0 + 1, int(np.searchsorted(times, s1)))
    seg_f0 = f0[i0:i1]
    seg_f0 = seg_f0[~np.isnan(seg_f0)]
    if len(seg_f0) > 0:
        midi_val = int(round(float(librosa.hz_to_midi(np.median(seg_f0)))))
        return max(40, min(84, midi_val))
    return fallback


def analyze_vocals_pitch(vocals_wav: Path, max_duration: float = 0.0):
    print("\n[STEP] Analyzing vocal pitch & BPM with librosa (pYIN)...", flush=True)
    y, sr = librosa.load(str(vocals_wav), sr=22050, mono=True)
    if max_duration > 0:
        y = y[: int(max_duration * sr)]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    if not (55.0 <= bpm <= 190.0):
        bpm = 110.0
    bpm = round(bpm, 1)
    dur_sec = len(y) / sr
    print(f"  Estimated BPM: {bpm}, audio length: {dur_sec:.1f}s", flush=True)

    hop = 256
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop,
    )
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    return y, sr, bpm, dur_sec, times, f0, rms


def extract_phrases_with_whisper(
    vocals_wav: Path,
    chimera_exe: Path,
    y: np.ndarray,
    sr: int,
    bpm: float,
    dur_sec: float,
    times: np.ndarray,
    f0: np.ndarray,
    rms: np.ndarray,
    language: Optional[str] = None,
    whisper_model_name: str = "small",
    work_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Transcribes vocals with faster-whisper (word_timestamps=True) and automatically
    fills active vocal intro/interlude regions (e.g. 'bum bum bum') that Whisper skips.
    """
    print(f"\n[STEP] Auto-extracting lyrics & timestamps with faster-whisper ({whisper_model_name})...", flush=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    # If trimmed by max_duration, write a temporary trimmed vocal wav for Whisper
    whisper_input = vocals_wav
    if work_dir is not None and dur_sec < (sf.info(str(vocals_wav)).duration - 0.5):
        whisper_input = work_dir / "vocals_trimmed.wav"
        sf.write(str(whisper_input), y, sr)

    # Try CUDA float16 first, safely fall back to CPU int8 if cuDNN DLLs are missing
    model = None
    try:
        import torch
        if torch.cuda.is_available():
            try:
                model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")
                print("  faster-whisper running on CUDA (float16)", flush=True)
            except Exception as cuda_err:
                print(f"  CUDA init fallback to CPU ({cuda_err})", flush=True)
    except Exception:
        pass

    if model is None:
        model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")
        print("  faster-whisper running on CPU (int8)", flush=True)

    lang_arg = None if (not language or language == "auto") else language
    segments_gen, info = model.transcribe(
        str(whisper_input),
        language=lang_arg,
        word_timestamps=True,
        vad_filter=True,
    )
    print(f"  Detected language: {info.language} (prob={info.language_probability:.2f})", flush=True)

    phrases: List[dict] = []
    covered_intervals: List[Tuple[float, float]] = []
    last_midi = 65

    for seg in segments_gen:
        seg_words = getattr(seg, "words", None) or []
        line_words: List[str] = []
        notes_sec: List[Tuple[int, float, float]] = []

        for w in seg_words:
            clean_w = sanitize_text(w.word)
            if not clean_w:
                continue
            w_start = max(0.0, float(w.start))
            w_end = min(dur_sec, max(w_start + 0.08, float(w.end)))
            if w_start >= dur_sec:
                continue

            n_syl = count_chimera_syllables(chimera_exe, clean_w)
            slot = max(0.06, (w_end - w_start) / max(1, n_syl))
            line_words.append(clean_w)
            for k in range(n_syl):
                s0 = w_start + k * slot
                s1 = s0 + slot
                m = sample_pitch_midi(times, f0, s0, s1, fallback=last_midi)
                last_midi = m
                notes_sec.append((m, s0, s1))

        if line_words and notes_sec:
            t_st = notes_sec[0][1]
            t_en = notes_sec[-1][2]
            covered_intervals.append((t_st, t_en))
            median_m = int(round(float(np.median([m for m, _, _ in notes_sec]))))
            line_text = " ".join(line_words)
            print(f"  [{t_st:5.1f}s - {t_en:5.1f}s] {line_text}", flush=True)
            phrases.append(
                {
                    "text": line_text,
                    "t_start": t_st,
                    "t_end": t_en,
                    "median_midi": median_m,
                    "notes_sec": notes_sec,
                }
            )

    # Detect active vocal regions NOT covered by Whisper (e.g. "bum bum bum" intro/scat!)
    rms_thresh = max(0.02, float(np.percentile(rms, 65)) * 0.30)
    active = (rms > rms_thresh) & (~np.isnan(f0))

    uncovered = active.copy()
    for c_st, c_en in covered_intervals:
        mask = (times >= c_st - 0.20) & (times <= c_en + 0.20)
        uncovered[mask] = False

    # Group contiguous uncovered vocal regions
    scat_regions: List[Tuple[float, float]] = []
    in_r = False
    r_st = 0.0
    for t, flag in zip(times, uncovered):
        if flag and not in_r:
            r_st = float(t)
            in_r = True
        elif not flag and in_r:
            if float(t) - r_st >= 0.35:
                scat_regions.append((r_st, float(t)))
            in_r = False
    if in_r and float(times[-1]) - r_st >= 0.35:
        scat_regions.append((r_st, float(times[-1])))

    # Merge nearby scat regions (< 0.45s gap)
    merged_scat: List[Tuple[float, float]] = []
    for st, en in scat_regions:
        if merged_scat and st - merged_scat[-1][1] < 0.45:
            merged_scat[-1] = (merged_scat[-1][0], en)
        else:
            merged_scat.append((st, en))

    # Subdivide each uncovered vocal region into 0.5-beat scat syllables ("bum")
    half_beat_sec = (60.0 / bpm) * 0.5
    scat_added = 0
    for st, en in merged_scat:
        seg_dur = en - st
        n_pulses = max(1, min(16, int(round(seg_dur / half_beat_sec))))
        step = seg_dur / n_pulses
        scat_words: List[str] = []
        scat_notes: List[Tuple[int, float, float]] = []
        for k in range(n_pulses):
            s0 = st + k * step
            s1 = s0 + step
            m = sample_pitch_midi(times, f0, s0, s1, fallback=last_midi)
            last_midi = m
            scat_words.append("bum")
            scat_notes.append((m, s0, s1))
        if scat_notes:
            median_m = int(round(float(np.median([m for m, _, _ in scat_notes]))))
            phrases.append(
                {
                    "text": " ".join(scat_words),
                    "t_start": st,
                    "t_end": en,
                    "median_midi": median_m,
                    "notes_sec": scat_notes,
                }
            )
            scat_added += 1
            print(f"  [{st:5.1f}s - {en:5.1f}s] (auto-scat) {' '.join(scat_words)}", flush=True)

    phrases.sort(key=lambda p: p["t_start"])
    if not phrases:
        raise RuntimeError("No vocal lyrics or pitches could be extracted from the audio.")
    print(f"  Total phrases: {len(phrases)} (including {scat_added} auto-scat intro/interlude regions)", flush=True)
    return phrases


def extract_phrases_manual_lyrics(
    lyric_lines: List[str],
    syl_counts: List[int],
    dur_sec: float,
    times: np.ndarray,
    f0: np.ndarray,
    rms: np.ndarray,
) -> List[dict]:
    rms_thresh = max(0.015, float(np.percentile(rms, 65)) * 0.25)
    active = (rms > rms_thresh) & (~np.isnan(f0))

    regions: List[Tuple[float, float]] = []
    in_reg = False
    reg_start = 0.0
    for t, is_act in zip(times, active):
        if is_act and not in_reg:
            reg_start = float(t)
            in_reg = True
        elif not is_act and in_reg:
            if float(t) - reg_start >= 0.15:
                regions.append((reg_start, float(t)))
            in_reg = False
    if in_reg and float(times[-1]) - reg_start >= 0.15:
        regions.append((reg_start, float(times[-1])))

    merged: List[Tuple[float, float]] = []
    for st, en in regions:
        if merged and st - merged[-1][1] < 0.45:
            merged[-1] = (merged[-1][0], en)
        else:
            merged.append((st, en))

    if not merged:
        merged = [(0.5, max(2.0, dur_sec - 0.5))]

    total_syls = sum(syl_counts)
    total_active_sec = sum(en - st for st, en in merged)

    def active_pos_to_time(frac: float) -> float:
        target = frac * total_active_sec
        acc = 0.0
        for st, en in merged:
            seg = en - st
            if acc + seg >= target:
                return st + (target - acc)
            acc += seg
        return merged[-1][1]

    phrases_out: List[dict] = []
    syl_cursor = 0
    last_midi = 65
    for text, n_syl in zip(lyric_lines, syl_counts):
        t_start = active_pos_to_time(syl_cursor / total_syls)
        t_end = active_pos_to_time((syl_cursor + n_syl) / total_syls)
        syl_cursor += n_syl

        slot_dur = max(0.10, (t_end - t_start) / max(1, n_syl))
        notes_sec: List[Tuple[int, float, float]] = []
        for k in range(n_syl):
            s0 = t_start + k * slot_dur
            s1 = s0 + slot_dur
            m = sample_pitch_midi(times, f0, s0, s1, fallback=last_midi)
            last_midi = m
            notes_sec.append((m, s0, s1))

        median_midi = int(round(float(np.median([m for m, _, _ in notes_sec]))))
        phrases_out.append(
            {
                "text": text,
                "t_start": t_start,
                "t_end": t_end,
                "median_midi": median_midi,
                "notes_sec": notes_sec,
            }
        )
    return phrases_out


def assign_voices(phrases: List[dict], voices: List[str], mode: str = "alternate") -> List[List[dict]]:
    n_v = len(voices)
    per_voice: List[List[dict]] = [[] for _ in range(n_v)]

    if mode == "register" and n_v >= 2:
        pitches = sorted(p["median_midi"] for p in phrases)
        for idx, p in enumerate(phrases):
            if idx == len(phrases) - 1 and n_v >= 2:
                for v_i in range(n_v):
                    per_voice[v_i].append(p)
            else:
                rank = pitches.index(p["median_midi"]) / max(1, len(pitches) - 1)
                v_i = min(n_v - 1, int(rank * n_v))
                per_voice[v_i].append(p)
    else:
        for idx, p in enumerate(phrases):
            if idx == len(phrases) - 1 and n_v >= 2:
                for v_i in range(n_v):
                    per_voice[v_i].append(p)
            else:
                per_voice[idx % n_v].append(p)

    return per_voice


def build_melody_json(assigned_phrases: List[dict], bpm: float, body_type: str) -> dict:
    spb = 60.0 / bpm
    notes: List[List[float]] = []
    lyrics_parts: List[str] = []
    cursor_beats = 0.0

    def fit_midi(m: int) -> int:
        if body_type == "male":
            while m > 64:
                m -= 12
            while m < 45:
                m += 12
        elif body_type == "child":
            while m < 64:
                m += 12
            while m > 81:
                m -= 12
        else:
            while m < 57:
                m += 12
            while m > 77:
                m -= 12
        return m

    for p in assigned_phrases:
        start_beats = max(cursor_beats, round(p["t_start"] / spb, 3))
        gap = round(start_beats - cursor_beats, 3)
        if gap >= 0.05:
            notes.append([0, gap])
            cursor_beats += gap

        lyrics_parts.append(p["text"])
        for midi_val, s0, s1 in p["notes_sec"]:
            dur_b = max(0.12, round((s1 - s0) / spb, 3))
            notes.append([fit_midi(midi_val), dur_b])
            cursor_beats += dur_b

    return {
        "bpm": bpm,
        "t0": CHIMERA_T0,
        "lyrics": " ".join(lyrics_parts),
        "notes": notes,
    }


def mix_mr_and_vocals(
    mr_wav: Path,
    vocal_wavs: List[Path],
    out_mix: Path,
    mr_vol: float = 0.55,
    vocal_vol: float = 1.0,
    max_duration: float = 0.0,
) -> None:
    print("\n[STEP] Mixing MR backing track + CHIMERA vocals into stereo mix.wav...", flush=True)
    vocals_data = []
    target_sr = 48000
    for vp in vocal_wavs:
        v_audio, v_sr = sf.read(str(vp), always_2d=False)
        target_sr = v_sr
        if v_audio.ndim > 1:
            v_audio = np.mean(v_audio, axis=1)
        vocals_data.append(v_audio.astype(np.float32))

    mr_audio, mr_sr = sf.read(str(mr_wav), always_2d=True)
    if mr_sr != target_sr:
        mr_ch = [
            librosa.resample(mr_audio[:, ch].astype(np.float32), orig_sr=mr_sr, target_sr=target_sr)
            for ch in range(mr_audio.shape[1])
        ]
        mr_audio = np.stack(mr_ch, axis=1)
    else:
        mr_audio = mr_audio.astype(np.float32)

    if mr_audio.shape[1] == 1:
        mr_audio = np.repeat(mr_audio, 2, axis=1)
    else:
        mr_audio = mr_audio[:, :2]

    if max_duration > 0:
        max_mr_samples = int(round(max_duration * target_sr))
        if len(mr_audio) > max_mr_samples:
            mr_audio = mr_audio[:max_mr_samples]
            fade = min(int(1.5 * target_sr), len(mr_audio))
            mr_audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]

    pad_samples = int(round(CHIMERA_T0 * target_sr))
    mr_padded = np.pad(mr_audio, ((pad_samples, 0), (0, 0)), mode="constant")

    max_len = max([len(mr_padded)] + [len(v) for v in vocals_data])
    mix = np.zeros((max_len, 2), dtype=np.float32)
    mix[: len(mr_padded)] += mr_padded * mr_vol

    n_v = len(vocals_data)
    for idx, v in enumerate(vocals_data):
        pan = 0.0 if n_v == 1 else -0.35 + (0.70 * idx / (n_v - 1))
        gain_l = math.cos((pan + 1) * math.pi / 4) * vocal_vol
        gain_r = math.sin((pan + 1) * math.pi / 4) * vocal_vol
        mix[: len(v), 0] += v * gain_l
        mix[: len(v), 1] += v * gain_r

    peak = float(np.max(np.abs(mix))) + 1e-6
    if peak > 0.92:
        mix = mix * (0.92 / peak)

    sf.write(str(out_mix), mix, target_sr, subtype="PCM_16")
    print(f"  Saved stereo mix: {out_mix} ({max_len/target_sr:.1f}s @ {target_sr} Hz)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate a CHIMERA 3D multi-voice show from audio.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--url", help="YouTube URL (requires yt-dlp)")
    src_group.add_argument("--audio", help="Path to local MP3/WAV audio file")
    parser.add_argument("--lyrics", default="", help="Optional UTF-8 lyrics file (omit to auto-transcribe with Whisper)")
    parser.add_argument("--lang", default="auto", help="Whisper language hint: auto, en, ko (default: auto)")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size (default: small)")
    parser.add_argument("--id", default="auto_show", help="Show folder ID under public/shows/<id>")
    parser.add_argument("--title", default="", help="Display title on the 3D show page")
    parser.add_argument("--voices", default="male,female,child", help="Comma-separated voices: male,female,child")
    parser.add_argument("--mode", choices=["alternate", "register"], default="alternate", help="Part assignment mode")
    parser.add_argument("--max-sec", type=float, default=0.0, help="Optional max duration in seconds (0 = full video)")
    parser.add_argument("--mr-vol", type=float, default=0.55, help="Background MR volume (default: 0.55)")
    parser.add_argument("--vocal-vol", type=float, default=1.0, help="CHIMERA vocal volume (default: 1.0)")
    args = parser.parse_args()

    repo_root = Path.cwd()
    chimera_exe = find_chimera_exe(repo_root)

    voices = [v.strip().lower() for v in args.voices.split(",") if v.strip()]
    for v in voices:
        if v not in {"male", "female", "child"}:
            raise ValueError(f"Unsupported voice '{v}'. Choose from: male, female, child")

    work_dir = repo_root / "_auto_work" / args.id
    show_dir = repo_root / "public" / "shows" / args.id
    work_dir.mkdir(parents=True, exist_ok=True)
    show_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source audio
    if args.url:
        audio_path = download_youtube_audio(args.url, work_dir)
    else:
        audio_path = Path(args.audio).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    find_ffmpeg_dir()

    # 2. Demucs vocal / MR separation
    vocals_wav, mr_wav = separate_stems_demucs(audio_path, work_dir)

    # 3. Pitch & BPM analysis
    y, sr, bpm, dur_sec, times, f0, rms = analyze_vocals_pitch(vocals_wav, max_duration=args.max_sec)

    # 4. Lyrics + timing (auto Whisper by default, or manual if --lyrics provided)
    manual_lines: List[str] = []
    if args.lyrics and Path(args.lyrics).exists():
        raw_lines = Path(args.lyrics).read_text(encoding="utf-8").splitlines()
        manual_lines = [sanitize_text(ln) for ln in raw_lines]
        manual_lines = [ln for ln in manual_lines if ln]

    if manual_lines:
        print("\n[STEP] Counting CHIMERA syllables for manual lyrics...", flush=True)
        syl_counts = [count_chimera_syllables(chimera_exe, ln) for ln in manual_lines]
        phrases = extract_phrases_manual_lyrics(manual_lines, syl_counts, dur_sec, times, f0, rms)
    else:
        phrases = extract_phrases_with_whisper(
            vocals_wav=vocals_wav,
            chimera_exe=chimera_exe,
            y=y,
            sr=sr,
            bpm=bpm,
            dur_sec=dur_sec,
            times=times,
            f0=f0,
            rms=rms,
            language=args.lang,
            whisper_model_name=args.whisper_model,
            work_dir=work_dir,
        )

    # 5. Assign phrases across singers
    assigned = assign_voices(phrases, voices, mode=args.mode)

    # 6. Render each singer with CHIMERA (--tele for 3D heads)
    rendered_wavs: List[Path] = []
    show_voices_meta = []

    for idx, body_type in enumerate(voices):
        if not assigned[idx]:
            continue
        label = f"{body_type.capitalize()} {idx + 1}" if voices.count(body_type) > 1 else body_type.capitalize()
        stem = f"voice_{idx + 1}_{body_type}"
        mel_path = show_dir / f"{stem}.json"
        tele_path = show_dir / f"{stem}.tele"
        wav_stem = show_dir / stem

        melody_obj = build_melody_json(assigned[idx], bpm, body_type)
        mel_path.write_text(json.dumps(melody_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        run_cmd(
            [
                str(chimera_exe),
                "render",
                "notes",
                str(mel_path),
                str(wav_stem),
                body_type,
                "--tele",
                str(tele_path),
            ],
            f"Rendering CHIMERA voice [{label}] ({body_type})",
        )
        rendered_wavs.append(show_dir / f"{stem}.wav")
        show_voices_meta.append(
            {
                "name": label,
                "tele": tele_path.name,
                "melody": mel_path.name,
            }
        )

    # 7. Mix MR + CHIMERA vocals
    mix_path = show_dir / "mix.wav"
    mix_mr_and_vocals(
        mr_wav,
        rendered_wavs,
        mix_path,
        mr_vol=args.mr_vol,
        vocal_vol=args.vocal_vol,
        max_duration=args.max_sec,
    )

    # 8. Write show.json
    show_title = args.title or f"{args.id} ({' · '.join(v['name'] for v in show_voices_meta)})"
    show_json = {
        "title": show_title,
        "audio": "mix.wav",
        "bpm": bpm,
        "downbeat": CHIMERA_T0,
        "voices": show_voices_meta,
    }
    (show_dir / "show.json").write_text(json.dumps(show_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========================================================", flush=True)
    print("DONE! Your 3D CHIMERA Show is ready:", flush=True)
    print(f"  Folder : {show_dir}", flush=True)
    print(f"  Web URL: http://127.0.0.1:5317/?show={args.id}", flush=True)
    print("========================================================\n", flush=True)


if __name__ == "__main__":
    main()