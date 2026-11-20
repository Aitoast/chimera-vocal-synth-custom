#!/usr/bin/env python3
"""
CHIMERA Auto-Show Pipeline (auto_show.py)
-----------------------------------------
1. Takes a YouTube URL (--url) or local audio file (--audio) + lyrics file (--lyrics)
2. Separates vocals & instrumental MR with Demucs (GPU accelerated if CUDA available)
3. Extracts BPM, phrase timing, and vocal melody MIDI pitches with librosa (pYIN)
4. Matches exact CHIMERA syllable counts via `chimera.exe syllables`
5. Automatically assigns lines to multiple singers (male / female / child)
6. Renders each voice's WAV + 3D animation telemetry (.tele) with `chimera.exe`
7. Mixes the extracted MR + all stereo-panned CHIMERA vocals into `mix.wav`
8. Writes `public/shows/<id>/show.json` so the CHIMERA web UI plays it immediately!

Usage:
  python auto_show.py --audio song.mp3 --lyrics lyrics.txt --id my_song --voices male,female,child
  python auto_show.py --url "https://www.youtube.com/watch?v=..." --lyrics lyrics.txt --id my_song --voices male,female
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
from typing import List, Tuple

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


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    out_tpl = str(work_dir / "source.%(ext)s")
    run_cmd(
        [
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
            url,
        ],
        "Downloading audio with yt-dlp",
    )
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
        # Fallback search in case of filename normalization
        vocals = next(sep_dir.rglob("vocals.wav"))
        no_vocals = next(sep_dir.rglob("no_vocals.wav"))
    return vocals, no_vocals


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
            # Fallback: count non-empty lines if output is line-delimited
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            if lines:
                return len(lines)
    # Fallback heuristic: Hangul chars = 1 each, Latin words = vowel groups
    hangul = len(re.findall(r"[가-힣]", clean))
    latin_words = re.findall(r"[A-Za-z']+", clean)
    latin_syls = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in latin_words)
    return max(1, hangul + latin_syls)


def sanitize_lyric_line(line: str) -> str:
    # Strip section headers like [Verse 1], punctuation that confuses G2P
    line = re.sub(r"\[[^\]]*\]", "", line).strip()
    line = re.sub(r"[^\w\s가-힣']", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def extract_vocal_phrases_and_notes(
    vocals_wav: Path,
    lyric_lines: List[str],
    syl_counts: List[int],
    max_duration: float = 0.0,
) -> Tuple[float, List[dict]]:
    """
    Loads isolated vocals, estimates BPM, extracts vocal pitch contour via pYIN,
    and maps each lyric line to a sequence of quantized [midi, beats] notes.
    """
    print("\n[STEP] Analyzing vocal pitch & rhythm with librosa (pYIN)...")
    y, sr = librosa.load(str(vocals_wav), sr=22050, mono=True)
    if max_duration > 0:
        y = y[: int(max_duration * sr)]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    if not (55.0 <= bpm <= 190.0):
        bpm = 110.0
    bpm = round(bpm, 1)
    spb = 60.0 / bpm
    print(f"  Estimated BPM: {bpm}, vocal length: {len(y)/sr:.1f}s")

    hop = 256
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop,
    )
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    # Active singing mask (energy above threshold)
    rms_thresh = max(0.015, float(np.percentile(rms, 65)) * 0.25)
    active = (rms > rms_thresh) & (~np.isnan(f0))

    # Find contiguous active singing regions (merge gaps < 0.35s)
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

    # Build active timeline spanning all detected singing (or full audio if sparse)
    total_dur = len(y) / sr
    if not merged:
        merged = [(0.5, max(2.0, total_dur - 0.5))]

    # Distribute lyric lines proportionally by syllable count across active singing regions
    total_syls = sum(syl_counts)
    total_active_sec = sum(en - st for st, en in merged)
    # Map each syllable position [0..total_syls] to a time in the vocal track
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
    for line_idx, (text, n_syl) in enumerate(zip(lyric_lines, syl_counts)):
        t_start = active_pos_to_time(syl_cursor / total_syls)
        t_end = active_pos_to_time((syl_cursor + n_syl) / total_syls)
        syl_cursor += n_syl

        # Subdivide [t_start, t_end] into n_syl slots and extract median MIDI per slot
        slot_dur = max(0.12, (t_end - t_start) / max(1, n_syl))
        notes_sec: List[Tuple[int, float, float]] = []
        valid_midis: List[int] = []

        for k in range(n_syl):
            s0 = t_start + k * slot_dur
            s1 = s0 + slot_dur
            i0 = int(np.searchsorted(times, s0))
            i1 = max(i0 + 1, int(np.searchsorted(times, s1)))
            seg_f0 = f0[i0:i1]
            seg_f0 = seg_f0[~np.isnan(seg_f0)]
            if len(seg_f0) > 0:
                midi_val = int(round(float(librosa.hz_to_midi(np.median(seg_f0)))))
                midi_val = max(40, min(84, midi_val))
            else:
                midi_val = valid_midis[-1] if valid_midis else 65
            valid_midis.append(midi_val)
            notes_sec.append((midi_val, s0, s1))

        median_midi = int(round(float(np.median(valid_midis)))) if valid_midis else 65
        phrases_out.append(
            {
                "line_idx": line_idx,
                "text": text,
                "t_start": t_start,
                "t_end": t_end,
                "median_midi": median_midi,
                "notes_sec": notes_sec,  # (midi, s0, s1)
            }
        )

    return bpm, phrases_out


def assign_voices(
    phrases: List[dict],
    voices: List[str],
    mode: str = "alternate",
) -> List[List[dict]]:
    """
    Assigns each phrase to one (or all, on the final phrase) of the requested voices.
    - alternate: rotates cleanly across singers line-by-line (plus tutti on last line)
    - register: assigns by phrase pitch (low -> male, mid -> female, high -> child)
    """
    n_v = len(voices)
    per_voice: List[List[dict]] = [[] for _ in range(n_v)]

    if mode == "register" and n_v >= 2:
        pitches = sorted(p["median_midi"] for p in phrases)
        for idx, p in enumerate(phrases):
            if idx == len(phrases) - 1 and n_v >= 2:
                # Everyone joins the final line!
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


def build_melody_json(
    assigned_phrases: List[dict],
    bpm: float,
    body_type: str,
) -> dict:
    """
    Converts assigned phrases into a CHIMERA `render notes` melody JSON
    starting at t = 0 (aligned to CHIMERA_T0 = 0.3s) with rests between phrases.
    """
    spb = 60.0 / bpm
    notes: List[List[float]] = []
    lyrics_parts: List[str] = []
    cursor_beats = 0.0

    # Shift octave into comfortable range per body type
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
        else:  # female
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
            dur_b = max(0.15, round((s1 - s0) / spb, 3))
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
) -> None:
    print("\n[STEP] Mixing MR backing track + CHIMERA vocals into stereo mix.wav...")
    vocals_data = []
    target_sr = 48000
    for vp in vocal_wavs:
        v_audio, v_sr = sf.read(str(vp), always_2d=False)
        target_sr = v_sr
        if v_audio.ndim > 1:
            v_audio = np.mean(v_audio, axis=1)
        vocals_data.append(v_audio.astype(np.float32))

    # Load & resample MR to match CHIMERA sample rate, pad by CHIMERA_T0 at start
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

    pad_samples = int(round(CHIMERA_T0 * target_sr))
    mr_padded = np.pad(mr_audio, ((pad_samples, 0), (0, 0)), mode="constant")

    max_len = max([len(mr_padded)] + [len(v) for v in vocals_data])
    mix = np.zeros((max_len, 2), dtype=np.float32)
    mix[: len(mr_padded)] += mr_padded * mr_vol

    # Stereo panning across voices (-0.4 .. +0.4)
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
    print(f"  Saved stereo mix: {out_mix} ({max_len/target_sr:.1f}s @ {target_sr} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate a CHIMERA 3D multi-voice show from audio + lyrics.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--url", help="YouTube URL (requires yt-dlp)")
    src_group.add_argument("--audio", help="Path to local MP3/WAV audio file")
    parser.add_argument("--lyrics", required=True, help="Path to UTF-8 text file containing lyrics (one line per phrase)")
    parser.add_argument("--id", default="auto_show", help="Show folder ID under public/shows/<id> (default: auto_show)")
    parser.add_argument("--title", default="", help="Display title on the 3D show page")
    parser.add_argument("--voices", default="male,female,child", help="Comma-separated voices: male,female,child")
    parser.add_argument("--mode", choices=["alternate", "register"], default="alternate", help="Part assignment mode")
    parser.add_argument("--max-sec", type=float, default=0.0, help="Optional max duration in seconds (e.g. 30 for quick test)")
    parser.add_argument("--mr-vol", type=float, default=0.55, help="Background MR volume (default: 0.55)")
    parser.add_argument("--vocal-vol", type=float, default=1.0, help="CHIMERA vocal volume (default: 1.0)")
    args = parser.parse_args()

    repo_root = Path.cwd()
    chimera_exe = find_chimera_exe(repo_root)

    voices = [v.strip().lower() for v in args.voices.split(",") if v.strip()]
    for v in voices:
        if v not in {"male", "female", "child"}:
            raise ValueError(f"Unsupported voice '{v}'. Choose from: male, female, child")

    raw_lines = Path(args.lyrics).read_text(encoding="utf-8").splitlines()
    lyric_lines = [sanitize_lyric_line(ln) for ln in raw_lines]
    lyric_lines = [ln for ln in lyric_lines if ln]
    if not lyric_lines:
        raise ValueError("Lyrics file has no usable lines.")

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

    # 2. Demucs vocal / MR separation
    vocals_wav, mr_wav = separate_stems_demucs(audio_path, work_dir)

    # 3. Count exact CHIMERA syllables per lyric line
    print("\n[STEP] Counting CHIMERA syllables per lyric line...")
    syl_counts = [count_chimera_syllables(chimera_exe, ln) for ln in lyric_lines]
    for ln, sc in zip(lyric_lines, syl_counts):
        print(f"  ({sc:2d} syl) {ln}")

    # 4. Extract vocal pitch & timing
    bpm, phrases = extract_vocal_phrases_and_notes(vocals_wav, lyric_lines, syl_counts, max_duration=args.max_sec)

    # 5. Assign lines across singers
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
    mix_mr_and_vocals(mr_wav, rendered_wavs, mix_path, mr_vol=args.mr_vol, vocal_vol=args.vocal_vol)

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

    print("\n========================================================")
    print("DONE! Your 3D CHIMERA Show is ready:")
    print(f"  Folder : {show_dir}")
    print(f"  Web URL: http://127.0.0.1:5317/?show={args.id}")
    print("========================================================\n")


if __name__ == "__main__":
    main()