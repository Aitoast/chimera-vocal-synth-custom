#!/usr/bin/env python3
"""
CHIMERA Auto-Show Pipeline v3 (auto_show.py)
--------------------------------------------
Upgrades over v2 (tuned for hard songs: fast chants, high-pitched anime/soprano vocals, scat):
  1. Raises pYIN pitch ceiling to C7 (2093 Hz) so high-pitched/screaming vocals don't become NaN.
  2. Disables Whisper's aggressive speech-only `vad_filter` so singing & rhythmic shouts aren't dropped.
  3. Replaces coarse 0.35s gap-filling with onset-driven vocal detection (`librosa.onset.onset_detect`)
     that captures every rapid staccato shout/scat hit down to 0.08s!
  4. Classifies each uncovered vocal onset's brightness (spectral centroid) into matching vocal
     syllables ("bum", "oo", "ah", "ha", "ya", "ee") instead of repeating "bum" everywhere.
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
        sys.executable,
        "-m",
        "yt_dlp",
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


_KANA_DIGRAPHS = {
    "きゃ": "캬", "きゅ": "큐", "きょ": "쿄",
    "しゃ": "샤", "しゅ": "슈", "しょ": "쇼",
    "ちゃ": "차", "ちゅ": "추", "ちょ": "초",
    "にゃ": "냐", "にゅ": "뉴", "にょ": "뇨",
    "ひゃ": "햐", "ひゅ": "휴", "ひょ": "효",
    "みゃ": "먀", "みゅ": "뮤", "みょ": "묘",
    "りゃ": "랴", "りゅ": "류", "りょ": "료",
    "ぎゃ": "갸", "ぎゅ": "규", "ぎょ": "교",
    "じゃ": "자", "じゅ": "주", "じょ": "조",
    "びゃ": "뱌", "びゅ": "뷰", "びょ": "뵤",
    "ぴゃ": "퍄", "ぴゅ": "퓨", "ぴょ": "표",
    "ふぁ": "파", "ふぃ": "피", "ふぇ": "페", "ふぉ": "포",
    "てぃ": "티", "でぃ": "디", "うぃ": "위", "うぇ": "웨", "うぉ": "워",
}

_KANA_SINGLE = {
    "あ": "아", "い": "이", "う": "우", "え": "에", "お": "오",
    "か": "카", "き": "키", "く": "쿠", "け": "케", "こ": "코",
    "さ": "사", "し": "시", "す": "스", "せ": "세", "そ": "소",
    "た": "타", "ち": "치", "つ": "츠", "て": "테", "と": "토",
    "な": "나", "に": "니", "ぬ": "누", "ね": "네", "の": "노",
    "は": "하", "ひ": "히", "ふ": "후", "へ": "헤", "ほ": "호",
    "ま": "마", "み": "미", "む": "무", "め": "메", "も": "모",
    "や": "야", "ゆ": "유", "よ": "요",
    "ら": "라", "り": "리", "る": "루", "れ": "레", "ろ": "로",
    "わ": "와", "を": "오", "ん": "응",
    "が": "가", "ぎ": "기", "ぐ": "구", "げ": "게", "ご": "고",
    "ざ": "자", "じ": "지", "ず": "즈", "ぜ": "제", "ぞ": "조",
    "だ": "다", "ぢ": "지", "づ": "즈", "で": "데", "ど": "도",
    "ば": "바", "び": "비", "ぶ": "부", "べ": "베", "ぼ": "보",
    "ぱ": "파", "ぴ": "피", "ぷ": "푸", "ぺ": "페", "ぽ": "포",
    "ぁ": "아", "ぃ": "이", "ぅ": "우", "ぇ": "에", "ぉ": "오",
    "ゃ": "야", "ゅ": "유", "ょ": "요", "ゎ": "와", "ゔ": "부",
}

_KAKASI = None


def has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", text))


def japanese_to_hangul(text: str) -> str:
    """Convert Japanese (Kanji via optional pykakasi + Hiragana/Katakana) into CHIMERA-ready Hangul."""
    global _KAKASI
    if not has_japanese(text):
        return text

    # Convert Kanji to Hiragana if pykakasi is installed
    if re.search(r"[\u4E00-\u9FFF]", text):
        try:
            if _KAKASI is None:
                import pykakasi
                _KAKASI = pykakasi.kakasi()
            text = "".join(item["hira"] for item in _KAKASI.convert(text))
        except Exception:
            pass

    # Normalize Katakana (U+30A1..U+30F6) to Hiragana (U+3041..U+3096)
    hira_chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            hira_chars.append(chr(code - 0x60))
        else:
            hira_chars.append(ch)
    s = "".join(hira_chars)

    out: List[str] = []
    i = 0
    while i < len(s):
        pair = s[i : i + 2]
        if pair in _KANA_DIGRAPHS:
            out.append(_KANA_DIGRAPHS[pair])
            i += 2
            continue
        ch = s[i]
        if ch in ("っ", "ー"):
            i += 1
            continue
        if ch in _KANA_SINGLE:
            out.append(_KANA_SINGLE[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def sanitize_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    if has_japanese(text):
        text = japanese_to_hangul(text)
    # Keep Korean, Latin, Cyrillic, and apostrophe (CHIMERA front-ends support EN/TR/KO/RU)
    text = re.sub(r"[^\w\s가-힣\u0400-\u04FF']", " ", text)
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
        return max(40, min(90, midi_val))
    return fallback


def pick_scat_syllable(centroid_hz: float, midi_val: int, is_short: bool) -> str:
    """Pick an expressive CHIMERA-safe syllable matching the acoustic brightness of the vocal hit."""
    if centroid_hz < 950:
        return "bum" if is_short else "oo"
    if centroid_hz < 1700:
        return "ah" if not is_short else "ha"
    if centroid_hz < 2600:
        return "ya"
    return "ee" if midi_val >= 76 else "ya"


def analyze_vocals_pitch(vocals_wav: Path, max_duration: float = 0.0):
    print("\n[STEP] Analyzing vocal pitch, onsets & BPM with librosa (pYIN up to C7)...", flush=True)
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
        fmax=librosa.note_to_hz("C7"),  # Raised to C7 for high-pitched anime/soprano vocals
        sr=sr,
        hop_length=hop,
    )
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    return y, sr, bpm, dur_sec, times, f0, rms, centroid


def extract_phrases_with_whisper(
    chimera_exe: Path,
    y: np.ndarray,
    sr: int,
    bpm: float,
    dur_sec: float,
    times: np.ndarray,
    f0: np.ndarray,
    rms: np.ndarray,
    centroid: np.ndarray,
    language: Optional[str] = None,
    whisper_model_name: str = "small",
) -> List[dict]:
    """
    1. Transcribes sung words via faster-whisper (with vad_filter=False so singing/chants aren't dropped).
    2. Detects EVERY remaining vocal onset in `vocals.wav` via librosa.onset.onset_detect
       (including rapid staccato shouts & high-pitched anime vocals where Whisper stays silent),
       groups them into rhythmic phrases, and assigns matching acoustic syllables (ya/ha/ah/bum/ee).
    """
    print(f"\n[STEP] Auto-extracting lyrics & timestamps with faster-whisper ({whisper_model_name})...", flush=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    audio_16k = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000).astype(np.float32)

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
        audio_16k,
        language=lang_arg,
        word_timestamps=True,
        vad_filter=False,  # Keep singing, chants, and short anime shouts!
        no_speech_threshold=0.85,
        log_prob_threshold=-1.5,
        condition_on_previous_text=False,
    )
    print(f"  Detected language: {info.language} (prob={info.language_probability:.2f})", flush=True)

    phrases: List[dict] = []
    covered_intervals: List[Tuple[float, float]] = []
    last_midi = 67

    for seg in segments_gen:
        seg_words = getattr(seg, "words", None) or []
        line_words: List[str] = []
        notes_sec: List[Tuple[int, float, float]] = []

        for w in seg_words:
            clean_w = sanitize_text(w.word)
            # Skip non-CHIMERA scripts (e.g. Japanese Kana/Kanji if Whisper outputs them) so onset-scat fills them cleanly
            if not clean_w or not re.search(r"[A-Za-z가-힣\u0400-\u04FF]", clean_w):
                continue
            w_start = max(0.0, float(w.start))
            w_end = min(dur_sec, max(w_start + 0.08, float(w.end)))
            if w_start >= dur_sec:
                continue
            # Reject stretched Whisper hallucinations (>1.25s) or low-confidence guesses on non-words/screams
            w_prob = float(getattr(w, "probability", 1.0) or 1.0)
            if (w_end - w_start) > 1.25 or w_prob < 0.35:
                continue

            # Verify there is actual vocal energy during this Whisper word (ignore Whisper hallucinations in silence)
            i0 = int(np.searchsorted(times, w_start))
            i1 = max(i0 + 1, int(np.searchsorted(times, w_end)))
            if float(np.max(rms[i0:i1])) < max(0.012, float(np.percentile(rms, 50)) * 0.6):
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
            print(f"  [{t_st:5.1f}s - {t_en:5.1f}s] (whisper) {line_text}", flush=True)
            phrases.append(
                {
                    "text": line_text,
                    "t_start": t_st,
                    "t_end": t_en,
                    "median_midi": median_m,
                    "notes_sec": notes_sec,
                }
            )

    # Onset-driven vocal capture for EVERY uncovered vocal sound (shouts, intros, fast chants!)
    print("\n[STEP] Detecting remaining vocal onsets (shouts / scat / intro)...", flush=True)
    rms_floor = max(0.015, float(np.percentile(rms, 55)) * 0.55)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=256,
        backtrack=False,
        delta=0.06,
        wait=7,  # minimum ~80ms between rapid staccato shouts
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

    def is_covered(t_val: float) -> bool:
        for c_st, c_en in covered_intervals:
            if c_st - 0.12 <= t_val <= c_en + 0.12:
                return True
        return False

    # Build individual uncovered onset notes (s0, s1, midi, syllable)
    raw_hits: List[Tuple[float, float, int, str]] = []
    for idx, t0 in enumerate(onset_times):
        t0 = float(t0)
        if t0 >= dur_sec - 0.05 or is_covered(t0):
            continue
        t_next = float(onset_times[idx + 1]) if idx + 1 < len(onset_times) else dur_sec
        t_end_max = min(dur_sec, t0 + 0.65, t_next - 0.02)
        if t_end_max <= t0 + 0.06:
            continue

        # Trim hit end where RMS drops below floor
        i0 = int(np.searchsorted(times, t0))
        i_max = max(i0 + 1, int(np.searchsorted(times, t_end_max)))
        seg_rms = rms[i0:i_max]
        if len(seg_rms) == 0 or float(np.max(seg_rms)) < rms_floor:
            continue

        above = np.where(seg_rms >= rms_floor * 0.65)[0]
        if len(above) == 0:
            continue
        t1 = min(t_end_max, float(times[min(len(times) - 1, i0 + int(above[-1]) + 1)]))
        t1 = max(t0 + 0.08, t1)

        m = sample_pitch_midi(times, f0, t0, t1, fallback=last_midi)
        last_midi = m
        cent = float(np.median(centroid[i0 : max(i0 + 1, int(np.searchsorted(times, t1)))]))
        syl = pick_scat_syllable(cent, m, is_short=(t1 - t0) < 0.24)
        raw_hits.append((t0, t1, m, syl))

    # Group consecutive onset hits into phrases (split on pauses > 0.65s or > 10 hits)
    scat_phrases = 0
    if raw_hits:
        cur_group: List[Tuple[float, float, int, str]] = [raw_hits[0]]

        def flush_group(grp: List[Tuple[float, float, int, str]]):
            nonlocal scat_phrases
            text = " ".join(h[3] for h in grp)
            notes_sec = [(h[2], h[0], h[1]) for h in grp]
            median_m = int(round(float(np.median([h[2] for h in grp]))))
            phrases.append(
                {
                    "text": text,
                    "t_start": grp[0][0],
                    "t_end": grp[-1][1],
                    "median_midi": median_m,
                    "notes_sec": notes_sec,
                }
            )
            scat_phrases += 1
            print(f"  [{grp[0][0]:5.1f}s - {grp[-1][1]:5.1f}s] (onset)   {text}", flush=True)

        for hit in raw_hits[1:]:
            if (hit[0] - cur_group[-1][1] > 0.65) or (len(cur_group) >= 10):
                flush_group(cur_group)
                cur_group = [hit]
            else:
                cur_group.append(hit)
        flush_group(cur_group)

    phrases.sort(key=lambda p: p["t_start"])
    if not phrases:
        raise RuntimeError("No vocal lyrics or pitches could be extracted from the audio.")
    print(f"  Total phrases: {len(phrases)} ({scat_phrases} onset-captured vocal phrases)", flush=True)
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
    active = rms > rms_thresh

    regions: List[Tuple[float, float]] = []
    in_reg = False
    reg_start = 0.0
    for t, is_act in zip(times, active):
        if is_act and not in_reg:
            reg_start = float(t)
            in_reg = True
        elif not is_act and in_reg:
            if float(t) - reg_start >= 0.12:
                regions.append((reg_start, float(t)))
            in_reg = False
    if in_reg and float(times[-1]) - reg_start >= 0.12:
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

        slot_dur = max(0.08, (t_end - t_start) / max(1, n_syl))
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


def build_melody_json(
    assigned_phrases: List[dict],
    bpm: float,
    body_type: str,
    target_midi: Optional[float] = None,
) -> dict:
    spb = 60.0 / bpm
    notes: List[List[float]] = []
    lyrics_parts: List[str] = []
    cursor_beats = 0.0

    def fit_midi(m: int) -> int:
        if target_midi is not None:
            lo = max(42, int(round(target_midi - 9)))
            hi = min(84, int(round(target_midi + 9)))
            while m > hi:
                m -= 12
            while m < lo:
                m += 12
            return m
        if body_type == "male":
            while m > 64:
                m -= 12
            while m < 45:
                m += 12
        elif body_type == "child":
            while m < 64:
                m += 12
            while m > 84:
                m -= 12
        else:
            while m < 57:
                m += 12
            while m > 78:
                m -= 12
        return m

    for p in assigned_phrases:
        start_beats = max(cursor_beats, round(p["t_start"] / spb, 3))
        gap = round(start_beats - cursor_beats, 3)
        if gap >= 0.04:
            notes.append([0, gap])
            cursor_beats += gap

        lyrics_parts.append(p["text"])
        for midi_val, s0, s1 in p["notes_sec"]:
            note_start_b = round(s0 / spb, 3)
            inner_gap = round(note_start_b - cursor_beats, 3)
            if inner_gap >= 0.06:
                notes.append([0, inner_gap])
                cursor_beats += inner_gap
            dur_b = max(0.10, min(2.5, round((s1 - s0) / spb, 3)))
            notes.append([fit_midi(midi_val), dur_b])
            cursor_beats += dur_b

    # Adaptive vibrato: crisp/dry for fast rap & chants, gentle for pop/ballads
    sounding_durs = [b * spb for m, b in notes if m > 0]
    med_sec = float(np.median(sounding_durs)) if sounding_durs else 0.3
    if med_sec < 0.28:
        vibrato = 4.0
    elif med_sec < 0.45:
        vibrato = 16.0
    else:
        vibrato = 28.0

    return {
        "bpm": bpm,
        "t0": CHIMERA_T0,
        "vibrato": vibrato,
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
    parser.add_argument("--lyrics", default="", help="Optional UTF-8 lyrics file (omit to auto-transcribe)")
    parser.add_argument("--lang", default="auto", help="Whisper language hint: auto, en, ko, ja (default: auto)")
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

    # Load custom voice presets from public/presets/voices.json if present
    presets_map = {}
    presets_file = repo_root / "public" / "presets" / "voices.json"
    if presets_file.exists():
        try:
            for item in json.loads(presets_file.read_text(encoding="utf-8")):
                if isinstance(item, dict) and item.get("id"):
                    presets_map[str(item["id"]).lower()] = item
        except Exception as pe:
            print(f"  Warning: failed to read {presets_file}: {pe}", flush=True)

    raw_voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    voice_specs = []
    for rv in raw_voices:
        key = rv.lower()
        if key in {"male", "female", "child"}:
            voice_specs.append({"body": key, "label": key.capitalize(), "target_midi": None})
        elif key in presets_map:
            pr = presets_map[key]
            cp = pr.get("chimera_preset", pr)
            body = cp.get("cli_body_by_timbre") or cp.get("cli_body") or "female"
            if body not in {"male", "female", "child"}:
                body = "female"
            t_midi = pr.get("acoustics", {}).get("median_midi")
            voice_specs.append({"body": body, "label": pr.get("name", rv), "target_midi": t_midi})
        else:
            raise ValueError(f"Unsupported voice '{rv}'. Use male, female, child, or a saved custom synth ID.")
    voices = [vs["body"] for vs in voice_specs]

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

    # 3. Pitch, onset & BPM analysis
    y, sr, bpm, dur_sec, times, f0, rms, centroid = analyze_vocals_pitch(vocals_wav, max_duration=args.max_sec)

    # 4. Lyrics + timing (auto Whisper + onset scat by default, or manual if --lyrics provided)
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
            chimera_exe=chimera_exe,
            y=y,
            sr=sr,
            bpm=bpm,
            dur_sec=dur_sec,
            times=times,
            f0=f0,
            rms=rms,
            centroid=centroid,
            language=args.lang,
            whisper_model_name=args.whisper_model,
        )

    # 5. Assign phrases across singers
    assigned = assign_voices(phrases, voices, mode=args.mode)

    # 6. Render each singer with CHIMERA (--tele for 3D heads)
    rendered_wavs: List[Path] = []
    show_voices_meta = []

    for idx, vs in enumerate(voice_specs):
        if not assigned[idx]:
            continue
        body_type = vs["body"]
        label = vs["label"]
        if sum(1 for x in voice_specs if x["label"] == label) > 1:
            label = f"{label} {idx + 1}"
        stem = f"voice_{idx + 1}_{body_type}"
        mel_path = show_dir / f"{stem}.json"
        tele_path = show_dir / f"{stem}.tele"
        wav_stem = show_dir / stem

        melody_obj = build_melody_json(assigned[idx], bpm, body_type, target_midi=vs["target_midi"])
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