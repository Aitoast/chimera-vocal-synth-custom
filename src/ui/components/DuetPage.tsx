/**
 * The duet page (`?show=<id>`): a pre-rendered song with two (or more) voices, each on its own 3D
 * head, side by side, moving exactly as the engine did when it sang.
 *
 * Includes `+ New Show` (auto Whisper lyrics + full video length default) and a 2-step `Delete`
 * button to clean up unwanted shows directly from the Show UI.
 */
import { useEffect, useRef, useState } from "react";
import { AnatomyView3D, type HeadFeed } from "./AnatomyView3D.tsx";
import { Chip, Screen } from "./Screen.tsx";
import { TeleTrack } from "../../show/tele.ts";
import { Transport } from "../../show/transport.ts";
import { getModel } from "../useModel.ts";
import { VoiceModel } from "../analysis.ts";
import { bodyLabel } from "../body.ts";
import type { ViewFlags } from "../render/anatomy.ts";
import { mouthWords } from "../mouthWords.ts";
import { V } from "../colors.ts";
import { g2p, zh } from "../speech.ts";
import { hasHan, lyricsToSyllables } from "../../wasm/engine.ts";
import { sungWords, wordAt, type Melody, type SungWords } from "../../show/words.ts";
import type { OverlayFn } from "./AnatomyView.tsx";

interface ShowFile {
  title: string;
  audio: string;
  bpm?: number;
  downbeat?: number;
  spell?: Record<string, string>;
  voices: { name: string; tele: string; melody?: string }[];
}

interface Loaded {
  show: ShowFile;
  voices: { name: string; track: TeleTrack; words: SungWords | null }[];
}

interface ShowSummary {
  id: string;
  title: string;
}

const VIEW: ViewFlags = { pressure: true, particles: true, grid: false, calipers: false };

async function fetchOk(url: string): Promise<Response> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return r;
}

const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function DuetPage(props: { id: string; onBack?: () => void }) {
  const [activeId, setActiveId] = useState(props.id);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [showList, setShowList] = useState<ShowSummary[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Streamlined Auto-Show panel state (default: full video length + auto Whisper lyrics)
  const [composerOpen, setComposerOpen] = useState(false);
  const [sourceInput, setSourceInput] = useState("");
  const [titleInput, setTitleInput] = useState("");
  const [idInput, setIdInput] = useState("");
  const [langInput, setLangInput] = useState<"auto" | "en" | "ko">("auto");
  const [selectedVoices, setSelectedVoices] = useState<string[]>(["male", "female", "child"]);
  const [assignMode, setAssignMode] = useState<"alternate" | "register">("alternate");
  const [maxSecInput, setMaxSecInput] = useState(""); // empty = full video length
  const [showManualLyrics, setShowManualLyrics] = useState(false);
  const [lyricsInput, setLyricsInput] = useState("");

  const [generating, setGenerating] = useState(false);
  const [genStep, setGenStep] = useState<string>("");
  const [genError, setGenError] = useState<string | null>(null);
  const [genLog, setGenLog] = useState<string>("");
  const logBoxRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    setActiveId(props.id);
    setConfirmDelete(false);
  }, [props.id]);

  const refreshShowList = async (): Promise<ShowSummary[]> => {
    try {
      const r = await fetch("/api/shows");
      if (!r.ok) return [];
      const d = await r.json();
      const list = Array.isArray(d?.shows) ? (d.shows as ShowSummary[]) : [];
      setShowList(list);
      return list;
    } catch {
      return [];
    }
  };

  useEffect(() => {
    void refreshShowList();
  }, []);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [genLog]);

  useEffect(() => {
    if (!generating) return;
    let alive = true;
    const poll = async () => {
      try {
        const r = await fetch("/api/generate-status");
        if (!r.ok || !alive) return;
        const st = await r.json();
        setGenStep(st.step || "");
        setGenLog(st.logs || "");
        if (st.done) {
          setGenerating(false);
          if (st.ok) {
            await refreshShowList();
            setComposerOpen(false);
            switchShow(st.id);
            setReloadNonce((n) => n + 1);
          } else {
            setGenError(st.error || "생성에 실패했습니다.");
          }
        }
      } catch {
        // ignore transient poll errors
      }
    };
    const timer = setInterval(poll, 600);
    void poll();
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [generating]);

  const switchShow = (nextId: string) => {
    setConfirmDelete(false);
    setActiveId(nextId);
    const url = new URL(window.location.href);
    url.searchParams.set("show", nextId);
    window.history.replaceState(null, "", url.toString());
  };

  const base = `${import.meta.env.BASE_URL}shows/${activeId}/`;
  const [transport, setTransport] = useState<Transport | null>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, setTick] = useState(0);

  useEffect(() => {
    const t = new Transport();
    setTransport(t);
    return () => t.dispose();
  }, []);

  useEffect(() => {
    if (!transport) return;
    let live = true;
    transport.pause();
    setLoaded(null);
    setError(null);
    const q = `?v=${reloadNonce}`;
    (async () => {
      const show = (await (await fetchOk(base + "show.json" + q)).json()) as ShowFile;
      if (!show.voices?.length || !show.audio) throw new Error("show.json needs audio and voices");
      const [audio, ...teles] = await Promise.all([
        fetchOk(base + show.audio + q).then((r) => r.arrayBuffer()),
        ...show.voices.map((v) => fetchOk(base + v.tele + q).then((r) => r.arrayBuffer())),
      ]);
      const melodies = await Promise.all(
        show.voices.map((v) => (v.melody ? fetchOk(base + v.melody + q).then((r) => r.json() as Promise<Melody>) : null))
      );
      if (melodies.some(Boolean)) await g2p.load().catch(() => {});
      if (melodies.some((m) => m && !m.syllables && hasHan(m.lyrics))) await zh.load().catch(() => {});
      const words = melodies.map((m) => {
        if (!m) return null;
        try {
          return sungWords(m, m.syllables ?? lyricsToSyllables(m.lyrics), show.spell);
        } catch (e) {
          console.warn("duet: no words for a voice:", e);
          return null;
        }
      });
      const voices = show.voices.map((v, i) => ({ name: v.name, track: new TeleTrack(teles[i]), words: words[i] }));
      await transport.load(audio);
      transport.seek(0);
      if (live) setLoaded({ show, voices });
    })().catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [base, transport, reloadNonce]);

  useEffect(() => {
    if (!transport) return;
    const off = transport.subscribe(() => setTick((n) => n + 1));
    const id = setInterval(() => transport.isPlaying && setTick((n) => n + 1), 250);
    return () => {
      off();
      clearInterval(id);
    };
  }, [transport]);

  const handleDeleteCurrent = async () => {
    if (!activeId || deleting) return;
    setDeleting(true);
    try {
      transport?.pause();
      const resp = await fetch(`/api/shows/${encodeURIComponent(activeId)}`, { method: "DELETE" });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        setError(data.error || "삭제에 실패했습니다.");
        return;
      }
      setConfirmDelete(false);
      const remaining = await refreshShowList();
      if (remaining.length > 0) {
        switchShow(remaining[0].id);
        setReloadNonce((n) => n + 1);
      } else {
        setLoaded(null);
        setError("저장된 쇼가 없습니다. 우측 상단 '+ New Show'로 새 곡을 만들어 주세요.");
        setComposerOpen(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  const toggleVoice = (v: string) => {
    setSelectedVoices((prev) => {
      if (prev.includes(v)) {
        return prev.length > 1 ? prev.filter((x) => x !== v) : prev;
      }
      return [...prev, v];
    });
  };

  const handleGenerate = async () => {
    if (!sourceInput.trim()) {
      setGenError("유튜브 URL(또는 로컬 오디오 경로)을 입력해 주세요.");
      return;
    }
    setGenError(null);
    setGenLog("");
    setGenStep("요청 전송 중...");
    try {
      const resp = await fetch("/api/generate-show", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: sourceInput.trim(),
          title: titleInput.trim(),
          id: idInput.trim(),
          lang: langInput,
          lyrics: showManualLyrics ? lyricsInput.trim() : "",
          voices: selectedVoices.join(","),
          mode: assignMode,
          maxSec: maxSecInput.trim() ? Number(maxSecInput) || 0 : 0,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        setGenError(data.error || "생성을 시작하지 못했습니다.");
        return;
      }
      setGenerating(true);
    } catch (e) {
      setGenError(e instanceof Error ? e.message : String(e));
    }
  };

  const st = transport?.state ?? { playing: false, time: 0, duration: 0, epoch: 0 };
  const title = loaded?.show.title ?? (error ? "Show" : "Loading…");
  const canDelete = showList.some((s) => s.id === activeId);

  return (
    <div className="duet">
      <header className="duet-top" style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
        {props.onBack && (
          <Chip on={false} onClick={props.onBack}>
            ← Back
          </Chip>
        )}
        <span className="wordmark">Chimera</span>
        <span className="duet-title">{title}</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
          {showList.length > 0 && (
            <select
              aria-label="Select Show"
              value={activeId}
              onChange={(e) => switchShow(e.target.value)}
              style={{
                background: "#080c14",
                color: "#9ae6ff",
                border: "1px solid rgba(90, 200, 255, 0.4)",
                borderRadius: "4px",
                padding: "0.25rem 0.5rem",
                fontSize: "0.8rem",
              }}
            >
              {!showList.some((s) => s.id === activeId) && <option value={activeId}>{activeId}</option>}
              {showList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title}
                </option>
              ))}
            </select>
          )}
          {canDelete && !confirmDelete && (
            <Chip on={false} onClick={() => setConfirmDelete(true)}>
              🗑 Delete
            </Chip>
          )}
          {canDelete && confirmDelete && (
            <>
              <button
                disabled={deleting}
                onClick={() => void handleDeleteCurrent()}
                style={{
                  background: "#991b1b",
                  color: "#fff",
                  border: "1px solid #f87171",
                  borderRadius: "4px",
                  padding: "0.25rem 0.6rem",
                  fontSize: "0.75rem",
                  cursor: "pointer",
                }}
              >
                {deleting ? "삭제 중..." : "삭제 확인"}
              </button>
              <Chip on={false} onClick={() => setConfirmDelete(false)}>
                취소
              </Chip>
            </>
          )}
          <Chip on={composerOpen} onClick={() => setComposerOpen((v) => !v)}>
            {composerOpen ? "✕ Close" : "+ New Show"}
          </Chip>
        </div>
      </header>

      {composerOpen && (
        <div
          style={{
            background: "rgba(6, 10, 20, 0.96)",
            borderBottom: "1px solid rgba(90, 200, 255, 0.35)",
            padding: "1rem 1.25rem",
            display: "grid",
            gap: "0.75rem",
            color: "#d8f3ff",
            fontSize: "0.85rem",
          }}
        >
          <div style={{ display: "grid", gridTemplateColumns: "2.2fr 1fr 1fr", gap: "0.6rem" }}>
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <span>YouTube URL 또는 로컬 MP3/WAV 경로 (가사 자동 추출)</span>
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=... 또는 C:\path\song.mp3"
                value={sourceInput}
                onChange={(e) => setSourceInput(e.target.value)}
                disabled={generating}
                style={{ background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.45rem 0.6rem", borderRadius: "4px" }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <span>곡 제목 (선택)</span>
              <input
                type="text"
                placeholder="예: Mr. Sandman"
                value={titleInput}
                onChange={(e) => setTitleInput(e.target.value)}
                disabled={generating}
                style={{ background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.45rem 0.6rem", borderRadius: "4px" }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <span>길이 제한(초 · 비우면 전체)</span>
              <input
                type="number"
                min={0}
                placeholder="전체 영상 길이"
                value={maxSecInput}
                onChange={(e) => setMaxSecInput(e.target.value)}
                disabled={generating}
                style={{ background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.45rem 0.6rem", borderRadius: "4px" }}
              />
            </label>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span>보이스:</span>
              {(["male", "female", "child"] as const).map((v) => (
                <Chip key={v} on={selectedVoices.includes(v)} onClick={() => !generating && toggleVoice(v)}>
                  {v.toUpperCase()}
                </Chip>
              ))}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span>배정:</span>
              <Chip on={assignMode === "alternate"} onClick={() => !generating && setAssignMode("alternate")}>
                교대
              </Chip>
              <Chip on={assignMode === "register"} onClick={() => !generating && setAssignMode("register")}>
                음역대별
              </Chip>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span>언어:</span>
              {(["auto", "en", "ko"] as const).map((lg) => (
                <Chip key={lg} on={langInput === lg} onClick={() => !generating && setLangInput(lg)}>
                  {lg === "auto" ? "자동" : lg.toUpperCase()}
                </Chip>
              ))}
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span>ID(선택):</span>
              <input
                type="text"
                placeholder="자동 생성"
                value={idInput}
                onChange={(e) => setIdInput(e.target.value)}
                disabled={generating}
                style={{ width: "110px", background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.25rem 0.5rem", borderRadius: "4px" }}
              />
            </label>
            <Chip on={showManualLyrics} onClick={() => !generating && setShowManualLyrics((v) => !v)}>
              {showManualLyrics ? "자동 가사 사용" : "수동 가사 입력"}
            </Chip>
          </div>

          {showManualLyrics && (
            <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <span>수동 가사 입력 (입력 시 Whisper 대신 이 가사 사용)</span>
              <textarea
                rows={4}
                placeholder="가사를 한 줄씩 입력..."
                value={lyricsInput}
                onChange={(e) => setLyricsInput(e.target.value)}
                disabled={generating}
                style={{ background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.5rem 0.6rem", borderRadius: "4px", fontFamily: "inherit" }}
              />
            </label>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <button className="primary" disabled={generating} onClick={() => void handleGenerate()} style={{ padding: "0.5rem 1.2rem" }}>
              {generating ? "생성 중..." : "생성 및 바로 재생"}
            </button>
            {genStep && (
              <span style={{ color: "#38e8ff", fontFamily: "monospace", fontSize: "0.8rem" }}>
                ▶ {genStep}
              </span>
            )}
            {genError && <span style={{ color: "#ff6b81" }}>{genError}</span>}
          </div>

          {(generating || genLog) && (
            <pre
              ref={logBoxRef}
              style={{
                maxHeight: "160px",
                overflow: "auto",
                background: "#02040a",
                border: "1px solid rgba(90, 200, 255, 0.2)",
                padding: "0.6rem",
                borderRadius: "4px",
                fontSize: "0.73rem",
                color: "#a5f3fc",
                margin: 0,
                whiteSpace: "pre-wrap",
              }}
            >
              {genLog || "로그 대기 중..."}
            </pre>
          )}
        </div>
      )}

      {error ? (
        <p className="duet-msg error">This show could not load: {error}</p>
      ) : (
        <main className="duet-heads" style={{ "--voices": loaded?.voices.length ?? 2 } as React.CSSProperties}>
          {loaded && transport ? (
            loaded.voices.map((v, i) => (
              <Head
                key={`${activeId}-${reloadNonce}-${i}`}
                name={v.name}
                track={v.track}
                words={v.words}
                show={loaded.show}
                transport={transport}
                facing={i > 0 && i === loaded.voices.length - 1 ? -1 : 1}
              />
            ))
          ) : (
            <p className="duet-msg">Loading the song…</p>
          )}
        </main>
      )}

      <footer className="duet-bar">
        <button className="primary large" disabled={!loaded} onClick={() => (st.playing ? transport?.pause() : transport?.play())}>
          {st.playing ? "Pause" : st.duration > 0 && st.time >= st.duration - 0.05 ? "Again" : "Play"}
        </button>
        <span className="duet-time">
          {clock(st.time)} / {clock(st.duration)}
        </span>
        <input
          type="range"
          aria-label="Position in the song"
          min={0}
          max={st.duration || 1}
          step={0.1}
          value={st.time}
          disabled={!loaded}
          onChange={(e) => transport?.seek(Number(e.target.value))}
          style={{ "--pos": `${st.duration ? (100 * st.time) / st.duration : 0}%` } as React.CSSProperties}
        />
      </footer>
    </div>
  );
}

/** one voice's head; the last of several turns to face the others */
function Head(props: { name: string; track: TeleTrack; words: SungWords | null; show: ShowFile; transport: Transport; facing: 1 | -1 }) {
  const { track, show, transport, words } = props;
  const feedRef = useRef<HeadFeed | null>(null);
  const overlayRef = useRef<OverlayFn | undefined>(undefined);
  if (words && !overlayRef.current) overlayRef.current = mouthWords(() => wordAt(words, transport.time()), V.green, props.facing);
  if (!feedRef.current) {
    const model = getModel(track.body, track.sampleRate);
    const beatStep = (t: number) => {
      if (!show.bpm) return -1;
      const s = Math.floor(((t - (show.downbeat ?? 0)) * show.bpm * 4) / 60);
      return s < 0 ? -1 : s % 16;
    };
    feedRef.current = {
      model,
      view: VIEW,
      maxDpr: 1.5,
      facing: props.facing,
      frame: () => {
        const t = transport.time();
        const i = track.indexAt(t);
        const msg = track.message(i, beatStep(t));
        if (!transport.isPlaying) return { tele: null, artic: VoiceModel.articFromArray(msg.t.artic), epoch: transport.epoch };
        return { tele: msg, artic: VoiceModel.articFromArray(msg.t.artic), epoch: transport.epoch };
      },
    };
  }
  return (
    <Screen className="duet-head" focus title={props.name} sub={bodyLabel(track.body)} label={`${props.name} voice, ${bodyLabel(track.body)}, in 3D`}>
      <AnatomyView3D feed={feedRef.current} readouts={false} overlay={overlayRef.current} />
    </Screen>
  );
}