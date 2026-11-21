import { lazy, Suspense, useEffect, useState } from "react";
import { useStore } from "./store.ts";
import { engine } from "../audio/AudioEngine.ts";
import { AnatomyView } from "./components/AnatomyView.tsx";
import { TubeView } from "./components/TubeView.tsx";
import { Stroboscope } from "./components/Stroboscope.tsx";
import { SpectrumView } from "./components/SpectrumView.tsx";
import { VowelMap } from "./components/VowelMap.tsx";
import { VoiceControls } from "./components/VoiceControls.tsx";
import { Keys, MidiButton } from "./components/Keys.tsx";
import { ActionBar, PrimaryButton } from "./components/ActionBar.tsx";
import { Segmented, Toggle } from "./components/Controls.tsx";
import { Chip, Screen } from "./components/Screen.tsx";
import { SpeakRight, speechInfo, speechSlots, speechWords } from "./components/SpeakPanel.tsx";
import { BestiaryRight } from "./components/BestiaryPanel.tsx";
import { CreatureView } from "./components/CreatureView.tsx";
import { BeatboxRight, BeatOverlay, beatInfo, beatSlots } from "./components/BeatboxPanel.tsx";
import { SingRight, singInfo, singWords } from "./components/SingPanel.tsx";
import type { Mode } from "./store.ts";
import type { BodyName } from "../worklet/protocol.ts";
import { BodyPad } from "./components/BodyPad.tsx";
import { presetOf, PRESETS } from "./body.ts";

export interface CustomVoicePreset {
  id: string;
  name: string;
  acoustics?: { median_f0_hz?: number; median_midi?: number };
  chimera_preset: {
    cli_body: string;
    cli_body_by_timbre?: string;
    body_blend: { male: number; female: number; child: number };
    play_voice: { pitch_hz: number; pressed_pct: number; vibrato_cents: number };
    play_tone: { breathy_pct: number; rough_pct: number; fry_pct: number };
    play_throat_larynx: { larynx_pct: number; twang_pct: number };
    sing_encoders: { vibrato_depth_cents: number; vibrato_rate_hz: number; breathy_pct: number; ring_pct: number };
  };
}

// three.js only loads when the 3D view is first shown
const AnatomyView3D = lazy(() => import("./components/AnatomyView3D.tsx").then((m) => ({ default: m.AnatomyView3D })));
// the duet page (`?show=<id>`): two (or more) 3D heads replaying a pre-rendered song
const DuetPage = lazy(() => import("./components/DuetPage.tsx").then((m) => ({ default: m.DuetPage })));

/** the show to play, from `?show=<id>` (a plain folder name under public/shows) */
function showId(): string | null {
  const id = new URLSearchParams(window.location.search).get("show");
  return id && /^[\w-]+$/.test(id) ? id : null;
}

function TopBar(props: { onOpenShow: (id: string) => void; onOpenAddSynth: () => void }) {
  const started = useStore((s) => s.started);
  const volume = useStore((s) => s.volume);
  const setVolume = useStore((s) => s.setVolume);
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const light = useStore((s) => s.lightAudio);
  const [load, setLoad] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setLoad(engine.load), 500);
    return () => clearInterval(id);
  }, []);
  return (
    <header className="topbar">
      <div className="brand">
        <span className="wordmark">Chimera</span>
        <span className="tagline">A voice grown from a simulated airway</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <Segmented<Mode>
          label="Mode"
          value={mode}
          onChange={setMode}
          options={[
            { value: "play", label: "Play" },
            { value: "speak", label: "Speak" },
            { value: "sing", label: "Sing" },
            { value: "bestiary", label: "Bestiary" },
            { value: "beatbox", label: "Beatbox" },
          ]}
        />
        <Chip on={false} onClick={() => props.onOpenShow("sandman")}>
          Show
        </Chip>
        <Chip on={false} onClick={props.onOpenAddSynth}>
          + Synth
        </Chip>
      </div>
      <div className="topbar-right">
        <label className="volume">
          <span>Volume</span>
          <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(e) => setVolume(Number(e.target.value))} style={{ "--pos": `${volume * 100}%` } as React.CSSProperties} />
        </label>
        {started && (
          <span className="load" title={"Share of the audio thread's time budget used by the simulation" + (light ? ". This device couldn't keep up at 48 kHz, so the voice runs at 32 kHz: lighter, a little duller. Reload to try 48 kHz again." : "")}>
            Engine load {Math.round(load * 100)} %{light && " · 32 kHz"}
          </span>
        )}
        <a className="repo-link" href="https://github.com/jamesondh/chimera-vocal-synth" target="_blank" rel="noreferrer" aria-label="Source code on GitHub" title="Source code on GitHub">
          <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
            <path
              fill="currentColor"
              d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
            />
          </svg>
        </a>
      </div>
    </header>
  );
}

function ViewToggles() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const bestiary = useStore((s) => s.mode === "bestiary");
  return (
    <div className="view-toggles">
      {!bestiary && <Toggle label="3D" checked={view.threeD} onChange={(v) => setView({ threeD: v })} />}
      <Toggle label="Sound pressure" checked={view.pressure} onChange={(v) => setView({ pressure: v })} />
      {!bestiary && <Toggle label="Airflow" checked={view.particles} onChange={(v) => setView({ particles: v })} />}
      <Toggle label={bestiary ? "Labels" : "Measurements"} checked={view.calipers} onChange={(v) => setView({ calipers: v })} />
      {!bestiary && <Toggle label="Grid" checked={view.grid} onChange={(v) => setView({ grid: v })} />}
    </div>
  );
}

/** The body, in the head's header: three presets, custom synths, and the triangle between them. */
function BodyControl(props: {
  customPresets: CustomVoicePreset[];
  activePresetId: string | null;
  onSelectPreset: (p: CustomVoicePreset) => void;
  onClearCustom: () => void;
}) {
  const body = useStore((s) => s.body);
  const setBody = useStore((s) => s.setBody);
  return (
    <div className="body-control" style={{ flexWrap: "wrap", gap: "0.4rem" }}>
      <Segmented<BodyName>
        label="Body"
        value={presetOf(body)}
        onChange={(b) => {
          props.onClearCustom();
          setBody(PRESETS[b]);
        }}
        options={[
          { value: "male", label: "Male" },
          { value: "female", label: "Female" },
          { value: "child", label: "Child" },
        ]}
      />
      <BodyPad />
      {props.customPresets.map((cp) => (
        <Chip
          key={cp.id}
          on={props.activePresetId === cp.id}
          onClick={() => props.onSelectPreset(cp)}
        >
          ★ {cp.name}
        </Chip>
      ))}
    </div>
  );
}

/** The tall left screen: the face (or the creature) with its overlays. No title: it shows itself. */
function LeftScreen(props: {
  customPresets: CustomVoicePreset[];
  activePresetId: string | null;
  onSelectPreset: (p: CustomVoicePreset) => void;
  onClearCustom: () => void;
}) {
  const mode = useStore((s) => s.mode);
  const threeD = useStore((s) => s.view.threeD);
  const bestiary = mode === "bestiary";
  const Face = threeD ? AnatomyView3D : AnatomyView;
  return (
    <Screen
      className="left"
      focus
      lead={
        bestiary ? undefined : (
          <BodyControl
            customPresets={props.customPresets}
            activePresetId={props.activePresetId}
            onSelectPreset={props.onSelectPreset}
            onClearCustom={props.onClearCustom}
          />
        )
      }
      label={bestiary ? "Live creature airway" : "Live vocal tract"}
      tools={<ViewToggles />}
    >
      <Suspense fallback={null}>
        {bestiary ? <CreatureView /> : mode === "speak" ? <Face slots={speechSlots} info={speechInfo} overlay={speechWords} /> : mode === "sing" ? <Face info={singInfo} overlay={singWords} /> : mode === "beatbox" ? <Face slots={beatSlots} info={beatInfo} showFormants={false} /> : <Face />}
      </Suspense>
      {mode === "beatbox" && <BeatOverlay />}
    </Screen>
  );
}

/** Play's main button: breathe (the voice sounds while it's on), and Whisper beside it. */
function PlayActions() {
  const hold = useStore((s) => s.hold);
  const toggleBreath = useStore((s) => s.toggleBreath);
  const voiced = useStore((s) => s.source.voiced);
  const setSource = useStore((s) => s.setSource);
  return (
    <ActionBar label="Voice" onSpace={() => void toggleBreath()}>
      <PrimaryButton label="Breathe" icon="breath" playing={hold} onClick={() => void toggleBreath()} />
      <Chip on={!voiced} onClick={() => setSource({ voiced: !voiced })}>
        Whisper
      </Chip>
    </ActionBar>
  );
}

function PlayRight() {
  const f0 = useStore((s) => s.source.f0);
  return (
    <div className="right right-play">
      <PlayActions />
      <Screen className="s-tube" title="Tube" sub="Glottis to lips">
        <TubeView />
      </Screen>
      <Screen className="s-folds" title="Vocal folds" sub="Slowed 100×">
        <Stroboscope />
      </Screen>
      <Screen className="s-spec" title="Spectrum">
        <SpectrumView />
      </Screen>
      <Screen className="s-vowels" title="Vowels" sub="F1 / F2">
        <VowelMap />
      </Screen>
      <Screen className="s-voice" title="Voice">
        <VoiceControls />
      </Screen>
      <Screen className="s-keys" title="Keys" sub={`${Math.round(f0)} Hz`} tools={<MidiButton />}>
        <Keys />
      </Screen>
    </div>
  );
}

function AddSynthModal(props: {
  presets: CustomVoicePreset[];
  onClose: () => void;
  onUpdated: (list: CustomVoicePreset[], newest?: CustomVoicePreset) => void;
}) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!name.trim() || !file) {
      setErr("신스 이름과 샘플 보이스 파일(WAV/MP3)을 모두 입력해 주세요.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const b64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("파일을 읽지 못했습니다."));
        reader.readAsDataURL(file);
      });
      const resp = await fetch("/api/voice-presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), filename: file.name, audioBase64: b64 }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        setErr(data.error || "신스 생성에 실패했습니다.");
        return;
      }
      setName("");
      setFile(null);
      props.onUpdated(data.presets || [], data.preset);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id: string) => {
    const resp = await fetch(`/api/voice-presets/${encodeURIComponent(id)}`, { method: "DELETE" });
    const data = await resp.json();
    if (resp.ok && data.ok) {
      props.onUpdated(data.presets || []);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgba(0,0,0,0.78)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "1rem",
      }}
    >
      <div
        style={{
          background: "#070b14",
          border: "1px solid rgba(90,200,255,0.45)",
          borderRadius: "8px",
          width: "100%",
          maxWidth: "520px",
          padding: "1.25rem",
          color: "#d8f3ff",
          display: "grid",
          gap: "0.85rem",
          fontSize: "0.85rem",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <strong style={{ fontSize: "1rem", color: "#38e8ff" }}>+ 새 커스텀 신스 추가 (샘플 보이스 분석)</strong>
          <Chip on={false} onClick={props.onClose}>
            ✕ Close
          </Chip>
        </div>

        <label style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
          <span>신스 이름</span>
          <input
            type="text"
            placeholder="예: CityPop Girl, Husky Male"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
            style={{ background: "#03060d", color: "#fff", border: "1px solid rgba(90,200,255,0.35)", padding: "0.45rem 0.6rem", borderRadius: "4px" }}
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
          <span>샘플 보이스 파일 (WAV / MP3, 10~30초 클린 보컬 추천)</span>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            disabled={busy}
            style={{ color: "#9ae6ff" }}
          />
        </label>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <button className="primary" disabled={busy} onClick={() => void handleCreate()} style={{ padding: "0.45rem 1rem" }}>
            {busy ? "보이스 분석 및 신스 생성 중..." : "분석 및 신스 추가"}
          </button>
          {err && <span style={{ color: "#ff6b81" }}>{err}</span>}
        </div>

        {props.presets.length > 0 && (
          <div style={{ borderTop: "1px solid rgba(90,200,255,0.2)", paddingTop: "0.75rem", display: "grid", gap: "0.4rem" }}>
            <span style={{ fontSize: "0.78rem", color: "#94a3b8" }}>저장된 커스텀 신스 (메인 화면 & Show 공용)</span>
            {props.presets.map((p) => (
              <div key={p.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#03060d", padding: "0.4rem 0.6rem", borderRadius: "4px" }}>
                <div>
                  <strong>★ {p.name}</strong>
                  <span style={{ marginLeft: "0.5rem", fontSize: "0.75rem", color: "#7dd3fc" }}>
                    {p.chimera_preset.cli_body} · {p.chimera_preset.play_voice.pitch_hz}Hz
                  </span>
                </div>
                <Chip on={false} onClick={() => void handleDelete(p.id)}>
                  삭제
                </Chip>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function App() {
  const [show, setShow] = useState<string | null>(() => showId());

  useEffect(() => {
    const onPop = () => setShow(showId());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const openShow = (id: string | null) => {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("show", id);
    else url.searchParams.delete("show");
    window.history.pushState(null, "", url.toString());
    setShow(id);
  };

  if (show)
    return (
      <Suspense fallback={null}>
        <DuetPage id={show} onBack={() => openShow(null)} />
      </Suspense>
    );
  return <Wall onOpenShow={openShow} />;
}

function Wall(props: { onOpenShow: (id: string) => void }) {
  const mode = useStore((s) => s.mode);
  const setBody = useStore((s) => s.setBody);
  const setSource = useStore((s) => s.setSource);
  const setArtic = useStore((s) => s.setArtic);

  const [customPresets, setCustomPresets] = useState<CustomVoicePreset[]>([]);
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [addSynthOpen, setAddSynthOpen] = useState(false);

  useEffect(() => {
    fetch("/api/voice-presets")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.presets) setCustomPresets(d.presets);
      })
      .catch(() => {});
  }, []);

  const applyCustomPreset = (p: CustomVoicePreset) => {
    setActivePresetId(p.id);
    const cp = p.chimera_preset;
    const blend = cp.body_blend;
    setBody({ female: blend.female, child: blend.child });
    const pv = cp.play_voice;
    const pt = cp.play_tone;
    const pl = cp.play_throat_larynx;
    setSource({
      f0: Math.max(40, Math.min(1100, pv.pitch_hz || 200)),
      tension: Math.max(0, Math.min(1, (pv.pressed_pct ?? 50) / 100)),
      vibratoDepth: Math.max(0, Math.min(120, pv.vibrato_cents ?? 25)),
      brightness: Math.max(0, Math.min(1, (pl.twang_pct ?? 50) / 100)),
      breathiness: Math.max(0, Math.min(1, (pt.breathy_pct ?? 20) / 100)),
      roughness: Math.max(0, Math.min(1, (pt.rough_pct ?? 10) / 100)),
      fry: Math.max(0, Math.min(1, (pt.fry_pct ?? 0) / 100)),
    });
    // Map larynx_pct (0..100) to CHIMERA's -1.5..+1.5 cm range, twang -> epilarynx (0..1), rough -> ventricular
    const larynxCm = (((pl.larynx_pct ?? 50) - 50) / 50) * 1.3;
    setArtic({
      larynx: Math.max(-1.5, Math.min(1.5, larynxCm)),
      epilarynx: Math.max(0, Math.min(1, (pl.twang_pct ?? 35) / 100)),
      ventricular: Math.max(0, Math.min(1, ((pt.rough_pct ?? 0) * 0.7) / 100)),
    });
  };

  return (
    <div className={`app text-colors mode-${mode}`}>
      <TopBar onOpenShow={props.onOpenShow} onOpenAddSynth={() => setAddSynthOpen(true)} />
      <main className={`wall wall-${mode}`}>
        <LeftScreen
          customPresets={customPresets}
          activePresetId={activePresetId}
          onSelectPreset={applyCustomPreset}
          onClearCustom={() => setActivePresetId(null)}
        />
        {mode === "play" ? <PlayRight /> : mode === "speak" ? <SpeakRight /> : mode === "sing" ? <SingRight /> : mode === "beatbox" ? <BeatboxRight /> : <BestiaryRight />}
      </main>
      {addSynthOpen && (
        <AddSynthModal
          presets={customPresets}
          onClose={() => setAddSynthOpen(false)}
          onUpdated={(list, newest) => {
            setCustomPresets(list);
            if (newest) applyCustomPreset(newest);
          }}
        />
      )}
    </div>
  );
}
