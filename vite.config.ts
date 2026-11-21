import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

interface JobState {
  running: boolean;
  done: boolean;
  ok: boolean;
  id: string;
  step: string;
  logs: string;
  error: string | null;
}

function readPresets(presetsPath: string): any[] {
  if (!fs.existsSync(presetsPath)) return [];
  try {
    const data = JSON.parse(fs.readFileSync(presetsPath, "utf-8"));
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function writePresets(presetsPath: string, list: any[]): void {
  fs.mkdirSync(path.dirname(presetsPath), { recursive: true });
  fs.writeFileSync(presetsPath, JSON.stringify(list, null, 2), "utf-8");
}

function autoShowPlugin(): Plugin {
  let job: JobState = {
    running: false,
    done: false,
    ok: false,
    id: "",
    step: "",
    logs: "",
    error: null,
  };

  return {
    name: "chimera-auto-show-api",
    configureServer(server) {
      const presetsPath = path.resolve(process.cwd(), "public/presets/voices.json");

      // GET / POST / DELETE /api/voice-presets — manage custom tuned synth voices
      server.middlewares.use("/api/voice-presets", (req, res, next) => {
        const subPath = (req.url || "").split("?")[0].replace(/^\/+/, "");

        if (req.method === "GET" && !subPath) {
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ presets: readPresets(presetsPath) }));
          return;
        }

        if (req.method === "DELETE" && subPath) {
          if (!/^[\w-]+$/.test(subPath)) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: false, error: "잘못된 프리셋 ID입니다." }));
            return;
          }
          const nextList = readPresets(presetsPath).filter((p) => p.id !== subPath);
          writePresets(presetsPath, nextList);
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ ok: true, id: subPath, presets: nextList }));
          return;
        }

        if (req.method === "POST" && !subPath) {
          const chunks: Buffer[] = [];
          req.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
          req.on("end", () => {
            try {
              const body = JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}");
              const name = String(body.name || "").trim();
              const audioBase64 = String(body.audioBase64 || "").trim();
              const filename = String(body.filename || "sample.wav").replace(/[^a-zA-Z0-9._-]/g, "_");
              const audioPathInput = String(body.audioPath || "").trim();

              if (!name) {
                res.statusCode = 400;
                res.setHeader("Content-Type", "application/json; charset=utf-8");
                res.end(JSON.stringify({ ok: false, error: "신스 이름을 입력해 주세요." }));
                return;
              }
              if (!audioBase64 && !audioPathInput) {
                res.statusCode = 400;
                res.setHeader("Content-Type", "application/json; charset=utf-8");
                res.end(JSON.stringify({ ok: false, error: "샘플 보이스 파일(WAV/MP3)을 선택해 주세요." }));
                return;
              }

              const slug =
                name
                  .toLowerCase()
                  .replace(/[^a-z0-9]+/g, "_")
                  .replace(/^_|_$/g, "") || `synth_${Date.now().toString(36)}`;
              const id = `custom_${slug}_${Date.now().toString(36).slice(-4)}`;

              const sampleDir = path.resolve(process.cwd(), "_auto_work/preset_samples");
              fs.mkdirSync(sampleDir, { recursive: true });

              let samplePath = audioPathInput;
              if (audioBase64) {
                const rawB64 = audioBase64.includes(",") ? audioBase64.split(",")[1] : audioBase64;
                samplePath = path.join(sampleDir, `${id}_${filename}`);
                fs.writeFileSync(samplePath, Buffer.from(rawB64, "base64"));
              }

              const outJsonPath = path.join(sampleDir, `${id}_preset.json`);
              const pyArgs = ["-u", "tune_voice.py", "--audio", samplePath, "--out", outJsonPath, "--compare-chimera"];

              const runTuner = (exe: string, triedFallback: boolean) => {
                let logs = "";
                const child = spawn(exe, pyArgs, {
                  cwd: process.cwd(),
                  env: { ...process.env, PYTHONUTF8: "1", PYTHONUNBUFFERED: "1" },
                });
                child.stdout.on("data", (d) => {
                  const s = d.toString("utf-8");
                  process.stdout.write(s);
                  logs += s;
                });
                child.stderr.on("data", (d) => {
                  const s = d.toString("utf-8");
                  process.stderr.write(s);
                  logs += s;
                });
                child.on("error", (err: NodeJS.ErrnoException) => {
                  if (!triedFallback && err.code === "ENOENT" && exe === "python") {
                    return runTuner("py", true);
                  }
                  res.statusCode = 500;
                  res.setHeader("Content-Type", "application/json; charset=utf-8");
                  res.end(JSON.stringify({ ok: false, error: `Python 실행 실패: ${err.message}`, log: logs }));
                });
                child.on("close", (code) => {
                  res.setHeader("Content-Type", "application/json; charset=utf-8");
                  if (code !== 0 || !fs.existsSync(outJsonPath)) {
                    res.statusCode = 500;
                    res.end(
                      JSON.stringify({
                        ok: false,
                        error: `보이스 분석 실패 (code ${code})`,
                        log: logs,
                      })
                    );
                    return;
                  }
                  try {
                    const tuned = JSON.parse(fs.readFileSync(outJsonPath, "utf-8"));
                    const preset = {
                      id,
                      name,
                      acoustics: tuned.acoustics,
                      chimera_preset: tuned.chimera_preset,
                    };
                    const list = readPresets(presetsPath);
                    list.push(preset);
                    writePresets(presetsPath, list);
                    res.end(JSON.stringify({ ok: true, preset, presets: list }));
                  } catch (e) {
                    res.statusCode = 500;
                    res.end(JSON.stringify({ ok: false, error: String(e), log: logs }));
                  }
                });
              };

              runTuner("python", false);
            } catch (err) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json; charset=utf-8");
              res.end(JSON.stringify({ ok: false, error: String(err) }));
            }
          });
          return;
        }

        return next();
      });

      // GET /api/shows & DELETE /api/shows/:id
      server.middlewares.use("/api/shows", (req, res, next) => {
        const subPath = (req.url || "").split("?")[0].replace(/^\/+/, "");

        if (req.method === "DELETE" && subPath) {
          if (!/^[\w-]+$/.test(subPath)) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: false, error: "잘못된 쇼 ID입니다." }));
            return;
          }
          try {
            const showDir = path.resolve(process.cwd(), "public/shows", subPath);
            const workDir = path.resolve(process.cwd(), "_auto_work", subPath);
            const workLyrics = path.resolve(process.cwd(), "_auto_work", `${subPath}_lyrics.txt`);
            if (fs.existsSync(showDir)) fs.rmSync(showDir, { recursive: true, force: true });
            if (fs.existsSync(workDir)) fs.rmSync(workDir, { recursive: true, force: true });
            if (fs.existsSync(workLyrics)) fs.rmSync(workLyrics, { force: true });
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: true, id: subPath }));
          } catch (err) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: false, error: String(err) }));
          }
          return;
        }

        if (req.method !== "GET" || subPath) return next();
        try {
          const root = path.resolve(process.cwd(), "public/shows");
          const shows: { id: string; title: string }[] = [];
          if (fs.existsSync(root)) {
            for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
              if (!entry.isDirectory()) continue;
              const metaPath = path.join(root, entry.name, "show.json");
              if (!fs.existsSync(metaPath)) continue;
              try {
                const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
                shows.push({ id: entry.name, title: meta.title || entry.name });
              } catch {
                shows.push({ id: entry.name, title: entry.name });
              }
            }
          }
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ shows }));
        } catch (err) {
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ error: String(err) }));
        }
      });

      // GET /api/generate-status — live progress & logs for the running auto_show.py job
      server.middlewares.use("/api/generate-status", (req, res, next) => {
        if (req.method !== "GET") return next();
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.end(JSON.stringify(job));
      });

      // POST /api/generate-show — start auto_show.py (lyrics optional; Whisper auto-transcribes)
      server.middlewares.use("/api/generate-show", (req, res, next) => {
        if (req.method !== "POST") return next();
        let raw = "";
        req.on("data", (chunk) => {
          raw += chunk;
        });
        req.on("end", () => {
          try {
            if (job.running) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json; charset=utf-8");
              res.end(JSON.stringify({ ok: false, error: "이미 다른 곡을 생성 중입니다." }));
              return;
            }

            const body = JSON.parse(raw || "{}");
            const source = String(body.source || "").trim();
            const lyrics = String(body.lyrics || "").trim();
            const lang = String(body.lang || "auto").trim();
            const voices = String(body.voices || "male,female,child").trim();
            const mode = body.mode === "register" ? "register" : body.mode === "harmony" ? "harmony" : "alternate";
            const maxSec = Number(body.maxSec || 0);
            const title = String(body.title || "").trim();
            const rawId = String(body.id || "").trim().replace(/[^a-zA-Z0-9_-]/g, "");
            const id = rawId || `show_${Date.now().toString(36)}`;

            if (!source) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json; charset=utf-8");
              res.end(JSON.stringify({ ok: false, error: "유튜브 URL(또는 로컬 오디오 경로)을 입력해 주세요." }));
              return;
            }

            const isUrl = /^https?:\/\//i.test(source);
            const scriptArgs = [
              "-u",
              "auto_show.py",
              isUrl ? "--url" : "--audio",
              source,
              "--id",
              id,
              "--voices",
              voices,
              "--mode",
              mode,
              "--lang",
              lang || "auto",
            ];

            if (lyrics) {
              const workDir = path.resolve(process.cwd(), "_auto_work");
              fs.mkdirSync(workDir, { recursive: true });
              const lyricsPath = path.join(workDir, `${id}_lyrics.txt`);
              fs.writeFileSync(lyricsPath, lyrics, "utf-8");
              scriptArgs.push("--lyrics", lyricsPath);
            }
            if (title) scriptArgs.push("--title", title);
            if (maxSec > 0) scriptArgs.push("--max-sec", String(maxSec));

            job = {
              running: true,
              done: false,
              ok: false,
              id,
              step: "Python 파이프라인 시작 중...",
              logs: "",
              error: null,
            };

            const appendLog = (chunk: string) => {
              process.stdout.write(chunk);
              job.logs = (job.logs + chunk).slice(-16000);
              const stepMatches = [...chunk.matchAll(/\[STEP\]\s*(.+)/g)];
              if (stepMatches.length > 0) {
                job.step = stepMatches[stepMatches.length - 1][1].trim();
              }
            };

            const runPython = (exe: string, triedFallback: boolean) => {
              const child = spawn(exe, scriptArgs, {
                cwd: process.cwd(),
                env: { ...process.env, PYTHONUTF8: "1", PYTHONUNBUFFERED: "1" },
              });

              child.stdout.on("data", (d) => appendLog(d.toString("utf-8")));
              child.stderr.on("data", (d) => appendLog(d.toString("utf-8")));

              child.on("error", (err: NodeJS.ErrnoException) => {
                if (!triedFallback && err.code === "ENOENT" && exe === "python") {
                  return runPython("py", true);
                }
                job.running = false;
                job.done = true;
                job.ok = false;
                job.step = "실행 실패";
                job.error = `Python 실행 실패: ${err.message}`;
              });

              child.on("close", (code) => {
                job.running = false;
                job.done = true;
                if (code === 0) {
                  job.ok = true;
                  job.step = "완료! 3D 쇼 로딩 중...";
                } else {
                  job.ok = false;
                  job.step = `오류 종료 (code ${code})`;
                  job.error = `auto_show.py가 종료 코드 ${code}로 실패했습니다.`;
                }
              });
            };

            runPython("python", false);

            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: true, id }));
          } catch (err) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ ok: false, error: String(err) }));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), autoShowPlugin()],
  worker: { format: "es" },
  build: { target: "es2022", sourcemap: true },
  server: { host: "127.0.0.1", port: 5317, strictPort: true, allowedHosts: true },
  preview: { host: "127.0.0.1", port: 5318, strictPort: true, allowedHosts: true },
});