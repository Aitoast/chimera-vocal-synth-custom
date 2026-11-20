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
      // GET /api/shows & DELETE /api/shows/:id
      server.middlewares.use("/api/shows", (req, res, next) => {
        const subPath = (req.url || "").split("?")[0].replace(/^\/+/, "");

        // DELETE /api/shows/<id> — remove public/shows/<id> and _auto_work/<id>
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

        // GET /api/shows — list all available shows under public/shows/
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
            const mode = body.mode === "register" ? "register" : "alternate";
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