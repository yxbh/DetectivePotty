import { spawn, type ChildProcess } from "node:child_process";
import { createConnection } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// `npm run dev` only serves the UI; the data comes from the FastAPI backend.
// Rather than make you start that in a second terminal, the dev server boots it
// for you (see backendAutostart below) and proxies /api to it.
const API_HOST = "127.0.0.1";
const API_PORT = 8000;
const API_TARGET = `http://${API_HOST}:${API_PORT}`;

// vite.config.ts lives at src/detectivepotty/web/frontend/, so the repo root
// (where config.yaml and `uv` live) is four levels up.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..", "..", "..");
const CONFIG = process.env.DETECTIVEPOTTY_CONFIG ?? "config.yaml";

function probePort(host: string, port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection({ host, port });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => resolve(false));
  });
}

async function waitForApi(timeoutMs: number, child: ChildProcess): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  let exited = false;
  child.once("exit", () => {
    exited = true;
  });
  while (Date.now() < deadline && !exited) {
    if (await probePort(API_HOST, API_PORT)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
}

// Defense-in-depth against silently reusing a backend started before a route
// existed: a stale `serve` (no --reload) keeps whatever app.py looked like when
// it launched, so it can be missing /api/stream and leave the Live feed stuck on
// "reconnecting". We can't safely kill a process we didn't start, so we just warn
// loudly. Returns "ok" when /api/stream serves SSE, "stale" when it's missing/wrong,
// "unknown" when the probe itself failed.
async function probeStreamContract(): Promise<"ok" | "stale" | "unknown"> {
  try {
    const controller = new AbortController();
    const res = await fetch(`${API_TARGET}/api/stream`, { signal: controller.signal });
    const ctype = res.headers.get("content-type") ?? "";
    controller.abort();
    if (res.status === 404) return "stale";
    return ctype.includes("text/event-stream") ? "ok" : "stale";
  } catch {
    return "unknown";
  }
}

// Boots `detectivepotty serve` alongside the dev server so `npm run dev` is a
// single, self-sufficient command. It reuses a backend you've already started,
// can be opted out of with DETECTIVEPOTTY_NO_BACKEND=1, and is torn down with the
// dev server. Only runs in dev (apply: "serve"); production uses the built dist/.
function backendAutostart(): Plugin {
  let child: ChildProcess | undefined;
  return {
    name: "detectivepotty-backend-autostart",
    apply: "serve",
    async configureServer(server) {
      const log = server.config.logger;
      if (process.env.DETECTIVEPOTTY_NO_BACKEND) {
        log.info("[backend] auto-start disabled (DETECTIVEPOTTY_NO_BACKEND set)");
        return;
      }
      if (await probePort(API_HOST, API_PORT)) {
        const contract = await probeStreamContract();
        if (contract === "stale") {
          log.warn(
            `[backend] reusing API at ${API_TARGET}, but it is missing /api/stream — ` +
              "this is likely a stale backend started before the live-stream route existed. " +
              "Stop it and re-run `npm run dev` (it now starts the backend with --reload) so " +
              "the Live feed can connect.",
          );
        } else {
          log.info(`[backend] reusing API already running at ${API_TARGET}`);
        }
        return;
      }

      log.info(`[backend] starting API: uv run detectivepotty serve --config ${CONFIG} --reload`);
      child = spawn("uv", ["run", "detectivepotty", "serve", "--config", CONFIG, "--reload"], {
        cwd: REPO_ROOT,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
        // Own process group so we can tear down uv *and* its python child together.
        detached: true,
      });
      const relay = (chunk: Buffer) => process.stdout.write(`[backend] ${chunk}`);
      child.stdout?.on("data", relay);
      child.stderr?.on("data", relay);
      child.on("error", (err) => {
        log.error(`[backend] failed to launch: ${err.message} — is \`uv\` installed and on PATH?`);
      });
      child.on("exit", (code) => {
        if (code) {
          log.error(
            `[backend] exited with code ${code} — check that ${CONFIG} exists ` +
              "(or set DETECTIVEPOTTY_CONFIG) and see the errors above.",
          );
        }
      });

      const shutdown = () => {
        if (child?.pid && child.exitCode === null) {
          try {
            process.kill(-child.pid, "SIGTERM");
          } catch {
            child.kill("SIGTERM");
          }
        }
      };
      server.httpServer?.once("close", shutdown);
      process.once("SIGINT", shutdown);
      process.once("SIGTERM", shutdown);
      process.once("exit", shutdown);

      if (await waitForApi(30_000, child)) {
        log.info(`[backend] API ready at ${API_TARGET}`);
      } else {
        log.warn("[backend] API is not ready yet — the portal will load it once it comes up.");
      }
    },
  };
}

// On the rare occasion the backend is down mid-session, surface a clear 502
// instead of an opaque proxy 500.
const PROXY_HINT =
  `The DetectivePotty API backend at ${API_TARGET} isn't responding. It is started ` +
  "automatically by `npm run dev`; check this terminal for [backend] errors.";

export default defineConfig({
  plugins: [svelte(), backendAutostart()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("error", (err, _req, res) => {
            console.error(`\n[vite proxy] ${PROXY_HINT}\n(${err.message})\n`);
            if (res && "writeHead" in res && !res.headersSent) {
              res.writeHead(502, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ detail: PROXY_HINT }));
            }
          });
        },
      },
    },
  },
});
