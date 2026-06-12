import { execFile, spawn, type ChildProcess } from "node:child_process";
import { createConnection } from "node:net";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// `npm run dev` only serves the UI; the data comes from the FastAPI backend.
// Rather than make you start that in a second terminal, the dev server boots it
// for you (see backendAutostart below) and proxies /api to it.
const API_HOST = "127.0.0.1";
const API_PORT = 8000;
const API_TARGET = `http://${API_HOST}:${API_PORT}`;
const STREAM_PROBE_TIMEOUT_MS = 2000;
const BACKEND_SHUTDOWN_GRACE_MS = 10_000;
const FRONTEND_CONNECTION_DRAIN_MS = 100;
const execFileAsync = promisify(execFile);

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

async function waitForPortFree(timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!(await probePort(API_HOST, API_PORT))) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
}

async function listenerPids(): Promise<number[]> {
  try {
    const { stdout } = await execFileAsync("lsof", [
      "-nP",
      `-iTCP:${API_PORT}`,
      "-sTCP:LISTEN",
      "-t",
    ]);
    return [...new Set(stdout.split(/\s+/).map(Number).filter(Number.isFinite))];
  } catch {
    return [];
  }
}

async function psField(pid: number, field: "command" | "pgid"): Promise<string> {
  const { stdout } = await execFileAsync("ps", ["-p", String(pid), "-o", `${field}=`]);
  return stdout.trim();
}

function isDevServeCommand(command: string): boolean {
  return (
    command.includes("detectivepotty serve") &&
    command.includes("--reload") &&
    command.includes("--config") &&
    command.includes(CONFIG)
  );
}

async function stopUnhealthyDevBackend(log: { warn: (message: string) => void }): Promise<boolean> {
  for (const pid of await listenerPids()) {
    let command = "";
    let pgid = pid;
    try {
      command = await psField(pid, "command");
      pgid = Number.parseInt(await psField(pid, "pgid"), 10) || pid;
    } catch {
      continue;
    }
    if (!isDevServeCommand(command)) {
      continue;
    }
    log.warn(
      `[backend] stopping unhealthy auto-started backend on ${API_TARGET} ` +
        `(pid ${pid}) before launching a fresh one`,
    );
    if (!signalProcessGroup(pgid, "SIGTERM")) {
      return false;
    }
    if (await waitForPortFree(5_000)) {
      return true;
    }
    log.warn(
      `[backend] auto-started backend on ${API_TARGET} ignored SIGTERM; ` +
        "forcing it to stop before restart",
    );
    if (!signalProcessGroup(pgid, "SIGKILL")) {
      return false;
    }
    return await waitForPortFree(5_000);
  }
  return false;
}

function unhealthyBackendMessage(): string {
  return (
    `[backend] ${API_TARGET} is occupied by an unhealthy process that could not ` +
    "be restarted automatically. Stop it manually, then re-run `npm run dev`."
  );
}

function signalProcessGroup(pid: number, signal: NodeJS.Signals): boolean {
  try {
    process.kill(-pid, signal);
    return true;
  } catch {
    try {
      process.kill(pid, signal);
      return true;
    } catch {
      return false;
    }
  }
}

// Defense-in-depth against silently reusing an unhealthy backend. Returns "ok"
// when /api/stream serves SSE, "stale" when it's missing/wrong, and "unknown"
// when the probe itself failed.
async function probeStreamContract(): Promise<"ok" | "stale" | "unknown"> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), STREAM_PROBE_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_TARGET}/api/stream`, { signal: controller.signal });
    const ctype = res.headers.get("content-type") ?? "";
    controller.abort();
    if (res.status === 404) return "stale";
    return ctype.includes("text/event-stream") ? "ok" : "stale";
  } catch {
    return "unknown";
  } finally {
    clearTimeout(timeout);
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
            `[backend] API at ${API_TARGET} is missing /api/stream — ` +
              "this is likely a stale backend started before the live-stream route existed. " +
              "Trying to restart the auto-started dev backend.",
          );
        } else if (contract === "unknown") {
          log.warn(
            `[backend] API at ${API_TARGET} did not answer the startup probe quickly. ` +
              "Trying to restart the auto-started dev backend.",
          );
        } else {
          log.info(`[backend] reusing API already running at ${API_TARGET}`);
          return;
        }
        if (!(await stopUnhealthyDevBackend(log))) {
          throw new Error(unhealthyBackendMessage());
        }
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

      let shutdownStarted = false;
      const shutdown = () => {
        if (shutdownStarted || !child?.pid || child.exitCode !== null) {
          return;
        }
        shutdownStarted = true;
        const pid = child.pid;
        server.httpServer?.closeAllConnections();
        setTimeout(() => {
          signalProcessGroup(pid, "SIGTERM");
        }, FRONTEND_CONNECTION_DRAIN_MS);
        setTimeout(() => {
          if (child?.exitCode === null) {
            log.warn(
              `[backend] still shutting down after ${BACKEND_SHUTDOWN_GRACE_MS}ms; ` +
                "forcing dev backend to exit",
            );
            signalProcessGroup(pid, "SIGKILL");
          }
        }, BACKEND_SHUTDOWN_GRACE_MS);
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
