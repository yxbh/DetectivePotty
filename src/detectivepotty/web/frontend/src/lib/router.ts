import { writable } from "svelte/store";

/** Current location as a reactive store, so views derive from the URL. */
export interface RouteState {
  path: string;
  query: URLSearchParams;
}

function snapshot(): RouteState {
  return {
    path: location.pathname,
    query: new URLSearchParams(location.search),
  };
}

export const route = writable<RouteState>(snapshot());

function sync(): void {
  route.set(snapshot());
}

// Back/forward buttons change the URL without a navigate() call; mirror them.
if (typeof window !== "undefined") {
  window.addEventListener("popstate", sync);
}

/**
 * Client-side navigation via the History API (clean URLs, no hash).
 *
 * Navigating to the URL already shown is a no-op: it neither pushes a duplicate
 * history entry nor re-sets the store. That property is what keeps the
 * "reflect the selected event in the URL" effect from looping — once the URL
 * matches the selection, further navigate() calls do nothing.
 */
export function navigate(to: string, opts: { replace?: boolean } = {}): void {
  const url = to.startsWith("/") ? to : `/${to}`;
  if (url === location.pathname + location.search) {
    return;
  }
  if (opts.replace) {
    history.replaceState({}, "", url);
  } else {
    history.pushState({}, "", url);
  }
  sync();
}

export type View = "review" | "live" | "tune";

/** Map a pathname to a top-level view (trailing slashes tolerated). */
export function routeToView(path: string): View {
  const normalized = path.replace(/\/+$/, "") || "/";
  if (normalized === "/live") {
    return "live";
  }
  if (normalized === "/tune") {
    return "tune";
  }
  return "review";
}
