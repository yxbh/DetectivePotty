export function observeResize(element: Element | null, callback: () => void): () => void {
  if (!element || typeof ResizeObserver === "undefined") {
    return () => undefined;
  }
  const observer = new ResizeObserver(() => callback());
  observer.observe(element);
  return () => observer.disconnect();
}
