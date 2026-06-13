export function errMsg(err: unknown, fallback?: string): string {
  return err instanceof Error ? err.message : (fallback ?? String(err));
}
