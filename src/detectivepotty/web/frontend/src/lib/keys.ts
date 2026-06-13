interface TypingTargetOptions {
  allowRange?: boolean;
}

export function isTypingTarget(
  target: EventTarget | null,
  options: TypingTargetOptions = {},
): boolean {
  const el = target as HTMLElement | null;
  if (!el) {
    return false;
  }
  if (el.isContentEditable) {
    return true;
  }
  if (el.tagName === "INPUT") {
    return !(options.allowRange && (el as HTMLInputElement).type === "range");
  }
  return el.tagName === "TEXTAREA" || el.tagName === "SELECT";
}
