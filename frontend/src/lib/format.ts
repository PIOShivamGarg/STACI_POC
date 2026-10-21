export const num = (n: number) => Math.round(n).toLocaleString("en-US");

/** "2026-10-29" -> "Oct 29". Parsed as a local date so the day never shifts with the time zone. */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export const signed = (n: number, digits = 1) => `${n > 0 ? "+" : ""}${n.toFixed(digits)}%`;
