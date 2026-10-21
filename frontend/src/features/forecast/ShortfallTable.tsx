import { useState } from "react";
import type { RiskRow } from "../../api/client";
import { Empty, Pill } from "../../components/ui";
import { fmtDate, num } from "../../lib/format";
import { AT_RISK, LEVEL_PILL, type RiskLevel } from "../../lib/risk";

const PAGE = 12;

export function asnStatus(r: RiskRow): { text: string; tone: "ok" | "warn" | "crit" } {
  if (!r.covered_by_asn) return { text: "No inbound ASN", tone: "crit" };
  const when = `${r.covered_by_asn} (ETA ${fmtDate(r.next_asn_date)})`;
  return r.asn_arrives_in_time ? { text: `Covered by ${when}`, tone: "ok" } : { text: `${when} — after stockout`, tone: "warn" };
}

export function runsOut(r: RiskRow): string {
  return r.projected_stockout_p50 ? `Runs out around ${fmtDate(r.projected_stockout_p50)}` : "No stockout in the next 13 weeks";
}

function Row({ r }: { r: RiskRow }) {
  const asn = asnStatus(r);
  const cover = r.days_of_cover < 1 ? "<1 day" : `${Math.round(r.days_of_cover)} days`;
  return (
    <div className="lrow">
      <span className="thumb" aria-hidden="true">{r.category.slice(0, 2).toUpperCase()}</span>
      <span>
        <span className="t">{r.description}</span>
        <span className="s mono block">{r.sku} · {r.client_name} · {r.warehouse} · {num(r.available_qty)} available · {r.avg_daily_usage_next4w}/day</span>
      </span>
      <span><b>{cover}</b><span className="s block">{runsOut(r)}</span></span>
      <span className="s" style={{ color: `var(--${asn.tone}-text)` }}>{asn.text}</span>
      <span className="actions">
        <Pill tone={LEVEL_PILL[r.risk_state]}>{r.risk_state}</Pill>
        <button className="btn sm" disabled title="Item pages are not part of this prototype">Open item</button>
        {!r.covered_by_asn && (
          <button className="btn sm dark" disabled title={`Suggested order: ${num(r.suggested_reorder_qty)} units`}>Create ASN</button>
        )}
      </span>
    </div>
  );
}

export default function ShortfallTable({ rows, counts }: { rows: RiskRow[]; counts: Record<RiskLevel, number> }) {
  const [level, setLevel] = useState<RiskLevel | "all">("all");
  const [all, setAll] = useState(false);
  const atRisk = rows.filter((r) => AT_RISK.includes(r.risk_state));
  const shown = atRisk.filter((r) => level === "all" || r.risk_state === level);
  const list = all ? shown : shown.slice(0, PAGE);
  return (
    <>
      <div className="seg" role="group" aria-label="Filter by risk level">
        {(["all", ...AT_RISK] as const).map((l) => (
          <button key={l} aria-pressed={level === l} onClick={() => { setLevel(l); setAll(false); }}>
            {l === "all" ? "All at risk" : l}
            <span className="count">{l === "all" ? AT_RISK.reduce((n, k) => n + counts[k], 0) : counts[l]}</span>
          </button>
        ))}
      </div>
      {shown.length === 0 ? <Empty>No items at this risk level.</Empty> : (
        <div className="list">{list.map((r) => <Row key={r.sku} r={r} />)}</div>
      )}
      {shown.length > PAGE && (
        <div className="listfoot">
          <button className="lnk" onClick={() => setAll(!all)}>{all ? "Show fewer" : `Show all ${shown.length}`}</button>
        </div>
      )}
    </>
  );
}
