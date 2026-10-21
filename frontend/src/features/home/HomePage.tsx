import { Link } from "react-router-dom";
import { useRisk, useVolume } from "../../api/client";
import { useFilters } from "../../components/FilterContext";
import { Card, ErrorBox, Loading, Pill } from "../../components/ui";
import { fmtDate, num } from "../../lib/format";
import { LEVEL_PILL } from "../../lib/risk";
import FilterBar from "../forecast/FilterBar";
import { asnStatus, runsOut } from "../forecast/ShortfallTable";
import VolumeChart from "../forecast/VolumeChart";
import { peakInfo } from "../forecast/volumeData";

export default function HomePage() {
  const { filters } = useFilters();
  const volume = useVolume(filters);
  const risk = useRisk(filters);
  const peak = volume.data ? peakInfo(volume.data) : null;
  const top = risk.data?.rows.filter((r) => r.risk_state === "High risk" || r.risk_state === "Watch").slice(0, 4) ?? [];

  return (
    <>
      <div className="head">
        <div><h1>Home</h1><p className="muted">Supply chain overview — forecast and shortfall risk.</p></div>
        <FilterBar />
      </div>
      <div className="grid g2">
        <Card title="Forecast and shortfall risk" sub="Next 13 weeks" action={<Link className="lnk small" to="/forecast">Forecasting</Link>}>
          {volume.isLoading ? <Loading rows={3} /> : volume.error ? <ErrorBox error={volume.error} /> : volume.data && (
            <>
              <VolumeChart data={volume.data} height={170} />
              {peak && <p className="small muted" style={{ margin: "8px 0 0" }}>
                Peak of about <b style={{ color: "var(--ink)" }}>{num(peak.orders)}</b> orders a week around {fmtDate(peak.week)} — {peak.pctVsLatest}% above the latest week.</p>}
            </>
          )}
          <div className="row" style={{ margin: "16px 0 4px" }}>
            <h3 className="segh">Shortfall risk</h3>
            <span className="muted small">{risk.data ? `${risk.data.shortfall_risk_skus} SKUs under 30 days of stock` : ""}</span>
          </div>
          {risk.isLoading ? <Loading rows={2} /> : risk.error ? <ErrorBox error={risk.error} /> : (
            <div className="list">
              {top.length === 0 && <div className="empty small">No SKUs under 30 days of stock.</div>}
              {top.map((r) => (
                <div className="lrow compact" key={r.sku}>
                  <span><span className="t">{r.description}</span><span className="s block">{r.warehouse} · {runsOut(r)}</span></span>
                  <span className="s" style={{ color: `var(--${asnStatus(r).tone}-text)` }}>{asnStatus(r).text}</span>
                  <Pill tone={LEVEL_PILL[r.risk_state]}>{Math.round(r.days_of_cover)} days</Pill>
                </div>
              ))}
            </div>
          )}
        </Card>
        <Card title="SKUs at shortfall risk" sub="Under 30 days of stock">
          <p className="bigstat" style={{ color: "var(--crit-text)" }}>{risk.data?.shortfall_risk_skus ?? "—"}</p>
          <p className="small muted">High risk {risk.data?.counts["High risk"] ?? "—"} · Watch {risk.data?.counts.Watch ?? "—"}</p>
          <Link className="lnk" to="/forecast">View all in Forecasting</Link>
        </Card>
      </div>
    </>
  );
}
