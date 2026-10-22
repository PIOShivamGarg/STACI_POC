import { useFva, useMl, useRisk, useSummary, useVolume } from "../../api/client";
import { useFilters } from "../../components/FilterContext";
import { Card, ErrorBox, Loading, Pill, StatTile } from "../../components/ui";
import { fmtDate, num, signed } from "../../lib/format";
import { FVA_PILL, riskCopy } from "../../lib/risk";
import MlPanel from "./MlPanel";
import FilterBar from "./FilterBar";
import ShortfallTable from "./ShortfallTable";
import VolumeChart from "./VolumeChart";
import { peakInfo } from "./volumeData";

const FVA_TONE = { "On track": "ok", Watch: "warn", "Off track": "crit" } as const;

export default function ForecastPage() {
  const { filters } = useFilters();
  const volume = useVolume(filters);
  const risk = useRisk(filters);
  const fva = useFva(filters);
  const summary = useSummary(filters);
  const ml = useMl(filters);
  const copy = riskCopy(ml.data?.method);
  const peak = volume.data ? peakInfo(volume.data) : null;
  const asOf = summary.data ? ` As of ${fmtDate(summary.data.as_of)}.` : "";

  return (
    <>
      <div className="head">
        <div>
          <h1>Forecasting</h1>
          <p className="muted">Expected order volume for the next 13 weeks, and items that could run short before new stock arrives.</p>
        </div>
        <FilterBar />
      </div>

      <div className="grid g4">
        <StatTile label="SKUs at shortfall risk" value={risk.data?.shortfall_risk_skus ?? "—"}
          sub={copy.shortfall} tone={risk.data?.shortfall_risk_skus ? "crit" : undefined} />
        <StatTile label="High risk" value={risk.data?.counts["High risk"] ?? "—"} sub={copy.high} />
        <StatTile label="Forecast peak" value={peak ? num(peak.orders) : "—"}
          sub={peak ? `Week of ${fmtDate(peak.week)} · ${peak.pctVsLatest >= 0 ? "+" : ""}${peak.pctVsLatest}% vs latest week` : ""} />
        <StatTile label="Forecast vs actual" value={fva.data ? signed(fva.data.variance_pct) : "—"}
          sub={fva.data ? `${num(fva.data.orders_received)} received vs ${num(fva.data.orders_forecast)} forecast · last 14 days` : ""}
          tone={fva.data ? FVA_TONE[fva.data.status] : undefined} />
      </div>

      <Card title="Weekly order volume" sub="16 weeks actual · 13 weeks forecast with likely range"
        action={fva.data && <Pill tone={FVA_PILL[fva.data.status]}>{fva.data.status}</Pill>}>
        {volume.isLoading ? <Loading rows={4} /> : volume.error ? <ErrorBox error={volume.error} /> : volume.data && (
          <>
            <VolumeChart data={volume.data} />
            {peak && <p className="small muted" style={{ margin: "12px 0 0" }}>
              Peak expected the week of {fmtDate(peak.week)}, about {peak.pctVsLatest}% above the latest full week.{asOf}</p>}
          </>
        )}
      </Card>

      <Card title="Items at risk of running short"
        sub={ml.data?.method === "ml" ? "Chance of running out in the next 30 days, from the ML model" : "Days of stock from available quantity and expected demand, checked against inbound ASNs"}>
        {risk.isLoading ? <Loading rows={5} /> : risk.error ? <ErrorBox error={risk.error} /> : risk.data && (
          <ShortfallTable rows={risk.data.rows} counts={risk.data.counts} />
        )}
      </Card>

      {ml.data?.method === "ml" && <MlPanel ml={ml.data} />}
    </>
  );
}
