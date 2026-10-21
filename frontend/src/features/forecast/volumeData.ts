import type { VolumeResponse } from "../../api/client";
import { fmtDate } from "../../lib/format";

export interface ChartPoint {
  week: string;
  label: string;
  actual?: number;
  forecast?: number;
  range?: [number, number];
}

/** Joins actual and forecast weeks into one series. The forecast line starts at the last actual point so the chart has no gap. */
export function buildSeries(v: VolumeResponse): ChartPoint[] {
  const pts: ChartPoint[] = v.actual.map((a) => ({ week: a.week_start, label: fmtDate(a.week_start), actual: a.actual }));
  if (pts.length) pts[pts.length - 1].forecast = pts[pts.length - 1].actual;
  for (const f of v.forecast) {
    pts.push({ week: f.week_start, label: fmtDate(f.week_start), forecast: f.p50, range: [f.p50, f.p90] });
  }
  return pts;
}

export function peakInfo(v: VolumeResponse) {
  if (!v.forecast.length || !v.actual.length) return null;
  const peak = v.forecast.reduce((m, f) => (f.p50 > m.p50 ? f : m));
  const latest = v.actual[v.actual.length - 1].actual;
  return { week: peak.week_start, orders: peak.p50, latest, pctVsLatest: latest ? Math.round((peak.p50 / latest - 1) * 100) : 0 };
}
