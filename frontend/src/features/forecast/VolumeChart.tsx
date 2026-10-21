import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { VolumeResponse } from "../../api/client";
import { num } from "../../lib/format";
import { buildSeries, peakInfo } from "./volumeData";

export default function VolumeChart({ data, height = 260 }: { data: VolumeResponse; height?: number }) {
  const series = buildSeries(data);
  const peak = peakInfo(data);
  const firstForecast = data.forecast[0] && series.find((p) => p.week === data.forecast[0].week_start)?.label;
  return (
    <div>
      <div className="legend" aria-hidden="true">
        <span><i style={{ background: "var(--brand)" }} />Actual</span>
        <span><i style={{ background: "var(--teal)" }} />Forecast (likely)</span>
        <span><i style={{ background: "rgba(33,180,232,.25)" }} />Range (likely to higher case)</span>
      </div>
      <div role="img" aria-label={peak ? `Weekly orders. Forecast peaks near ${num(peak.orders)} orders.` : "Weekly orders"}>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={series} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted)" }} interval="preserveStartEnd" minTickGap={24} />
            <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} width={44} tickFormatter={num} />
            <Tooltip
              formatter={(value: number | [number, number], name: string) =>
                Array.isArray(value) ? [`${num(value[0])} – ${num(value[1])}`, "Range"] : [num(value), name]}
              labelFormatter={(l) => `Week of ${l}`}
            />
            <Area dataKey="range" name="Range" stroke="none" fill="var(--teal)" fillOpacity={0.22} isAnimationActive={false} />
            <Line dataKey="actual" name="Actual" stroke="var(--brand)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
            <Line dataKey="forecast" name="Forecast" stroke="var(--teal)" strokeWidth={2.5} strokeDasharray="6 4" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {firstForecast && <p className="tiny muted" style={{ margin: "6px 0 0" }}>Forecast starts week of {firstForecast}.</p>}
    </div>
  );
}
