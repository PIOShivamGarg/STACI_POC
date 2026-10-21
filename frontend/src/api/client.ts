import { useQuery } from "@tanstack/react-query";
import type { FvaStatus, RiskLevel } from "../lib/risk";

export interface Filters {
  model: string;
  warehouse: string; // "" = all
}

export interface VolumeResponse {
  actual: { week_start: string; actual: number }[];
  forecast: { week_start: string; p50: number; p90: number }[];
}

export interface RiskRow {
  sku: string;
  client_id: string;
  client_name: string;
  category: string;
  description: string;
  warehouse: string;
  available_qty: number;
  avg_daily_usage_next4w: number;
  days_of_cover: number;
  lead_time_days: number;
  covered_by_asn: string | null;
  next_asn_date: string | null;
  asn_arrives_in_time: boolean;
  projected_stockout_p50: string | null;
  projected_stockout_p90: string | null;
  risk_state: RiskLevel;
  reason: string;
  suggested_reorder_qty: number;
}

export interface RiskResponse {
  counts: Record<RiskLevel, number>;
  shortfall_risk_skus: number;
  rows: RiskRow[];
}

export interface FvaResponse {
  orders_received: number;
  orders_forecast: number;
  variance_pct: number;
  status: FvaStatus;
  days: { date: string; orders_received: number; orders_forecast: number }[];
}

export interface SummaryResponse {
  as_of: string;
  method_version: string;
}

async function get<T>(path: string, params: Record<string, string>): Promise<T> {
  const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v)).toString();
  const res = await fetch(`/api/${path}${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

const scope = (f: Filters) => ({ model: f.model, warehouse: f.warehouse });

export const useModels = () => useQuery({ queryKey: ["models"], queryFn: () => get<string[]>("models", {}) });
export const useWarehouses = (f: Filters) =>
  useQuery({
    queryKey: ["filters", f.model],
    queryFn: () => get<{ warehouses: string[] }>("filters", { model: f.model }),
  });
export const useSummary = (f: Filters) =>
  useQuery({ queryKey: ["summary", f.model], queryFn: () => get<SummaryResponse>("summary", { model: f.model }) });
export const useVolume = (f: Filters) =>
  useQuery({ queryKey: ["volume", f], queryFn: () => get<VolumeResponse>("volume", scope(f)) });
export const useRisk = (f: Filters) =>
  useQuery({ queryKey: ["risk", f], queryFn: () => get<RiskResponse>("risk", scope(f)) });
export const useFva = (f: Filters) =>
  useQuery({ queryKey: ["fva", f], queryFn: () => get<FvaResponse>("forecast-vs-actual", scope(f)) });
