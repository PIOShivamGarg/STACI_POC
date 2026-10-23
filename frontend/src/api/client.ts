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
  supplier_delay_days?: number;
  next_asn_likely_date?: string | null;   // next_asn_date + the supplier's recent average delay
  asn_arrives_in_time: boolean;
  projected_stockout_p50: string | null;
  projected_stockout_p90: string | null;
  risk_state: RiskLevel;
  reason: string;
  suggested_reorder_qty: number;
  // set when the ML risk model is on (RISK_METHOD=ml)
  risk_probability?: number | null;
  overstock_probability?: number | null;   // chance stock + inbound is not used up within 120 days
  top_drivers?: string | null;
  rule_risk_state?: RiskLevel | null;
  explanation?: string | null;
  explanation_source?: string | null;
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

interface MlSplit {
  rows: number;
  stockout_rate: number;
  ml: { pr_auc: number; roc_auc: number; at_same_flag_count: { precision: number; recall: number } };
  rule: { pr_auc: number; roc_auc: number; at_30_days: { precision: number; recall: number } };
}

export interface MlResponse {
  method: "ml" | "rules";
  metrics: {
    horizon_days: number;
    validation: MlSplit;
    test_recent: MlSplit;
  } | null;
  insight: { text: string; source: string } | null;
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
export const useMl = (f: Filters) =>
  useQuery({ queryKey: ["ml", f.model], queryFn: () => get<MlResponse>("ml", { model: f.model }) });
export const useVolume = (f: Filters) =>
  useQuery({ queryKey: ["volume", f], queryFn: () => get<VolumeResponse>("volume", scope(f)) });
export const useRisk = (f: Filters) =>
  useQuery({ queryKey: ["risk", f], queryFn: () => get<RiskResponse>("risk", scope(f)) });
export const useFva = (f: Filters) =>
  useQuery({ queryKey: ["fva", f], queryFn: () => get<FvaResponse>("forecast-vs-actual", scope(f)) });
