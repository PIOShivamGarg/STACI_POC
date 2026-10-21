import { useModels, useWarehouses } from "../../api/client";
import { useFilters } from "../../components/FilterContext";

const MODEL_NAME: Record<string, string> = { lgbm: "LightGBM", xgboost: "XGBoost", chronos: "Chronos-2", stats: "Statistical" };

export default function FilterBar() {
  const { filters, setFilters } = useFilters();
  const models = useModels();
  const wh = useWarehouses(filters);
  return (
    <div className="filters">
      <label>
        <span>Warehouse</span>
        <select value={filters.warehouse} onChange={(e) => setFilters({ ...filters, warehouse: e.target.value })}>
          <option value="">All warehouses</option>
          {wh.data?.warehouses.map((w) => <option key={w} value={w}>{w}</option>)}
        </select>
      </label>
      <label>
        <span>Forecast model</span>
        <select value={filters.model} onChange={(e) => setFilters({ ...filters, model: e.target.value })}>
          {(models.data ?? [filters.model]).map((m) => <option key={m} value={m}>{MODEL_NAME[m] ?? m}</option>)}
        </select>
      </label>
    </div>
  );
}
