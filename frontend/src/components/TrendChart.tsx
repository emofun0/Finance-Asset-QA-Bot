import type { HistoryPoint } from "../types/api";

interface TrendChartProps {
  symbol: string;
  points: HistoryPoint[];
  trend: string | null;
  changePct: number | null;
}

const CHART_WIDTH = 640;
const CHART_HEIGHT = 320;
const PADDING_X = 18;
const PADDING_Y = 20;

function formatChange(changePct: number | null): string {
  if (changePct === null) {
    return "--";
  }
  const sign = changePct > 0 ? "+" : "";
  return `${sign}${changePct.toFixed(2)}%`;
}

function toPolyline(points: HistoryPoint[]) {
  const closes = points.map((point) => point.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const innerWidth = CHART_WIDTH - PADDING_X * 2;
  const innerHeight = CHART_HEIGHT - PADDING_Y * 2;

  const coords = points.map((point, index) => {
    const x = PADDING_X + (index / Math.max(points.length - 1, 1)) * innerWidth;
    const normalized = (point.close - min) / range;
    const y = CHART_HEIGHT - PADDING_Y - normalized * innerHeight;
    return { x, y, close: point.close, date: point.timestamp.slice(0, 10) };
  });

  return {
    min,
    max,
    coords,
    line: coords.map((coord) => `${coord.x},${coord.y}`).join(" "),
    area: `${PADDING_X},${CHART_HEIGHT - PADDING_Y} ${coords.map((coord) => `${coord.x},${coord.y}`).join(" ")} ${CHART_WIDTH - PADDING_X},${CHART_HEIGHT - PADDING_Y}`,
  };
}

export function TrendChart({ symbol, points, trend, changePct }: TrendChartProps) {
  const { min, max, coords, line, area } = toPolyline(points);
  const firstPoint = coords[0];
  const lastPoint = coords[coords.length - 1];

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">价格走势</p>
          <h3>{symbol} 日线走势</h3>
        </div>
        <div className="chart-metrics">
          <span>{trend ?? "未分类"}</span>
          <strong>{formatChange(changePct)}</strong>
        </div>
      </div>

      <div className="chart-canvas">
        <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={`${symbol} 价格走势图`}>
          <defs>
            <linearGradient id="trendArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(20, 101, 217, 0.30)" />
              <stop offset="100%" stopColor="rgba(20, 101, 217, 0.04)" />
            </linearGradient>
          </defs>
          {[0, 1, 2, 3].map((tick) => {
            const y = PADDING_Y + (tick / 3) * (CHART_HEIGHT - PADDING_Y * 2);
            return <line key={tick} x1={PADDING_X} y1={y} x2={CHART_WIDTH - PADDING_X} y2={y} className="chart-grid" />;
          })}
          <polygon points={area} fill="url(#trendArea)" />
          <polyline points={line} fill="none" stroke="#1465d9" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
          {firstPoint ? <circle cx={firstPoint.x} cy={firstPoint.y} r="4" fill="#1465d9" /> : null}
          {lastPoint ? <circle cx={lastPoint.x} cy={lastPoint.y} r="4.5" fill="#0c8f72" /> : null}
        </svg>
      </div>

      <div className="chart-axis">
        <span>{firstPoint?.date ?? "--"}</span>
        <strong>
          {min.toFixed(2)} - {max.toFixed(2)}
        </strong>
        <span>{lastPoint?.date ?? "--"}</span>
      </div>
    </section>
  );
}
