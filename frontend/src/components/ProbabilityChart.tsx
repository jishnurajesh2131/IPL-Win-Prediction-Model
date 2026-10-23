import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { ProbabilityPoint } from '../types/prediction';
import { getTeamColor } from '../utils/teamVisualConfig';
import { formatCricketBall } from '../utils/deliveryFormat';

interface ProbabilityChartProps {
  points: ProbabilityPoint[];
  team1Name: string;
  team2Name: string;
  currentEventIndex?: number;
  onPointClick?: (eventIndex: number) => void;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  points,
  team1Name,
  team2Name,
  currentEventIndex,
  onPointClick,
}) => {
  const chartData = useMemo(() => {
    return points.map((p) => {
      const overStr = p.event_index === 0 ? 'T0' : `Inn ${p.innings}: ${formatCricketBall(p.ball, p.overs_completed)}`;

      return {
        event_index: p.event_index,
        label: overStr,
        innings: p.innings,
        overs_completed: p.overs_completed,
        score: p.score,
        wickets: p.wickets,
        prob_team_1: Math.round(p.team_1_probability * 1000) / 10,
        prob_team_2: Math.round(p.team_2_probability * 1000) / 10,
        prediction_type: p.prediction_type,
      };
    });
  }, [points]);

  // Find innings break event index if present
  const inningsBreakIndex = useMemo(() => {
    for (let i = 1; i < points.length; i++) {
      if (points[i].innings === 2 && points[i - 1].innings === 1) {
        return points[i].event_index;
      }
    }
    return null;
  }, [points]);

  // Current selected label for vertical reference line
  const activeLabel = useMemo(() => {
    if (currentEventIndex === undefined || currentEventIndex === null) return null;
    const item = chartData.find((d) => d.event_index === currentEventIndex);
    return item ? item.label : null;
  }, [chartData, currentEventIndex]);

  if (points.length === 0) {
    return (
      <div className="chart-empty-card">
        <p>No probability trajectory data recorded yet.</p>
      </div>
    );
  }

  const team1Col = getTeamColor(team1Name);
  const team2Col = getTeamColor(team2Name);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title">
          <h3>WIN PROBABILITY TIMELINE</h3>
          <span className="text-muted">Click any point to jump replay position</span>
        </div>
        <div className="chart-legend-custom">
          <div className="legend-item">
            <span className="legend-color-box" style={{ backgroundColor: team1Col }} />
            <span>{team1Name}</span>
          </div>
          <div className="legend-item">
            <span className="legend-color-box" style={{ backgroundColor: team2Col }} />
            <span>{team2Name}</span>
          </div>
        </div>
      </div>

      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 20, left: -10, bottom: 5 }}
            onClick={(e) => {
              if (onPointClick && e && e.activePayload && e.activePayload.length > 0) {
                const targetIdx = e.activePayload[0].payload.event_index;
                if (targetIdx !== undefined) {
                  onPointClick(targetIdx);
                }
              }
            }}
          >
            <defs>
              <linearGradient id="colorTeam1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={team1Col} stopOpacity={0.35} />
                <stop offset="95%" stopColor={team1Col} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorTeam2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={team2Col} stopOpacity={0.35} />
                <stop offset="95%" stopColor={team2Col} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
            <XAxis
              dataKey="label"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              interval="preserveStartEnd"
              minTickGap={35}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
              ticks={[0, 25, 50, 75, 100]}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="custom-tooltip">
                    <div className="tooltip-header">
                      <strong>{d.label}</strong>
                      {d.event_index > 0 && (
                        <span className="tooltip-score">
                          {d.score}/{d.wickets}
                        </span>
                      )}
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: team1Col, fontWeight: 'bold' }}>{team1Name}:</span>
                      <strong style={{ color: '#ffffff' }}>{d.prob_team_1.toFixed(1)}%</strong>
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: team2Col, fontWeight: 'bold' }}>{team2Name}:</span>
                      <strong style={{ color: '#ffffff' }}>{d.prob_team_2.toFixed(1)}%</strong>
                    </div>
                    <div className="tooltip-action-hint">Click to jump replay here</div>
                  </div>
                );
              }}
            />
            {/* 50% parity line */}
            <ReferenceLine
              y={50}
              stroke="#475569"
              strokeDasharray="4 4"
              label={{ value: '50% Parity', fill: '#64748b', fontSize: 10, position: 'insideRight' }}
            />

            {/* Innings Break divider if present */}
            {inningsBreakIndex !== null && (
              <ReferenceLine
                x="Inn 2: 0.1"
                stroke="#f59e0b"
                strokeDasharray="3 3"
                label={{ value: 'Innings Break', fill: '#f59e0b', fontSize: 10, position: 'top' }}
              />
            )}

            {/* Selected ball reference marker */}
            {activeLabel && (
              <ReferenceLine
                x={activeLabel}
                stroke="#38bdf8"
                strokeWidth={2}
                label={{ value: 'Active', fill: '#38bdf8', fontSize: 10, position: 'insideTopLeft' }}
              />
            )}

            <Area
              type="monotone"
              dataKey="prob_team_1"
              name={team1Name}
              stroke={team1Col}
              fillOpacity={1}
              fill="url(#colorTeam1)"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 6, fill: team1Col, stroke: '#ffffff', strokeWidth: 2 }}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="prob_team_2"
              name={team2Name}
              stroke={team2Col}
              fillOpacity={1}
              fill="url(#colorTeam2)"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 6, fill: team2Col, stroke: '#ffffff', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
