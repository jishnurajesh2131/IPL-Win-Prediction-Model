import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { fetchEraShiftMonitor } from '../../api/intelligence';
import { EraShiftMonitorData } from '../../types/intelligence';
import {
  Activity,
  AlertTriangle,
  Zap,
  Info,
} from 'lucide-react';

export const EraShiftMonitorTab: React.FC = () => {
  const [data, setData] = useState<EraShiftMonitorData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchEraShiftMonitor()
      .then((res) => {
        if (isMounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load era-shift monitor data');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = data?.seasons.map((s) => ({
    season: s.season,
    matches: s.matches,
    avgScore: s.avg_first_innings_score,
    scores200Pct: s.scores_200_plus_pct,
    brier: s.brier_score,
    accuracyPct: Math.round(s.favorite_accuracy * 1000) / 10,
    isImpactEra: ['2023', '2024', '2025', '2026'].includes(s.season),
  })) || [];

  return (
    <div className="intelligence-tab-content">
      {/* Header */}
      <div className="intel-section-header">
        <div>
          <h2>IPL Era-Shift Monitor</h2>
          <p className="intel-subtitle">
            Longitudinal monitoring of tournament-wide scoring distribution shifts versus model calibration resilience.
          </p>
        </div>

        <div className="intel-badge warning">
          <Zap size={14} /> Impact Player Rule Era: 2023–2026
        </div>
      </div>

      {loading ? (
        <div className="intel-loading-card">
          <Activity className="spin" size={32} />
          <p>Auditing 19 seasons of chronological scoring distributions and model stability...</p>
        </div>
      ) : error ? (
        <div className="intel-error-card">
          <AlertTriangle size={28} color="#ef4444" />
          <p>{error}</p>
        </div>
      ) : data ? (
        <>
          {/* Era Definitions Overview Cards */}
          <div className="intel-eras-grid">
            {data.eras.map((era, idx) => {
              const isLatest = idx === data.eras.length - 1;
              return (
                <div key={era.era_name} className={`intel-era-card ${isLatest ? 'impact-era' : ''}`}>
                  <div className="era-badge-row">
                    <span className="era-tag">{era.season_range}</span>
                    {isLatest && <span className="era-highlight-tag">Tactical Disruption</span>}
                  </div>
                  <h3 className="era-name">{era.era_name}</h3>
                  <p className="era-desc">{era.description}</p>

                  <div className="era-stats-row">
                    <div className="era-stat">
                      <span className="stat-label">Avg 1st Inn Score</span>
                      <span className="stat-val highlight">{era.avg_first_innings_score}</span>
                    </div>
                    <div className="era-stat">
                      <span className="stat-label">Model Brier</span>
                      <span className="stat-val">{era.brier_score.toFixed(4)}</span>
                    </div>
                    <div className="era-stat">
                      <span className="stat-label">Matches</span>
                      <span className="stat-val">{era.matches}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Statistical Shift Hypothesis Tests */}
          <div className="intel-card full-width">
            <div className="intel-card-header">
              <div className="intel-card-title-group">
                <h3>Structural Shift Hypothesis Testing</h3>
                <span className="intel-badge">Welch's Two-Sample t-Tests</span>
              </div>
              <span className="intel-badge info">
                Baseline (2018–2022) vs Modern Impact Era (2023–2026)
              </span>
            </div>

            <div className="shift-tests-grid">
              {data.shift_tests.map((test, idx) => {
                const isSig = test.is_statistically_significant;
                const isDistShift = test.analysis_type === 'scoring_distribution_shift';

                return (
                  <div
                    key={idx}
                    className={`shift-test-card ${isDistShift && isSig ? 'detected' : 'stable'}`}
                  >
                    <div className="test-header">
                      <div className="test-title">{test.comparison_name}</div>
                      <span className={`badge-pill ${isSig ? 'warning' : 'success'}`}>
                        {test.status === 'statistically_significant'
                          ? 'Shift Detected'
                          : 'Resilient / Stable'}
                      </span>
                    </div>

                    <div className="test-metric-diff">
                      <div className="metric-compare">
                        <span className="lbl">{test.baseline_era}</span>
                        <span className="val">{test.baseline_mean}</span>
                      </div>
                      <div className="metric-arrow">➔</div>
                      <div className="metric-compare">
                        <span className="lbl">{test.recent_era}</span>
                        <span className="val highlight">{test.recent_mean}</span>
                      </div>
                    </div>

                    <div className="test-stats-row">
                      <span>
                        Delta: <strong>{test.absolute_change > 0 ? `+${test.absolute_change}` : test.absolute_change}</strong> ({test.percentage_change > 0 ? `+${test.percentage_change}%` : `${test.percentage_change}%`})
                      </span>
                      <span>
                        p-value: <strong>{test.p_value != null ? test.p_value.toFixed(4) : 'N/A'}</strong>
                      </span>
                    </div>

                    <p className="test-notes">{test.notes}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Longitudinal Chart: 19 Seasons Scoring Trends vs Model Accuracy */}
          <div className="intel-card full-width">
            <div className="intel-card-header">
              <div className="intel-card-title-group">
                <h3>Seasonal Scoring Trends &amp; Model Accuracy (2008–2026)</h3>
                <span className="intel-badge">19 Consecutive Seasons</span>
              </div>
            </div>

            <p className="intel-chart-description">
              Notice the divergence in 2023–2026: while 1st innings scores surged from ~165 to 193 runs and 200+ scores jumped from 15% to 51%,
              model accuracy and calibration remained stable (53–54% favorite accuracy; live model test accuracy 68.3%).
            </p>

            <div style={{ width: '100%', height: 380, marginTop: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
                  <XAxis
                    dataKey="season"
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    interval={0}
                    angle={-30}
                    textAnchor="end"
                  />
                  <YAxis
                    yAxisId="left"
                    domain={[130, 210]}
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    label={{ value: 'Avg 1st Innings Runs', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, 80]}
                    stroke="#38bdf8"
                    tick={{ fill: '#38bdf8', fontSize: 11 }}
                    label={{ value: '200+ Scores % / Win Accuracy %', angle: 90, position: 'insideRight', fill: '#38bdf8' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="intel-custom-tooltip">
                            <div className="tooltip-title">IPL Season {d.season}</div>
                            <div className="tooltip-row">
                              <span>Avg 1st Innings Score:</span>
                              <strong>{d.avgScore} runs</strong>
                            </div>
                            <div className="tooltip-row">
                              <span>200+ Totals Frequency:</span>
                              <strong style={{ color: '#f59e0b' }}>{d.scores200Pct}%</strong>
                            </div>
                            <div className="tooltip-row">
                              <span>Model Favorite Accuracy:</span>
                              <strong style={{ color: '#10b981' }}>{d.accuracyPct}%</strong>
                            </div>
                            <div className="tooltip-row">
                              <span>Model Brier Score:</span>
                              <span>{d.brier.toFixed(4)}</span>
                            </div>
                            <div className="tooltip-row">
                              <span>Matches in Season:</span>
                              <span>{d.matches}</span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ color: '#cbd5e1' }} />
                  {/* Vertical Reference Line for Impact Player Rule */}
                  <ReferenceLine
                    yAxisId="left"
                    x="2023"
                    stroke="#ef4444"
                    strokeDasharray="4 4"
                    label={{ value: 'Impact Player Rule (2023)', fill: '#ef4444', fontSize: 11, position: 'top' }}
                  />
                  {/* Bars for 200+ score percentage */}
                  <Bar
                    yAxisId="right"
                    dataKey="scores200Pct"
                    name="200+ Scores (%)"
                    fill="#f59e0b"
                    opacity={0.3}
                    radius={[4, 4, 0, 0]}
                  />
                  {/* Line for Average 1st Innings Score */}
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="avgScore"
                    name="Avg 1st Innings Runs"
                    stroke="#e11d48"
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#e11d48' }}
                  />
                  {/* Line for Model Accuracy */}
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="accuracyPct"
                    name="Model Accuracy (%)"
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    dot={{ r: 4, fill: '#10b981' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Caveats & Methodological Transparency */}
          <div className="intel-card full-width">
            <div className="intel-card-header">
              <h3>Methodology &amp; Known Limitations</h3>
              <Info size={16} color="#94a3b8" />
            </div>
            <p className="intel-subtext">{data.methodology}</p>
            <ul className="intel-limitations-list">
              {data.limitations.map((lim, idx) => (
                <li key={idx}>
                  <AlertTriangle size={14} color="#f59e0b" />
                  <span>{lim}</span>
                </li>
              ))}
            </ul>
          </div>
        </>
      ) : null}
    </div>
  );
};
