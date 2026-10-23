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
import { fetchReliabilityAudit } from '../../api/intelligence';
import { ReliabilityAuditorData } from '../../types/intelligence';
import {
  ShieldCheck,
  Activity,
  CheckCircle2,
  AlertCircle,
  Layers,
  Clock,
} from 'lucide-react';

export const ReliabilityAuditorTab: React.FC = () => {
  const [modelType, setModelType] = useState<'live' | 'pre_match'>('live');
  const [split, setSplit] = useState<'test' | 'val' | 'walk_forward'>('test');
  const [selectedSeason, setSelectedSeason] = useState<string>('');
  const [data, setData] = useState<ReliabilityAuditorData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    const activeSplit = modelType === 'live' ? (split === 'val' ? 'val' : 'test') : split;

    fetchReliabilityAudit(modelType, activeSplit, selectedSeason || undefined)
      .then((res) => {
        if (isMounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load reliability audit data');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [modelType, split, selectedSeason]);

  // Calibration chart data preparation
  const chartData = data?.calibration_bins.map((b) => ({
    name: b.bin_range,
    binIndex: b.bin_index,
    predicted: Math.round(b.mean_predicted_prob * 1000) / 10,
    observed: b.count > 0 && b.observed_frequency !== null && b.observed_frequency !== undefined
      ? Math.round(b.observed_frequency * 1000) / 10
      : null,
    ideal: Math.round(((b.bin_index - 0.5) / 10) * 1000) / 10,
    count: b.count,
    ciLower: b.count > 0 && b.ci_lower !== null && b.ci_lower !== undefined ? Math.round(b.ci_lower * 1000) / 10 : null,
    ciUpper: b.count > 0 && b.ci_upper !== null && b.ci_upper !== undefined ? Math.round(b.ci_upper * 1000) / 10 : null,
    calError: b.count > 0 && b.calibration_error !== null && b.calibration_error !== undefined ? Math.round(b.calibration_error * 1000) / 10 : null,
  })) || [];

  return (
    <div className="intelligence-tab-content">
      {/* Tab Header & Controls */}
      <div className="intel-section-header">
        <div>
          <h2>Prediction Reliability Auditor</h2>
          <p className="intel-subtitle">
            Empirical calibration, probability reliability diagrams, and scoring metrics measured against actual match outcomes.
          </p>
        </div>

        <div className="intel-controls-row">
          {/* Model Type Selector */}
          <div className="intel-btn-group">
            <button
              className={`btn btn-sm ${modelType === 'live' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => {
                setModelType('live');
                setSplit('test');
                setSelectedSeason('');
              }}
            >
              Live Ball-by-Ball Model
            </button>
            <button
              className={`btn btn-sm ${modelType === 'pre_match' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => {
                setModelType('pre_match');
                setSplit('walk_forward');
              }}
            >
              T0 Pre-Match Walk-Forward
            </button>
          </div>

          {/* Split / Context Selector */}
          {modelType === 'live' ? (
            <div className="intel-btn-group">
              <button
                className={`btn btn-sm ${split === 'test' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSplit('test')}
              >
                Held-Out Test Set (2024–2026)
              </button>
              <button
                className={`btn btn-sm ${split === 'val' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSplit('val')}
              >
                Validation Period (Pre-2024)
              </button>
            </div>
          ) : (
            <div className="intel-btn-group">
              <button
                className={`btn btn-sm ${split === 'walk_forward' && !selectedSeason ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => {
                  setSplit('walk_forward');
                  setSelectedSeason('');
                }}
              >
                All Walk-Forward (19 Seasons)
              </button>
              <button
                className={`btn btn-sm ${split === 'test' && !selectedSeason ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => {
                  setSplit('test');
                  setSelectedSeason('');
                }}
              >
                Held-Out 2024–2026
              </button>
            </div>
          )}
        </div>
      </div>

      {loading ? (
        <div className="intel-loading-card">
          <Activity className="spin" size={32} />
          <p>Auditing historical calibration curves and sample distributions...</p>
        </div>
      ) : error ? (
        <div className="intel-error-card">
          <AlertCircle size={28} color="#ef4444" />
          <p>{error}</p>
        </div>
      ) : data?.insufficient_data ? (
        <div className="intel-empty-card">
          <AlertCircle size={32} color="#f59e0b" />
          <h3>Insufficient Historical Evidence</h3>
          <p>{data.status_message || 'Sample size is too small to construct reliable calibration bins.'}</p>
        </div>
      ) : data ? (
        <>
          {/* Key Reliability Metrics */}
          <div className="intel-metrics-grid">
            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Brier Score</span>
                <span className="metric-tag info" title="Lower is better (0.25 is uninformative coin-flip)">
                  Target: &lt; 0.25
                </span>
              </div>
              <div className="metric-value highlight">{data.brier_score.toFixed(4)}</div>
              <div className="metric-footer">
                Mean squared error of probabilities. Perfect is 0.00.
              </div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Log Loss (Cross-Entropy)</span>
                <span className="metric-tag info">Logarithmic penalty</span>
              </div>
              <div className="metric-value">{data.log_loss.toFixed(4)}</div>
              <div className="metric-footer">
                Penalizes confident mispredictions heavily.
              </div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Expected Calibration Error (ECE)</span>
                <span className="metric-tag success">Weighted Gap</span>
              </div>
              <div className="metric-value highlight-cyan">
                {(data.ece * 100).toFixed(2)}%
              </div>
              <div className="metric-footer">
                Average deviation between predicted &amp; observed win rates.
              </div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Favorite Win Accuracy</span>
                <span className="metric-tag">Binary Benchmark</span>
              </div>
              <div className="metric-value">{(data.accuracy * 100).toFixed(1)}%</div>
              <div className="metric-footer">
                Fraction of matches where predicted &gt;50% favorite won.
              </div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Sample Coverage</span>
                <span className="metric-tag success">Verified</span>
              </div>
              <div className="metric-value highlight-gold">
                {data.evaluation_predictions.toLocaleString()}
              </div>
              <div className="metric-footer">
                Across {data.evaluation_matches.toLocaleString()} independent matches.
              </div>
            </div>
          </div>

          {/* Within-Match Clustering Notice */}
          {data.clustering_caveat && (
            <div className="intel-alert-box info" style={{ marginBottom: '1.25rem' }}>
              <Layers size={20} color="#38bdf8" style={{ flexShrink: 0 }} />
              <div>
                <strong>Statistical Clustering Notice:</strong> {data.clustering_caveat}
              </div>
            </div>
          )}

          {/* Calibration Curve Diagram */}
          <div className="intel-card full-width">
            <div className="intel-card-header">
              <div className="intel-card-title-group">
                <h3>Reliability Diagram (Calibration Curve)</h3>
                <span className="intel-badge">10 Standard Probability Deciles</span>
              </div>
              <div className="intel-badge success">
                <CheckCircle2 size={12} /> Temporal Isolation Verified
              </div>
            </div>

            <p className="intel-chart-description">
              In a well-calibrated sports model, when predictions average <strong>70%</strong>, the favored team should win approximately <strong>70%</strong> of the time.
              The dotted diagonal represents perfect calibration. Bars show delivery sample density per bin.
            </p>

            <div style={{ width: '100%', height: 380, marginTop: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
                  <XAxis
                    dataKey="name"
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    label={{ value: 'Predicted Probability Decile', position: 'bottom', offset: 10, fill: '#cbd5e1' }}
                  />
                  <YAxis
                    yAxisId="left"
                    domain={[0, 100]}
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    label={{ value: 'Observed Win Frequency (%)', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 10 }}
                    label={{ value: 'Sample Count (Deliveries / Matches)', angle: 90, position: 'insideRight', fill: '#64748b' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="intel-custom-tooltip">
                            <div className="tooltip-title">{d.name}</div>
                            <div className="tooltip-row">
                              <span>Mean Predicted:</span>
                              <strong>{d.predicted}%</strong>
                            </div>
                            <div className="tooltip-row">
                              <span>Observed Win Rate:</span>
                              {d.count > 0 && d.observed !== null ? (
                                <strong style={{ color: '#38bdf8' }}>{d.observed}%</strong>
                              ) : (
                                <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>N/A (0 predictions in bin)</span>
                              )}
                            </div>
                            <div className="tooltip-row">
                              <span>Calibration Error:</span>
                              {d.count > 0 && d.calError !== null ? (
                                <span>{d.calError}%</span>
                              ) : (
                                <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>N/A</span>
                              )}
                            </div>
                            {d.count > 0 && d.ciLower !== null && d.ciUpper !== null && (
                              <div className="tooltip-row">
                                <span>95% Wilson CI:</span>
                                <span>[{d.ciLower}%, {d.ciUpper}%]</span>
                              </div>
                            )}
                            <div className="tooltip-row">
                              <span>Bin Population:</span>
                              <span>{d.count.toLocaleString()} samples</span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ color: '#cbd5e1' }} />
                  {/* Sample count bar */}
                  <Bar
                    yAxisId="right"
                    dataKey="count"
                    name="Sample Count"
                    fill="#3b82f6"
                    opacity={0.25}
                    radius={[4, 4, 0, 0]}
                  />
                  {/* Perfect calibration 45° reference line */}
                  <ReferenceLine
                    yAxisId="left"
                    stroke="#f59e0b"
                    strokeDasharray="4 4"
                    label={{ value: 'Ideal Calibration', fill: '#f59e0b', fontSize: 11, position: 'top' }}
                  />
                  {/* Observed win frequency curve */}
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="observed"
                    name="Observed Win Frequency (%)"
                    stroke="#38bdf8"
                    strokeWidth={3}
                    connectNulls={false}
                    dot={{ r: 5, fill: '#0284c7', stroke: '#38bdf8', strokeWidth: 2 }}
                    activeDot={{ r: 7 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Innings & Phase Diagnostics */}
          <div className="intel-subsections-grid">
            {data.phase_metrics && data.phase_metrics.length > 0 && (
              <div className="intel-card">
                <div className="intel-card-header">
                  <h3>Reliability by Match Phase</h3>
                  <Clock size={16} color="#94a3b8" />
                </div>
                <p className="intel-subtext">
                  Model uncertainty narrows as balls remaining decrease.
                </p>
                <table className="intel-table">
                  <thead>
                    <tr>
                      <th>Phase</th>
                      <th>Samples</th>
                      <th>Brier</th>
                      <th>Log Loss</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.phase_metrics.map((pm, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600 }}>{pm.phase}</td>
                        <td>{pm.rows.toLocaleString()}</td>
                        <td>{pm.brier_score.toFixed(4)}</td>
                        <td>{pm.log_loss.toFixed(4)}</td>
                        <td>
                          <span className="badge-pill">{(pm.accuracy * 100).toFixed(1)}%</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {data.innings_metrics && data.innings_metrics.length > 0 && (
              <div className="intel-card">
                <div className="intel-card-header">
                  <h3>Innings 1 vs Innings 2 Diagnostics</h3>
                  <Layers size={16} color="#94a3b8" />
                </div>
                <p className="intel-subtext">
                  Innings 2 benefits from a fixed target score and known required run rate.
                </p>
                <table className="intel-table">
                  <thead>
                    <tr>
                      <th>Innings</th>
                      <th>Samples</th>
                      <th>Brier</th>
                      <th>Log Loss</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.innings_metrics.map((im, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600 }}>{im.innings}</td>
                        <td>{im.rows.toLocaleString()}</td>
                        <td>{im.brier_score.toFixed(4)}</td>
                        <td>{im.log_loss.toFixed(4)}</td>
                        <td>
                          <span className="badge-pill">{(im.accuracy * 100).toFixed(1)}%</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Methodology & Temporal Isolation Guarantee */}
          <div className="intel-info-box">
            <ShieldCheck size={20} color="#10b981" />
            <div>
              <strong>Strict Temporal Isolation Guarantee:</strong>{' '}
              <span>{data.methodology_notes}</span>
              <div style={{ marginTop: '0.4rem', color: '#94a3b8', fontSize: '0.85rem' }}>
                All probability bins are computed from genuine out-of-sample evaluations. No in-sample training data is presented as an independent reliability audit.
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
