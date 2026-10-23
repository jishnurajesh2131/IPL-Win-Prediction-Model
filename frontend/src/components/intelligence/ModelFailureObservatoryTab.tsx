import React, { useState, useEffect, useMemo } from 'react';
import { fetchFailureObservatory } from '../../api/intelligence';
import { FailureObservatoryData } from '../../types/intelligence';
import {
  AlertTriangle,
  Activity,
  CheckCircle2,
  HelpCircle,
  Filter,
} from 'lucide-react';

export const ModelFailureObservatoryTab: React.FC = () => {
  const [data, setData] = useState<FailureObservatoryData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDimension, setSelectedDimension] = useState<string>('All');

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchFailureObservatory()
      .then((res) => {
        if (isMounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load model failure observatory data');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const dimensions = useMemo(() => {
    if (!data) return ['All'];
    const dims = Array.from(new Set(data.subgroups.map((s) => s.dimension)));
    return ['All', ...dims];
  }, [data]);

  const filteredSubgroups = useMemo(() => {
    if (!data) return [];
    if (selectedDimension === 'All') return data.subgroups;
    return data.subgroups.filter((s) => s.dimension === selectedDimension);
  }, [data, selectedDimension]);

  return (
    <div className="intelligence-tab-content">
      {/* Header */}
      <div className="intel-section-header">
        <div>
          <h2>Model Failure Observatory</h2>
          <p className="intel-subtitle">
            Systematic audit of recurring historical error patterns across venues, seasons, and match contexts.
          </p>
        </div>

        {/* Dimension Filter Buttons */}
        <div className="intel-controls-row">
          <div className="intel-filter-label">
            <Filter size={14} /> Filter Subgroups:
          </div>
          <div className="intel-btn-group">
            {dimensions.map((dim) => (
              <button
                key={dim}
                className={`btn btn-sm ${selectedDimension === dim ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedDimension(dim)}
              >
                {dim}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="intel-loading-card">
          <Activity className="spin" size={32} />
          <p>Analyzing historical error clusters and significance bounds...</p>
        </div>
      ) : error ? (
        <div className="intel-error-card">
          <AlertTriangle size={28} color="#ef4444" />
          <p>{error}</p>
        </div>
      ) : data ? (
        <>
          {/* Baseline Summary Row */}
          <div className="intel-metrics-grid four-cols">
            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Audited Matches</span>
                <span className="metric-tag success">100% Complete</span>
              </div>
              <div className="metric-value highlight-gold">
                {data.total_matches_audited.toLocaleString()}
              </div>
              <div className="metric-footer">Across all 19 historical IPL seasons.</div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Audited Historical Errors</span>
                <span className="metric-tag info">Favorite Upset</span>
              </div>
              <div className="metric-value highlight-rose">
                {data.total_errors_audited.toLocaleString()}
              </div>
              <div className="metric-footer">Matches where predicted &gt;50% favorite lost.</div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Baseline Error Rate</span>
                <span className="metric-tag">Global Benchmark</span>
              </div>
              <div className="metric-value">{(data.overall_error_rate * 100).toFixed(1)}%</div>
              <div className="metric-footer">Expected baseline prediction error rate.</div>
            </div>

            <div className="intel-metric-card">
              <div className="metric-header">
                <span className="metric-title">Baseline Accuracy</span>
                <span className="metric-tag success">Favorite Win</span>
              </div>
              <div className="metric-value highlight">{(data.baseline_accuracy * 100).toFixed(1)}%</div>
              <div className="metric-footer">Global out-of-sample favorite accuracy.</div>
            </div>
          </div>

          {/* Mandatory Non-Causal Guidance Alert */}
          <div className="intel-alert-box warning">
            <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0 }} />
            <div>
              <strong>Scientific &amp; Methodological Notice:</strong> {data.non_causal_statement}
            </div>
          </div>

          {/* Subgroup Error Patterns List */}
          <div className="intel-card full-width">
            <div className="intel-card-header">
              <div className="intel-card-title-group">
                <h3>Subgroup Error Patterns ({filteredSubgroups.length} Analyzed)</h3>
                <span className="intel-badge">Sorted by Error Rate</span>
              </div>
              <div className="intel-legend-inline">
                <span className="badge-validated">Validated Pattern</span>
                <span className="badge-exploratory">Exploratory</span>
                <span className="badge-insufficient">Insufficient Evidence</span>
              </div>
            </div>

            <div className="intel-subgroup-list">
              {filteredSubgroups.map((sg, idx) => {
                const diffPct = (sg.difference_from_baseline * 100).toFixed(1);
                const isAboveBaseline = sg.difference_from_baseline > 0;

                return (
                  <div key={idx} className="intel-subgroup-row">
                    <div className="subgroup-col-main">
                      <div className="subgroup-dim-tag">{sg.dimension}</div>
                      <div className="subgroup-name">{sg.subgroup}</div>
                      <div className="subgroup-caveat">{sg.caveat}</div>
                    </div>

                    <div className="subgroup-col-stats">
                      <div className="stat-item">
                        <span className="stat-label">Matches</span>
                        <span className="stat-val">{sg.total_matches}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Errors</span>
                        <span className="stat-val">{sg.total_errors}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">High-Conf Errors</span>
                        <span className="stat-val">{sg.high_confidence_errors}</span>
                      </div>
                    </div>

                    <div className="subgroup-col-meter">
                      <div className="meter-label-row">
                        <span className="stat-label">Error Rate</span>
                        <span className="stat-val-bold">
                          {(sg.error_rate * 100).toFixed(1)}%
                          <span
                            className={`delta-tag ${isAboveBaseline ? 'negative' : 'positive'}`}
                            title={`Versus rest-of-matches baseline (${(sg.baseline_error_rate * 100).toFixed(1)}%)`}
                          >
                            {isAboveBaseline ? `+${diffPct}%` : `${diffPct}%`}
                          </span>
                        </span>
                      </div>

                      {/* Visual bar comparing to baseline */}
                      <div className="meter-track">
                        <div
                          className="meter-bar"
                          style={{
                            width: `${Math.min(100, sg.error_rate * 100)}%`,
                            backgroundColor:
                              sg.error_rate > 0.52
                                ? '#f43f5e'
                                : sg.error_rate < 0.40
                                ? '#10b981'
                                : '#38bdf8',
                          }}
                        />
                        {/* Global Baseline Marker */}
                        <div
                          className="meter-baseline-pin"
                          style={{ left: `${data.overall_error_rate * 100}%` }}
                          title={`Global baseline: ${(data.overall_error_rate * 100).toFixed(1)}%`}
                        />
                      </div>

                      <div className="meter-ci-row">
                        <span>
                          95% Wilson CI: [{(sg.ci_lower * 100).toFixed(1)}%, {(sg.ci_upper * 100).toFixed(1)}%]
                        </span>
                        {sg.p_value !== null && sg.p_value !== undefined && (
                          <span>
                            p = {sg.p_value.toFixed(4)}
                            {sg.p_value_adjusted !== null && sg.p_value_adjusted !== undefined && (
                              <span title="Benjamini-Hochberg False Discovery Rate Adjusted p-value (q-value)">
                                {' '}(FDR q = {sg.p_value_adjusted.toFixed(4)})
                              </span>
                            )}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="subgroup-col-badge">
                      {sg.status === 'validated' && (
                        <span className="badge-status validated" title="Statistically supported divergence from rest of matches after Benjamini-Hochberg FDR control (q < 0.05, N ≥ 50)">
                          <CheckCircle2 size={12} /> Validated Finding
                        </span>
                      )}
                      {sg.status === 'exploratory' && (
                        <span className="badge-status exploratory" title="Exploratory subgroup finding (15 ≤ N < 50 or FDR q ≥ 0.05)">
                          <HelpCircle size={12} /> Exploratory
                        </span>
                      )}
                      {sg.status === 'insufficient_evidence' && (
                        <span className="badge-status insufficient" title="Sample size N < 15 matches. High sampling variance; cannot draw reliable conclusions.">
                          <AlertTriangle size={12} /> Insufficient Evidence
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Methodology & Multiple Testing Controls */}
          <div className="intel-info-box" style={{ marginTop: '1.5rem' }}>
            <CheckCircle2 size={20} color="#10b981" style={{ flexShrink: 0 }} />
            <div>
              <strong>Statistical Comparison &amp; Multiple Testing Controls:</strong>{' '}
              <span>{data.methodology}</span>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
