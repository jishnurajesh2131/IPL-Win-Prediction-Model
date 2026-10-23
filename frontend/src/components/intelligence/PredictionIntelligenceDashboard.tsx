import React, { useState, useEffect } from 'react';
import { ReliabilityAuditorTab } from './ReliabilityAuditorTab';
import { ModelFailureObservatoryTab } from './ModelFailureObservatoryTab';
import { EraShiftMonitorTab } from './EraShiftMonitorTab';
import { fetchIntelligenceSummary } from '../../api/intelligence';
import { IntelligenceSummaryData } from '../../types/intelligence';
import {
  ShieldCheck,
  Activity,
  TrendingUp,
  Search,
} from 'lucide-react';

export const PredictionIntelligenceDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'reliability' | 'observatory' | 'era_shift'>('reliability');
  const [summary, setSummary] = useState<IntelligenceSummaryData | null>(null);

  useEffect(() => {
    fetchIntelligenceSummary()
      .then((data) => setSummary(data))
      .catch((err) => console.error('Failed to load intelligence summary:', err));
  }, []);

  return (
    <div className="intelligence-dashboard-container">
      {/* Top Banner: Module Overview & Integrity Guarantee */}
      <div className="intel-hero-banner">
        <div className="hero-content">
          <div className="hero-badge">
            <ShieldCheck size={14} color="#10b981" />
            <span>Independent Prediction Research &amp; Monitoring Layer</span>
          </div>
          <h1>Prediction Intelligence</h1>
          <p className="hero-description">
            Rigorous statistical evaluation of probability calibration, recurring failure modes, and tournament era shifts.
            Operating strictly read-only with zero in-sample leakage.
          </p>
        </div>

        {summary && (
          <div className="hero-stats-row">
            <div className="hero-stat-pill">
              <span className="lbl">Live Test Brier</span>
              <span className="val highlight">{summary.live_model_test_brier.toFixed(4)}</span>
              <span className="sub">213 Held-Out Matches</span>
            </div>
            <div className="hero-stat-pill">
              <span className="lbl">Live Test Accuracy</span>
              <span className="val">{(summary.live_model_test_accuracy * 100).toFixed(1)}%</span>
              <span className="sub">50,867 Deliveries</span>
            </div>
            <div className="hero-stat-pill">
              <span className="lbl">Walk-Forward Pre-Match</span>
              <span className="val highlight-gold">{summary.pre_match_walkforward_matches} Matches</span>
              <span className="sub">19 Seasons Covered</span>
            </div>
            <div className="hero-stat-pill">
              <span className="lbl">Era Shift Status</span>
              <span className="val highlight-cyan">Confirmed Shift</span>
              <span className="sub">Impact Player Era</span>
            </div>
          </div>
        )}
      </div>

      {/* Primary Sub-Navigation Tabs */}
      <div className="intel-tab-nav">
        <button
          className={`intel-tab-btn ${activeTab === 'reliability' ? 'active' : ''}`}
          onClick={() => setActiveTab('reliability')}
        >
          <Activity size={16} />
          <span>Reliability Auditor</span>
        </button>

        <button
          className={`intel-tab-btn ${activeTab === 'observatory' ? 'active' : ''}`}
          onClick={() => setActiveTab('observatory')}
        >
          <Search size={16} />
          <span>Model Failure Observatory</span>
        </button>

        <button
          className={`intel-tab-btn ${activeTab === 'era_shift' ? 'active' : ''}`}
          onClick={() => setActiveTab('era_shift')}
        >
          <TrendingUp size={16} />
          <span>IPL Era-Shift Monitor</span>
        </button>
      </div>

      {/* Tab Panels */}
      <div className="intel-tab-panel">
        {activeTab === 'reliability' && <ReliabilityAuditorTab />}
        {activeTab === 'observatory' && <ModelFailureObservatoryTab />}
        {activeTab === 'era_shift' && <EraShiftMonitorTab />}
      </div>
    </div>
  );
};
