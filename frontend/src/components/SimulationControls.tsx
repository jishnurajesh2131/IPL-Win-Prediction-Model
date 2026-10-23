import React, { useState, useEffect, useMemo } from 'react';
import {
  FastForward,
  RefreshCcw,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Target,
  Search,
  X,
} from 'lucide-react';
import { ProbabilityPoint } from '../types/prediction';
import { formatCricketBall } from '../utils/deliveryFormat';

interface SimulationControlsProps {
  status: string;
  eventIndex: number;
  totalEvents: number;
  isStepping: boolean;
  isRunning: boolean;
  fullTrajectory?: ProbabilityPoint[];
  onNextBall: () => void;
  onPreviousBall: () => void;
  onGoToBall: (index: number) => void;
  onRunToEnd: () => void;
  onReset: () => void;
  onOpenSelector: () => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  status,
  eventIndex,
  totalEvents,
  isStepping,
  isRunning,
  fullTrajectory = [],
  onNextBall,
  onPreviousBall,
  onGoToBall,
  onRunToEnd,
  onReset,
  onOpenSelector,
}) => {
  const isCompleted = status === 'completed';
  const isBusy = isStepping || isRunning;
  const canGoPrevious = eventIndex > 0 && !isBusy;
  const canGoNext = !isCompleted && !isBusy;

  // Go to Ball Modal State
  const [isGoToOpen, setIsGoToOpen] = useState(false);
  const [targetInnings, setTargetInnings] = useState<number>(1);
  const [targetOver, setTargetOver] = useState<string>('0');
  const [targetBall, setTargetBall] = useState<string>('1');
  const [directInput, setDirectInput] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);

  // Search Ball Dropdown State
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Keyboard navigation: Left Arrow (Prev), Right Arrow / Space (Next)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'SELECT') {
        return;
      }
      if (e.code === 'ArrowLeft' && canGoPrevious) {
        e.preventDefault();
        onPreviousBall();
      } else if ((e.code === 'ArrowRight' || e.code === 'Space') && canGoNext) {
        e.preventDefault();
        onNextBall();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [canGoPrevious, canGoNext, onPreviousBall, onNextBall]);

  // Build searchable ball catalog from fullTrajectory
  const availableDeliveries = useMemo(() => {
    return fullTrajectory.map((p, idx) => {
      const inn = p.innings || 1;
      const overDecimal = formatCricketBall(p.ball, p.overs_completed);
      let label = idx === 0 ? 'T0 · Pre-Match Anchor' : `Inn ${inn} · Ball ${overDecimal} (${p.score}/${p.wickets})`;
      if (idx > 0 && p.is_legal === false) {
        label += ' · EXTRA';
      }
      return {
        eventIndex: p.event_index !== undefined ? p.event_index : idx,
        innings: inn,
        ballStr: overDecimal,
        label,
        score: p.score,
        wickets: p.wickets,
        isLegal: p.is_legal,
      };
    });
  }, [fullTrajectory]);

  // Filter deliveries based on search query
  const filteredDeliveries = useMemo(() => {
    if (!searchQuery.trim()) return availableDeliveries.slice(0, 40);
    const q = searchQuery.toLowerCase();
    return availableDeliveries.filter(
      (d) =>
        d.ballStr.includes(q) ||
        d.label.toLowerCase().includes(q) ||
        `inn ${d.innings}`.includes(q)
    ).slice(0, 40);
  }, [availableDeliveries, searchQuery]);

  // Execute Go To Ball
  const handleExecuteGoTo = () => {
    setValidationError(null);

    // If direct input format like "2.17.3" or "17.3"
    let parsedInn = targetInnings;
    let parsedOver = parseInt(targetOver, 10);
    let parsedBall = parseInt(targetBall, 10);

    if (directInput.trim()) {
      const parts = directInput.trim().split('.');
      if (parts.length === 3) {
        parsedInn = parseInt(parts[0], 10);
        parsedOver = parseInt(parts[1], 10);
        parsedBall = parseInt(parts[2], 10);
      } else if (parts.length === 2) {
        parsedOver = parseInt(parts[0], 10);
        parsedBall = parseInt(parts[1], 10);
      } else {
        setValidationError('Enter format like "17.3" or "2.17.3" (Innings.Over.Ball)');
        return;
      }
    }

    if (isNaN(parsedInn) || (parsedInn !== 1 && parsedInn !== 2)) {
      setValidationError('Innings must be 1 or 2');
      return;
    }
    if (isNaN(parsedOver) || parsedOver < 0 || parsedOver > 19) {
      setValidationError('Over must be between 0 and 19');
      return;
    }
    if (isNaN(parsedBall) || parsedBall < 1 || parsedBall > 6) {
      setValidationError('Ball must be between 1 and 6');
      return;
    }

    const expectedBallFloat = parseFloat(`${parsedOver}.${parsedBall}`);

    // Search matching delivery in trajectory: handle multiple events sharing ball notation (e.g. extras vs legal)
    const matches = availableDeliveries.filter(
      (d) => d.innings === parsedInn && Math.abs(parseFloat(d.ballStr) - expectedBallFloat) < 0.05
    );

    if (matches.length > 0) {
      // If multiple events share the same cricket ball notation, prefer the legal ball completion
      const chosen = matches.find((m) => m.isLegal !== false) || matches[0];
      onGoToBall(chosen.eventIndex);
      setIsGoToOpen(false);
      setDirectInput('');
      setValidationError(null);
    } else {
      setValidationError(`Delivery Innings ${parsedInn}, ${parsedOver}.${parsedBall} not found in this match sequence.`);
    }
  };

  return (
    <div className="controls-panel">
      {/* Top Status & Progress */}
      <div className="controls-status-bar">
        <div className="status-indicator-group">
          <span className="status-label">REPLAY STATE:</span>
          <span className={`status-tag ${status}`}>
            {isCompleted ? (
              <>
                <CheckCircle size={12} /> COMPLETED
              </>
            ) : status === 'active' ? (
              'REPLAY ACTIVE'
            ) : (
              'READY (T0)'
            )}
          </span>
        </div>

        <div className="progress-counter">
          <span>
            Event {eventIndex} of {totalEvents > 0 ? totalEvents - 1 : 0}
          </span>
          {totalEvents > 0 && (
            <div className="progress-mini-bar">
              <div
                className="progress-mini-fill"
                style={{ width: `${Math.min(100, (eventIndex / (totalEvents - 1)) * 100)}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Main Buttons Row */}
      <div className="controls-buttons-row">
        {/* PREVIOUS BALL */}
        <button
          type="button"
          className="btn btn-secondary btn-lg"
          onClick={onPreviousBall}
          disabled={!canGoPrevious}
          title="Move one delivery backward (Shortcut: ← Left Arrow)"
        >
          <ChevronLeft size={18} />
          <span>← PREVIOUS BALL</span>
          <kbd className="key-hint">←</kbd>
        </button>

        {/* NEXT BALL */}
        <button
          type="button"
          className="btn btn-primary btn-lg"
          onClick={onNextBall}
          disabled={!canGoNext}
          title="Advance exactly one delivery event (Shortcut: Spacebar or → Right Arrow)"
        >
          <span>{isStepping ? 'Stepping...' : 'NEXT BALL →'}</span>
          <ChevronRight size={18} />
          <kbd className="key-hint">Space</kbd>
        </button>

        {/* GO TO BALL */}
        <button
          type="button"
          className="btn btn-outline"
          onClick={() => setIsGoToOpen(true)}
          disabled={isBusy}
          title="Jump directly to an exact delivery"
        >
          <Target size={16} />
          <span>GO TO BALL</span>
        </button>

        {/* SEARCH BALL DROPDOWN TOGGLE */}
        <div className="search-ball-wrapper">
          <button
            type="button"
            className={`btn btn-outline search-ball-trigger ${isSearchOpen ? 'active' : ''}`}
            onClick={() => setIsSearchOpen(!isSearchOpen)}
            disabled={isBusy}
            title="Search and select from ball index"
          >
            <Search size={15} />
            <span>SEARCH BALL</span>
          </button>

          {isSearchOpen && (
            <div className="search-ball-dropdown">
              <div className="dropdown-search-bar">
                <input
                  type="text"
                  placeholder="Type over e.g. 17.3..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  autoFocus
                />
                <button
                  type="button"
                  className="btn-clear-search"
                  onClick={() => setIsSearchOpen(false)}
                >
                  <X size={14} />
                </button>
              </div>

              <div className="search-results-list">
                {filteredDeliveries.map((item) => (
                  <div
                    key={item.eventIndex}
                    className={`search-result-item ${item.eventIndex === eventIndex ? 'active-event' : ''}`}
                    onClick={() => {
                      onGoToBall(item.eventIndex);
                      setIsSearchOpen(false);
                    }}
                  >
                    <span className="search-item-ball">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* RUN TO END */}
        <button
          type="button"
          className="btn btn-accent"
          onClick={onRunToEnd}
          disabled={isCompleted || isBusy}
          title="Replay all remaining delivery events to match conclusion"
        >
          <FastForward size={16} />
          <span>{isRunning ? 'Simulating...' : 'RUN TO END'}</span>
        </button>

        {/* RESTART MATCH */}
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onReset}
          disabled={isBusy}
          title="Reset simulation to T0 pre-match anchor"
        >
          <RefreshCcw size={15} />
          <span>Restart</span>
        </button>

        {/* SELECT ANOTHER MATCH */}
        <button
          type="button"
          className="btn btn-outline"
          onClick={onOpenSelector}
          disabled={isBusy}
          title="Browse and select a different historical IPL match"
        >
          Select Match
        </button>
      </div>

      {/* GO TO BALL MODAL */}
      {isGoToOpen && (
        <div className="modal-backdrop" onClick={() => setIsGoToOpen(false)}>
          <div className="go-to-ball-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4>GO TO BALL</h4>
              <button
                type="button"
                className="btn-close-modal"
                onClick={() => setIsGoToOpen(false)}
              >
                <X size={16} />
              </button>
            </div>

            <div className="modal-body">
              <div className="goto-field-group">
                <label>Direct Input (e.g. 17.3 or 2.17.3)</label>
                <input
                  type="text"
                  placeholder="e.g. 17.3 or 2.17.3"
                  value={directInput}
                  onChange={(e) => setDirectInput(e.target.value)}
                  className="input-text"
                />
              </div>

              <div className="modal-divider">
                <span>OR SELECT INNINGS / OVER / BALL</span>
              </div>

              <div className="goto-selectors-row">
                <div className="goto-select-group">
                  <label>Innings</label>
                  <select
                    value={targetInnings}
                    onChange={(e) => setTargetInnings(parseInt(e.target.value, 10))}
                  >
                    <option value={1}>Innings 1</option>
                    <option value={2}>Innings 2</option>
                  </select>
                </div>

                <div className="goto-select-group">
                  <label>Over (0–19)</label>
                  <input
                    type="number"
                    min={0}
                    max={19}
                    value={targetOver}
                    onChange={(e) => setTargetOver(e.target.value)}
                  />
                </div>

                <div className="goto-select-group">
                  <label>Ball (1–6)</label>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={targetBall}
                    onChange={(e) => setTargetBall(e.target.value)}
                  />
                </div>
              </div>

              {validationError && (
                <div className="goto-error-msg text-red">{validationError}</div>
              )}

              <div className="goto-actions-row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setIsGoToOpen(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleExecuteGoTo}
                >
                  GO TO BALL →
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
