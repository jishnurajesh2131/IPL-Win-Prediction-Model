import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { MatchHeader } from './components/MatchHeader';
import { WinProbability } from './components/WinProbability';
import { CurrentPlayers } from './components/CurrentPlayers';
import { DeliveryCard } from './components/DeliveryCard';
import { SimulationControls } from './components/SimulationControls';
import { ProbabilityChart } from './components/ProbabilityChart';
import { ProbabilitySwingWpa } from './components/ProbabilitySwingWpa';
import { MatchVolatilityMeter } from './components/MatchVolatilityMeter';
import { PlayerImpactCard } from './components/PlayerImpactCard';
import { MatchDnaCard } from './components/MatchDnaCard';
import { DualInningsScorecard } from './components/DualInningsScorecard';
import { MatchSelector } from './components/MatchSelector';
import { LoadingState } from './components/LoadingState';
import { FinalTestDashboard } from './components/FinalTestDashboard';

import { useMatches } from './hooks/useMatches';
import { useSimulation } from './hooks/useSimulation';
import { AlertCircle, RefreshCw } from 'lucide-react';

const DEFAULT_MATCH_ID = '1426274'; // KKR vs RCB 2024 thriller

export const App: React.FC = () => {
  const [mode, setMode] = useState<'historical' | 'evaluation'>('historical');
  const [isSelectorOpen, setIsSelectorOpen] = useState<boolean>(false);

  const {
    matches,
    total,
    selectedSeason,
    setSelectedSeason,
    selectedTeam,
    setSelectedTeam,
    availableSeasons,
    availableTeams,
    selectedMatch,
    selectMatch,
    loading: matchesLoading,
    error: matchesError,
    refreshMatches,
  } = useMatches();

  const {
    matchId,
    status,
    eventIndex,
    totalEvents,
    winner,
    currentPrediction,
    currentState,
    currentDelivery,
    probabilityHistory,
    fullTrajectory,
    isStarting,
    isStepping,
    isRunning,
    error: simError,
    startSimulation,
    nextBall,
    previousBall,
    goToBall,
    runToEnd,
  } = useSimulation();

  // Load default match on initial mount if none selected
  useEffect(() => {
    if (!selectedMatch && !matchId) {
      selectMatch(DEFAULT_MATCH_ID).then((details) => {
        if (details) {
          startSimulation(DEFAULT_MATCH_ID);
        }
      });
    }
  }, [selectedMatch, matchId, selectMatch, startSimulation]);

  const handleChooseMatch = async (chosenMatchId: string) => {
    setMode('historical');
    const details = await selectMatch(chosenMatchId);
    if (details) {
      startSimulation(chosenMatchId);
    }
  };

  const handleRestart = () => {
    if (matchId) {
      startSimulation(matchId);
    }
  };

  const activeError = matchesError || simError;

  return (
    <div className="app-container">
      {/* Top Navigation & Status Bar */}
      <Header
        onOpenMatchSelector={() => setIsSelectorOpen(true)}
        selectedMatchId={matchId}
        mode={mode}
        onToggleMode={(newMode) => {
          setMode(newMode);
          if (newMode === 'evaluation') {
            setSelectedSeason('2024');
          } else if (newMode === 'historical') {
            setSelectedSeason('2023');
          }
        }}
      />

      {/* Main Content Area */}
      <main className="dashboard-main">
        {activeError && mode === 'historical' && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span className="error-text">{activeError}</span>
            <button className="btn btn-sm btn-outline" onClick={refreshMatches}>
              <RefreshCw size={12} /> Retry
            </button>
          </div>
        )}

        {mode === 'evaluation' ? (
          <FinalTestDashboard onSimulateMatch={handleChooseMatch} />
        ) : isStarting ? (
          <LoadingState message="Initializing historical match simulation & T0 pre-match anchor..." />
        ) : selectedMatch && currentPrediction && currentState ? (
          <div className="dashboard-grid">
            {/* Section 1: Match Header */}
            <div className="grid-full-width">
              <MatchHeader match={selectedMatch} winner={winner} status={status} />
            </div>

            {/* Section 2: Current Match State & Win Probability */}
            <div className="grid-full-width">
              <WinProbability prediction={currentPrediction} />
            </div>

            {/* Active Players & Last Delivery Mini-Cards */}
            <div className="grid-full-width live-state-subgrid">
              <CurrentPlayers
                state={currentState}
                team1Name={selectedMatch.team_1}
                team2Name={selectedMatch.team_2}
              />
              <DeliveryCard delivery={currentDelivery} />
            </div>

            {/* Section 3: Replay Controls */}
            <div className="grid-full-width">
              <SimulationControls
                status={status}
                eventIndex={eventIndex}
                totalEvents={totalEvents}
                isStepping={isStepping}
                isRunning={isRunning}
                fullTrajectory={fullTrajectory}
                onNextBall={nextBall}
                onPreviousBall={previousBall}
                onGoToBall={goToBall}
                onRunToEnd={runToEnd}
                onReset={handleRestart}
                onOpenSelector={() => setIsSelectorOpen(true)}
              />
            </div>

            {/* Section 4: Probability Timeline Chart */}
            <div className="grid-full-width">
              <ProbabilityChart
                points={probabilityHistory}
                team1Name={selectedMatch.team_1}
                team2Name={selectedMatch.team_2}
                currentEventIndex={eventIndex}
                onPointClick={goToBall}
              />
            </div>

            {/* Section 5: Analytics Row (4 Primary Intelligence Modules) */}
            <div className="grid-full-width analytics-modules-grid">
              {/* Module A: Probability Swing / WPA */}
              <div className="analytics-col">
                <ProbabilitySwingWpa
                  currentPrediction={currentPrediction}
                  currentDelivery={currentDelivery}
                  currentState={currentState}
                  eventIndex={eventIndex}
                  probabilityHistory={probabilityHistory}
                  onSeekToBall={goToBall}
                />
              </div>

              {/* Module B: Match Volatility Meter */}
              <div className="analytics-col">
                <MatchVolatilityMeter
                  probabilityHistory={probabilityHistory}
                  eventIndex={eventIndex}
                />
              </div>

              {/* Module C: Player Impact */}
              <div className="analytics-col">
                <PlayerImpactCard
                  currentState={currentState}
                  probabilityHistory={probabilityHistory}
                  eventIndex={eventIndex}
                  team1Name={selectedMatch.team_1}
                  team2Name={selectedMatch.team_2}
                />
              </div>

              {/* Module D: Match DNA */}
              <div className="analytics-col">
                <MatchDnaCard
                  currentState={currentState}
                  probabilityHistory={probabilityHistory}
                  eventIndex={eventIndex}
                />
              </div>
            </div>

            {/* Section 6: Two Complete Innings Scorecards */}
            <div className="grid-full-width">
              <DualInningsScorecard
                state={currentState}
                team1Name={selectedMatch.team_1}
                team2Name={selectedMatch.team_2}
              />
            </div>
          </div>
        ) : (
          <div className="welcome-placeholder">
            <h2>Welcome to IPL Match Intelligence Analytics</h2>
            <p>Select a historical match to launch delivery-by-delivery live simulation.</p>
            <button className="btn btn-primary btn-lg" onClick={() => setIsSelectorOpen(true)}>
              Browse 1,243 Matches
            </button>
          </div>
        )}
      </main>

      {/* Match Selector Modal */}
      <MatchSelector
        isOpen={isSelectorOpen}
        onClose={() => setIsSelectorOpen(false)}
        matches={matches}
        total={total}
        loading={matchesLoading}
        selectedSeason={selectedSeason}
        onSelectSeason={setSelectedSeason}
        selectedTeam={selectedTeam}
        onSelectTeam={setSelectedTeam}
        availableSeasons={availableSeasons}
        availableTeams={availableTeams}
        onChooseMatch={handleChooseMatch}
      />
    </div>
  );
};
