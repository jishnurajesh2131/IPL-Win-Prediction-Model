import { useState, useCallback } from 'react';
import {
  createSimulation,
  advanceSimulation,
  previousSimulation,
  seekSimulation,
  runSimulation,
  fetchProbabilityHistory,
  getSimulationState,
} from '../api/simulations';
import { MatchStateSnapshot, DeliverySnapshot, MatchIntelligenceSnapshot } from '../types/simulation';
import { PredictionResponse, ProbabilityPoint } from '../types/prediction';

export function useSimulation() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [matchId, setMatchId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'ready' | 'active' | 'completed' | 'error'>('idle');
  const [eventIndex, setEventIndex] = useState<number>(0);
  const [totalEvents, setTotalEvents] = useState<number>(0);
  const [winner, setWinner] = useState<string | null>(null);

  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [currentState, setCurrentState] = useState<MatchStateSnapshot | null>(null);
  const [currentDelivery, setCurrentDelivery] = useState<DeliverySnapshot | null>(null);
  const [matchIntelligence, setMatchIntelligence] = useState<MatchIntelligenceSnapshot | null>(null);
  const [probabilityHistory, setProbabilityHistory] = useState<ProbabilityPoint[]>([]);
  const [fullTrajectory, setFullTrajectory] = useState<ProbabilityPoint[]>([]);

  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [isStepping, setIsStepping] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const startSimulation = useCallback(async (mid: string) => {
    setIsStarting(true);
    setError(null);
    try {
      const resp = await createSimulation(mid);
      setSessionId(resp.session_id);
      setMatchId(resp.match_id);
      setStatus(resp.status as any);
      setEventIndex(resp.current_event_index);
      setTotalEvents(resp.total_events);
      setCurrentPrediction(resp.current_prediction);
      setWinner(null);

      // Fetch the full initial state snapshot from the backend
      const stateResp = await getSimulationState(resp.session_id);
      setCurrentState(stateResp.state);
      setCurrentDelivery(stateResp.delivery);
      setMatchIntelligence(stateResp.match_intelligence || null);

      // Fetch initial probability history (event 0)
      const histResp = await fetchProbabilityHistory(resp.session_id);
      setProbabilityHistory(histResp.points);

      // Fetch full trajectory for the complete match for seamless chart interaction & seeking
      try {
        const fullHist = await fetchProbabilityHistory(resp.session_id, true);
        setFullTrajectory(fullHist.points);
      } catch {
        setFullTrajectory(histResp.points);
      }

      return resp;
    } catch (err: any) {
      setError(err.message || 'Failed to start simulation session.');
      setStatus('error');
      return null;
    } finally {
      setIsStarting(false);
    }
  }, []);

  const nextBall = useCallback(async () => {
    if (!sessionId || status === 'completed') return;

    setIsStepping(true);
    setError(null);
    try {
      const stepResp = await advanceSimulation(sessionId);
      setEventIndex(stepResp.event_index);
      setStatus(stepResp.status as any);
      setCurrentDelivery(stepResp.delivery);
      setCurrentState(stepResp.state);
      setCurrentPrediction(stepResp.prediction);
      setMatchIntelligence(stepResp.match_intelligence || null);

      // Append new point to history
      const newPoint: ProbabilityPoint = {
        event_index: stepResp.event_index,
        innings: stepResp.state.innings,
        ball: stepResp.delivery ? stepResp.delivery.ball : 0.0,
        overs_completed: stepResp.state.overs_completed,
        score: stepResp.state.score,
        wickets: stepResp.state.wickets,
        team_1_probability: stepResp.prediction.team_1_probability,
        team_2_probability: stepResp.prediction.team_2_probability,
        prediction_type: stepResp.prediction.prediction_type,
      };

      setProbabilityHistory((prev) => {
        const updated = [...prev.slice(0, stepResp.event_index), newPoint];
        return updated;
      });

      if (stepResp.status === 'completed') {
        if (stepResp.prediction.team_1_probability === 1.0) {
          setWinner(stepResp.prediction.team_1);
        } else if (stepResp.prediction.team_2_probability === 1.0) {
          setWinner(stepResp.prediction.team_2);
        } else {
          setWinner('Regulation Tie');
        }
      }

      return stepResp;
    } catch (err: any) {
      setError(err.message || 'Failed to advance simulation step.');
      return null;
    } finally {
      setIsStepping(false);
    }
  }, [sessionId, status]);

  const previousBall = useCallback(async () => {
    if (!sessionId || eventIndex <= 0) return;

    setIsStepping(true);
    setError(null);
    try {
      const stepResp = await previousSimulation(sessionId);
      setEventIndex(stepResp.event_index);
      setStatus(stepResp.status as any);
      setCurrentDelivery(stepResp.delivery);
      setCurrentState(stepResp.state);
      setCurrentPrediction(stepResp.prediction);
      setMatchIntelligence(stepResp.match_intelligence || null);
      setWinner(null);

      // Symmetrically trim probability history back to previous position
      setProbabilityHistory((prev) => prev.slice(0, stepResp.event_index + 1));

      return stepResp;
    } catch (err: any) {
      setError(err.message || 'Failed to step backward.');
      return null;
    } finally {
      setIsStepping(false);
    }
  }, [sessionId, eventIndex]);

  const goToBall = useCallback(async (targetIndex: number) => {
    if (!sessionId) return;

    setIsStepping(true);
    setError(null);
    try {
      const stepResp = await seekSimulation(sessionId, targetIndex);
      setEventIndex(stepResp.event_index);
      setStatus(stepResp.status as any);
      setCurrentDelivery(stepResp.delivery);
      setCurrentState(stepResp.state);
      setCurrentPrediction(stepResp.prediction);
      setMatchIntelligence(stepResp.match_intelligence || null);

      // Rebuild probability history up to targetIndex using fullTrajectory or API
      if (fullTrajectory.length > targetIndex) {
        setProbabilityHistory(fullTrajectory.slice(0, targetIndex + 1));
      } else {
        const histResp = await fetchProbabilityHistory(sessionId);
        setProbabilityHistory(histResp.points);
      }

      if (stepResp.status === 'completed') {
        if (stepResp.prediction.team_1_probability === 1.0) {
          setWinner(stepResp.prediction.team_1);
        } else if (stepResp.prediction.team_2_probability === 1.0) {
          setWinner(stepResp.prediction.team_2);
        } else {
          setWinner('Regulation Tie');
        }
      } else {
        setWinner(null);
      }

      return stepResp;
    } catch (err: any) {
      setError(err.message || 'Failed to navigate to ball.');
      return null;
    } finally {
      setIsStepping(false);
    }
  }, [sessionId, fullTrajectory]);

  const runToEnd = useCallback(async () => {
    if (!sessionId || status === 'completed') return;

    setIsRunning(true);
    setError(null);
    try {
      const runResp = await runSimulation(sessionId, true);
      setStatus('completed');
      setEventIndex(runResp.total_events - 1);
      setWinner(runResp.winner);
      setCurrentState(runResp.final_state);
      setCurrentPrediction(runResp.final_prediction);
      if (runResp.delivery) {
        setCurrentDelivery(runResp.delivery);
      }
      if (runResp.match_intelligence) {
        setMatchIntelligence(runResp.match_intelligence);
      }
      if (runResp.history) {
        setProbabilityHistory(runResp.history);
        setFullTrajectory(runResp.history);
      }
      return runResp;
    } catch (err: any) {
      setError(err.message || 'Failed to complete simulation run.');
      return null;
    } finally {
      setIsRunning(false);
    }
  }, [sessionId, status]);

  const resetSimulation = useCallback(() => {
    setSessionId(null);
    setMatchId(null);
    setStatus('idle');
    setEventIndex(0);
    setTotalEvents(0);
    setWinner(null);
    setCurrentPrediction(null);
    setCurrentState(null);
    setCurrentDelivery(null);
    setMatchIntelligence(null);
    setProbabilityHistory([]);
    setFullTrajectory([]);
    setError(null);
  }, []);

  return {
    sessionId,
    matchId,
    status,
    eventIndex,
    totalEvents,
    winner,
    currentPrediction,
    currentState,
    currentDelivery,
    matchIntelligence,
    probabilityHistory,
    fullTrajectory,
    isStarting,
    isStepping,
    isRunning,
    error,
    startSimulation,
    nextBall,
    previousBall,
    goToBall,
    runToEnd,
    resetSimulation,
  };
}
