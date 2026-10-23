import { request } from './client';
import {
  SimulationCreateResponse,
  SimulationStepResponse,
  SimulationRunResponse,
  HealthResponse,
} from '../types/simulation';
import { ProbabilityHistoryResponse } from '../types/prediction';

export async function checkBackendHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export async function createSimulation(matchId: string): Promise<SimulationCreateResponse> {
  return request<SimulationCreateResponse>('/simulations', {
    method: 'POST',
    body: JSON.stringify({ match_id: matchId }),
  });
}

export async function getSimulationState(sessionId: string): Promise<SimulationStepResponse> {
  return request<SimulationStepResponse>(`/simulations/${encodeURIComponent(sessionId)}`);
}

export async function advanceSimulation(sessionId: string): Promise<SimulationStepResponse> {
  return request<SimulationStepResponse>(`/simulations/${encodeURIComponent(sessionId)}/next`, {
    method: 'POST',
  });
}

export async function previousSimulation(sessionId: string): Promise<SimulationStepResponse> {
  return request<SimulationStepResponse>(`/simulations/${encodeURIComponent(sessionId)}/previous`, {
    method: 'POST',
  });
}

export async function seekSimulation(sessionId: string, eventIndex: number): Promise<SimulationStepResponse> {
  return request<SimulationStepResponse>(`/simulations/${encodeURIComponent(sessionId)}/seek`, {
    method: 'POST',
    body: JSON.stringify({ event_index: eventIndex }),
  });
}

export async function fetchProbabilityHistory(
  sessionId: string,
  includeAll: boolean = false
): Promise<ProbabilityHistoryResponse> {
  return request<ProbabilityHistoryResponse>(
    `/simulations/${encodeURIComponent(sessionId)}/probabilities${includeAll ? '?all=true' : ''}`
  );
}

export async function runSimulation(
  sessionId: string,
  includeHistory: boolean = true
): Promise<SimulationRunResponse> {
  return request<SimulationRunResponse>(
    `/simulations/${encodeURIComponent(sessionId)}/run?include_history=${includeHistory}`,
    {
      method: 'POST',
    }
  );
}
