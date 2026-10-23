import { request } from './client';
import {
  ReliabilityAuditorData,
  FailureObservatoryData,
  EraShiftMonitorData,
  IntelligenceSummaryData,
} from '../types/intelligence';

export async function fetchIntelligenceSummary(): Promise<IntelligenceSummaryData> {
  return request<IntelligenceSummaryData>('/intelligence/summary');
}

export async function fetchReliabilityAudit(
  modelType: 'live' | 'pre_match' = 'live',
  split: 'test' | 'val' | 'walk_forward' = 'test',
  season?: string
): Promise<ReliabilityAuditorData> {
  const params = new URLSearchParams({
    model_type: modelType,
    split,
  });
  if (season) {
    params.append('season', season);
  }
  return request<ReliabilityAuditorData>(`/intelligence/reliability?${params.toString()}`);
}

export async function fetchFailureObservatory(): Promise<FailureObservatoryData> {
  return request<FailureObservatoryData>('/intelligence/failures');
}

export async function fetchEraShiftMonitor(): Promise<EraShiftMonitorData> {
  return request<EraShiftMonitorData>('/intelligence/era-shifts');
}
