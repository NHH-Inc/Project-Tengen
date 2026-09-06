import { useCallback, useEffect, useState } from 'react';
import { getApi } from '../api';
import type { ContractViolation, Track } from '../contracts';

interface JobTracks {
  tracks: Track[];
  boxSampleRate: number;
  loading: boolean;
  error: string | null;
  violations: ContractViolation[];
}

const EMPTY: JobTracks = {
  tracks: [],
  boxSampleRate: 0,
  loading: true,
  error: null,
  violations: [],
};

/** Poll the job-level YOLO track file so the player can show boxes during analysis. */
export function useJobTracks(jobId: string | null, enabled: boolean, watch = false) {
  const [data, setData] = useState<JobTracks>(EMPTY);

  const load = useCallback(async () => {
    if (!jobId || !enabled) {
      setData({ ...EMPTY, loading: false });
      return;
    }
    try {
      const api = await getApi();
      const response = await api.getJobTracks(jobId);
      setData({
        tracks: response.data.tracks,
        boxSampleRate: response.data.boxSampleRate,
        loading: false,
        error: null,
        violations: response.violations,
      });
    } catch (error) {
      setData((current) => ({ ...current, loading: false, error: (error as Error).message }));
    }
  }, [enabled, jobId]);

  useEffect(() => {
    void load();
    if (!watch) return;
    const timer = window.setInterval(() => void load(), 1500);
    return () => window.clearInterval(timer);
  }, [load, watch]);

  return data;
}
