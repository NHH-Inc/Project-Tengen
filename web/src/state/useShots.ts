import { useCallback, useEffect, useState } from 'react';
import { getApi } from '../api';
import { EMPTY_SHOT_RESPONSE, type ShotResponse } from '../api/shots';

export interface ShotState extends ShotResponse {
  loading: boolean;
  error: string | null;
}

const EMPTY: ShotState = { ...EMPTY_SHOT_RESPONSE, loading: false, error: null };

export function useShots(jobId: string | null, ready: boolean) {
  const [state, setState] = useState<ShotState>(EMPTY);

  const load = useCallback(async () => {
    if (!jobId || !ready) {
      setState(EMPTY);
      return;
    }
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const api = await getApi();
      const response = await api.getShots(jobId);
      setState({ ...response, loading: false, error: null });
    } catch (error) {
      setState({ ...EMPTY_SHOT_RESPONSE, loading: false, error: (error as Error).message });
    }
  }, [jobId, ready]);

  useEffect(() => {
    void load();
  }, [load]);

  return { ...state, reload: load };
}
