import { useCallback, useEffect, useRef, useState } from 'react';
import { getApi, type CreateJobInput } from '../api';
import type { ContractViolation, Job } from '../contracts';
import { ACTIVE_STATUSES } from '../lib/format';

export interface JobsState {
  jobs: Job[];
  loading: boolean;
  error: string | null;
  violations: ContractViolation[];
}

/**
 * The job queue. Doc 3: "The UI needs a queue with visible status per video: queued,
 * downloading, analyzing, done, failed."
 *
 * Polls GET /api/jobs while anything is still moving and stops once everything has settled,
 * so an idle tab is not hammering the ingest service.
 */
export function useJobs(pollMs = 1500) {
  const [state, setState] = useState<JobsState>({
    jobs: [],
    loading: true,
    error: null,
    violations: [],
  });
  const timer = useRef<number | null>(null);
  const alive = useRef(true);

  const load = useCallback(async () => {
    try {
      const api = await getApi();
      const { data, violations } = await api.listJobs();
      if (!alive.current) return;
      setState({ jobs: data, loading: false, error: null, violations });
    } catch (e) {
      if (!alive.current) return;
      setState((s) => ({ ...s, loading: false, error: (e as Error).message }));
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void load();
    return () => {
      alive.current = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [load]);

  // Reschedule after every settle, rather than a fixed interval, so a slow response cannot
  // stack up requests behind it.
  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    const busy = state.jobs.some((j) => ACTIVE_STATUSES.has(j.status));
    if (!busy) return;
    timer.current = window.setTimeout(() => void load(), pollMs);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [state.jobs, load, pollMs]);

  const createJob = useCallback(
    async (input: CreateJobInput) => {
      const api = await getApi();
      const { data } = await api.createJob(input);
      await load();
      return data;
    },
    [load]
  );

  const deleteJob = useCallback(
    async (jobId: string) => {
      const api = await getApi();
      await api.deleteJob(jobId);
      await load();
    },
    [load]
  );

  /** Re-run the stored video without creating a second job or requiring the link again. */
  const retryJob = useCallback(
    async (job: Job) => {
      const api = await getApi();
      // v2: retry reuses the job id and increments attempt.
      const { data } = await api.retryJob(job.jobId);
      await load();
      return data;
    },
    [load]
  );

  return { ...state, reload: load, createJob, deleteJob, retryJob };
}
