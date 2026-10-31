import { useEffect, useState } from 'react';
import { getApi } from '../api';
import type { TeamProfile } from '../api/strategy';

// Fetch and cache team profiles for the Match Plan and Pick List views. It accumulates: growing the
// set of teams only fetches the new ones, so toggling an alliance member does not refetch the rest.
// The strategy engines (lib/strategyEngine.ts) run over the Map this returns.
export function useTeamProfiles(teams: number[]): { profiles: Map<number, TeamProfile>; loading: boolean } {
  const [profiles, setProfiles] = useState<Map<number, TeamProfile>>(new Map());
  const [loading, setLoading] = useState(false);
  const want = [...new Set(teams)].sort((a, b) => a - b).join(',');

  useEffect(() => {
    const need = want ? want.split(',').map(Number).filter((t) => !profiles.has(t)) : [];
    if (need.length === 0) return;
    let live = true;
    setLoading(true);
    getApi()
      .then((api) => Promise.all(need.map(async (t) => [t, (await api.getTeamProfile(t))?.profile] as const)))
      .then((entries) => {
        if (!live) return;
        setProfiles((prev) => {
          const next = new Map(prev);
          for (const [team, profile] of entries) if (profile) next.set(team, profile);
          return next;
        });
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // `want` captures the requested set; `profiles` is intentionally not a dep (it would loop).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [want]);

  return { profiles, loading };
}
