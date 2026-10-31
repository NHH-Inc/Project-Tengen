// A TypeScript port of the Python strategy engine (ingest/strategy/engine.py) and pick engine
// (picklist.py), run client-side over the TeamProfile objects the web already fetches via
// getTeamProfile. The web derives views from fetched data all the time (the heat-map density grid,
// team stats); this is the same, so the drive team gets match plans and pick lists with no backend
// round-trip and the fixture/demo works offline.
//
// It mirrors the Python field-for-field. The Python side carries the authoritative tests
// (ingest/tests/test_engine.py, test_picklist.py); keep the two in sync, and prefer changing the
// Python first so those tests pin the behaviour.

import type { TeamProfile } from '../api/strategy';

// --- shared thresholds (engine.py) -----------------------------------------------------------
const AUTO_HOT_INTENSITY = 0.4; // an auto_map cell this bright counts as part of a team's footprint
const AUTO_PREDICTABLE_SHARE = 0.4; // one auto cell holding this share, over 2+ matches, is predictable
const DEFENSE_SHARE = 0.5; // marked an opponent in this share of matches => a defender
const CONFLICT_PENALTY = 2.0; // a unit of conflict is worth this many shots/match of raw power

const round3 = (n: number) => Math.round(n * 1000) / 1000;
const round2 = (n: number) => Math.round(n * 100) / 100;
const round1 = (n: number) => Math.round(n * 10) / 10;

// --- conflict model (engine.py / picklist.py) ------------------------------------------------

function autoHotCells(p: TeamProfile): Map<string, number> {
  const cells = new Map<string, number>();
  for (const c of p.auto_map?.heatmap ?? []) {
    if (c.intensity >= AUTO_HOT_INTENSITY) cells.set(`${c.x},${c.y}`, c.intensity);
  }
  return cells;
}

function effectiveZoneWeights(p: TeamProfile): Map<string, number> {
  const zones = new Map<string, number>();
  for (const z of p.scoring_map?.effective_zones ?? []) zones.set(`${z.zone}|${z.lane}`, z.weight);
  return zones;
}

export interface Conflict { auto: number; zones: number; total: number; }

/** How much two robots would step on each other: shared hot auto cells + shared scoring zones. */
export function conflict(a: TeamProfile, b: TeamProfile): Conflict {
  const fa = autoHotCells(a), fb = autoHotCells(b);
  let auto = 0;
  for (const [k, v] of fa) { const o = fb.get(k); if (o !== undefined) auto += Math.min(v, o); }
  const za = effectiveZoneWeights(a), zb = effectiveZoneWeights(b);
  let zones = 0;
  for (const [k, v] of za) { const o = zb.get(k); if (o !== undefined) zones += Math.min(v, o); }
  return { auto: round3(auto), zones: round3(zones), total: round3(auto + zones) };
}

export function offensivePower(p: TeamProfile): number {
  return p.launches_per_match ?? 0;
}

// --- match plan (engine.py) ------------------------------------------------------------------

export interface AutoConflict { x: number; y: number; teams: number[]; intensity: number; }
export interface PredictableAuto { team: number; x: number; y: number; share: number; matches: number; }
export interface PooledZone { zone: string; lane: string; weight: number; teams: number[]; }
export interface DefenseWatch { team: number; matches_played: number; of: number; }
export interface Advice { id: string; advice: string; evidence: string; }
export interface MatchPlan {
  alliance: number[];
  opponents: number[];
  missing: number[];
  auto: { partner_conflicts: AutoConflict[]; contest_opponent_autos: PredictableAuto[] };
  teleop: { deny: PooledZone[]; our_strengths: PooledZone[]; defense_watch: DefenseWatch[] };
  advice: Advice[];
}

function autoConflicts(profiles: TeamProfile[]): AutoConflict[] {
  const byCell = new Map<string, Array<{ team: number; intensity: number }>>();
  for (const p of profiles) {
    for (const [key, intensity] of autoHotCells(p)) {
      (byCell.get(key) ?? byCell.set(key, []).get(key)!).push({ team: p.team, intensity });
    }
  }
  const out: AutoConflict[] = [];
  for (const [key, hits] of byCell) {
    const teams = hits.map((h) => h.team).filter((t) => t != null).sort((a, b) => a - b);
    if (teams.length >= 2) {
      const [x, y] = key.split(',').map(Number);
      out.push({ x, y, teams, intensity: round3(Math.max(...hits.map((h) => h.intensity))) });
    }
  }
  return out.sort((a, b) => b.intensity - a.intensity);
}

function predictableAutos(profiles: TeamProfile[]): PredictableAuto[] {
  const out: PredictableAuto[] = [];
  for (const p of profiles) {
    const cells = p.auto_map?.heatmap ?? [];
    const total = cells.reduce((s, c) => s + c.weight, 0);
    if (!total) continue;
    const top = cells[0]; // heatmap is sorted by weight
    const share = top.weight / total;
    if (share >= AUTO_PREDICTABLE_SHARE && top.matches >= 2) {
      out.push({ team: p.team, x: top.x, y: top.y, share: round2(share), matches: top.matches });
    }
  }
  return out.sort((a, b) => b.share - a.share);
}

export function aggregateZones(profiles: TeamProfile[]): PooledZone[] {
  const agg = new Map<string, PooledZone>();
  for (const p of profiles) {
    for (const z of p.scoring_map?.effective_zones ?? []) {
      const key = `${z.zone}|${z.lane}`;
      const entry = agg.get(key) ?? { zone: z.zone, lane: z.lane, weight: 0, teams: [] };
      entry.weight += z.weight;
      if (p.team != null && !entry.teams.includes(p.team)) entry.teams.push(p.team);
      agg.set(key, entry);
    }
  }
  const out = [...agg.values()];
  for (const e of out) { e.weight = round2(e.weight); e.teams.sort((a, b) => a - b); }
  return out.sort((a, b) => b.weight - a.weight);
}

function defenseWatch(profiles: TeamProfile[]): DefenseWatch[] {
  const out: DefenseWatch[] = [];
  for (const p of profiles) {
    const d = p.tendencies?.defense;
    const of = d?.of ?? 0;
    if (of && (d!.matches_played ?? 0) / of >= DEFENSE_SHARE) {
      out.push({ team: p.team, matches_played: d!.matches_played, of });
    }
  }
  return out;
}

const zoneLabel = (zone: string, lane: string) => `${zone.replace(/_/g, ' ')} / ${lane} lane`;

function advise(plan: Omit<MatchPlan, 'advice'>): Advice[] {
  const advice: Advice[] = [];
  for (const c of plan.auto.partner_conflicts) {
    advice.push({
      id: 'auto_partner_conflict',
      advice: `Split ${c.teams.join(' and ')} in auto: both camp ~(${c.x}, ${c.y}) ft, so give one that spot and send the other elsewhere.`,
      evidence: `Both run auto in the same cell (intensity ${c.intensity.toFixed(2)}).`,
    });
  }
  for (const a of plan.auto.contest_opponent_autos) {
    advice.push({
      id: 'contest_opponent_auto',
      advice: `Contest team ${a.team}'s auto near (${a.x}, ${a.y}) ft — it is predictable, so plan your auto onto its game pieces.`,
      evidence: `${Math.round(a.share * 100)}% of its auto there over ${a.matches} matches.`,
    });
  }
  const deny = plan.teleop.deny[0];
  if (deny) {
    advice.push({
      id: 'deny_scoring',
      advice: `Deny ${zoneLabel(deny.zone, deny.lane)}: it is the opponents' best scoring ground, so contesting it prevents the most points.`,
      evidence: `Opponents (${deny.teams.join(', ')}) score most there (weight ${deny.weight}).`,
    });
  }
  for (const w of plan.teleop.defense_watch) {
    advice.push({
      id: 'expect_defense',
      advice: `Expect defense from team ${w.team}: keep a scoring route that avoids it.`,
      evidence: `Played defense in ${w.matches_played} of ${w.of} matches.`,
    });
  }
  const strength = plan.teleop.our_strengths[0];
  if (strength) {
    advice.push({
      id: 'play_to_strength',
      advice: `Feed our scorers at ${zoneLabel(strength.zone, strength.lane)} — that is where we convert best.`,
      evidence: `Our alliance scores most there (weight ${strength.weight}).`,
    });
  }
  return advice;
}

/** An alliance plan for one match. `profiles` maps team -> profile (with auto_map/scoring_map). */
export function buildMatchPlan(ourTeams: number[], theirTeams: number[],
                               profiles: Map<number, TeamProfile>): MatchPlan {
  const our = ourTeams.map((t) => profiles.get(t)).filter((p): p is TeamProfile => !!p);
  const their = theirTeams.map((t) => profiles.get(t)).filter((p): p is TeamProfile => !!p);
  const base = {
    alliance: ourTeams,
    opponents: theirTeams,
    missing: [...ourTeams, ...theirTeams].filter((t) => !profiles.has(t)),
    auto: { partner_conflicts: autoConflicts(our), contest_opponent_autos: predictableAutos(their) },
    teleop: { deny: aggregateZones(their), our_strengths: aggregateZones(our), defense_watch: defenseWatch(their) },
  };
  return { ...base, advice: advise(base) };
}

// --- pick list (picklist.py) -----------------------------------------------------------------

export interface PartnerValue { team: number; power: number; conflict: number; score: number; }

function partnerValue(our: TeamProfile[], candidate: TeamProfile): PartnerValue {
  const power = offensivePower(candidate);
  const totalConflict = our.reduce((s, m) => s + conflict(m, candidate).total, 0);
  return { team: candidate.team, power: round1(power), conflict: round3(totalConflict),
           score: round2(power - CONFLICT_PENALTY * totalConflict) };
}

/** Our pick list: available teams ranked best partner first, by power minus conflict with us. */
export function rankPartners(ourTeams: number[], availableTeams: number[],
                             profiles: Map<number, TeamProfile>, limit = 18): PartnerValue[] {
  const our = ourTeams.map((t) => profiles.get(t)).filter((p): p is TeamProfile => !!p);
  const ranked = availableTeams
    .filter((t) => !ourTeams.includes(t) && profiles.has(t))
    .map((t) => partnerValue(our, profiles.get(t)!));
  ranked.sort((a, b) => b.score - a.score || b.power - a.power);
  return ranked.slice(0, limit);
}

export interface AllianceStrength { teams: number[]; power: number; internal_conflict: number; score: number; }

export function allianceStrength(teams: number[], profiles: Map<number, TeamProfile>): AllianceStrength {
  const members = teams.map((t) => profiles.get(t)).filter((p): p is TeamProfile => !!p);
  const power = members.reduce((s, m) => s + offensivePower(m), 0);
  let internal = 0;
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) internal += conflict(members[i], members[j]).total;
  }
  return { teams, power: round1(power), internal_conflict: round3(internal),
           score: round2(power - CONFLICT_PENALTY * internal) };
}
