// The strategy layer's output, as the ingest service serves it (ingest/hosted.py). These are the
// Python strategy layer's own records (ingest/strategy/features.py, profile.py), passed through
// unchanged, so their keys stay snake_case here rather than being remapped field by field.
// Positions are in the ALLIANCE FRAME: feet from the robot's own alliance wall.

export interface Spot {
  ax: number;
  ay: number;
  launches: number;
  share?: number | null;
  matches?: number;
  distance_to_hub_ft?: number;
}

export interface RobotFeatures {
  job_id: string | null;
  match_id: string | null;
  robot_key: string;
  team: number | null;
  alliance: 'red' | 'blue' | null;
  identity_issues: string[];
  coverage: { match_seconds: number; observed_s: number; positioned_s: number; positioned_share: number | null };
  launches: { total: number; by_phase: Record<string, number> };
  positions?: null;
  movement?: {
    distance_ft: number;
    median_moving_speed_ftps: number | null;
    top_speed_ftps: number | null;
    stationary_share: number | null;
    longest_stop_s: number;
  };
  zones?: { auto: Record<string, number>; teleop: Record<string, number>; teleop_lanes: Record<string, number>; own_half_share: number | null };
  start?: { ax: number; ay: number; zone: string | null; lane: string | null } | null;
  auto_path?: { points: Array<[number, number] | null>; observed_share: number | null; distance_ft: number };
  shooting?: {
    launches: number;
    located_launches: number;
    spots: Spot[];
    volleys: number;
    median_launches_per_volley: number | null;
    median_volley_interval_s: number | null;
    median_dwell_s: number | null;
    on_the_move_share: number | null;
    median_distance_to_hub_ft: number | null;
  };
  defense?: { opponent_half_s: number; near_opponent_s: number | null; marked: string | null; marked_s?: number | null };
  endgame?: { last_launch_mt: number | null; stopped_scoring_before_endgame_s: number | null };
}

export interface MatchStrategy {
  job_id: string;
  match_id: string | null;
  clock: { auto_start_t: number | null; teleop_start_t: number | null; source: string; notes: string[] } | null;
  robots: RobotFeatures[];
}

export interface StrategyTeam {
  team: number;
  matches: Array<string | null>;
  with_positions: number;
}

export interface Counter {
  id: string;
  advice: string;
  evidence: string;
  matches?: Array<string | null>;
}

export interface HeatCell {
  x: number;
  y: number;
  launches: number;
  weight: number;
  matches: number;
  intensity: number; // 0..1, relative to the team's hottest cell
}

export interface AutoCell {
  x: number;
  y: number;
  samples: number; // per-second auto-path points that fell in this cell
  weight: number;
  matches: number;
  intensity: number; // 0..1, relative to the team's busiest auto cell
}

export interface ZoneScore {
  zone: string;
  lane: string;
  launches: number;
  weight: number;
  matches: number;
  share: number;
  effective: boolean;
}

// Where a team scores across matches, weighted toward recent matches, practice scoring kept.
// See ingest/strategy/heatmap.py and docs/PLAN-2026.md section 5.
export interface ScoringMap {
  cell_ft: number;
  recency_half_life_matches: number;
  n_matches: number;
  n_practice: number;
  hot_spot: { x: number; y: number } | null;
  heatmap: HeatCell[];
  effective_zones: ZoneScore[];
  zones: ZoneScore[];
}

// Where a team drives in autonomous, over OFFICIAL matches only (practice auto is noise -- the
// team is testing routines). See ingest/strategy/heatmap.py auto_map and docs/PLAN-2026.md section 5.
export interface AutoMap {
  cell_ft: number;
  recency_half_life_matches: number;
  n_matches: number; // official matches behind the map
  n_excluded_practice: number;
  busiest_cell: { x: number; y: number } | null;
  heatmap: AutoCell[];
}

export interface TeamProfile {
  team: number;
  matches: Array<string | null>;
  n_matches: number;
  n_with_positions: number;
  scoring_map?: ScoringMap;
  auto_map?: AutoMap;
  launches_per_match: number | null;
  timing?: {
    median_volley_interval_s: number | null;
    median_launches_per_volley: number | null;
    median_stopped_scoring_before_endgame_s: number | null;
  };
  tendencies: null | {
    start: { lane: string | null; lane_matches: number; known: number; mean_ax: number | null; mean_ay: number | null };
    auto: { routines: Array<{ matches: number; match_ids: Array<string | null>; path: Array<[number, number] | null> }>; median_distance_ft: number | null };
    shooting: { spots: Spot[]; median_volley_interval_s: number | null; median_dwell_s: number | null; on_the_move_share: number | null; median_distance_to_hub_ft: number | null };
    zones: Record<string, number | null>;
    lanes: Record<string, number | null>;
    movement: { top_speed_ftps: number | null; stationary_share: number | null };
    defense: { matches_played: number; of: number; median_near_opponent_s: number | null };
    endgame: { median_stopped_scoring_before_endgame_s: number | null };
  };
  prediction: null | {
    start_lane: string | null;
    auto_routine: number | null;
    auto_routine_confidence: number | null;
    primary_spot: { ax: number; ay: number } | null;
    launches: number | null;
    plays_defense: boolean | null;
  };
  counters: Counter[];
}

export interface TeamProfileResponse {
  profile: TeamProfile;
  matches: RobotFeatures[];
}
