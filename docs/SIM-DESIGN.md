# §2 Simulation / prediction testbed — design

**Status (2026-10-31):** scope approved and **all four tiers (T0–T3) are shipped**, Python core in
`ingest/sim/`. The sim now has real motion, realistic corners, robot-robot blocking (defense), and a
**config-driven scoring framework** that reads goals + point values from the season config (today's
values are placeholders — fill them in and scoring comes alive). What remains for §2 is not a tier
but the **thin web view** to drive it interactively, and richer scoring once the 2026 rules are set.
See the per-tier status in §3 and `ingest/sim/`.

Decided in [PLAN-2026 §2](PLAN-2026.md): **the full physics field sim (b)**, not just the lightweight
replay (a). This doc records how we get there in a way that pays off early and doesn't balloon.

---

## 1. What it's actually for

From the chat and the plan, the sim exists to **test predictions**:

- "If we think team X runs auto Y, replay it and check." — does a predicted auto actually clear the
  pieces we think it does, in the time it has?
- Make the **heatmaps and the pick/strategy engines trustworthy** — a prediction you can replay and
  watch is one a drive coach will believe.
- Roleplay: "if we defend 254 with our #2 robot, where does the match go?"

It is **not** a game-design simulator or a driver-practice trainer. Fidelity only matters where it
changes a *strategy* answer. That single rule decides most of the scope below.

## 2. What we can build on (already in the repo)

- **Tracks** — every robot's field position at 10 Hz, in WPILib field coordinates (feet), per match.
  This is both the sim's *input* (where robots really went) and its *yardstick* (did the sim match
  reality?).
- **Profiles** (`ingest/strategy/profile.py`) — per team: auto routines (waypoint paths),
  `movement.top_speed_ftps`, shooting spots, start lane, predicted auto. The sim's robots are driven
  by these.
- **Geometry** (`contracts/strategy/<season>.json`) — field size, zones, lanes, hub positions.
- **Season config** — auto/teleop/endgame seconds.
- **AdvantageScope export** (`advantagescope.py`, §4) — already turns per-robot pose timeseries into a
  `.wpilog`. The sim can emit its output in the *same* shape, so a simulated match plays back in
  AdvantageScope next to the real one.
- **Web `FieldSketch` + heat-map canvas** — the rendering surface for an interactive sim.

The upshot: we already have the field, the robots' real trajectories, their motion limits, and a
viewer. The sim is the missing middle — a motion model that turns *intentions* into *trajectories*.

## 3. Fidelity tiers (the scope decision)

Each tier is a superset of the one before. Higher tiers cost more and help fewer real decisions.

| Tier | What it models | Effort | Strategy value |
|---|---|---|---|
| **T0 — Kinematic replay** | Play predicted waypoint paths and actual tracks on one field, time-aligned. No forces. | S | High — the (a) stepping stone; makes everything else checkable. | ✅ shipped (`validate.py`) |
| **T1 — Motion dynamics** | Robots as point masses with **max velocity + acceleration** (from each team's measured `top_speed`). Waypoint-following controller. Produces *realistic* trajectories and **arrival times**. | M | High — "can this auto reach 3 pieces in 15 s?" needs real accel limits. | ✅ shipped (`model.py`) |
| **T2 — Interactions** | Robot–robot **collision/blocking** as soft bodies (separation + removing the velocity driving them together), so **defense** is simulable; plus a turn-rate limit for non-holonomic drive. | M–L | Medium — defense is the main multi-robot question. | ✅ shipped (`match.py`) |
| **T3 — Game pieces & scoring** | Goals, pieces and point values **read from the season config**; a shooter scores while in range, defense-aware. | L–XL | Season-dependent; the approved end target. | ✅ shipped (`game.py`, `scoring.py`) |

**Approved target: T3.** Build order is still T0 → T1 → T2 → T3, each independently useful, because
each tier is the foundation (and the validation substrate) for the next. T0/T1 are done; T2 adds the
turn-rate limit the T1 model omits plus robot–robot blocking; T3 then adds game pieces and scoring on
a trusted motion base. A 2D physics engine (planck/pymunk) comes in at T2 only if the hand-rolled
separation model proves insufficient.

## 4. Architecture

Two surfaces, one core:

- **Core motion model** — a small, deterministic, headless library. Candidate home: a new
  `ingest/sim/` package (Python), so it can batch-validate against every analysed match and feed the
  strategy engines. Deterministic + seeded, like `profile.py`, so results are checkable.
- **Interactive front-end** — a web view (`web/src/views/Sim.tsx`) on the existing `FieldSketch`,
  where a coach sets up a scenario (our autos vs theirs, or "defend X") and scrubs the result. For
  T1 a hand-rolled integrator is enough; **no physics-engine dependency** until T2, and even then a
  light one (planck.js / matter.js on the web; `pymunk` only if we go headless-T2).
- **Interchange** — the sim emits the same per-robot pose timeseries the AdvantageScope export
  already speaks, so any sim run also opens in AdvantageScope and renders with `FieldSketch`.

Where the motion model *lives* (Python core vs TypeScript, like we did for the strategy engines) is
an open question below — it's the same port-vs-endpoint trade-off as #8.

## 5. How we know it's right (validation)

The sim is only useful if it tracks reality. The harness, from day one (T0/T1):

1. Take a real match's **actual auto** (from tracks).
2. Feed its **start pose + waypoints** to the sim.
3. Compare the **simulated** trajectory to the **actual** one: mean/max position error (ft), and
   arrival-time error at each waypoint.
4. Report a per-team, per-match divergence number. A sim that can't reproduce a known auto within a
   few feet isn't ready to predict an unknown one.

This reuses the same tracks the heatmaps use, so there's ground truth for free.

## 6. Proposed phasing

- **Phase 0 (T0):** replay predicted + actual paths on one field, time-aligned, in the web; emit to
  AdvantageScope. Ships the validation harness. *(This is the (a) work the §2 note said to do first.)*
- **Phase 1 (T1):** motion model with per-team velocity/accel/turn limits + a waypoint controller;
  validate against real autos; expose "simulate this predicted auto" in the web.
- **Phase 2 (T2):** robot–robot blocking, so defense scenarios run; roleplay UI ("defend 254").
- **Phase 3 (T3):** *only if approved* — game pieces + scoring.

Each phase is its own PR and is independently useful.

## 7. Decisions (resolved 2026-10-31)

1. **Fidelity target** — ✅ **T3** (full game physics + scoring), reached by building T0→T1→T2→T3.
2. **Core language** — ✅ **Python core** in `ingest/sim/`, thin web view for interaction.
3. **Scenario scope for v1** — replay + validate autos first (done); "what-if" setups (defense,
   alternate autos) arrive with T2's interaction model.
4. **Season/game-piece assumption** — game-agnostic (robots + field) through T2; game pieces + a
   scoring model are T3, the approved end target.
