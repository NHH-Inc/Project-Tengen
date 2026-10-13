# Tengen as a strategy predictor: the plan

Drafted 2026-10-12, decisions recorded the same day. Supersedes the shot-counting direction in
`PLAN.md`; that file stays as history.

## The new goal, in one paragraph

Tengen used to answer *"how many points did each robot score?"* It now answers **"how does this
team play, and how do we beat it?"** From video of one match, or of a whole event, it should
work out where each robot goes, where it shoots from and how long it sits there, how it cycles,
what it does in auto, whether it plays defense, and how it finishes the match. It then turns that
into a **pattern** per team across matches, a **prediction** for that team's next match, and a
**counter**: concrete advice a drive team can use. The best case is during an event, ready
before we play them. The fallback is after the event.

Scoring numbers become a minor input. **Behaviour is now the product.**

---

## Where we actually are (measured, not hoped)

The strategy layer needs two inputs above all: **which team each robot is**, and **where it is
on the field**. Today both are the weak links.

| Run | Robot track fragments (6 robots on field) | Tracks with a team | Samples with field position |
|---|---:|---:|---:|
| `real-2026tuis_qm29` (v2, this PC) | 45 | 11 | 0 |
| `real-2026tuis_qm29-v3` (v3, this PC) | 53 | 0 | 0 |
| Robert's desktop run, `refM5LLkuJ8` (Mac) | **6** | assigned by hand | **18,113** (AprilTag calibration worked) |

What this means:

- **Detection is good enough.** v3 at 960 px: 0 of 400 held-out frames blind, mean 6.34
  robots per frame.
- **Identity is the #1 blocker.** A strategy profile for "track 37" is worthless. Bumper OCR
  attributes 2 of 51 tracks. On the **C++** engine, tracks also break into ~9 fragments per robot.
  The **Python** engine already has a six-robot re-identification layer
  (`docs/TRACKING-IDENTITY.md`), which is why Robert's run came out as exactly 6 tracks. That is
  one more reason to standardise on the Python engine (D2).
- **Field position is the #2 blocker.** It works on some broadcast matches (Robert's run) and
  fails on most. On a camera **we** mount and calibrate once, it should work every time.
- **Ball tracking is useful, re-purposed.** We no longer care much whether a shot went in. We
  care **when** and **from where** a robot shot. `launch_signals.py` and the launch part of
  `ball_scouting.py` give exactly that.

---

## What came in since the last session (2026-09-25 → 2026-09-28)

**Robert (`mr-garfield42`), merged as `NHH-Inc/Project-Tengen#1`:**

- **`desktop/`**: a native PySide6 app (`python tengen.py`). It opens a local video, runs the
  Python YOLO+ByteTrack analyzer in a subprocess, and plays the result with overlays (boxes,
  shot trails, goal regions). It also has a time-limited heat map, team assignment, event
  editing with undo, and JSON/CSV export. It needs no server, browser, or database. Runs are
  plain files in `data/desktop/runs/`. He verified it on his Mac with playback benchmarks
  (60 fps holds at 60).
- **`analysis/src/fuel_tracker.cpp`**: an experimental standalone C++ tool. It uses hand-drawn
  hub polygons, yellow-blob detection, and counts fuel that disappears inside the hub. Robert's
  own write-up says honestly that it isn't reliable yet (70 robot IDs in 10 s, 0 counts
  attributed).
- `track_yolo.py` now lets FFmpeg open the YouTube media URL directly instead of piping
  yt-dlp. That fixes hangs. Default model path moved to
  `data/models/yolo-v3-960-20260906/weights/best.pt`.

**Justin + Claude (2026-09-25, before the event):** upload a phone/webcam recording
(`POST /api/jobs/upload`), analyse uploads full-frame instead of through the broadcast crop,
keep the player's crop in sync, fix fullscreen box alignment, and export to a single `Tengen`
tab in Sheets.

**Checked on this PC 2026-10-12:** 514 Python tests pass, the smoke test passes 73/73, and the
C++ (including the fuel tracker) builds with vcpkg OpenCV 4.12 and passes 4/4 CTest. One test
isolation bug was fixed in `0ca1708` (see its message).

**Problems found (and what was done about them):**

1. **The shared Supabase database refuses connections** ("tenant/user not found"). The project
   looks paused or deleted. The web app can't start against it. The desktop app doesn't need it.
2. `data/models/yolo-v3-960-20260906/` exists only on Robert's Mac. Here the same weights are
   `data/robot-v3.pt`. Fixed in `82e4e9c`: the desktop and API defaults use whichever exists.
3. **2026 match timing was wrong in the season file** (15 s auto / 135 / 20). The scoreboard
   clock on two real matches and WPILib's 2026 game data agree: **20 s auto, then 2:20 of teleop
   whose last 30 s are the endgame**. Under Doc 0's phase rule (`teleop_seconds` includes the
   endgame) that is **20 / 140 / 30**. A first fix wrote 110 for teleop, following `phase_at` in
   the YOLO path, which turned out to be the one implementation that disagreed with Doc 0; it
   was corrected instead. Robert and Nathaniel should confirm: it is a contract value.
4. **Clips do not start at match time zero.** A typical upload has a title card, a frozen
   clock until ~7 s, and a ~4 s pause between auto and teleop. Match time now comes from the
   scoreboard clock, fitted robustly (`ingest/strategy/clock.py`); a Championship broadcast had
   also broken the clock reader, because blue was on the left.
5. `desktop/` is a fourth top-level directory that imports `ingest`. Doc 0 says "three
   top-level directories, no cross-imports." The pivot is the moment to rewrite Doc 0 rather
   than quietly break it (Decision D4).

---

## Architecture: five layers, each with one job

```
 video (file / our camera / stream)
   │
 [1] PERCEPTION      detect → track → stitch → identify team → field position → launch events
   │                 (exists, needs identity + calibration work)
   ▼
 [2] ROBOT TIMELINE  ONE canonical record per robot per match: 10 Hz field x/y, heading, speed,
   │                 team, alliance, phase, zone, carrying?, launch events.   ← NEW, the hinge
   ▼
 [3] FEATURES        per robot per match: ~40 behaviour numbers (catalogue below)   ← NEW
   ▼
 [4] PROFILE         per team across matches: tendencies with sample sizes + predictions  ← NEW
   ▼
 [5] COUNTER + UI    counter-strategy cards, pre-match briefing, evidence clips  ← NEW
```

**The robot timeline (layer 2) is the most important new thing.** Today every consumer reads raw
`tracks.jsonl` boxes, at whatever rate and in whatever coordinates the analyzer happened to use.
A fixed contract gives three things:

- Features never touch pixels.
- Any perception improvement immediately improves every strategy output.
- The 2027 game only changes the season file, not the code.

It becomes a new Contract (`contracts/timeline.schema.json`).

**Season-agnostic by construction.** Zones (shooting spots, loading stations, scoring areas,
starting positions, endgame area) are **polygons in field feet inside the season JSON**, not code.
The 2027 game is revealed in January. Swapping the season file should be most of the port.

---

## The feature catalogue (layer 3)

Each feature lists its data dependency. **ID** = needs team identity, **XY** = needs field
position, **L** = needs launch detection. Almost everything needs XY, which is why calibration
matters so much.

### Positioning and movement
1. **Zone dwell map**: % of teleop in each season zone. *XY*
2. **Heat map** per team, aggregated across matches (exists per match in desktop). *XY, ID*
3. **Preferred side/lane**: left/centre/right lane share, and routes between zones. *XY*
4. **Distance driven** and **top/typical speed** (already in tracks as `speed_ftps`). *XY*
5. **Time stationary vs moving**, and where it parks. *XY*
6. **Field-half split**: own half vs opponent half. *XY*

### Scoring
7. **Shooting spots**: launch positions clustered into 1–3 "favourite spots" per team. *XY, L*
8. **Dwell at shooting spot**: how long it sits before and while shooting (the user's ask). *XY, L*
9. **Shots per visit**, and **volley rhythm** (time between launches in a burst). *L*
10. **Shoot-on-the-move vs stop-to-shoot** (speed at launch). *XY, L*
11. **Shooting range**: distance from launch spot to goal. *XY, L*
12. Made/missed per spot, **only where ball tracking is confident**. Optional; never required. *L*

### Cycling
13. **Cycle time** (Doc 0's cycle: reload to reload, which needs intake events) and, until
    those exist, the **volley interval**: start-to-start time between shooting visits. *XY, L*
14. **Source preference**: floor vs loading station vs human player. *XY*
15. **Cycles per match** and whether it slows down late (fatigue/brownout signature). *XY, L*

### Autonomous
16. **Starting position** (identified pre-match, before robots move). *XY, ID*
17. **Auto path fingerprint**: the 15 s trajectory, compared across matches. Same path every time
    means it is predictable. *XY*
18. **Auto variants**: how many distinct autos it ran, and which one when. *XY*

### Defense and interaction
19. **Defense time**: time in the opponent's half near opponent robots (contract already has
    `defense_start/end`). *XY*
20. **Who it defends**: the opponent robot it shadows most. *XY, ID*
21. **Defended-against response**: does its cycle time collapse when someone sits on it? *XY*
22. **Contact/pin events**: two robots within a bumper width, both slow. *XY*

### Endgame and reliability
23. **Endgame start time**: when it abandons scoring for the endgame. *XY*
24. **Endgame position / success** (season-specific). *XY*
25. **Immobile periods**: dead robot, brownout, tipped (contract has `immobile_start/end`). *XY*
26. **Consistency score**: how similar its matches are to each other. *(all)*

### Alliance level
27. **Role**: scorer / feeder / defender / hybrid, from the mix above.
28. **Alliance coordination**: who takes which zone, and spacing between partners.
29. **Congestion**: shooting spots where two partners collide.

Features 1–11, 13 (as the volley interval), 16–17 and 19 are the **first wave**. They cover most counter-strategies, and
all of them come from the same two inputs: identity and field position.

---

## Patterns, predictions and counters (layers 4–5)

**Start with statistics and rules, not machine learning.** A team plays roughly 10–12 quals per
event. That is far too little for a learned model per team. Honest summaries with sample sizes
beat a neural net here:

- **Profile:** each feature as a distribution across the team's matches, always with *n*.
  "Shoots from the left trench spot in 7 of 8 matches." Never "always."
- **Prediction:** the next match's most likely start position, auto path, shooting spots, cycle
  count, and defense probability, each with a confidence.
- **Counters:** a rule library that maps tendencies to advice. Each card shows *why* and links
  the **evidence clips** (video timestamps):
  - Fixed auto path → "contest the pieces at X in auto."
  - One shooting spot and stop-to-shoot → "defend spot X; their cycle time goes from 9 s to
    16 s when defended (2 matches of evidence)."
  - Late endgame start → "pressure them before T-25 s."
  - Collapses under defense → "assign a defender."
  - Same loading station every cycle → "block lane Y."
- **Learned models later** (once we have hundreds of team-matches): cluster robots into roles
  across the 50-match corpus, and estimate how much each tendency actually changes match
  outcome. Machine learning goes where data volume supports it, which is cross-team, not
  per-team.

**How we know it works: the next-match test.** For every team with at least 4 analysed matches,
build the profile from matches 1..k, predict match k+1, and score it. Did the predicted shooting
spot match? Was the predicted start position correct? Was the cycle count within ±2? This is the
strategy equivalent of the held-out viewpoint pack, and it is the number we report. It is also
exactly the kind of evidence FRC awards judges respond to.

---

## One match, or the whole video

Both work with the same pipeline. A **match** is the unit everything is keyed on.

- **One match:** a clip, or a start/end time marked in the player. Analyse only that window.
- **Whole video** (a full day's stream, or our own camera's 6-hour recording):
  1. **Segment** it into matches first, cheaply. The scoreboard timer comes from OCR
     (`scoreboard.py` exists), match-start detection from the auto start, and both are
     cross-checked against the TBA schedule for match numbers and alliances.
  2. Analyse **only** the 2:30 inside each match. Skipping breaks, replays and award ceremonies
     is the single biggest speed-up for whole videos (most of a stream isn't match play).
  3. Matches are independent, so analyse them in parallel.

---

## Live at an event

**The realistic target is "near-live":** a match's strategy data is ready **within ~2 minutes
after it ends**. Teams play several matches apart, so a team's profile updates long before we
face them. True frame-by-frame live is a stretch goal, used for a live dashboard rather than for
decisions.

What that needs:

- **Our own fixed camera, where the event allows one** (phone or webcam on a tripod in the stands, full field in view). It
  is calibrated **once per event** with five hand-marked carpet points (`--method carpet`
  exists). It has no camera cuts and no broadcast overlays, so the broadcast crop and most
  fragmentation go away. This removes blocker #2 almost entirely. The stream becomes the backup,
  not the main source.
- **Analysis faster than real time on a laptop.** Today Robert measured 19 fps at 960 px on
  Mac CPU with every frame analysed. For positions, **10 Hz is plenty**. Analysing every third
  frame of a 30 fps source gives roughly 3× headroom before any other optimisation.
- **Identity fixed at match start.** Pull the 6 teams and alliances from TBA, take alliance from
  bumper colour, and have the operator click each robot once before the match (6 clicks, about
  10 s). The tracker and stitcher (below) carry the identity from there. Bumper OCR becomes a
  confirmation, not a dependency.

---

## Performance and efficiency

In order of payoff:

1. **Analyse match windows only** (segmentation above).
2. **Decouple rates.** Robots at 10 Hz, ball/launch detection at full rate **only in short
   windows** where a robot is at a shooting spot or motion says a launch is likely.
3. **Hardware inference.** ONNX Runtime with **DirectML** on this PC's RX 7800 XT (Windows, no
   ROCm needed), **CoreML** on Robert's Mac, CUDA where available. FP16 export. Today
   everything is CPU.
4. **Hardware decode.** FFmpeg `-hwaccel d3d11va` / `videotoolbox`, decoding once and feeding
   both robots and ball.
5. **Track stitching instead of re-detection.** Join the ~9 fragments per robot globally, using
   motion continuity, appearance, and the hard constraint that exactly 3 robots per alliance are
   on the field. That is an assignment problem (Hungarian), and it is cheap.
6. **One engine.** We currently maintain the C++ analyzer, the Python YOLO tracker, the C++
   fuel tracker, and two UIs. Each duplicate costs every future feature twice (Decision D2).

---

## Network and Wi-Fi

**Design so the network is a bonus, not a requirement.** Venue rules on team Wi-Fi vary.
Historically FRC event rules banned teams' own 802.11 access points (the 2022 manual's E301, as
quoted on Chief Delphi). That arguably includes phone hotspots, but rules and enforcement differ
by year and by event. **Check the 2027 manual and each event's rules.** Bluetooth or USB
tethering to a phone has generally been treated as fine. Venue Wi-Fi is often unusable anyway.

- Everything (analysis, profiles, counters) runs **on one laptop in the stands with no network**.
- **Moving data to people without a network:** a **QR code** per strategy card (scouting apps do
  this already), printed or PDF briefings, and a USB stick.
- **When a connection exists** (hotspot if allowed, phone tether, venue, hotel): sync runs and
  profiles to Sheets/Drive, or to the web app, in small JSON deltas, never video.
- **Stream bandwidth, when we use a stream at all:** fetch 720p, not 1080p, for robot positions
  (measure first that recall holds). Fetch higher resolution only for launch windows.

## UI

A UI built around the new questions, not around the analyzer's output:

1. **Event dashboard:** the schedule from TBA, which matches are analysed, and our upcoming
   matches with the opponents highlighted.
2. **Pre-match briefing** (the money screen): our next match's three opponents and two
   partners, each with predicted start/auto/shooting spots on a field diagram, the top 3
   counters, and confidence. One screen, printable, QR-shareable.
3. **Team page:** the profile across matches, an aggregated heat map, shooting-spot clusters,
   auto path overlays (all of a team's autos drawn on one field), the cycle-time trend, and an
   evidence clip for every claim.
4. **Match replay:** the existing player plus a **2D field view in sync with the video** (robots
   as dots with trails). Analysts need this most, and it is cheap once the timeline exists.
5. **Live view:** field view and running feature counters during a match (stretch).
6. **Identity panel:** the 6-click assignment at match start, and a fragment-merge review
   ("these two tracks are the same robot?").

---

## Decisions (recorded 2026-10-12)

| # | Decision | Status |
|---|---|---|
| D1 | **Desktop is the primary app; the web app stays too.** The desktop app is the offline analysis station at events. The web app remains for viewing and sharing between our own machines, and reads the same files and contracts, so neither forks the data. | **Decided** |
| D2 | **One analysis engine: the Python/Ultralytics pipeline** (`track_yolo` + launch detection + six-robot re-ID). The C++ RF-DETR analyzer and `fuel_tracker` are frozen as reference. | **Decided** |
| D3 | **Video source: our own camera if the event allows, broadcast otherwise.** Both must work. The broadcast path cannot be an afterthought, so calibration and identity work targets broadcast footage first. | **Depends on the event** |
| D4 | Doc 0 rewrite for `desktop/` and the strategy layer. Keep "components talk through files." | **Done 2026-10-13**; Robert and Nathaniel to confirm |
| D5 | Stop investing in made/missed and goal counting. **Keep launch detection.** | **Decided** |
| D6 | The dead Supabase DB: leave it down until the web app needs a shared store again. | **Decided** |

### Hardware: an RTX 5070 Ti is likely coming

That changes three things for the better:

- **Training moves back to this PC.** CUDA PyTorch trains the detector here instead of on
  Robert's Mac, so model iterations stop waiting on file transfers. A YOLO11 retrain at 960 px
  takes minutes per epoch, not hours.
- **Every-frame analysis becomes cheap.** TensorRT or CUDA ONNX Runtime with FP16 should run
  many times faster than real time at 960 px. That makes true live analysis realistic, and lets
  launch detection run on every frame instead of in windows.
- **NVDEC hardware decode** takes video decoding off the CPU.

Until it arrives, everything stays CPU-runnable (this PC's RX 7800 XT via DirectML is the
interim GPU option).

### Our robot's own camera and logs

**Camera feed.** During a match our robot's camera reaches **only our driver station**, over the
field network, within the bandwidth limit the robot rules set. Tengen could run on the
driver-station laptop and **record** that feed for analysis after the match. A first-person view
does not see the whole field, so it complements the field camera rather than replacing it. Two
cautions:

- Getting the feed from the driver station to the stands *during* a match would need a network
  link the rules restrict.
- Anything on the driver-station laptop must never slow the driver station down mid-match.

Check the 2027 robot rules (bandwidth, what may connect to the driver station) before building
anything.

**Robot logs are the more valuable idea.** Our robot already knows its own field pose
precisely, from odometry and AprilTag localisation, and logs it every match (WPILib DataLog /
AdvantageKit). That is **ground truth for Tengen's position accuracy**: align our robot's logged
pose with Tengen's track of our robot and measure the error in feet, every match, for free. No
other way of checking homography accuracy is this cheap or this exact. *(Added to M2.)*

**Driver assist.** Further out, the robot's vision coprocessor could run Tengen's detector to see
opponent robots live. That is robot code, not Tengen, but the trained model is shareable.

## Milestones

Today is 2026-10-12. The 2027 game is revealed at kickoff in early January. Awards are judged
in the 2027 season. **Rough owners are suggestions; adjust them in Discord.**

### M1: the timeline and identity (Oct 12 → Nov 2)
- **Started 2026-10-12:** `ingest/strategy/` (timeline → features → profiles → counters →
  next-match test), `contracts/strategy/2026.json` (alliance-frame zones from the surveyed tag
  layout), `python -m ingest.strategy`. *(Justin + Claude)*
- `contracts/timeline.schema.json`, so the timeline is a contract and not just one module's output.
- **First real run (Championship Curie qm10, 2026-10-12):** 6 tracks, trustworthy calibration,
  524 of 529 launches attributed, paths and shooting spots consistent with the video. But one
  identity slot went to a tiny false positive at the blue alliance wall (median box a fifth of a
  robot's width), and robot 4027 plus that slot both ended at 73 s, so only 4 of 6 robots
  were tracked for the last ~100 s. The re-ID layer needs a minimum box size before a track
  may claim a slot. *(Robert: `track_yolo.py` identity layer.)* The miner now flags this as
  `undersized`.
- Measure the Python engine's six-robot re-ID on more broadcast matches. Target: ≤ 8 tracks for
  6 robots on broadcast footage, and exactly 6 on our own camera.
- Identity panel: TBA schedule + bumper colour + 6-click assignment. *(Robert, in desktop)*
- Season zones as polygons in `contracts/seasons/2026.json`.
- **Done when:** one real match produces 6 timelines, each with a team and field positions.

### M2: own-camera capture at a local event or practice (Nov)
- Record full matches from a tripod (we already handle uploads). Calibrate with carpet points.
  Measure fragmentation and positional error against a few hand-checked frames.
- **Ground truth from our robot's logs:** align its logged pose with Tengen's track of it and
  report the position error in feet.
- **Done when:** a full recorded session is segmented into matches automatically.

### M3: features + profiles + the next-match test (Nov → mid-Dec)
- First-wave features (1–11, 13, 16–17, 19), team profiles, and predictions.
- Run the next-match test on whatever multi-match data we have (our recordings plus streams of a
  2026 event with good calibration).
- **Done when:** we can report "start position predicted correctly X% of the time; favourite
  shooting spot Y%" with *n*.

### M4: counters + briefing UI + speed (Dec)
- Counter rule library with evidence clips. Pre-match briefing, team page, synced field view.
- DirectML/CoreML inference, decoupled rates, match-window-only analysis.
- **Done when:** a match is analysed in under its own length on a laptop, and a briefing comes
  out as a PDF and a QR code.

### Continuous: the detector keeps improving (every milestone)
- Every analysed match feeds **hard-example mining**: frames where a track breaks, a robot
  vanishes mid-field, or a box appears for under a second go into the next label pack.
- Retrain on this PC once the 5070 Ti is in. Keep the 400-frame held-out viewpoint pack as the
  yardstick, and never report a val-split number as progress.
- From kickoff, add 2027 robots: week-0 and early-event footage is labelled first.

### M5: port to the 2027 game (January, after kickoff)
- New season file and zones. Check the detector still finds 2027 robots. Bumpers are
  rules-constant, so it probably generalises, **but measure it on week-0 footage**. Retrain if
  recall drops.
- Re-tune the launch detector for the new game piece.

### M6: field it (Feb → competition season)
- Use it at a week-0 or early event, keep the next-match test numbers, and document it for
  judges.

---

## What is explicitly *not* in scope yet
- Predicting match **scores** or win probability. That needs real `point_values` and many
  matches; Statbotics/OPR already do it from official data.
- Per-team machine-learning models (not enough data per team; see above).
- Anything that needs the field's own network or a venue Wi-Fi network.

---

## Appendix: the Doc 0 amendment (D4)

Applied on 2026-10-13 at Justin's request, as Doc 0's "2026-10-13 revision". Robert and Nathaniel
still need to confirm it. The text proposed here, for reference:

**Repository layout.** Replace "one repo, three top-level directories, no cross-imports" with:

> One repo. `/analysis` (C++, frozen reference), `/ingest` (Python: service, analysis engine
> orchestration, strategy layer), `/desktop` (Python: the primary app), `/web` (TypeScript).
> `/desktop` may import `/ingest`; nothing imports `/desktop`. `/ingest` and `/desktop` never
> import `/training` at runtime: analyzer output crosses that line as files. `/web` talks to
> `/ingest` over HTTP only.

**Contracts.** Add `timeline.schema.json` (robot timeline) and `strategy/<year>.json`
(alliance-frame zones, lanes, hubs).

**Shared vocabulary.** Add these terms, with exactly these meanings:

- **timeline**: one robot's resampled positions, phases, zones and launches for one match.
- **alliance frame**: field feet with the robot's own alliance wall at x=0, x toward the
  opponent, y to the driver's left. Red positions are rotated 180° about the field centre.
- **zone / lane**: a named x-band / y-band of the alliance frame, defined per season.
- **launch**: a game piece seen leaving a robot. Says nothing about whether it scored.
- **spot**: a cluster of one team's launch positions (4 ft radius).
- **volley**: launches from one robot less than 3 s apart.
- **volley interval**: start-to-start time between volleys. Not a **cycle**, which stays
  reload to reload.
- **profile**: one team's tendencies across matches, each with its number of matches.
- **counter**: advice derived from a profile, always shown with its evidence.
