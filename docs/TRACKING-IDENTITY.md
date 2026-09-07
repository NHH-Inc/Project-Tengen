# Robot identity and re-identification

The detector answers “where is a robot?” and ByteTrack/BoT-SORT groups nearby detections into raw
tracklets. Neither raw tracker IDs nor alliance colour are robot identities. The job-level identity
layer is deliberately separate:

```text
detector -> raw tracker tracklets -> global six-robot ReID -> OCR/roster reconciliation -> review
                  |                         |                       |
           tracks.raw.jsonl           tracks.jsonl       optional attribution fields
```

## Failure modes corrected

The previous `AppearanceTrackMemory.resolve` processed boxes in detector order. It put a Boolean
“same colour” ahead of appearance in a tuple sort, let the sole same-colour candidate bypass the
distance gate, selected each detection before seeing its competitors, and blended every accepted
crop into one template. Motion was calculated only after the stable ID had already been selected.
Consequently two red robots could swap, a wrong association could poison future comparisons, and a
recycled raw ID could be trusted for too long.

`team_id.py` already constrained OCR to the known match roster and abstained on tied reads, but
`attribute_tracks.py` previously attributed every track independently. It did not merge two
non-overlapping fragments with the same strong team evidence or detect the impossible case where
two simultaneous tracks claimed one roster team.

## Online identity policy

`ReIDConfig` contains the tunable thresholds. `AppearanceTrackMemory.resolve_frame` constructs one
score matrix for the whole frame and solves an exact maximum-weight one-to-one assignment. Because
there can be at most six identities, its dynamic program remains small even if the detector emits
extra boxes.

A candidate is rejected before scoring when:

- two known alliances conflict;
- elapsed time exceeds the memory window;
- image- or field-coordinate travel is physically impossible; or
- a new raw tracklet conflicts with the protected appearance gallery.

Accepted scores combine the best gallery appearance, residual from velocity-predicted position,
elapsed time, edge-consistent re-entry, and a small revalidated raw-ID continuity bonus. Alliance
colour contributes no positive score. Field positions are preferred when a trustworthy homography
exists; callers can also supply global camera motion to compensate image-space predictions.

Every proposed match is compared with the best global assignment that excludes it. If the total
score does not fall by the configured margin, the match is ambiguous and the layer abstains. New
identities are allocated only from `robot1` through `robot6`; once those slots are occupied,
unassigned observations remain null in the raw sidecar.

The gallery stores multiple diverse descriptors rather than an exponential moving average. A new
view must pass detector-confidence, assignment-score, appearance-consistency, and consecutive-frame
confirmation gates before it is added. A single wrong match therefore cannot overwrite the identity
anchor.

## Offline OCR and roster reconciliation

`attribute_tracks.py` still writes a new attributed file and never modifies its input. After
roster-constrained OCR voting, strong same-team fragments may merge only when they do not overlap
and their alliances are compatible. Duplicate team claims that overlap, disagree on alliance, or
have weak evidence are set back to `team: null` and marked `identity_status: review` with an
`identity_issues` reason.

The required Contract C fields and schema version remain unchanged. Stable tracks may add
`raw_track_ids`; OCR reconciliation may add `merged_track_ids`, `identity_status`, and
`identity_issues`. Full detector/tracker provenance is written separately to `tracks.raw.jsonl`
under `raw-tracklets.schema.json`, including explicit null stable IDs for abstentions.

## Tracker choice

ByteTrack remains the default. The BoT-SORT profile enables sparse-optical-flow global motion
compensation and native ReID (`model: auto`) to produce better raw tracklets. Native ReID adds cost
and can still confuse visually similar alliance partners, so it does not replace the custom global
identity or OCR/roster layers. See the [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/).
