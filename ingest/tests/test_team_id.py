"""Attribution logic, tested without a video or an OCR engine.

The scoring, the alliance decision and the vote are pure, and they are where a wrong answer would
be expensive: attributing a track to the wrong team credits one team's actions to another, and
nothing downstream would notice. So the cases here are mostly about *refusing* to answer.
"""

from ingest.scoreboard import roster_from_counts
from ingest.attribute_tracks import reconcile_ocr_fragments
from ingest.team_id import (
    MIN_CONFIDENCE,
    TrackVote,
    decide_alliance,
    score_read,
    similarity,
    tally_track,
)

ROSTER = {"red": [8242, 10308, 11244], "blue": [8557, 9601, 11281]}


class TestSimilarity:
    def test_exact_read_is_one(self):
        assert similarity("8242", "8242") == 1.0

    def test_a_dropped_digit_still_scores_well(self):
        # The commonest real failure: "824" for 8242. It should still be strong evidence.
        assert similarity("824", "8242") == 0.75

    def test_it_separates_the_two_numbers_that_start_alike(self):
        # 11244 and 11281 share their first three digits, which is exactly the confusion that
        # would silently swap two teams on the same field.
        assert similarity("11244", "11244") > similarity("11244", "11281")

    def test_empty_reads_score_nothing(self):
        assert similarity("", "8242") == 0.0
        assert similarity("8242", "") == 0.0


class TestScoreRead:
    def test_a_clear_read_picks_its_team(self):
        assert score_read("8242", ROSTER["red"]) == (8242, 1.0)

    def test_a_partial_read_still_picks(self):
        team, score = score_read("824", ROSTER["red"])
        assert team == 8242 and 0.6 <= score < 1.0

    def test_a_read_equidistant_from_two_candidates_picks_neither(self):
        # Evidence for two teams equally is evidence for neither. Returning the first would
        # manufacture confidence that was never earned.
        assert score_read("11", [1123, 1132]) is None

    def test_garbage_picks_nothing(self):
        assert score_read("777", ROSTER["red"]) is None

    def test_no_candidates_is_not_a_crash(self):
        assert score_read("8242", []) is None


class TestDecideAlliance:
    def test_a_clear_majority_wins(self):
        assert decide_alliance(["red"] * 9 + ["blue"]) == "red"

    def test_frames_that_saw_no_colour_do_not_count_against_it(self):
        assert decide_alliance([None, None, "red", "red", None]) == "red"

    def test_a_split_track_refuses_to_choose(self):
        # Half red and half blue means the track probably covers two robots. Picking one would
        # hand a whole alliance's worth of actions to the wrong side.
        assert decide_alliance(["red"] * 5 + ["blue"] * 5) is None

    def test_nothing_seen_is_none(self):
        assert decide_alliance([None, None]) is None
        assert decide_alliance([]) is None


class TestTallyTrack:
    def test_the_alliance_narrows_the_candidates(self):
        # A blue track must never collect votes for a red team, however the digits read. This is
        # a real failure that happened before the alliance was decided up front.
        vote = tally_track(["8242"] * 5, "blue", ROSTER)
        assert 8242 not in vote.tally

    def test_no_alliance_falls_back_to_all_six(self):
        # Doc 0: a job with no TBA data is still valid, and component 1 falls back to raw OCR
        # without elimination.
        vote = tally_track(["8242"] * 5, None, ROSTER)
        assert vote.tally.get(8242)

    def test_agreeing_reads_produce_a_confident_track(self):
        team, confidence = tally_track(["8242"] * 10, "red", ROSTER).resolve()
        assert team == 8242 and confidence == 1.0

    def test_one_stray_read_does_not_flip_a_track(self):
        team, confidence = tally_track(["8242"] * 10 + ["10308"], "red", ROSTER).resolve()
        assert team == 8242 and confidence > MIN_CONFIDENCE

    def test_a_split_vote_returns_null(self):
        # Half the reads say one team and half say another. There is no answer here, and
        # inventing one is worse than leaving a human to attribute the track.
        team, confidence = tally_track(["8242"] * 5 + ["10308"] * 5, "red", ROSTER).resolve()
        assert team is None
        assert confidence < MIN_CONFIDENCE

    def test_too_little_evidence_returns_null(self):
        # One good read is not a track attribution, however clean it looked.
        assert tally_track(["8242"], "red", ROSTER).resolve() == (None, 0.0)

    def test_unreadable_track_returns_null(self):
        assert tally_track([], "red", ROSTER).resolve() == (None, 0.0)

    def test_confidence_is_the_winning_share(self):
        vote = TrackVote(alliance="red", tally={8242: 3.0, 10308: 1.0})
        team, confidence = vote.resolve()
        assert team == 8242 and confidence == 0.75

    def test_narrow_vote_margin_abstains_even_at_confidence_floor(self):
        vote = TrackVote(alliance="red", tally={8242: 5.5, 10308: 4.5})
        team, confidence = vote.resolve()
        assert team is None
        assert confidence == 0.55


class TestRosterFromCounts:
    def test_the_persistent_numbers_are_the_roster(self):
        counts = {"red": {8242: 7, 10308: 7, 11244: 7, 98: 2, 33: 1},
                  "blue": {8557: 7, 9601: 7, 11281: 6}}
        assert roster_from_counts(counts, 7) == {"red": [8242, 10308, 11244],
                                                 "blue": [8557, 9601, 11281]}

    def test_a_missing_team_stays_missing(self):
        # Two teams and a gap beats three teams one of which is the best available noise.
        counts = {"red": {8242: 7, 10308: 7, 4: 1, 7: 1}, "blue": {}}
        assert roster_from_counts(counts, 7) == {"red": [8242, 10308], "blue": []}

    def test_scores_and_timers_do_not_survive(self):
        # A score is read every frame too, but it is a different number each time, so no single
        # value reaches a majority. That is the whole trick.
        counts = {"red": {8242: 6, 12: 1, 24: 1, 47: 1, 63: 1, 98: 1}, "blue": {}}
        assert roster_from_counts(counts, 6)["red"] == [8242]


class TestOcrFragmentReconciliation:
    @staticmethod
    def track(track_id, team, start, end, confidence=0.95, alliance="red"):
        return {
            "schema_version": 3,
            "track_id": track_id,
            "robot_name": f"robot{track_id}",
            "team": team,
            "alliance": alliance,
            "team_confidence": confidence,
            "boxes": [
                {"t": start, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                {"t": end, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
            ],
            "gaps": [],
            "raw_track_ids": [track_id * 10],
        }

    def test_strong_non_overlapping_same_team_fragments_merge(self):
        reconciled = reconcile_ocr_fragments([
            self.track(1, 8242, 0.0, 2.0),
            self.track(4, 8242, 4.0, 6.0),
        ], fps=10.0)
        assert len(reconciled) == 1
        assert reconciled[0]["track_id"] == 1
        assert reconciled[0]["merged_track_ids"] == [1, 4]
        assert reconciled[0]["raw_track_ids"] == [10, 40]

    def test_conflicting_team_number_ocr_never_merges_fragments(self):
        reconciled = reconcile_ocr_fragments([
            self.track(1, 8242, 0.0, 2.0),
            self.track(2, 10308, 3.0, 5.0),
        ])
        assert len(reconciled) == 2
        assert [track["team"] for track in reconciled] == [8242, 10308]
        assert all("merged_track_ids" not in track for track in reconciled)

    def test_overlapping_duplicate_ocr_is_flagged_for_review(self):
        reconciled = reconcile_ocr_fragments([
            self.track(1, 8242, 0.0, 4.0),
            self.track(2, 8242, 2.0, 5.0),
        ])
        assert len(reconciled) == 2
        assert all(track["team"] is None for track in reconciled)
        assert all(track["identity_status"] == "review" for track in reconciled)
        assert all(
            "overlapping_duplicate_team_ocr" in track["identity_issues"]
            for track in reconciled
        )
