#ifndef CONTRACT_MODELS_H
#define CONTRACT_MODELS_H

// Contracts A, B, C and D at SCHEMA_VERSION 3.
//
// These structs mirror /contracts/*.schema.json. Doc 0 is normative; if this file and a
// schema disagree, the schema wins. Do NOT build the event row from document 1 -- doc 1's
// "Output format" section used to list an eight-field subset that breaks the corrections
// layer, the accuracy comparison and the training-data export.

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// nlohmann/json does not serialise std::optional out of the box. Without this, a null
// `team` or `track_id` would be dropped from the object entirely -- but Contract B requires
// the key to be PRESENT with a null value ("Everything else is required"). Component 3's
// parser reads `raw.team ?? null`, so a missing key silently becomes null there too, which
// is exactly the kind of quiet divergence doc 0 exists to prevent.
namespace nlohmann {
template <typename T>
struct adl_serializer<std::optional<T>> {
    static void to_json(json& j, const std::optional<T>& opt) {
        if (opt.has_value()) {
            j = *opt;
        } else {
            j = nullptr;
        }
    }
    static void from_json(const json& j, std::optional<T>& opt) {
        if (j.is_null()) {
            opt = std::nullopt;
        } else {
            opt = j.get<T>();
        }
    }
};
}  // namespace nlohmann

namespace frc {

constexpr int SCHEMA_VERSION = 3;

// ---------------------------------------------------------------- closed sets
//
// Doc 0: "Closed sets. Adding a value is a contract change. Anything unrecognized is a bug,
// not a fallback."

namespace phase {
inline constexpr const char* kAuto = "auto";
inline constexpr const char* kTeleop = "teleop";
inline constexpr const char* kEndgame = "endgame";
inline constexpr const char* kUnknown = "unknown";
}  // namespace phase

namespace stage {
inline constexpr const char* kDownloading = "downloading";
inline constexpr const char* kDecoding = "decoding";
inline constexpr const char* kDetecting = "detecting";
inline constexpr const char* kTracking = "tracking";
inline constexpr const char* kOcr = "ocr";
inline constexpr const char* kEvents = "events";
}  // namespace stage

namespace error_code {
inline constexpr const char* kVideoUnavailable = "video_unavailable";
inline constexpr const char* kDownloadFailed = "download_failed";
inline constexpr const char* kRateLimited = "rate_limited";
inline constexpr const char* kNoMatchData = "no_match_data";
inline constexpr const char* kAnalysisFailed = "analysis_failed";
inline constexpr const char* kTimeout = "timeout";
inline constexpr const char* kInternal = "internal";
}  // namespace error_code

namespace gap_reason {
inline constexpr const char* kShotChange = "shot_change";
inline constexpr const char* kOcclusion = "occlusion";
inline constexpr const char* kOutOfFrame = "out_of_frame";
inline constexpr const char* kDetectionLost = "detection_lost";
}  // namespace gap_reason

namespace event_type {
inline constexpr const char* kMatchStart = "match_start";
inline constexpr const char* kMatchEnd = "match_end";
inline constexpr const char* kPhaseChange = "phase_change";
inline constexpr const char* kShotAttempt = "shot_attempt";
inline constexpr const char* kShotMade = "shot_made";
inline constexpr const char* kReload = "reload";
inline constexpr const char* kDefenseStart = "defense_start";
inline constexpr const char* kDefenseEnd = "defense_end";
inline constexpr const char* kImmobileStart = "immobile_start";
inline constexpr const char* kImmobileEnd = "immobile_end";
inline constexpr const char* kFoul = "foul";
}  // namespace event_type

/** Match-level events belong to the match, not a robot: team/track_id/field_* are all null. */
inline bool is_match_level(const std::string& type) {
    return type == event_type::kMatchStart || type == event_type::kMatchEnd ||
           type == event_type::kPhaseChange;
}

/** Only a shot can carry a `goal`. */
inline bool is_shot(const std::string& type) {
    return type == event_type::kShotAttempt || type == event_type::kShotMade;
}

// ---------------------------------------------------------------- season config
//
// Doc 0: "/contracts/seasons/<year>.json. Selected by the `season` field on the job record,
// so old footage stays analyzable after the game changes." Passed in via --season.

struct SeasonConfig {
    int season = 0;
    double field_length_ft = 0.0;
    double field_width_ft = 0.0;
    double auto_seconds = 0.0;
    double teleop_seconds = 0.0;
    double endgame_seconds = 0.0;
    std::vector<std::string> game_pieces;
    std::vector<std::string> goals;
    json point_values;
};

/**
 * Doc 0: "phase is a pure function of match-relative time and the season config. Both
 * component 1 and component 3 compute it with the same function from the same file, so they
 * cannot disagree."
 *
 * `t_match` is t_seconds minus the time of match_start. Nobody hardcodes 15, 135 or 20 --
 * hardcoding them will silently disagree with the frontend's timeline bands.
 */
inline std::string phase_at(double t_match, const SeasonConfig& cfg) {
    const double auto_end = cfg.auto_seconds;
    const double teleop_end = cfg.auto_seconds + cfg.teleop_seconds - cfg.endgame_seconds;
    const double match_end = cfg.auto_seconds + cfg.teleop_seconds;
    if (t_match < 0.0) return phase::kUnknown;
    if (t_match < auto_end) return phase::kAuto;
    if (t_match < teleop_end) return phase::kTeleop;
    if (t_match <= match_end) return phase::kEndgame;
    return phase::kUnknown;
}

// ---------------------------------------------------------------- Contract A

struct Alliances {
    std::vector<int> red;
    std::vector<int> blue;
};

struct Job {
    int schema_version = SCHEMA_VERSION;
    std::string job_id;
    std::optional<std::string> match_id;
    int season = 0;
    std::string video_id;
    std::optional<std::string> local_path;
    double start_offset = 0.0;
    // Nullable until the download reports them; guaranteed non-null once status is
    // downloaded / analyzing / complete, which is when component 1 ever sees a job.
    std::optional<double> duration;
    std::optional<double> fps;
    std::optional<int> width;
    std::optional<int> height;
    std::string status;
    std::optional<std::string> stage;
    std::optional<double> progress;
    std::optional<std::string> error_code;
    std::optional<std::string> error;
    int attempt = 1;
    std::string created_at;
    std::string updated_at;
    // Null when TBA has no data. Component 1 then falls back to raw OCR without elimination.
    std::optional<Alliances> alliances;
    json tba_score;
};

// ---------------------------------------------------------------- Contract B

struct Event {
    int schema_version = SCHEMA_VERSION;
    std::string job_id;
    std::string match_id;
    /** UUIDv4, generated HERE -- corrections reference it before anything is stored. */
    std::string event_id;
    std::optional<int> team;
    /** Null for match-level events, which belong to no robot. */
    std::optional<int> track_id;
    double t_seconds = 0.0;
    std::string phase;
    std::string event_type;
    double confidence = 0.0;
    std::optional<double> field_x;
    std::optional<double> field_y;
    /**
     * v3. Which goal a shot went into; null when unknown or not a shot.
     *
     * NOT a closed set in doc 0 -- legal values are the `goals` array of the season
     * config passed via --season, because goal names change every season. Validate
     * against that file; there is no enum here to check.
     */
    std::optional<std::string> goal;
    std::string source = "model";
};

// ---------------------------------------------------------------- Contract C

struct Box {
    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    // Optional field geometry. Pixel boxes remain valid when calibration is absent or a shot cut
    // makes the mapping unsafe; these values are populated only for a valid observed position.
    std::optional<double> field_x;
    std::optional<double> field_y;
    std::optional<double> velocity_x_ftps;
    std::optional<double> velocity_y_ftps;
    std::optional<double> speed_ftps;
    // Direction of travel, not robot body orientation. A box alone cannot reveal body heading.
    std::optional<double> motion_heading_rad;
};

/**
 * An interval where the robot was NOT observed.
 *
 * Doc 1: skipped shot-change segments "must be reported explicitly in each track's `gaps`
 * array... Without this, a four-second hole is indistinguishable from a low sample rate and
 * the frontend draws a robot gliding through footage nobody analyzed."
 */
struct Gap {
    double start = 0.0;
    double end = 0.0;
    std::string reason;
};

struct Track {
    int schema_version = SCHEMA_VERSION;
    int track_id = 0;
    std::optional<int> team;
    std::optional<std::string> alliance;
    /** Confidence in the WHOLE track's identity, separate from per-event confidence. */
    std::optional<double> team_confidence;
    /** Calibration source for optional field positions. */
    std::optional<std::string> position_source;
    std::vector<Box> boxes;
    /**
     * REQUIRED, possibly empty. Do NOT split a track at a gap: re-identification exists to
     * stitch fragments into one logical track, and splitting undoes it.
     */
    std::vector<Gap> gaps;
};

// ---------------------------------------------------------------- Contract D

struct RunResult {
    int schema_version = SCHEMA_VERSION;
    std::string job_id;
    std::string model_version;
    /** Hertz -- samples per second, not a frame interval. */
    double box_sample_rate = 0.0;
    bool homography_ok = false;
    std::optional<std::string> homography_source;
    int frames_total = 0;
    int frames_analyzed = 0;
    int frames_skipped_shot_change = 0;
    int tracks_emitted = 0;
    int events_emitted = 0;
    /** Null while the season's point values are placeholders. */
    json reconstructed_score;
    std::string started_at;
    std::string finished_at;
};

// ---------------------------------------------------------------- JSON conversion

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Alliances, red, blue)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SeasonConfig, season, field_length_ft, field_width_ft,
                                   auto_seconds, teleop_seconds, endgame_seconds, game_pieces,
                                   goals, point_values)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Job, schema_version, job_id, match_id, season, video_id,
                                   local_path, start_offset, duration, fps, width, height,
                                   status, stage, progress, error_code, error, attempt,
                                   created_at, updated_at, alliances, tba_score)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Event, schema_version, job_id, match_id, event_id, team,
                                   track_id, t_seconds, phase, event_type, confidence, field_x,
                                   field_y, goal, source)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, t, x, y, w, h, field_x, field_y,
                                   velocity_x_ftps, velocity_y_ftps, speed_ftps,
                                   motion_heading_rad)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Gap, start, end, reason)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Track, schema_version, track_id, team, alliance,
                                   team_confidence, position_source, boxes, gaps)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RunResult, schema_version, job_id, model_version,
                                   box_sample_rate, homography_ok, homography_source, frames_total,
                                   frames_analyzed, frames_skipped_shot_change, tracks_emitted,
                                   events_emitted, reconstructed_score, started_at, finished_at)

}  // namespace frc

#endif  // CONTRACT_MODELS_H
