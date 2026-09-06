// Component 1 entry point -- Contract D at SCHEMA_VERSION 3.
//
//     analysis --job <path/to/job.json> --season <path/to/season.json> --out <output/dir>
//
// On success: exit 0, and <out>/events.jsonl, <out>/tracks.jsonl, <out>/result.json exist.
// On failure: nonzero exit, human-readable reason on stderr, and an error_code enum value as
// the LAST LINE of stderr so component 2 can classify without parsing prose.
// Progress goes to stdout as one JSON object per line: {"progress": 0.42, "stage": "tracking"}
//
// The detection/tracking/OCR pipeline is not implemented yet. What is implemented is the
// contract surface, so component 2 and component 3 can integrate against a real binary.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "ContractModels.h"
#include "Homography.h"
#include "IoUTracker.h"
#include "RobotDetector.h"

namespace fs = std::filesystem;

namespace {

/** Progress line. `stage` MUST be a value from the stage enum -- component 3 renders it. */
void print_progress(double progress, const char* stage) {
    const json msg = {{"progress", progress}, {"stage", stage}};
    std::cout << msg.dump() << std::endl;
}

/**
 * Fail the way Contract D says to: reason on stderr, then the error_code alone on the last
 * line. Component 2 reads that last line rather than pattern-matching the message.
 */
[[nodiscard]] int fail(const std::string& reason, const char* code) {
    std::cerr << reason << std::endl;
    std::cerr << code << std::endl;
    return 1;
}

/** UUIDv4. Contract B: event_id is generated here, because corrections reference it. */
std::string uuid4() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s(36, '-');
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) continue;
        s[i] = kHex[rng() & 0xF];
    }
    s[14] = '4';                                  // version
    s[19] = kHex[(rng() & 0x3) | 0x8];            // variant
    return s;
}

std::string iso_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool is_camera_cut(const cv::Mat& bgr, cv::Mat& previous_histogram, double threshold) {
    cv::Mat thumbnail;
    cv::resize(bgr, thumbnail, cv::Size(96, 54), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat hsv;
    cv::cvtColor(thumbnail, hsv, cv::COLOR_BGR2HSV);
    const int channels[] = {0, 1};
    const int histogram_size[] = {12, 8};
    const float hue_range[] = {0.0F, 180.0F};
    const float saturation_range[] = {0.0F, 256.0F};
    const float* ranges[] = {hue_range, saturation_range};
    cv::Mat histogram;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), histogram, 2, histogram_size, ranges, true, false);
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    if (previous_histogram.empty()) {
        previous_histogram = histogram;
        return false;
    }
    const double distance = cv::compareHist(previous_histogram, histogram, cv::HISTCMP_BHATTACHARYYA);
    previous_histogram = histogram;
    return distance >= threshold;
}

frc::Event match_event(const frc::Job& job, const frc::SeasonConfig& season, double t_seconds,
                       const char* type) {
    frc::Event event;
    event.job_id = job.job_id;
    event.match_id = *job.match_id;
    event.event_id = uuid4();
    event.t_seconds = t_seconds;
    event.phase = frc::phase_at(t_seconds, season);
    event.event_type = type;
    event.confidence = 1.0;
    event.source = "model";
    return event;
}

bool gap_between(const frc::Track& track, double start, double end) {
    for (const auto& gap : track.gaps) {
        if (gap.start >= start && gap.end <= end) return true;
    }
    return false;
}

void annotate_field_motion(std::vector<frc::Track>& tracks,
                           const frc::vision::Homography& homography,
                           int image_width, int image_height) {
    for (auto& track : tracks) {
        std::optional<std::pair<double, double>> previous;
        double previous_time = 0.0;
        for (auto& box : track.boxes) {
            const double pixel_x = (box.x + box.w / 2.0) * image_width;
            const double pixel_y = (box.y + box.h) * image_height;
            const auto position = homography.box_to_field(
                box.x, box.y, box.w, box.h, image_width, image_height);
            if (!position || !homography.on_field(pixel_x, pixel_y) ||
                !std::isfinite(position->first) || !std::isfinite(position->second)) {
                previous.reset();
                continue;
            }

            box.field_x = position->first;
            box.field_y = position->second;
            if (previous.has_value() && !gap_between(track, previous_time, box.t)) {
                const double dt = box.t - previous_time;
                if (dt > 0.0) {
                    const double vx = (position->first - previous->first) / dt;
                    const double vy = (position->second - previous->second) / dt;
                    const double speed = std::hypot(vx, vy);
                    if (speed <= frc::vision::kMaxPlausibleFtps) {
                        box.velocity_x_ftps = vx;
                        box.velocity_y_ftps = vy;
                        box.speed_ftps = speed;
                        box.motion_heading_rad = std::atan2(vy, vx);
                    } else {
                        // Keep the position, but prevent an identity swap from contaminating the
                        // next finite difference too.
                        previous.reset();
                    }
                }
            }
            previous = position;
            previous_time = box.t;
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string job_path;
    std::string season_path;
    std::string out_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--job" && i + 1 < argc) {
            job_path = argv[++i];
        } else if (arg == "--season" && i + 1 < argc) {
            season_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }

    if (job_path.empty() || out_dir.empty() || season_path.empty()) {
        return fail(
            "Usage: analysis --job <job.json> --season <season.json> --out <dir>",
            frc::error_code::kInternal);
    }

    const std::string started_at = iso_now();

    try {
        std::ifstream job_file(job_path);
        if (!job_file.is_open()) {
            return fail("Failed to open job file: " + job_path, frc::error_code::kInternal);
        }
        json job_json;
        job_file >> job_json;
        const auto job = job_json.get<frc::Job>();

        if (job.schema_version != frc::SCHEMA_VERSION) {
            return fail("Job schema_version " + std::to_string(job.schema_version) +
                            " != " + std::to_string(frc::SCHEMA_VERSION),
                        frc::error_code::kInternal);
        }

        // Contract B requires a string match_id on every event, so a job that never resolved
        // one cannot produce a valid event stream.
        if (!job.match_id.has_value()) {
            return fail("Job has no match_id; cannot attribute events to a match",
                        frc::error_code::kNoMatchData);
        }
        // Contract A guarantees these are non-null once status is downloaded or later, which
        // is the only state component 2 ever hands us a job in.
        if (!job.local_path.has_value() || !job.fps.has_value() || !job.duration.has_value()) {
            return fail("Job is missing media metadata (local_path/fps/duration)",
                        frc::error_code::kInternal);
        }

        std::ifstream season_file(season_path);
        if (!season_file.is_open()) {
            return fail("Failed to open season config: " + season_path,
                        frc::error_code::kInternal);
        }
        json season_json;
        season_file >> season_json;
        const auto season = season_json.get<frc::SeasonConfig>();
        if (season.season != job.season) {
            return fail("Season config is for " + std::to_string(season.season) +
                            " but the job says " + std::to_string(job.season),
                        frc::error_code::kInternal);
        }

        fs::create_directories(out_dir);

        cv::VideoCapture video(*job.local_path);
        if (!video.isOpened()) {
            return fail("Failed to open video: " + *job.local_path,
                        frc::error_code::kVideoUnavailable);
        }

        // Model configuration is deliberately local: Contract A jobs stay portable and never
        // name a machine-specific model file. With no config, this remains an honest decode-only
        // run with zero tracks -- the old diagnostic box is intentionally gone.
        const frc::vision::DetectorConfig detector_config = frc::vision::load_detector_config();
        // Optional. Absent or untrustworthy leaves homography_ok false and field
        // coordinates null, which is the honest output rather than a guessed one.
        const auto homography = frc::vision::load_homography(
            season.field_length_ft, season.field_width_ft);
        const frc::vision::RobotDetector detector(detector_config);
        const bool detector_enabled = detector.enabled();

        // Which part of each frame is the field. Broadcasts that composite a second camera view
        // show the same six robots twice, and without this each becomes two tracks and two team
        // numbers for a human to type. Keyed by the source's stem; an unlisted source keeps the
        // whole frame. See ingest/collection/calibrate_region.py.
        const auto view_region = frc::vision::region_for(detector.config(), *job.local_path);

        print_progress(0.05, frc::stage::kDecoding);
        cv::Mat frame;
        int decoded_frames = 0;
        int decoded_width = 0;
        int decoded_height = 0;
        int frames_analyzed = 0;
        int frames_skipped_shot_change = 0;
        const double reported_fps = video.get(cv::CAP_PROP_FPS);
        const double decode_fps = reported_fps > 0.0 ? reported_fps : *job.fps;
        if (decode_fps <= 0.0) {
            return fail("Video reports an invalid frame rate", frc::error_code::kAnalysisFailed);
        }
        const int sample_interval_frames = std::max(
            1, static_cast<int>(std::llround(decode_fps / detector_config.sample_rate_hz)));
        const double box_sample_rate = decode_fps / sample_interval_frames;
        frc::vision::IoUTracker tracker;
        cv::Mat previous_histogram;
        while (video.read(frame) && !frame.empty()) {
            if (decoded_frames == 0) {
                decoded_width = frame.cols;
                decoded_height = frame.rows;
            }
            if (detector_enabled && decoded_frames % sample_interval_frames == 0) {
                const double t_seconds = decoded_frames / decode_fps;
                const bool camera_cut = is_camera_cut(
                    frame, previous_histogram, detector_config.shot_change_threshold);
                if (camera_cut) {
                    ++frames_skipped_shot_change;
                    tracker.update(t_seconds, {}, true);
                } else {
                    print_progress(0.25, frc::stage::kDetecting);
                    const auto detections = detector.infer(frame, view_region);
                    ++frames_analyzed;
                    print_progress(0.55, frc::stage::kTracking);
                    tracker.update(t_seconds, detections, false);
                }
            }
            ++decoded_frames;
        }
        if (decoded_width <= 0 || decoded_height <= 0 || decoded_frames <= 0) {
            return fail("Video contains no decodable frames: " + *job.local_path,
                        frc::error_code::kAnalysisFailed);
        }

        const double match_end_t = std::max(0.0, (decoded_frames - 1) / decode_fps);
        tracker.finish(match_end_t);
        std::vector<frc::Track> tracks = tracker.tracks();
        if (homography.has_value()) {
            annotate_field_motion(tracks, *homography, decoded_width, decoded_height);
            for (auto& track : tracks) track.position_source = homography->source();
        }
        // Action/OCR inference needs reviewed data. Until it exists, emit only safe match-level
        // events; their required nullable fields are supplied by ContractModels.h.
        std::vector<frc::Event> events;
        events.push_back(match_event(job, season, 0.0, frc::event_type::kMatchStart));
        events.push_back(match_event(job, season, match_end_t, frc::event_type::kMatchEnd));
        print_progress(0.80, frc::stage::kOcr);
        print_progress(0.95, frc::stage::kEvents);

        std::ofstream events_file(fs::path(out_dir) / "events.jsonl");
        for (const auto& e : events) {
            events_file << json(e).dump() << "\n";
        }

        std::ofstream tracks_file(fs::path(out_dir) / "tracks.jsonl");
        for (const auto& t : tracks) {
            tracks_file << json(t).dump() << "\n";
        }

        frc::RunResult result;
        result.job_id = job.job_id;
        result.model_version = detector_enabled ? detector_config.model_version : "detector-unconfigured";
        result.box_sample_rate = box_sample_rate;
        result.homography_ok = homography.has_value();
        if (homography.has_value()) result.homography_source = homography->source();
        result.frames_total = decoded_frames;
        result.frames_analyzed = frames_analyzed;
        result.frames_skipped_shot_change = frames_skipped_shot_change;
        result.tracks_emitted = static_cast<int>(tracks.size());
        result.events_emitted = static_cast<int>(events.size());
        // Null while the season's point values are zero placeholders. Doc 0: "Do not invent
        // values to make a test pass."
        result.reconstructed_score = nullptr;
        result.started_at = started_at;
        result.finished_at = iso_now();

        std::ofstream result_file(fs::path(out_dir) / "result.json");
        result_file << json(result).dump(2) << std::endl;

        print_progress(1.0, frc::stage::kEvents);
        return 0;

    } catch (const json::exception& e) {
        return fail(std::string("Malformed JSON: ") + e.what(), frc::error_code::kInternal);
    } catch (const std::exception& e) {
        return fail(std::string("Analysis failed: ") + e.what(),
                    frc::error_code::kAnalysisFailed);
    }
}
