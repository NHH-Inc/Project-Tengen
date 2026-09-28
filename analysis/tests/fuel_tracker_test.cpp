// Exercise the imported runtime directly: no duplicate tracking implementation.
#define FRC_FUEL_TRACKER_NO_MAIN
#include "../src/fuel_tracker.cpp"

static int failures = 0;
static int checks = 0;
static void check(bool ok, const std::string& what) {
    ++checks;
    std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
    if (!ok) ++failures;
}

template <typename F> static void rejects(F fn, const std::string& what) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    check(threw, what);
}

struct Scenario {
    std::map<int, FuelTrack> tracks;
    int nextId = 1, human = 0, unknown = 0, frame = 0;
    std::map<std::string, int> scores;
    std::vector<cv::Point> hub{{80, 80}, {120, 80}, {120, 120}, {80, 120}};
    void step(std::vector<FuelCandidate> pts,
              std::vector<std::pair<int, Box>> robots = {},
              std::optional<std::vector<cv::Point>> humanZone = std::nullopt) {
        updateFuelTracks(tracks, nextId, scores, human, unknown, pts, robots, hub, humanZone, frame++);
    }
    void expire() { for (int i = 0; i <= FUEL_MAX_MISSES; ++i) step({}); }
};

int main() {
    cv::Mat frame(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, {50, 50}, 7, {0, 255, 255}, -1);
    cv::circle(frame, {150, 50}, 7, {0, 0, 255}, -1);
    cv::circle(frame, {250, 50}, 25, {0, 255, 255}, -1);
    auto found = detectYellowCandidates(frame, std::nullopt);
    check(found.size() == 1 && std::abs(found[0].x - 50) <= 1,
          "isolated yellow fuel detected; red and oversized blobs rejected");
    found = detectYellowCandidates(frame, Box{40, 40, 70, 70});
    check(found.size() == 1 && found[0].x == 50 && found[0].y == 50,
          "ROI detection maps back to full-frame coordinates");
    check(detectYellowCandidates(frame, Box{400, 400, 500, 500}).empty(), "off-image ROI is empty");

    SimpleTracker robot;
    auto first = robot.update({{10, 10, 50, 50}});
    auto second = robot.update({{15, 10, 55, 50}});
    check(first[0].first == second[0].first, "robot ID persists under small motion");
    check(robot.update({}).empty(), "missing robots are not returned as current observations");
    check(robot.update({{20, 10, 60, 50}})[0].first == first[0].first,
          "robot track survives a short occlusion internally");
    for (int i = 0; i < 11; ++i) robot.update({});
    check(robot.update({{20, 10, 60, 50}})[0].first != first[0].first, "stale robot IDs retire");

    Scenario s;
    s.step({{65, 100, 5}}, {{7, {30, 80, 60, 120}}});
    s.step({{90, 100, 5}});
    for (int i = 0; i < FUEL_MAX_MISSES; ++i) s.step({});
    check(s.scores.empty(), "brief disappearance does not count prematurely");
    s.step({});
    check(s.scores["R7"] == 1 && s.tracks.empty(), "robot-owned fuel disappearing in hub counted once");
    s.expire();
    check(s.scores["R7"] == 1, "expired fuel cannot count twice");

    Scenario unknown;
    unknown.step({{100, 100, 5}});
    unknown.expire();
    check(unknown.unknown == 1 && unknown.human == 0, "first seen in hub counts as unknown, never invented human");
    Scenario human;
    human.step({{100, 100, 5}}, {}, human.hub);
    human.expire();
    check(human.human == 1 && human.unknown == 0, "explicit human zone attribution works");
    Scenario outside;
    outside.step({{30, 100, 5}});
    outside.expire();
    check(outside.unknown == 0 && outside.scores.empty(), "disappearance outside hub does not count");
    Scenario exiting;
    exiting.step({{90, 100, 5}});
    exiting.step({{130, 100, 5}});
    exiting.expire();
    check(exiting.unknown == 0, "fuel leaving hub before disappearing does not count");
    Scenario pending;
    pending.step({{100, 100, 5}});
    check(pending.unknown == 0 && pending.tracks.size() == 1, "end of clip alone cannot confirm a count");

    // YOLO raw tensor: two duplicate robot boxes and an overlapping nonrobot with a higher score.
    int dims[] = {1, 6, 10};
    cv::Mat output(3, dims, CV_32F, cv::Scalar(0));
    for (int b = 0; b < 3; ++b) {
        output.ptr<float>(0, 0)[b] = 320;
        output.ptr<float>(0, 1)[b] = 320;
        output.ptr<float>(0, 2)[b] = 100;
        output.ptr<float>(0, 3)[b] = 50;
    }
    output.ptr<float>(0, 4)[0] = .9f;
    output.ptr<float>(0, 4)[1] = .8f;
    output.ptr<float>(0, 5)[2] = .99f;
    auto boxes = decodeYoloOutput(output, {1280, 720}, .5, 0, 140, 0);
    check(boxes.size() == 1 && boxes[0].classId == 0, "robot-only NMS suppresses duplicates without losing robots to other classes");
    check(boxes.size() == 1 && boxes[0].box.x1 == 540 && boxes[0].box.y1 == 310 &&
          boxes[0].box.x2 == 740 && boxes[0].box.y2 == 410, "YOLO letterbox coordinates decode correctly");
    rejects([&] { decodeYoloOutput(cv::Mat::zeros(10, 6, CV_32F), {100, 100}, 1, 0, 0, 0); },
            "unsupported output rank fails clearly");
    rejects([&] { decodeYoloOutput(output, {100, 100}, 1, 0, 0, 2); }, "invalid model class fails clearly");
    rejects([&] { readPolygon(nlohmann::json::array({{0, 0}, {1, 1}}), {100, 100}); }, "short polygon rejected");
    rejects([&] { readPolygon(nlohmann::json::array({{0, 0}, {50, 50}, {101, 0}}), {100, 100}); },
            "out-of-image polygon rejected");
    rejects([&] { readPolygon(nlohmann::json::array({{0, 0}, {10, 10}, {20, 20}}), {100, 100}); },
            "degenerate polygon rejected");

    // Characterize limitations separately: these are measured failures, not promises of accuracy.
    SimpleTracker fast;
    std::vector<int> ids;
    for (int i = 0; i < 9; ++i) ids.push_back(fast.update({{i * 35, 0, i * 35 + 40, 40}}).back().first);
    std::sort(ids.begin(), ids.end());
    std::cout << "LIMITATION fast robot (35px/frame, 40px width): "
              << std::distance(ids.begin(), std::unique(ids.begin(), ids.end())) << " IDs for 1 robot\n";
    Scenario occlusion;
    occlusion.step({{100, 100, 5}});
    occlusion.expire();
    std::cout << "LIMITATION occlusion in hub: " << occlusion.unknown << " inferred count without proof of a basket\n";
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
