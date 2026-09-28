// fuel_tracker.cpp
//
// Imported C++ port of the Python FRC robot+fuel runtime tracker.
// Integrated build, command-line usage and limitations: docs/FUEL-TRACKER.md.
//
// Build (once you have OpenCV 4.x with the dnn module installed):
//   cmake -S analysis -B analysis/build && cmake --build analysis/build
//
// Requirements / notes on what changed vs. the Python version:
//   1. Ultralytics' `model.track(persist=True)` runs ByteTrack internally to give
//      robots persistent IDs across frames. There is no equivalent "just works"
//      dependency-free C++ call, so this file implements a small IOU-based
//      greedy tracker (SimpleTracker) that assigns/keeps robot IDs frame-to-frame.
//      It has no Kalman motion model and can lose IDs under fast motion.
//      See the measured limitations in docs/FUEL-TRACKER.md.
//   2. YOLO inference is done via cv::dnn::readNetFromONNX. Your model must be
//      exported first; pass the export image size through --imgsz.
//      This code assumes the standard Ultralytics v8 export output shape
//      [1, 4+num_classes, num_boxes] (no separate objectness column). If your
//      export differs, adjust decodeYoloOutput().
//   3. Yellow-fuel detection and greedy association preserve the supplied
//      implementation. CLI, validation, reporting, and counting fixes are
//      documented in docs/FUEL-TRACKER.md.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------
// Runtime tuning (mirrors Python constants)
// -----------------------------
static const std::string WINDOW_NAME = "FRC Robot + Fuel Runtime";

static const int    YOLO_IMGSZ            = 640;
static const float  YOLO_CONF             = 0.18f;
static const float  YOLO_NMS_IOU          = 0.5f;
static const int    ROBOT_CLASS_ID        = 0;   // custom model class id for "robot"
static const double FUEL_MAX_ASSOC_DIST   = 55.0;
static const int    FUEL_MAX_MISSES       = 5;
static const size_t FUEL_TRAIL_LEN        = 10;
static const int    HUB_MARGIN            = 35;
static const int    FRAME_SCALE_MAX_W     = 1500;
static const int    FRAME_SCALE_MAX_H     = 950;
static const bool   DEFAULT_HUMAN_IF_UNKNOWN = false;
static const bool   SHOW_FUEL_LABELS      = true;
static const bool   SHOW_ROBOT_IDS        = true;
static const double ROBOT_MATCH_IOU_MIN   = 0.3; // for SimpleTracker association

// Yellow fuel HSV range - tune if needed
static const cv::Scalar HSV_LOW(15, 90, 90);
static const cv::Scalar HSV_HIGH(40, 255, 255);

// -----------------------------
// Small geometry helpers
// -----------------------------
struct Box {
    int x1, y1, x2, y2;
    double iou(const Box& o) const {
        int ix1 = std::max(x1, o.x1), iy1 = std::max(y1, o.y1);
        int ix2 = std::min(x2, o.x2), iy2 = std::min(y2, o.y2);
        int iw = std::max(0, ix2 - ix1), ih = std::max(0, iy2 - iy1);
        double inter = static_cast<double>(iw) * ih;
        double a1 = static_cast<double>(x2 - x1) * (y2 - y1);
        double a2 = static_cast<double>(o.x2 - o.x1) * (o.y2 - o.y1);
        double uni = a1 + a2 - inter;
        return uni <= 0 ? 0.0 : inter / uni;
    }
};

static cv::Point centerOfBox(const Box& b) {
    return cv::Point((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2);
}

static Box clampBox(double x1, double y1, double x2, double y2, const cv::Size& shape) {
    int ix1 = static_cast<int>(std::round(x1));
    int iy1 = static_cast<int>(std::round(y1));
    int ix2 = static_cast<int>(std::round(x2));
    int iy2 = static_cast<int>(std::round(y2));
    int w = shape.width, h = shape.height;
    ix1 = std::clamp(ix1, 0, w - 1);
    ix2 = std::clamp(ix2, 0, w - 1);
    iy1 = std::clamp(iy1, 0, h - 1);
    iy2 = std::clamp(iy2, 0, h - 1);
    if (ix2 <= ix1) ix2 = std::min(w - 1, ix1 + 1);
    if (iy2 <= iy1) iy2 = std::min(h - 1, iy1 + 1);
    return {ix1, iy1, ix2, iy2};
}

static bool pointInPoly(const cv::Point2f& pt, const std::vector<cv::Point>& poly) {
    if (poly.size() < 3) return false;
    double res = cv::pointPolygonTest(poly, pt, false);
    return res >= 0;
}

static double pointToBoxDistance(const cv::Point2f& pt, const Box& b) {
    double cx = std::clamp<double>(pt.x, b.x1, b.x2);
    double cy = std::clamp<double>(pt.y, b.y1, b.y2);
    return std::hypot(pt.x - cx, pt.y - cy);
}

static Box polyBBox(const std::vector<cv::Point>& poly) {
    cv::Rect r = cv::boundingRect(poly);
    return {r.x, r.y, r.x + r.width, r.y + r.height};
}

static void drawText(cv::Mat& img, const std::string& text, cv::Point xy,
                      cv::Scalar color = cv::Scalar(255, 255, 255),
                      double scale = 0.6, int thickness = 2, bool bg = true) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    if (bg) {
        cv::rectangle(img,
                       cv::Point(xy.x, xy.y - ts.height - 6),
                       cv::Point(xy.x + ts.width + 4, xy.y + 4),
                       cv::Scalar(0, 0, 0), -1);
    }
    cv::putText(img, text, cv::Point(xy.x + 2, xy.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

static cv::Mat fitFrame(const cv::Mat& frame, double& scaleOut,
                         int maxW = FRAME_SCALE_MAX_W, int maxH = FRAME_SCALE_MAX_H) {
    double scale = std::min({static_cast<double>(maxW) / frame.cols,
                              static_cast<double>(maxH) / frame.rows, 1.0});
    cv::Mat out;
    cv::resize(frame, out, cv::Size(static_cast<int>(frame.cols * scale),
                                     static_cast<int>(frame.rows * scale)));
    scaleOut = scale;
    return out;
}

// -----------------------------
// Video / model discovery
// -----------------------------
static fs::path findVideo(const fs::path& projectDir) {
    fs::path videosDir = projectDir / "videos";
    std::vector<fs::path> roots = {projectDir};
    if (fs::exists(videosDir)) roots.push_back(videosDir);

    std::vector<std::string> exts = {".mp4", ".MP4", ".mov", ".MOV", ".m4v", ".M4V"};
    std::vector<fs::path> candidates;
    for (auto& root : roots) {
        if (!fs::exists(root)) continue;
        for (auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                candidates.push_back(entry.path());
            }
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("No video found in project or videos directory");
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
}

static fs::path findBestModel(const fs::path& projectDir) {
    fs::path runsDir = projectDir / "runs" / "detect";
    std::vector<fs::path> weights;
    if (fs::exists(runsDir)) {
        for (auto& entry : fs::recursive_directory_iterator(runsDir)) {
            if (entry.path().filename() == "best.onnx" &&
                entry.path().parent_path().filename() == "weights") {
                weights.push_back(entry.path());
            }
        }
    }
    if (weights.empty()) {
        fs::path direct = projectDir / "best.onnx";
        if (fs::exists(direct)) return direct;
        throw std::runtime_error(
            "Could not find best.onnx. Export your best.pt first: "
            "yolo export model=best.pt format=onnx");
    }
    std::sort(weights.begin(), weights.end(), [](const fs::path& a, const fs::path& b) {
        return fs::last_write_time(a) > fs::last_write_time(b);
    });
    return weights.front();
}

// -----------------------------
// Yellow fuel detection
// -----------------------------
struct FuelCandidate { int x, y, r; };

static std::vector<FuelCandidate> detectYellowCandidates(
        const cv::Mat& frame, std::optional<Box> searchRect) {
    cv::Mat roi;
    int offX = 0, offY = 0;
    if (searchRect) {
        cv::Rect r(searchRect->x1, searchRect->y1,
                   searchRect->x2 - searchRect->x1, searchRect->y2 - searchRect->y1);
        r = r & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width <= 0 || r.height <= 0) return {};
        roi = frame(r);
        offX = r.x; offY = r.y;
    } else {
        roi = frame;
    }

    cv::Mat hsv, mask;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, HSV_LOW, HSV_HIGH, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::medianBlur(mask, mask, 5);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<FuelCandidate> pts;
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < 10 || area > 350) continue;
        cv::Point2f center; float radius;
        cv::minEnclosingCircle(c, center, radius);
        if (radius < 2 || radius > 18) continue;
        pts.push_back({static_cast<int>(center.x) + offX,
                        static_cast<int>(center.y) + offY,
                        static_cast<int>(radius)});
    }
    return pts;
}

// -----------------------------
// Polygon selection UI
// -----------------------------
static std::vector<cv::Point> g_polyPoints;

static void onMousePoly(int event, int x, int y, int /*flags*/, void* /*param*/) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_polyPoints.emplace_back(x, y);
    }
}

// Returns std::nullopt if cancelled (Esc/q). Caller decides whether that's fatal.
static std::optional<std::vector<cv::Point>> selectPolygon(const cv::Mat& baseImg,
                                                             const std::string& title) {
    g_polyPoints.clear();
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::setMouseCallback(title, onMousePoly, nullptr);

    while (true) {
        cv::Mat vis = baseImg.clone();
        drawText(vis, title + ": left-click points | U undo | Enter finish | Esc cancel",
                  cv::Point(10, 28), cv::Scalar(0, 255, 0), 0.65);
        if (!g_polyPoints.empty()) {
            for (size_t i = 0; i < g_polyPoints.size(); ++i) {
                cv::circle(vis, g_polyPoints[i], 4, cv::Scalar(0, 255, 255), -1);
                if (i > 0) cv::line(vis, g_polyPoints[i - 1], g_polyPoints[i],
                                     cv::Scalar(0, 255, 255), 2);
            }
            if (g_polyPoints.size() > 2) {
                cv::line(vis, g_polyPoints.back(), g_polyPoints.front(),
                          cv::Scalar(0, 255, 255), 1);
            }
        }
        cv::imshow(title, vis);
        int key = cv::waitKey(20) & 0xFF;
        if (key == 13 || key == 10) { // Enter
            if (g_polyPoints.size() >= 3) {
                auto result = g_polyPoints;
                cv::destroyWindow(title);
                return result;
            }
        } else if (key == 27 || key == 'q') { // Esc or q
            cv::destroyWindow(title);
            return std::nullopt;
        } else if (key == 'u' || key == 'U') {
            if (!g_polyPoints.empty()) g_polyPoints.pop_back();
        }
    }
}

// Returns {frame_index, frame}. Throws on cancel.
static std::pair<int, cv::Mat> chooseSetupFrame(const fs::path& videoPath) {
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + videoPath.string());
    }
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int idx = 0;
    cv::namedWindow("Choose Setup Frame", cv::WINDOW_NORMAL);

    while (true) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat frame;
        if (!cap.read(frame)) break;

        double scale;
        cv::Mat display = fitFrame(frame, scale);
        drawText(display, "A/D: +/-15 frames | J/L: +/-60 | Space/Enter: choose | Q/Esc: quit",
                  cv::Point(10, 28), cv::Scalar(0, 255, 0), 0.65);
        drawText(display, "Frame " + std::to_string(idx) + "/" + std::to_string(std::max(total - 1, 0)),
                  cv::Point(10, 60), cv::Scalar(255, 255, 255), 0.7);
        cv::imshow("Choose Setup Frame", display);
        int key = cv::waitKey(0) & 0xFF;

        if (key == ' ' || key == 13 || key == 10) {
            cap.release();
            cv::destroyWindow("Choose Setup Frame");
            return {idx, frame};
        } else if (key == 'a' || key == 'A') {
            idx = std::max(0, idx - 15);
        } else if (key == 'd' || key == 'D') {
            idx = std::min(std::max(total - 1, 0), idx + 15);
        } else if (key == 'j' || key == 'J') {
            idx = std::max(0, idx - 60);
        } else if (key == 'l' || key == 'L') {
            idx = std::min(std::max(total - 1, 0), idx + 60);
        } else if (key == 27 || key == 'q' || key == 'Q') {
            cap.release();
            cv::destroyWindow("Choose Setup Frame");
            throw std::runtime_error("Cancelled by user.");
        }
    }
    cap.release();
    cv::destroyWindow("Choose Setup Frame");
    throw std::runtime_error("Could not choose setup frame.");
}

// -----------------------------
// Fuel tracks
// -----------------------------
struct FuelTrack {
    int tid;
    cv::Point center;
    int radius;
    std::deque<cv::Point> trail;
    int birthFrame;
    int lastSeen;
    int misses = 0;
    bool enteredHub = false;
    bool confirmed = false;
    std::optional<int> ownerId;
    std::optional<std::string> ownerLabel;
    bool human = false;

    FuelTrack(int id, cv::Point pt, int r, int frameIdx)
        : tid(id), center(pt), radius(r), birthFrame(frameIdx), lastSeen(frameIdx) {
        trail.push_back(center);
    }

    void update(cv::Point pt, int r, int frameIdx) {
        center = pt;
        radius = r;
        trail.push_back(center);
        if (trail.size() > FUEL_TRAIL_LEN) trail.pop_front();
        lastSeen = frameIdx;
        misses = 0;
    }

    cv::Point2f velocity() const {
        if (trail.size() < 2) return {0.f, 0.f};
        auto it2 = trail.rbegin();
        cv::Point p2 = *it2;
        cv::Point p1 = *(++it2);
        return {static_cast<float>(p2.x - p1.x), static_cast<float>(p2.y - p1.y)};
    }

    cv::Point estimatedOrigin(int backSteps = 4) const {
        if (trail.size() < 2) return center;
        cv::Point2f v = velocity();
        return {static_cast<int>(center.x - v.x * backSteps),
                static_cast<int>(center.y - v.y * backSteps)};
    }
};

// -----------------------------
// Simple IOU-based robot tracker
// (stand-in for ultralytics model.track(persist=True) / ByteTrack)
// -----------------------------
class SimpleTracker {
public:
    struct TrackedRobot { int id; Box box; int misses = 0; };
    int tracksCreated() const { return nextId_ - 1; }

    std::vector<std::pair<int, Box>> update(const std::vector<Box>& detections) {
        std::vector<bool> used(detections.size(), false);

        // Greedy match existing tracks to detections by best IOU.
        for (auto& tr : tracks_) {
            int bestIdx = -1;
            double bestIou = ROBOT_MATCH_IOU_MIN;
            for (size_t i = 0; i < detections.size(); ++i) {
                if (used[i]) continue;
                double iouVal = tr.box.iou(detections[i]);
                if (iouVal > bestIou) {
                    bestIou = iouVal;
                    bestIdx = static_cast<int>(i);
                }
            }
            if (bestIdx >= 0) {
                tr.box = detections[bestIdx];
                tr.misses = 0;
                used[bestIdx] = true;
            } else {
                tr.misses++;
            }
        }

        // Spawn new tracks for unmatched detections.
        for (size_t i = 0; i < detections.size(); ++i) {
            if (used[i]) continue;
            tracks_.push_back({nextId_++, detections[i], 0});
        }

        // Drop stale tracks.
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                          [](const TrackedRobot& t) { return t.misses > 10; }),
                      tracks_.end());

        std::vector<std::pair<int, Box>> result;
        for (auto& t : tracks_) {
            if (t.misses == 0) result.emplace_back(t.id, t.box);
        }
        return result;
    }

private:
    std::vector<TrackedRobot> tracks_;
    int nextId_ = 1;
};

// -----------------------------
// YOLO ONNX inference (Ultralytics v8-style output)
// -----------------------------
struct Detection { Box box; int classId; float conf; };

static std::vector<Detection> decodeYoloOutput(const cv::Mat& output, const cv::Size& frameSize,
                                                double scale, int padX, int padY, int robotClass);

static std::vector<Detection> runYoloInference(cv::dnn::Net& net, const cv::Mat& frame,
                                                int imageSize, int robotClass) {
    // Letterbox resize to a square imageSize x imageSize canvas.
    int w = frame.cols, h = frame.rows;
    double scale = std::min(static_cast<double>(imageSize) / w,
                             static_cast<double>(imageSize) / h);
    int newW = static_cast<int>(std::round(w * scale));
    int newH = static_cast<int>(std::round(h * scale));
    int padX = (imageSize - newW) / 2;
    int padY = (imageSize - newH) / 2;

    cv::Mat resized, canvas;
    cv::resize(frame, resized, cv::Size(newW, newH));
    canvas = cv::Mat(imageSize, imageSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(imageSize, imageSize),
                                           cv::Scalar(), true, false);
    net.setInput(blob);
    cv::Mat output = net.forward();
    return decodeYoloOutput(output, frame.size(), scale, padX, padY, robotClass);
}

static std::vector<Detection> decodeYoloOutput(const cv::Mat& output, const cv::Size& frameSize,
                                                double scale, int padX, int padY, int robotClass) {
    if (output.dims != 3 || output.size[0] != 1 || output.size[1] < 5 ||
        output.size[1] > 512 || output.size[2] <= output.size[1] ||
        output.type() != CV_32F || !output.isContinuous()) {
        throw std::runtime_error("Expected raw YOLO output [1, 4+classes, boxes]; export without NMS.");
    }
    const int numClasses = output.size[1] - 4;
    if (robotClass < 0 || robotClass >= numClasses)
        throw std::runtime_error("--robot-class is outside the model class range.");

    // Reshape to [4+numClasses, N] then transpose to [N, 4+numClasses]
    int rows = output.size[1];
    int numBoxes = output.size[2];
    cv::Mat data(rows, numBoxes, CV_32F, const_cast<float*>(output.ptr<float>()));
    cv::Mat dataT;
    cv::transpose(data, dataT); // [N, 4+numClasses]

    std::vector<cv::Rect> boxesForNms;
    std::vector<float> scoresForNms;
    std::vector<int> classIds;

    for (int i = 0; i < numBoxes; ++i) {
        const float* row = dataT.ptr<float>(i);
        float cx = row[0], cy = row[1], bw = row[2], bh = row[3];

        int bestClass = -1;
        float bestScore = 0.f;
        for (int c = 0; c < numClasses; ++c) {
            float s = row[4 + c];
            if (s > bestScore) { bestScore = s; bestClass = c; }
        }
        if (bestScore < YOLO_CONF || bestClass != robotClass) continue;
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(bw) ||
            !std::isfinite(bh) || bw <= 0 || bh <= 0) continue;

        // Undo letterbox to original frame coordinates.
        double x1 = (cx - bw / 2.0 - padX) / scale;
        double y1 = (cy - bh / 2.0 - padY) / scale;
        double x2 = (cx + bw / 2.0 - padX) / scale;
        double y2 = (cy + bh / 2.0 - padY) / scale;

        boxesForNms.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                  cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        scoresForNms.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxesForNms, scoresForNms, YOLO_CONF, YOLO_NMS_IOU, keep);

    std::vector<Detection> results;
    for (int idx : keep) {
        cv::Rect r = boxesForNms[idx];
        Box b = clampBox(r.x, r.y, r.x + r.width, r.y + r.height, frameSize);
        if (b.x2 > b.x1 && b.y2 > b.y1)
            results.push_back({b, classIds[idx], scoresForNms[idx]});
    }
    return results;
}

// The score is a disappearance-in-hub heuristic, not a verified made basket.
static void updateFuelTracks(std::map<int, FuelTrack>& fuelTracks, int& nextFuelId,
        std::map<std::string, int>& scoreByRobot, int& humanScore, int& unattributed,
        const std::vector<FuelCandidate>& pts, const std::vector<std::pair<int, Box>>& robotBoxes,
        const std::vector<cv::Point>& hubPoly,
        const std::optional<std::vector<cv::Point>>& humanPoly, int frameIdx) {
    // ---- Associate fuel detections to existing tracks ----
    std::vector<bool> used(pts.size(), false);
    for (auto& [tid, tr] : fuelTracks) {
        int bestI = -1;
        double bestD = FUEL_MAX_ASSOC_DIST;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (used[i]) continue;
            double d = std::hypot(pts[i].x - tr.center.x, pts[i].y - tr.center.y);
            if (d < bestD) { bestD = d; bestI = static_cast<int>(i); }
        }
        if (bestI >= 0) {
            used[bestI] = true;
            tr.update({pts[bestI].x, pts[bestI].y}, pts[bestI].r, frameIdx);
            if (pointInPoly(tr.center, hubPoly)) tr.enteredHub = true;
        } else {
            tr.misses++;
        }
    }

    // ---- Create new tracks for leftover detections ----
    for (size_t i = 0; i < pts.size(); ++i) {
        if (used[i]) continue;
        FuelTrack tr(nextFuelId, {pts[i].x, pts[i].y}, pts[i].r, frameIdx);
        tr.enteredHub = pointInPoly(tr.center, hubPoly);

        cv::Point origin = tr.estimatedOrigin(4);
        std::optional<int> bestOwner;
        double bestDist = 1e9;
        for (auto& [rid, box] : robotBoxes) {
            double d = pointToBoxDistance(origin, box);
            if (d < bestDist) { bestDist = d; bestOwner = rid; }
        }
        if (bestOwner && bestDist < 140.0) {
            tr.ownerId = bestOwner;
            tr.ownerLabel = "R" + std::to_string(*bestOwner);
        } else if (humanPoly && pointInPoly(origin, *humanPoly)) {
            tr.human = true;
            tr.ownerLabel = "HP";
        }

        fuelTracks.emplace(nextFuelId, std::move(tr));
        nextFuelId++;
    }

    // ---- Confirm / remove stale tracks ----
    for (auto it = fuelTracks.begin(); it != fuelTracks.end(); ) {
        FuelTrack& tr = it->second;
        if (tr.misses > FUEL_MAX_MISSES) {
            if (tr.enteredHub && pointInPoly(tr.center, hubPoly) && !tr.confirmed) {
                tr.confirmed = true;
                if (tr.ownerId) {
                    scoreByRobot[*tr.ownerLabel]++;
                } else if (tr.human || DEFAULT_HUMAN_IF_UNKNOWN) {
                    humanScore++;
                } else {
                    unattributed++;
                }
            }
            it = fuelTracks.erase(it);
        } else {
            ++it;
        }
    }
}

struct RuntimeOptions {
    fs::path video, model, config, saveConfig, output, report;
    int imageSize = YOLO_IMGSZ;
    int robotClass = ROBOT_CLASS_ID;
    int startFrame = 0;
    int maxFrames = 0;
    bool headless = false;
    bool help = false;
};

static RuntimeOptions parseOptions(int argc, char** argv) {
    RuntimeOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { o.help = true; continue; }
        if (arg == "--headless") { o.headless = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[++i];
        auto integer = [&]() {
            size_t consumed = 0;
            int n = std::stoi(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("Invalid integer for " + arg);
            return n;
        };
        if (arg == "--video") o.video = value;
        else if (arg == "--model") o.model = value;
        else if (arg == "--config") o.config = value;
        else if (arg == "--save-config") o.saveConfig = value;
        else if (arg == "--output") o.output = value;
        else if (arg == "--report") o.report = value;
        else if (arg == "--imgsz") o.imageSize = integer();
        else if (arg == "--robot-class") o.robotClass = integer();
        else if (arg == "--start-frame") o.startFrame = integer();
        else if (arg == "--max-frames") o.maxFrames = integer();
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.imageSize <= 0 || o.imageSize % 32 != 0 || o.robotClass < 0 ||
        o.startFrame < 0 || o.maxFrames < 0)
        throw std::runtime_error("Use a positive multiple of 32 for --imgsz and nonnegative frame/class values.");
    if (!o.help && o.headless && o.config.empty())
        throw std::runtime_error("--headless requires --config with field and hub polygons.");
    return o;
}

static std::vector<cv::Point> readPolygon(const nlohmann::json& value, const cv::Size& size) {
    if (!value.is_array() || value.size() < 3)
        throw std::runtime_error("A polygon needs at least three [x,y] pixel points.");
    std::vector<cv::Point> points;
    for (const auto& point : value) {
        if (!point.is_array() || point.size() != 2 || !point[0].is_number() || !point[1].is_number())
            throw std::runtime_error("Polygon points must be [x,y] pixel coordinates.");
        double x = point[0].get<double>(), y = point[1].get<double>();
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x >= size.width || y >= size.height)
            throw std::runtime_error("Polygon point is outside the video frame.");
        points.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
    if (std::abs(cv::contourArea(points)) < 1)
        throw std::runtime_error("Polygon must have nonzero area.");
    return points;
}

static void writeJson(const fs::path& path, const nlohmann::json& document) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream || !(stream << document.dump(2) << '\n'))
        throw std::runtime_error("Could not write " + path.string());
}

static nlohmann::json polygonJson(const std::vector<cv::Point>& points) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& p : points) out.push_back({p.x, p.y});
    return out;
}

// -----------------------------
// Main
// -----------------------------
#ifndef FRC_FUEL_TRACKER_NO_MAIN
int main(int argc, char** argv) {
    try {
        const RuntimeOptions options = parseOptions(argc, argv);
        if (options.help) {
            std::cout << "Experimental robot + fuel tracker (heuristic counts)\n"
                      << "Usage: fuel_tracker --video VIDEO --model MODEL.onnx [options]\n"
                      << "  --imgsz N            Model input size (default 640; local model uses 960)\n"
                      << "  --robot-class N      Robot class ID (default 0)\n"
                      << "  --config FILE        Pixel polygons: image_width, image_height, field, hub, human(optional)\n"
                      << "  --save-config FILE   Save selected polygons for repeat runs\n"
                      << "  --headless           Disable windows; requires --config\n"
                      << "  --start-frame N      Start frame with --config (default 0)\n"
                      << "  --max-frames N       Limit processed frames (0 = all)\n"
                      << "  --output FILE.mp4    Annotated video\n"
                      << "  --report FILE.json   Counts and performance report\n";
            return 0;
        }
        const fs::path projectDir = fs::current_path();
        const fs::path modelPath = options.model.empty() ? findBestModel(projectDir) : options.model;
        const fs::path videoPath = options.video.empty() ? findVideo(projectDir) : options.video;
        if (!fs::is_regular_file(modelPath)) throw std::runtime_error("Model not found: " + modelPath.string());
        if (!fs::is_regular_file(videoPath)) throw std::runtime_error("Video not found: " + videoPath.string());
        // Do not let output options overwrite an input or each other.
        std::vector<fs::path> outputs;
        for (const auto& out : {options.output, options.report, options.saveConfig}) {
            if (out.empty()) continue;
            const auto normalized = fs::weakly_canonical(out);
            for (const auto& input : {videoPath, modelPath, options.config})
                if (!input.empty() && normalized == fs::weakly_canonical(input))
                    throw std::runtime_error("Output path must differ from input paths.");
            if (std::find(outputs.begin(), outputs.end(), normalized) != outputs.end())
                throw std::runtime_error("Output paths must be distinct.");
            outputs.push_back(normalized);
        }
        std::cout << "Using model: " << modelPath << "\nUsing video: " << videoPath << "\n";

        int setupIdx = options.startFrame;
        cv::Mat setupFrame;
        std::vector<cv::Point> fieldPoly, hubPoly;
        std::optional<std::vector<cv::Point>> humanPoly;
        if (!options.config.empty()) {
            cv::VideoCapture setup(videoPath.string());
            if (!setup.isOpened()) throw std::runtime_error("Could not open video.");
            setup.set(cv::CAP_PROP_POS_FRAMES, setupIdx);
            if (!setup.read(setupFrame)) throw std::runtime_error("Could not read --start-frame.");
            std::ifstream stream(options.config);
            if (!stream) throw std::runtime_error("Could not open config: " + options.config.string());
            nlohmann::json config;
            stream >> config;
            if (config.at("image_width") != setupFrame.cols || config.at("image_height") != setupFrame.rows)
                throw std::runtime_error("Config image dimensions must match the video.");
            fieldPoly = readPolygon(config.at("field"), setupFrame.size());
            hubPoly = readPolygon(config.at("hub"), setupFrame.size());
            if (config.contains("human") && !config["human"].is_null())
                humanPoly = readPolygon(config["human"], setupFrame.size());
        } else {
            auto chosen = chooseSetupFrame(videoPath);
            setupIdx = chosen.first;
            setupFrame = chosen.second;
            double setupScale;
            cv::Mat displaySetup = fitFrame(setupFrame, setupScale);
            auto field = selectPolygon(displaySetup, "Draw FIELD polygon");
            if (!field) throw std::runtime_error("Field polygon cancelled.");
            auto hub = selectPolygon(displaySetup, "Draw HUB polygon");
            if (!hub) throw std::runtime_error("Hub polygon cancelled.");
            auto human = selectPolygon(displaySetup, "Draw HUMAN zone (optional)");
            auto scaleBack = [setupScale](const std::vector<cv::Point>& pts) {
                std::vector<cv::Point> out;
                for (const auto& p : pts) out.emplace_back(static_cast<int>(p.x / setupScale),
                                                          static_cast<int>(p.y / setupScale));
                return out;
            };
            fieldPoly = readPolygon(polygonJson(scaleBack(*field)), setupFrame.size());
            hubPoly = readPolygon(polygonJson(scaleBack(*hub)), setupFrame.size());
            if (human) humanPoly = readPolygon(polygonJson(scaleBack(*human)), setupFrame.size());
        }
        if (!options.saveConfig.empty()) {
            nlohmann::json config = {{"image_width", setupFrame.cols}, {"image_height", setupFrame.rows},
                                    {"field", polygonJson(fieldPoly)}, {"hub", polygonJson(hubPoly)}};
            if (humanPoly) config["human"] = polygonJson(*humanPoly);
            writeJson(options.saveConfig, config);
        }

        Box hubBBox = polyBBox(hubPoly);
        Box searchRect{
            std::max(0, hubBBox.x1 - HUB_MARGIN),
            std::max(0, hubBBox.y1 - HUB_MARGIN),
            hubBBox.x2 + HUB_MARGIN,
            hubBBox.y2 + HUB_MARGIN
        };

        // ---- Load model ----
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath.string());
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // ---- Reopen video for full run ----
        cv::VideoCapture cap(videoPath.string());
        if (!cap.isOpened()) throw std::runtime_error("Could not open video: " + videoPath.string());
        cap.set(cv::CAP_PROP_POS_FRAMES, setupIdx);
        if (!options.headless) cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
        cv::VideoWriter writer;
        if (!options.output.empty()) {
            if (options.output.has_parent_path()) fs::create_directories(options.output.parent_path());
            double displayScale;
            const cv::Mat display = fitFrame(setupFrame, displayScale);
            const double fps = cap.get(cv::CAP_PROP_FPS);
            if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("Video has invalid FPS.");
            writer.open(options.output.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, display.size());
            if (!writer.isOpened()) throw std::runtime_error("Could not open output video.");
        }
        int framesProcessed = 0;
        int robotDetections = 0;
        int fuelDetections = 0;
        const auto started = std::chrono::steady_clock::now();

        std::map<int, FuelTrack> fuelTracks;
        int nextFuelId = 1;
        std::map<std::string, int> scoreByRobot;
        int humanScore = 0;
        int unattributed = 0;
        int frameIdx = setupIdx;

        SimpleTracker robotTracker;

        while (options.maxFrames == 0 || framesProcessed < options.maxFrames) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            // ---- Robot detection + tracking ----
            std::vector<Detection> detections = runYoloInference(net, frame, options.imageSize, options.robotClass);
            std::vector<Box> robotDetBoxes;
            for (auto& d : detections) {
                if (d.classId != options.robotClass) continue;
                cv::Point c = centerOfBox(d.box);
                if (pointInPoly(c, fieldPoly)) robotDetBoxes.push_back(d.box);
            }
            std::vector<std::pair<int, Box>> robotBoxes = robotTracker.update(robotDetBoxes);

            // ---- Fuel detection (near hub only) ----
            std::vector<FuelCandidate> pts = detectYellowCandidates(frame, searchRect);

            updateFuelTracks(fuelTracks, nextFuelId, scoreByRobot, humanScore, unattributed,
                             pts, robotBoxes, hubPoly, humanPoly, frameIdx);

            ++framesProcessed;
            robotDetections += static_cast<int>(robotBoxes.size());
            fuelDetections += static_cast<int>(pts.size());
            ++frameIdx;
            if (options.headless && !writer.isOpened()) continue;

            // ---- Draw ----
            double scale;
            cv::Mat display = fitFrame(frame, scale);

            auto scalePoly = [scale](const std::vector<cv::Point>& poly) {
                std::vector<cv::Point> out;
                out.reserve(poly.size());
                for (auto& p : poly) out.emplace_back(static_cast<int>(p.x * scale), static_cast<int>(p.y * scale));
                return out;
            };

            cv::polylines(display, scalePoly(fieldPoly), true, cv::Scalar(0, 255, 0), 2);
            cv::polylines(display, scalePoly(hubPoly), true, cv::Scalar(255, 0, 0), 2);
            if (humanPoly) cv::polylines(display, scalePoly(*humanPoly), true, cv::Scalar(0, 165, 255), 2);

            for (auto& [rid, box] : robotBoxes) {
                int x1 = static_cast<int>(box.x1 * scale), y1 = static_cast<int>(box.y1 * scale);
                int x2 = static_cast<int>(box.x2 * scale), y2 = static_cast<int>(box.y2 * scale);
                cv::rectangle(display, {x1, y1}, {x2, y2}, cv::Scalar(0, 255, 255), 2);
                if (SHOW_ROBOT_IDS) {
                    drawText(display, "R" + std::to_string(rid), {x1, std::max(24, y1)},
                              cv::Scalar(0, 255, 255), 0.75, 2);
                }
            }

            for (auto& [tid, tr] : fuelTracks) {
                int x = static_cast<int>(tr.center.x * scale);
                int y = static_cast<int>(tr.center.y * scale);
                int r = std::max(3, static_cast<int>(tr.radius * scale));
                cv::Scalar color = tr.enteredHub ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                cv::circle(display, {x, y}, r, color, 2);

                std::vector<cv::Point> trailScaled;
                for (auto& p : tr.trail) {
                    trailScaled.emplace_back(static_cast<int>(p.x * scale), static_cast<int>(p.y * scale));
                }
                for (size_t i = 1; i < trailScaled.size(); ++i) {
                    cv::line(display, trailScaled[i - 1], trailScaled[i], color, 1);
                }
                if (SHOW_FUEL_LABELS) {
                    std::string ownerText = tr.ownerLabel ? *tr.ownerLabel : (tr.human ? "HP" : "?");
                    drawText(display, "f" + std::to_string(tid) + "->" + ownerText,
                              {x + 8, y - 8}, color, 0.7, 2);
                }
            }

            int y0 = 28;
            drawText(display, "Human: " + std::to_string(humanScore), {10, y0}, cv::Scalar(0, 165, 255), 0.75, 2);
            y0 += 30;
            drawText(display, "Unattributed: " + std::to_string(unattributed), {10, y0}, cv::Scalar(200, 200, 200), 0.75, 2);
            y0 += 30;
            for (auto& [label, score] : scoreByRobot) {
                drawText(display, label + ": " + std::to_string(score), {10, y0}, cv::Scalar(255, 255, 255), 0.75, 2);
                y0 += 30;
            }

            drawText(display, "q quit", {display.cols - 90, 28}, cv::Scalar(255, 255, 255), 0.65, 2);

            if (writer.isOpened()) writer.write(display);
            if (!options.headless) {
                cv::imshow(WINDOW_NAME, display);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 27) break;
            }
        }

        cap.release();
        writer.release();
        if (!options.headless) cv::destroyAllWindows();
        if (framesProcessed == 0) throw std::runtime_error("No video frames processed.");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        nlohmann::json report = {
            {"experimental", true}, {"count_method", "disappearance_in_hub"},
            {"video", videoPath.string()}, {"model", modelPath.string()},
            {"image_size", options.imageSize}, {"start_frame", setupIdx},
            {"frames_processed", framesProcessed}, {"elapsed_seconds", elapsed},
            {"processing_fps", framesProcessed / elapsed}, {"robot_detection_observations", robotDetections},
            {"robot_tracks_created", robotTracker.tracksCreated()},
            {"fuel_detection_observations", fuelDetections}, {"fuel_tracks_created", nextFuelId - 1},
            {"pending_fuel_tracks", fuelTracks.size()}, {"counts_by_robot", scoreByRobot},
            {"human_count", humanScore}, {"unattributed_count", unattributed}
        };
        if (!options.report.empty()) writeJson(options.report, report);
        std::cout << report.dump(2) << "\n";

        std::cout << "\n===== HEURISTIC FUEL COUNTS =====\n";
        for (auto& [label, score] : scoreByRobot) {
            std::cout << label << ": " << score << "\n";
        }
        std::cout << "Human: " << humanScore << "\n";
        std::cout << "Unattributed: " << unattributed << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#endif  // FRC_FUEL_TRACKER_NO_MAIN
