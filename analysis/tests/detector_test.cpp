// Detector post-processing, written against the gap that stopped the pipeline working.
//
// The analyzer spoke only RF-DETR: two outputs, ImageNet normalisation, a plain stretch to
// square, no suppression. Both trained models are YOLO: one output, /255, letterboxed, thousands
// of overlapping candidates. So the analyzer decoded zero frames of every real match while
// reporting success, and the numbers in result.json looked like an honest decode-only run.
//
// Everything below is the pure half of that fix -- the arithmetic between a raw tensor and a box
// on a robot. It needs no model and no ONNX Runtime, which is what makes the cases that actually
// go wrong (a 16:9 frame in a square input, one robot as a cluster of anchors, a robot half off
// the edge) cheap to pin down here rather than by staring at rendered frames.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/RobotDetector.h"

using frc::vision::Detection;
using frc::vision::DetectorConfig;
using frc::vision::Letterbox;
using frc::vision::clipped_by_tile;
using frc::vision::decode_yolo;
using frc::vision::letterbox_for;
using frc::vision::ViewRegion;
using frc::vision::map_from_region;
using frc::vision::map_from_tile;
using frc::vision::region_for;
using frc::vision::non_max_suppression;
using frc::vision::tile_origins;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (ok) {
        std::printf("  ok   %s\n", what.c_str());
    } else {
        ++failures;
        std::printf("  FAIL %s\n", what.c_str());
    }
}

bool near(double a, double b, double tolerance = 1e-6) { return std::fabs(a - b) <= tolerance; }

// A single-class YOLO tensor laid out as (4 + 1 attributes, anchors) -- the modern export order.
// Values are in model-input pixels, which is what the model actually emits.
struct Tensor {
    std::vector<float> attributes_first;  // (5, anchors)
    std::vector<float> anchors_first;     // (anchors, 5)
    int64_t anchors = 0;
};

// Padded out to a realistic anchor count. A real export has 18,900 anchors against 5
// attributes, and that lopsidedness is exactly what tells the two layouts apart. A tensor with
// three anchors is genuinely ambiguous -- and is not a tensor any export produces -- so testing
// against one would be testing a shape the code will never see.
constexpr size_t kAnchors = 64;

Tensor build(const std::vector<std::array<double, 5>>& rows) {
    Tensor t;
    t.anchors = static_cast<int64_t>(kAnchors);
    t.attributes_first.assign(5 * kAnchors, 0.0F);  // the rest score zero and fall below any threshold
    t.anchors_first.assign(5 * kAnchors, 0.0F);
    for (size_t anchor = 0; anchor < rows.size() && anchor < kAnchors; ++anchor) {
        for (size_t attribute = 0; attribute < 5; ++attribute) {
            const auto value = static_cast<float>(rows[anchor][attribute]);
            t.attributes_first[attribute * kAnchors + anchor] = value;
            t.anchors_first[anchor * 5 + attribute] = value;
        }
    }
    return t;
}

// Centre/size in model-input pixels for a box given in source pixels, under a letterbox. This is
// the forward direction of what decode_yolo undoes, so a round trip has a known answer.
std::array<double, 5> row_for(double x0, double y0, double x1, double y1,
                              const Letterbox& box, double score) {
    const double mx0 = x0 * box.scale + box.pad_x;
    const double my0 = y0 * box.scale + box.pad_y;
    const double mx1 = x1 * box.scale + box.pad_x;
    const double my1 = y1 * box.scale + box.pad_y;
    return {(mx0 + mx1) / 2.0, (my0 + my1) / 2.0, mx1 - mx0, my1 - my0, score};
}

// --- Fitting a 16:9 frame into a square input --------------------------------------------------
// Broadcast footage is 1920x1080 and the model takes 960x960. Stretching to fill the square is
// what the RF-DETR path did; for YOLO it feeds the model an image with the wrong proportions and
// every box comes back displaced vertically.
void letterbox_geometry() {
    std::printf("letterbox of a 1920x1080 frame into 960x960\n");
    const Letterbox box = letterbox_for(1920, 1080, 960);
    check(near(box.scale, 0.5), "long edge sets the scale (960/1920 = 0.5)");
    check(box.pad_x == 0, "no padding on the long edge");
    check(box.pad_y == 210, "short edge is centred: (960 - 540) / 2 = 210 rows of grey");

    const Letterbox square = letterbox_for(720, 720, 640);
    check(square.pad_x == 0 && square.pad_y == 0, "a square frame needs no padding");

    const Letterbox portrait = letterbox_for(1080, 1920, 960);
    check(portrait.pad_y == 0 && portrait.pad_x == 210, "a tall frame pads left and right instead");
}

// --- A box comes back where it started ---------------------------------------------------------
void round_trip() {
    std::printf("decoding a box back to source coordinates\n");
    const Letterbox box = letterbox_for(1920, 1080, 960);
    // A robot occupying source pixels x 800..1000, y 500..650.
    const Tensor t = build({row_for(800, 500, 1000, 650, box, 0.90)});

    const auto out = decode_yolo(t.attributes_first.data(), 5, t.anchors, box, 1920, 1080, 0.25, 0);
    check(out.size() == 1, "one anchor over threshold gives one box");
    if (out.size() == 1) {
        check(near(out[0].x, 800.0 / 1920.0, 1e-5), "x is the source position, normalized");
        check(near(out[0].y, 500.0 / 1080.0, 1e-5), "y undoes the 210px pad, not just the scale");
        check(near(out[0].w, 200.0 / 1920.0, 1e-5), "width is the source width, normalized");
        check(near(out[0].h, 150.0 / 1080.0, 1e-5), "height is the source height, normalized");
        check(near(out[0].confidence, 0.90, 1e-6), "confidence is the class score, not a sigmoid of it");
    }

    // The bug this whole file exists for: skip the padding and the box lands somewhere else
    // entirely, with nothing to signal that anything went wrong.
    Letterbox unpadded = box;
    unpadded.pad_y = 0;
    const auto wrong = decode_yolo(t.attributes_first.data(), 5, t.anchors, unpadded, 1920, 1080, 0.25, 0);
    check(wrong.size() == 1 && wrong[0].y > 0.8,
          "ignoring the pad puts a mid-field robot near the bottom edge -- silently wrong, not empty");
}

// --- Either tensor layout, same answer ----------------------------------------------------------
// YOLOv8/11 export (1, 4+classes, anchors); older exports use the transpose. Reading the wrong
// axis produces boxes that are plausible and meaningless, so the axis is chosen by length.
void both_layouts() {
    std::printf("tensor layout is detected, not assumed\n");
    const Letterbox box = letterbox_for(1920, 1080, 960);
    const Tensor t = build({
        row_for(100, 200, 260, 340, box, 0.80),
        row_for(900, 400, 1080, 560, box, 0.70),
        row_for(1500, 600, 1660, 740, box, 0.60),
    });

    const auto modern = decode_yolo(t.attributes_first.data(), 5, t.anchors, box, 1920, 1080, 0.25, 0);
    const auto legacy = decode_yolo(t.anchors_first.data(), t.anchors, 5, box, 1920, 1080, 0.25, 0);
    check(modern.size() == 3 && legacy.size() == 3, "both layouts yield three boxes");
    bool same = modern.size() == legacy.size();
    for (size_t i = 0; same && i < modern.size(); ++i) {
        same = near(modern[i].x, legacy[i].x, 1e-6) && near(modern[i].y, legacy[i].y, 1e-6) &&
               near(modern[i].w, legacy[i].w, 1e-6) && near(modern[i].h, legacy[i].h, 1e-6);
    }
    check(same, "(attributes, anchors) and (anchors, attributes) decode identically");
}

// --- Thresholds and classes ---------------------------------------------------------------------
void filtering() {
    std::printf("score threshold and class filter\n");
    const Letterbox box = letterbox_for(1920, 1080, 960);
    const Tensor t = build({
        row_for(100, 100, 200, 200, box, 0.80),
        row_for(400, 100, 500, 200, box, 0.30),
        row_for(700, 100, 800, 200, box, 0.10),
    });
    const auto out = decode_yolo(t.attributes_first.data(), 5, t.anchors, box, 1920, 1080, 0.25, 0);
    check(out.size() == 2, "anchors below the threshold are dropped");

    // With one class, asking for class 1 must yield nothing rather than falling back to class 0.
    const auto other = decode_yolo(t.attributes_first.data(), 5, t.anchors, box, 1920, 1080, 0.25, 1);
    check(other.empty(), "a class the model does not have returns nothing");

    const auto degenerate = decode_yolo(t.attributes_first.data(), 4, 4, box, 1920, 1080, 0.25, 0);
    check(degenerate.empty(), "a tensor too narrow to hold box plus class is refused, not guessed");
}

// --- Suppression --------------------------------------------------------------------------------
// A raw YOLO tensor holds a prediction per anchor, so a single robot arrives as a cluster. Without
// this step every count downstream -- robots on the field, tracks, per-team statistics -- is
// inflated several-fold while looking entirely reasonable.
void suppression() {
    std::printf("collapsing anchor clusters to one box per robot\n");
    std::vector<Detection> cluster;
    for (int i = 0; i < 7; ++i) {
        Detection d;
        d.x = 0.40 + i * 0.002;
        d.y = 0.50 + i * 0.002;
        d.w = 0.08;
        d.h = 0.12;
        d.confidence = 0.90 - i * 0.01;
        cluster.push_back(d);
    }
    const auto kept = non_max_suppression(cluster, 0.50);
    check(kept.size() == 1, "seven near-identical anchors become one robot");
    check(kept.size() == 1 && near(kept[0].confidence, 0.90), "the survivor is the most confident");

    // Two robots close together are not one robot. The threshold has to leave them alone.
    std::vector<Detection> pair = {cluster[0]};
    Detection neighbour = cluster[0];
    neighbour.x += 0.09;  // adjacent, barely touching
    neighbour.confidence = 0.85;
    pair.push_back(neighbour);
    check(non_max_suppression(pair, 0.50).size() == 2, "two adjacent robots both survive");

    check(non_max_suppression({}, 0.50).empty(), "no detections suppress to no detections");
}

// --- Robots at the frame edge -------------------------------------------------------------------
void frame_edges() {
    std::printf("boxes that run off the edge\n");
    const Letterbox box = letterbox_for(1920, 1080, 960);
    // A robot half out of shot on the left: source x from -100 to 100.
    const Tensor t = build({row_for(-100, 400, 100, 560, box, 0.90)});
    const auto out = decode_yolo(t.attributes_first.data(), 5, t.anchors, box, 1920, 1080, 0.25, 0);
    check(out.size() == 1, "a partly visible robot is still a detection");
    if (out.size() == 1) {
        check(near(out[0].x, 0.0, 1e-6), "the box starts at the edge");
        check(near(out[0].w, 100.0 / 1920.0, 1e-5),
              "width is the visible half, so x + w stays inside the frame");
    }

    // Entirely off-screen: source x from -400 to -200. Nothing to see, so nothing to report.
    const Tensor gone = build({row_for(-400, 400, -200, 560, box, 0.90)});
    check(decode_yolo(gone.attributes_first.data(), 5, gone.anchors, box, 1920, 1080, 0.25, 0).empty(),
          "a box entirely outside the frame is dropped rather than flattened onto the edge");
}

// --- Slicing the frame ---------------------------------------------------------------------------
// A 1920x1080 frame letterboxed into a 960x960 input arrives at half scale, and a robot far from
// the camera falls below what the model resolves. Running it again on native-resolution crops
// recovered about half as many robots again on a real match -- 35 detections became 55.
void tiling_geometry() {
    std::printf("tiles across a 1920x1080 frame\n");
    const auto across = tile_origins(1920, 960, 0.25);
    check(across.size() == 3, "1920 needs three 960 tiles at 25% overlap (" +
                                  std::to_string(across.size()) + ")");
    check(!across.empty() && across.front() == 0, "the first tile starts at the edge");
    check(!across.empty() && across.back() == 960,
          "the last tile ends flush with the frame, so no strip is never looked at");

    const auto down = tile_origins(1080, 960, 0.25);
    check(down.size() == 2 && down.back() == 120, "1080 needs two rows, the second flush at 120");

    check(tile_origins(800, 960, 0.25).size() == 1,
          "a frame smaller than a tile is one tile, not zero");
    check(tile_origins(1920, 0, 0.25).empty(), "a zero tile size produces no tiles");

    // Every column of the frame must fall inside some tile, or robots there are invisible.
    bool covered = true;
    for (int x = 0; x < 1920; ++x) {
        bool inside = false;
        for (const int start : across) inside = inside || (x >= start && x < start + 960);
        covered = covered && inside;
    }
    check(covered, "every column of the frame is inside at least one tile");
}

void tile_coordinates() {
    std::printf("mapping a tile box back to the frame\n");
    // A box filling the middle of the second tile across (starting at 720) and second row (120).
    Detection in;
    in.x = 0.25; in.y = 0.50; in.w = 0.25; in.h = 0.25; in.confidence = 0.8;
    const Detection out = map_from_tile(in, 720, 120, 960, 960, 1920, 1080);
    check(near(out.x, (0.25 * 960 + 720) / 1920.0, 1e-9), "x lands where the tile puts it");
    check(near(out.y, (0.50 * 960 + 120) / 1080.0, 1e-9), "y lands where the tile puts it");
    check(near(out.w, 0.25 * 960 / 1920.0, 1e-9), "width shrinks by the tile's share of the frame");
    check(near(out.h, 0.25 * 960 / 1080.0, 1e-9), "height shrinks by its own axis, not x's");
    check(near(out.confidence, 0.8), "confidence is carried through unchanged");

    // A tile that IS the frame must be a no-op, or single-tile frames would be distorted.
    const Detection same = map_from_tile(in, 0, 0, 1920, 1080, 1920, 1080);
    check(near(same.x, in.x) && near(same.y, in.y) && near(same.w, in.w) && near(same.h, in.h),
          "a tile covering the whole frame changes nothing");
}

// The artifact that made this worth testing: on a real frame, tile seams fell at x=960 and a
// robot straddling one came back as a 0.26-confidence sliver beside the real box. Suppression
// cannot remove it -- a sliver barely overlaps anything -- so it had to be dropped at the source.
void tile_seams() {
    std::printf("a robot cut by a tile seam\n");
    Detection at_right;
    at_right.x = 0.90; at_right.y = 0.40; at_right.w = 0.10; at_right.h = 0.10;
    check(clipped_by_tile(at_right, 0, 0, 960, 960, 1920, 1080),
          "a box against an interior right edge is a cut robot");

    Detection at_left;
    at_left.x = 0.0; at_left.y = 0.40; at_left.w = 0.10; at_left.h = 0.10;
    check(clipped_by_tile(at_left, 720, 0, 960, 960, 1920, 1080),
          "a box against an interior left edge is a cut robot");

    // The frame's own edges are different: a robot genuinely leaving shot is real, and dropping
    // it would blind the tracker exactly where robots enter and exit.
    check(!clipped_by_tile(at_left, 0, 0, 960, 960, 1920, 1080),
          "the frame's own left edge is not a seam");
    Detection at_frame_right;
    at_frame_right.x = 0.90; at_frame_right.y = 0.40; at_frame_right.w = 0.10; at_frame_right.h = 0.10;
    check(!clipped_by_tile(at_frame_right, 960, 0, 960, 960, 1920, 1080),
          "the frame's own right edge is not a seam");

    Detection middle;
    middle.x = 0.40; middle.y = 0.40; middle.w = 0.10; middle.h = 0.10;
    check(!clipped_by_tile(middle, 0, 0, 960, 960, 1920, 1080),
          "a box well inside its tile is kept");

    Detection at_bottom;
    at_bottom.x = 0.40; at_bottom.y = 0.92; at_bottom.w = 0.10; at_bottom.h = 0.08;
    check(clipped_by_tile(at_bottom, 0, 0, 960, 960, 1920, 1080),
          "an interior bottom edge cuts too, not just the sides");
    check(!clipped_by_tile(at_bottom, 0, 120, 960, 960, 1920, 1080),
          "the same box is fine when that edge is the bottom of the frame");
}


// --- a composited second camera view ------------------------------------------------------------

Detection box_at(double x, double y, double w = 0.05, double h = 0.05) {
    Detection d;
    d.x = x; d.y = y; d.w = w; d.h = h; d.confidence = 0.9;
    return d;
}

void view_regions() {
    std::printf("\na frame with a second camera view composited under the first\n");

    const ViewRegion whole;
    check(whole.is_full_frame(), "the default region is the whole frame");
    check(whole.contains_centre(box_at(0.5, 0.97)), "and it keeps a box near the bottom edge");

    const ViewRegion top{0.0, 0.0, 1.0, 0.70, false};
    check(!top.is_full_frame(), "a region that stops short of the bottom is not the full frame");
    check(top.contains_centre(box_at(0.5, 0.30)), "a robot in the upper panel is kept");
    check(!top.contains_centre(box_at(0.5, 0.90)), "the same robot in the lower panel is dropped");

    // The centre decides, so a robot lying across the seam belongs to one panel rather than to
    // both or neither. Testing edges instead would double-count it or lose it.
    check(top.contains_centre(box_at(0.5, 0.62, 0.05, 0.10)),
          "a box straddling the seam, mostly above it, is kept");
    check(!top.contains_centre(box_at(0.5, 0.68, 0.05, 0.10)),
          "a box straddling the seam, mostly below it, is dropped");

    // Mapping back after a crop. Getting this wrong is silent: every box stays a valid normalized
    // box, and every robot simply reports higher up the frame than it really is.
    const Detection inside = box_at(0.25, 0.50, 0.10, 0.20);
    const Detection mapped = map_from_region(inside, top);
    check(near(mapped.y, 0.35), "y is scaled by the region's height, not left alone");
    check(near(mapped.h, 0.14), "height is scaled too");
    check(near(mapped.x, 0.25), "x is untouched by a full-width region");
    check(near(mapped.w, 0.10), "and so is width");
    check(near(map_from_region(inside, whole).y, 0.50),
          "mapping through the full frame changes nothing");

    const ViewRegion offset{0.20, 0.30, 0.50, 0.50, true};
    const Detection corner = map_from_region(box_at(0.0, 0.0), offset);
    check(near(corner.x, 0.20) && near(corner.y, 0.30),
          "a box at a crop's origin lands at the region's origin");

    // Lookup by source, which is how a per-venue setting reaches a run.
    DetectorConfig config;
    config.regions["ulD5ox6pFdU_00000_00215"] = top;
    check(near(region_for(config, "data/segments/ulD5ox6pFdU_00000_00215.mp4").h, 0.70),
          "a listed source gets its own region, found by the path's stem");
    check(near(region_for(config, "C:\\data\\segments\\ulD5ox6pFdU_00000_00215.mp4").h, 0.70),
          "backslashes separate a path too");
    check(region_for(config, "data/segments/unlisted_00000_00215.mp4").is_full_frame(),
          "an unlisted source keeps the whole frame rather than borrowing another's crop");
    check(region_for(config, "no_extension").is_full_frame(),
          "a path without an extension does not confuse the lookup");
}

}  // namespace

int main() {
    std::printf("\nDetector post-processing\n\n");
    letterbox_geometry();
    round_trip();
    both_layouts();
    filtering();
    suppression();
    frame_edges();
    tiling_geometry();
    tile_coordinates();
    tile_seams();
    view_regions();
    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
