#ifndef ROBOT_DETECTOR_H
#define ROBOT_DETECTOR_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace frc::vision {

/** A normalized top-left box from the detector. It never crosses Contract D directly. */
struct Detection {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    double confidence = 0.0;
    int class_id = 0;
};

/**
 * The part of a frame that holds the field, for broadcasts that composite a second camera view
 * into one picture.
 *
 * Both panels show the same six robots, so without this every robot is detected twice, the
 * tracker makes two tracks of it, and -- because team attribution is per track -- a human is
 * asked for the same team number twice while every shot on that robot is counted twice. A
 * duplicate is worse for the numbers than a miss.
 *
 * The seam cannot be read from a single frame; it is calibrated per source by
 * `ingest.collection.calibrate_region`, which samples frames and looks for two separated bands of
 * robots. An absent entry means the whole frame, which is the safe default.
 */
struct ViewRegion {
    double x = 0.0;
    double y = 0.0;
    double w = 1.0;
    double h = 1.0;
    //: false filters boxes after inference, true crops the frame before it. Filtering measured
    //: better at a 640px export -- 4.22 boxes per frame against 4.05 -- because the model was
    //: trained on whole frames, and moving the aspect ratio away from that costs more than the
    //: extra pixels return. At 960 they tie (4.50 against 4.52), and they cost the same either
    //: way, since both letterbox into the same square. Cropping is kept because a model trained
    //: on cropped frames would change that, not because it helps today.
    //: See ingest/collection/view_region.py.
    bool crop = false;

    [[nodiscard]] bool is_full_frame() const {
        return x == 0.0 && y == 0.0 && w == 1.0 && h == 1.0;
    }

    /**
     * Whether a box belongs to this view, judged by its centre.
     *
     * The centre, not the edges: a robot straddling the seam belongs to whichever panel it is
     * mostly in, and testing edges would either drop it from both or keep it in both.
     */
    [[nodiscard]] bool contains_centre(const Detection& detection) const {
        const double cx = detection.x + detection.w / 2.0;
        const double cy = detection.y + detection.h / 2.0;
        return cx >= x && cx <= x + w && cy >= y && cy <= y + h;
    }
};

/** Which export the ONNX file is. The two families disagree on every step of pre- and
 *  post-processing, so running one as the other produces plausible, meaningless boxes. */
enum class DetectorFamily {
    kAuto,    //!< Decide from the loaded model's outputs: one means YOLO, two mean RF-DETR.
    kYolo,    //!< One tensor, (1, 4+classes, anchors) or its transpose. /255, letterboxed, needs NMS.
    kRfDetr,  //!< Two tensors, dets + labels. ImageNet-normalised, stretched to square, no NMS.
};

/**
 * Local, non-contract model settings. Configure with FRC_DETECTOR_CONFIG; never put a model
 * path in a job record because Contract A must remain portable and model-independent.
 */
struct DetectorConfig {
    std::string model_path;
    std::string model_version = "robot-onnx-0.1.0";
    DetectorFamily family = DetectorFamily::kAuto;
    std::string input_name = "images";
    std::string boxes_output_name = "dets";
    std::string logits_output_name = "labels";
    int input_width = 640;
    int input_height = 640;
    //: Whether input_width/input_height came from the file rather than from these defaults. An
    //: export records its own training resolution; trusting a default over that silently
    //: mis-scales every box, so an absent setting is filled from the model and a present one
    //: that contradicts the model is refused.
    bool input_size_from_config = false;
    int robot_class_id = 0;
    int background_class_id = -1;
    double score_threshold = 0.50;
    //: IoU above which two candidates are the same object. A raw YOLO tensor holds a prediction
    //: per anchor, so one robot arrives as a cluster; without suppression every count downstream
    //: is inflated. RF-DETR needs none -- its queries are already one-per-object.
    double nms_iou = 0.50;
    //: Side of the square crops the frame is additionally sliced into, in source pixels; 0 is
    //: off. A 1920x1080 frame letterboxed into a 960x960 input halves everything, and a robot
    //: far from the camera falls below what the model resolves. Running it again on native-
    //: resolution crops recovered about half as many robots again on a real match.
    int tile_size = 0;
    //: How much neighbouring tiles share. Without overlap a robot on a seam is cut in two and
    //: recognised as neither half.
    double tile_overlap = 0.25;
    double sample_rate_hz = 2.0;
    double shot_change_threshold = 0.55;
    //: Applied when a source has no entry of its own.
    ViewRegion region;
    //: Per-source regions, keyed by the video file's stem. Written by
    //: `ingest.collection.calibrate_region`; sources composite differently and 35 of 60 measured
    //: sources carry a second view, so one setting cannot serve them all.
    std::map<std::string, ViewRegion> regions;
};

/** The region for one video, by its path's stem, falling back to the config's default. */
ViewRegion region_for(const DetectorConfig& config, const std::string& video_path);

/** A box found inside a crop, expressed relative to the whole frame. */
Detection map_from_region(const Detection& detection, const ViewRegion& region);

/** Reads FRC_DETECTOR_CONFIG. No environment setting means detector work is deliberately off. */
DetectorConfig load_detector_config();

// --- pure post-processing, exposed so it can be tested without a model or ONNX Runtime --------

/** How a source frame was fitted into the model's square input, preserving aspect ratio. */
struct Letterbox {
    double scale = 1.0;  //!< model-input pixels per source pixel
    int pad_x = 0;       //!< left padding, in model-input pixels
    int pad_y = 0;       //!< top padding, in model-input pixels
};

Letterbox letterbox_for(int source_width, int source_height, int input_size);

/**
 * Decode a raw YOLO tensor into normalized boxes, before suppression.
 *
 * The layout is checked rather than assumed: YOLOv8/11 export (1, 4+classes, anchors) and older
 * exports (1, anchors, 4+classes). Transposing the wrong one produces boxes that look plausible
 * and are meaningless, so the axis matching the attribute count decides.
 */
std::vector<Detection> decode_yolo(const float* data, int64_t dim1, int64_t dim2,
                                   const Letterbox& box, int source_width, int source_height,
                                   double score_threshold, int robot_class_id);

/** Greedy suppression: keep the most confident box, drop what overlaps it, repeat. */
std::vector<Detection> non_max_suppression(std::vector<Detection> boxes, double iou_threshold);

/** Where each tile starts along one axis, always including a final tile flush with the end. */
std::vector<int> tile_origins(int total, int tile, double overlap);

/** A tile-relative normalized box, expressed relative to the whole frame. */
Detection map_from_tile(const Detection& detection, int tile_x, int tile_y, int tile_w, int tile_h,
                        int frame_w, int frame_h);

/**
 * Whether a box runs into a tile edge that is not also a frame edge.
 *
 * A robot straddling a seam is cut, and the model happily labels the visible sliver as a whole
 * robot with modest confidence. Suppression does not remove it -- a sliver barely overlaps the
 * real box -- so it survives as a phantom robot beside a real one. The overlapping neighbour
 * tile sees the same robot whole, so dropping the cut copy loses nothing.
 */
bool clipped_by_tile(const Detection& detection, int tile_x, int tile_y, int tile_w, int tile_h,
                     int frame_w, int frame_h);

/** Runs a trained ONNX detector when a local model is configured. */
class RobotDetector {
  public:
    explicit RobotDetector(DetectorConfig config);
    ~RobotDetector();
    RobotDetector(RobotDetector&&) noexcept;
    RobotDetector& operator=(RobotDetector&&) noexcept;
    RobotDetector(const RobotDetector&) = delete;
    RobotDetector& operator=(const RobotDetector&) = delete;

    [[nodiscard]] bool enabled() const;
    [[nodiscard]] const DetectorConfig& config() const;
    [[nodiscard]] std::vector<Detection> infer(const cv::Mat& bgr_frame) const;
    /** As above, restricted to one view of a composited frame. */
    [[nodiscard]] std::vector<Detection> infer(const cv::Mat& bgr_frame,
                                               const ViewRegion& region) const;

  private:
    /** One pass over one image. `infer` calls this for the whole frame and again per tile. */
    [[nodiscard]] std::vector<Detection> infer_once(const cv::Mat& bgr_frame) const;

    struct Impl;
    DetectorConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace frc::vision

#endif  // ROBOT_DETECTOR_H
