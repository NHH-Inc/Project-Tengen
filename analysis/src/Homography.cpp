#include "Homography.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace frc::vision {

Homography::Homography(std::array<double, 9> matrix, double reprojection_ft, int point_count,
                       double field_length_ft, double field_width_ft,
                       std::optional<double> plane_height_ft, std::string source,
                       std::optional<bool> trusted_override)
    : matrix_(matrix),
      reprojection_ft_(reprojection_ft),
      point_count_(point_count),
      field_length_ft_(field_length_ft),
      field_width_ft_(field_width_ft),
      plane_height_ft_(plane_height_ft),
      source_(std::move(source)),
      trusted_override_(trusted_override) {}

bool Homography::trustworthy() const {
    if (trusted_override_.has_value()) return point_count_ >= 4 && *trusted_override_;
    if (point_count_ < 4) return false;
    // A four-point solve is geometrically valid but unverified; its zero error proves nothing.
    // Treated as usable and flagged via has_redundancy() rather than silently called confirmed.
    if (!has_redundancy()) return true;
    return reprojection_ft_ <= kMaxReprojectionFt;
}

std::optional<std::pair<double, double>> Homography::to_field(double x, double y) const {
    const double denom = matrix_[6] * x + matrix_[7] * y + matrix_[8];
    if (std::abs(denom) < 1e-12) return std::nullopt;
    return std::pair<double, double>{
        (matrix_[0] * x + matrix_[1] * y + matrix_[2]) / denom,
        (matrix_[3] * x + matrix_[4] * y + matrix_[5]) / denom,
    };
}

std::optional<std::pair<double, double>> Homography::box_to_field(
    double x, double y, double w, double h, int image_w, int image_h) const {
    return to_field((x + w / 2.0) * image_w, (y + h) * image_h);
}

bool Homography::on_field(double x, double y) const {
    const auto mapped = to_field(x, y);
    if (!mapped) return false;
    const auto [fx, fy] = *mapped;
    if (!std::isfinite(fx) || !std::isfinite(fy)) return false;
    return fx >= -kFieldMarginFt && fx <= field_length_ft_ + kFieldMarginFt &&
           fy >= -kFieldMarginFt && fy <= field_width_ft_ + kFieldMarginFt;
}

std::optional<Homography> solve_homography(const std::vector<PointPair>& points,
                                           double field_length_ft, double field_width_ft) {
    if (points.size() < 4) return std::nullopt;

    std::vector<cv::Point2d> src;
    std::vector<cv::Point2d> dst;
    src.reserve(points.size());
    dst.reserve(points.size());
    for (const auto& p : points) {
        src.emplace_back(p.image_x, p.image_y);
        dst.emplace_back(p.field_x, p.field_y);
    }

    const cv::Mat found = cv::findHomography(src, dst, 0);
    if (found.empty() || found.rows != 3 || found.cols != 3) return std::nullopt;

    std::array<double, 9> matrix{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const double value = found.at<double>(r, c);
            if (!std::isfinite(value)) return std::nullopt;
            matrix[static_cast<std::size_t>(r * 3 + c)] = value;
        }
    }

    // Reproject the inputs and measure. A solution that cannot reproduce the points it was fitted
    // to is describing a different geometry than the one it was given.
    Homography candidate(matrix, 0.0, static_cast<int>(points.size()), field_length_ft,
                         field_width_ft);
    double worst = 0.0;
    for (const auto& p : points) {
        const auto mapped = candidate.to_field(p.image_x, p.image_y);
        if (!mapped) return std::nullopt;
        worst = std::max(worst, std::hypot(mapped->first - p.field_x,
                                           mapped->second - p.field_y));
    }

    return Homography(matrix, worst, static_cast<int>(points.size()), field_length_ft,
                      field_width_ft);
}

std::optional<Homography> load_homography(double field_length_ft, double field_width_ft) {
    const char* path = std::getenv("FRC_HOMOGRAPHY_CONFIG");
    if (path == nullptr || *path == '\0') return std::nullopt;

    std::ifstream stream(path);
    if (!stream) return std::nullopt;

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception&) {
        // A malformed calibration is not a reason to abort a run. Falling back to no homography
        // loses field coordinates and keeps every other output correct, which beats failing a
        // whole match analysis over a stray comma.
        return std::nullopt;
    }

    const std::string source = document.value("mapping_source", "tag_plane");
    const std::optional<double> plane_height =
        document.contains("plane_height_ft") && document["plane_height_ft"].is_number()
        ? std::optional<double>(document["plane_height_ft"].get<double>())
        : std::nullopt;
    const std::optional<bool> trusted_override =
        document.contains("trustworthy") && document["trustworthy"].is_boolean()
        ? std::optional<bool>(document["trustworthy"].get<bool>())
        : std::nullopt;

    // New calibrations persist the image-to-carpet matrix so the runtime does not need to redo
    // pose solving. Keep point-based files below for compatibility with older calibrations.
    if (document.contains("matrix") && document["matrix"].is_array()) {
        const auto& raw = document["matrix"];
        std::array<double, 9> matrix{};
        if (raw.size() == 3 && raw[0].is_array() && raw[1].is_array() && raw[2].is_array() &&
            raw[0].size() == 3 && raw[1].size() == 3 && raw[2].size() == 3) {
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    if (!raw[r][c].is_number()) return std::nullopt;
                    matrix[r * 3 + c] = raw[r][c].get<double>();
                }
            }
        } else if (raw.size() == 9) {
            for (size_t i = 0; i < 9; ++i) {
                if (!raw[i].is_number()) return std::nullopt;
                matrix[i] = raw[i].get<double>();
            }
        } else {
            return std::nullopt;
        }
        const int point_count = document.value("point_count", 4);
        const double reprojection = document.value("reprojection_ft", 0.0);
        Homography loaded(matrix, reprojection, point_count, field_length_ft, field_width_ft,
                          plane_height, source, trusted_override);
        return loaded.trustworthy() ? std::optional<Homography>(loaded) : std::nullopt;
    }

    if (!document.contains("points") || !document["points"].is_array()) return std::nullopt;

    std::vector<PointPair> points;
    for (const auto& entry : document["points"]) {
        if (!entry.contains("image") || !entry.contains("field")) continue;
        const auto& image = entry["image"];
        const auto& field = entry["field"];
        if (!image.is_array() || image.size() != 2) continue;
        if (!field.is_array() || field.size() != 2) continue;
        points.push_back(PointPair{image[0].get<double>(), image[1].get<double>(),
                                   field[0].get<double>(), field[1].get<double>()});
    }

    auto solved = solve_homography(points, field_length_ft, field_width_ft);
    if (!solved || !solved->trustworthy()) return std::nullopt;
    if (trusted_override.has_value() && !*trusted_override) return std::nullopt;
    return Homography(solved->matrix(), solved->reprojection_ft(), solved->point_count(),
                      field_length_ft, field_width_ft, plane_height, source, trusted_override);
}

}  // namespace frc::vision
