// Image-to-field mapping for a fixed camera looking at the flat field.
//
// A homography turns pixel positions into field feet, which is what makes speed and distance
// comparable between venues, camera positions and zoom levels. A pixel means something different
// in every shot; a foot does not.
//
// The maths here is a port of ingest/collection/homography.py, which was validated against
// synthetic geometry where the correct answer is known exactly. Porting proven arithmetic is
// deliberate: a wrong homography does not crash, it reports confident, wrong distances, and every
// position and speed derived from it inherits the error in silence.
//
// Calibration is supplied, never guessed. No AprilTag layout is invented in this file.

#ifndef FRC_HOMOGRAPHY_H
#define FRC_HOMOGRAPHY_H

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace frc::vision {

// Reprojection error, in feet, above which a solution is not trusted.
inline constexpr double kMaxReprojectionFt = 1.5;

// Slack outside the field before a mapped point counts as off-field. Bumpers overhang and the
// camera sees past the guardrail, so the exact boundary is not a hard edge.
inline constexpr double kFieldMarginFt = 4.0;
// Used to reject identity swaps and bad mappings before they become scouting metrics.
inline constexpr double kMaxPlausibleFtps = 20.0;

struct PointPair {
    double image_x = 0.0;
    double image_y = 0.0;
    double field_x = 0.0;
    double field_y = 0.0;
};

class Homography {
public:
    Homography() = default;
    Homography(std::array<double, 9> matrix, double reprojection_ft, int point_count,
               double field_length_ft, double field_width_ft,
               std::optional<double> plane_height_ft = std::nullopt,
               std::string source = "unknown",
               std::optional<bool> trusted_override = std::nullopt);

    // Four correspondences determine a homography exactly, so the fit reproduces them with zero
    // error however wrong they are -- mistype a corner by twenty feet and the residual is still
    // zero. Error only becomes evidence from the fifth point on. Callers should not read a zero
    // from a four-point solve as confirmation of anything.
    [[nodiscard]] bool has_redundancy() const { return point_count_ >= 5; }
    [[nodiscard]] bool trustworthy() const;

    [[nodiscard]] double reprojection_ft() const { return reprojection_ft_; }
    [[nodiscard]] int point_count() const { return point_count_; }
    [[nodiscard]] const std::array<double, 9>& matrix() const { return matrix_; }
    [[nodiscard]] const std::optional<double>& plane_height_ft() const { return plane_height_ft_; }
    [[nodiscard]] const std::string& source() const { return source_; }

    // Pixel -> field feet. Returns nullopt for a point on the plane's horizon line, which has no
    // finite field position.
    [[nodiscard]] std::optional<std::pair<double, double>> to_field(double x, double y) const;

    // Field position of a normalised detection box, taken from its BOTTOM-CENTRE. A robot stands
    // on the carpet and the carpet is the plane being modelled; the box centre floats up the
    // robot's body and maps to a point behind where it actually is.
    [[nodiscard]] std::optional<std::pair<double, double>> box_to_field(
        double x, double y, double w, double h, int image_w, int image_h) const;

    [[nodiscard]] bool on_field(double x, double y) const;

private:
    std::array<double, 9> matrix_{1, 0, 0, 0, 1, 0, 0, 0, 1};
    double reprojection_ft_ = 0.0;
    int point_count_ = 0;
    double field_length_ft_ = 0.0;
    double field_width_ft_ = 0.0;
    std::optional<double> plane_height_ft_;
    std::string source_ = "unknown";
    std::optional<bool> trusted_override_;
};

// Solve from at least four correspondences. Returns nullopt when the input cannot describe a
// mapping at all -- too few points, or a degenerate arrangement such as points on a line.
std::optional<Homography> solve_homography(const std::vector<PointPair>& points,
                                           double field_length_ft, double field_width_ft);

// Load calibration from the path in FRC_HOMOGRAPHY_CONFIG, if set. Absent or unreadable yields
// nullopt, which leaves the analyzer reporting homography_ok=false and null field coordinates --
// the same honest output it produced before this existed.
std::optional<Homography> load_homography(double field_length_ft, double field_width_ft);

}  // namespace frc::vision

#endif  // FRC_HOMOGRAPHY_H
