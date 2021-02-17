#pragma once

// OpenCV
#include "opencv2/core.hpp"

// Sophus
#include "sophus/se3.hpp"

// Standard Library
#include <memory>

namespace estimators
{

/// \brief Struct for 3D-2D pose estimation results
struct PoseEstimate
{
  Sophus::SE3d pose_W_C;                        /// Camera pose in the world.
  Eigen::Matrix<double, 6, 6> poseCovariance;   /// Pose covariance
  std::vector<cv::Point2f> image_inlier_points; /// 2D inlier image points.
  std::vector<cv::Point3f> world_inlier_points; /// 3D inlier world points.
  std::pair<double, double> cost;

  /// \brief Checks if estimation succeeded.
  /// \return True if result was found.
  bool isFound() const
  {
    // Default identity orientation means looking away,
    // therefore using default value when no valid estimate was found.
    return !pose_W_C.rotationMatrix().isIdentity(1e-8);
  }
};

/// \brief Interface for 3D-2D pose estimators.
class PoseEstimator
{
public:
  virtual ~PoseEstimator() = default;

  /// \brief Estimates camera pose from 3D-2D correspondences.
  /// \param image_points 2D image points.
  /// \param world_points 3D world points.
  /// \return The results. Check PoseEstimate::isFound() to check if solution was found.
  virtual PoseEstimate estimate(const std::vector<cv::Point2f>& image_points,
                                const std::vector<cv::Point3f>& world_points,
                                const std::vector<float>& matched_distances) = 0;

  /// \brief Shared pointer to PoseEstimator, for convenience.
  using Ptr = std::shared_ptr<PoseEstimator>;
};

} // namespace estimators

