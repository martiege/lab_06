#pragma once

// Local 
#include "estimators/pose_estimator.h"
#include "utilities/camera_projection_measurement.h"

namespace estimators
{

/// \brief Iterative pose estimator for calibrated camera with 3D-2D correspondences.
/// This pose estimator need another pose estimator,
/// which it will use to initialize estimate and find inliers.
class MobaPoseEstimator : public PoseEstimator
{
public:
  /// \brief Constructs pose estimator.
  /// \param initial_pose_estimator Pointer to a pose estimator for initialization and inlier extraction.
  /// \param principal_point Principal point from camera calibration.
  /// \param focal_lengths Focal lengths from camera calibration.
  MobaPoseEstimator(PoseEstimator::Ptr initial_pose_estimator,
      const Eigen::Vector2d& principal_point,
      const Eigen::Vector2d& focal_lengths,
      bool use_distances_as_std);

  /// \brief Estimates camera pose from 3D-2D correspondences.
  /// \param image_points 2D image points.
  /// \param world_points 3D planar world points.
  /// \return The results. Check PoseEstimate::isFound() to check if solution was found.
  PoseEstimate estimate(const std::vector<cv::Point2f>& image_points,
                        const std::vector<cv::Point3f>& world_points,
                        const std::vector<float>& matched_distances) override;

private:
  PoseEstimator::Ptr initial_pose_estimator_;
  Eigen::Vector2d principal_point_;
  Eigen::Vector2d focal_lengths_;
  bool use_distances_as_std_;

  // Gauss-Newton optimization, minimizing reprojection error
  void optimize(const std::vector<utilities::CameraProjectionMeasurement>& measurements,
                PoseEstimate& initial_pose);

  // Transforming pixels to normalized image coordinates.
  Eigen::Vector2d toNormalized(const Eigen::Vector2d& pixel);
};

} // namespace estimators

