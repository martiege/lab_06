#pragma once

// OpenCV
#include <opencv2/viz.hpp>

// Local
#include "estimators/pose_estimator.h"
#include "models/plane_world_model.h"

namespace visualise
{

/// \brief 3D visualizer.
class Scene3D
{
public:
  /// \brief Constructs a 3D representation of the planar world model.
  /// \param world World model.
  explicit Scene3D(const models::PlaneWorldModel& world);

  /// \brief Updated the visualization.
  /// \param image Current frame.
  /// \param estimate Current pose estimate.
  /// \param K Current camera calibration matrix.
  void update(const cv::Mat& image, const estimators::PoseEstimate& estimate, const Eigen::Matrix3d& K);

private:
  cv::viz::Viz3d vis_3d_;
  bool has_camera_;
};

} // namespace visualise

