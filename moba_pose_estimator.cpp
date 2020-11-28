#include "moba_pose_estimator.h"
#include <utility>

MobaPoseEstimator::MobaPoseEstimator(PoseEstimator::Ptr initial_pose_estimator,
                                     const Eigen::Vector2d& principal_point,
                                     const Eigen::Vector2d& focal_lengths,
                                     bool use_distances_as_std)
  : initial_pose_estimator_{std::move(initial_pose_estimator)}
  , principal_point_{principal_point}
  , focal_lengths_{focal_lengths}
  , use_distances_as_std_{use_distances_as_std}
{

}

PoseEstimate
MobaPoseEstimator::estimate(const std::vector<cv::Point2f>& image_points,
                            const std::vector<cv::Point3f>& world_points,
                            const std::vector<float>& matched_distances)
{
  // Get initial pose estimate.
  PoseEstimate init_estimate = initial_pose_estimator_->estimate(image_points, world_points, matched_distances);

  if (!init_estimate.isFound())
  {
    return init_estimate;
  }

  // Create measurement set.
  const size_t num_inliers = init_estimate.image_inlier_points.size();
  std::vector<CameraProjectionMeasurement> measurements;
  for (size_t i = 0; i < num_inliers; ++i)
  {
    Eigen::Vector2d image_point{init_estimate.image_inlier_points[i].x,
                                init_estimate.image_inlier_points[i].y};

    Eigen::Vector3d world_point{init_estimate.world_inlier_points[i].x,
                                init_estimate.world_inlier_points[i].y,
                                init_estimate.world_inlier_points[i].z};

    measurements.emplace_back(
      toNormalized(image_point),
      world_point,
      (use_distances_as_std_) ? matched_distances[i] : 1
    );
  }

  // Optimize and update estimate.
  optimize(measurements, init_estimate);

  return init_estimate;
}

Eigen::Vector2d MobaPoseEstimator::toNormalized(const Eigen::Vector2d& pixel)
{
  return (pixel - principal_point_).array() / focal_lengths_.array();
}

void
MobaPoseEstimator::optimize(const std::vector<CameraProjectionMeasurement>& measurements,
                            PoseEstimate& initial_pose)
{
  const size_t max_iterations = 5;
  const int measure_dim = CameraProjectionMeasurement::measure_dim;
  const int state_dim = CameraProjectionMeasurement::state_dim;

  Eigen::MatrixXd A(measure_dim * measurements.size(), state_dim);
  Eigen::VectorXd b(measure_dim * measurements.size());

  Eigen::MatrixXd covariance(state_dim, state_dim);

  Sophus::SE3d current_state = initial_pose.pose_W_C;

  // Comment when done!
  // std::cout << "---------" << std::endl;

  size_t iteration = 0;
  // double curr_cost = 0.0f;
  while (iteration < max_iterations)
  {
    // Linearize.
    // Build A and b from each measurement.
    for (size_t j=0; j < measurements.size(); ++j)
    {
      const LinearizedCameraProjectionMeasurement linearization = measurements[j].linearize(current_state);

      A.block(measure_dim*j, 0, measure_dim, state_dim) = linearization.A;
      b.segment(measure_dim*j, measure_dim) = linearization.b;
    }

    // Compute current cost.
    // curr_cost = b.squaredNorm();

    // Remove when done!
    // std::cout << "Cost before update: " << curr_cost << std::endl;

    // Solve linearized system.
    Sophus::SE3d::Tangent update = A.householderQr().solve(b);
    current_state = current_state * Sophus::SE3d::exp(update);
//    Eigen::MatrixXd ATA = A.transpose() * A;
//    Eigen::MatrixXd diagonalATA = ATA.diagonal().asDiagonal().toDenseMatrix();
//
//    Sophus::SE3d::Tangent update = (ATA + lambda * diagonalATA).householderQr().solve(A.transpose() * b);
//
//    Sophus::SE3d updated_state = current_state * Sophus::SE3d::exp(update);
//
//    for (size_t j=0; j < measurements.size(); ++j)
//    {
//      const LinearizedCameraProjectionMeasurement linearization = measurements[j].linearize(updated_state, measurement_noise_sqrt_lu_);
//
//      A.block(measure_dim*j, 0, measure_dim, state_dim) = linearization.A;
//      b.segment(measure_dim*j, measure_dim) = linearization.b;
//    }
//
//    if (b.squaredNorm() < curr_cost)
//    {
//      // Update state.
//      current_state = updated_state;
//      lambda /= 10;
//    }
//    else
//    {
//      lambda *= 10;
//    }
//

    ++iteration;
  }

  covariance = (A.transpose() * A).inverse();

  initial_pose.pose_W_C = current_state;
  initial_pose.poseCovariance = covariance;
}


