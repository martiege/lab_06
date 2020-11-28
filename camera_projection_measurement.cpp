#include "camera_projection_measurement.h"
#include <unsupported/Eigen/MatrixFunctions>

CameraProjectionMeasurement::CameraProjectionMeasurement(
    const Eigen::Vector2d& normalized_plane_point,
    const Eigen::Vector3d& world_point,
    const double& point_std)
    : world_point_{world_point}
    , normalized_plane_point_{normalized_plane_point}
    , inv_root_point_covariance_{1.0 / std::abs(point_std)}
{

}

LinearizedCameraProjectionMeasurement CameraProjectionMeasurement::linearize(const Sophus::SE3d& current_state) const
{
  // TODO 7.1: Use current_state (T_w_c) and world_point_ (x_w) to predict x_c.
  // Transform world point to camera coordinate frame based on current state estimate.
  Eigen::Vector3d x_c_pred;
  x_c_pred = current_state.inverse() * world_point_; 

  // TODO 7.2: Use x_c_pred to predict x_n.
  // Predict normalized image coordinate based on current state estimate.
  Eigen::Vector2d x_n_pred;
  x_n_pred = x_c_pred.hnormalized(); 

  // Construct linearization object.
  LinearizedCameraProjectionMeasurement linearization;

  // TODO 7.3: Use normalized_plane_point_ to compute the measurement error.
  // Compute measurement error.
  linearization.b = inv_root_point_covariance_ * (normalized_plane_point_ - x_n_pred);

  // TODO 7.4: Use the predicted x_c_pred and x_n_pred to compute the measurement Jacobian.
  // Compute measurement Jacobian.
  double d = 1.0 / x_c_pred.z(); 
  double x_n = x_n_pred.x(); 
  double y_n = x_n_pred.y(); 
  double x_n_y_n = x_n * y_n; 
  linearization.A << -d,  0, d * x_n, x_n_y_n,       -1 - x_n * x_n,  y_n, 
                      0, -d, d * y_n, 1 + y_n * y_n, - x_n_y_n,      -x_n; 

  linearization.A = inv_root_point_covariance_ * linearization.A;

  return linearization;
}
