#include "lab_6.h"

// Matplot
#include <matplot/matplot.h>

// Standard Library
#include <map>
#include <set>
#include <chrono>
#include <vector>
#include <string>

// Visualising 
#include "visualise/ar_example.h"
#include "models/plane_world_model.h"
#include "visualise/scene_3d.h"

// Estimators
#include "estimators/homography_pose_estimator.h"
#include "estimators/pnp_pose_estimator.h"
#include "estimators/moba_pose_estimator.h"

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

// Make shorthand aliases for timing tools.
using Clock = std::chrono::high_resolution_clock;
using DurationInMs = std::chrono::duration<double, std::milli>;

// Convenient shorthand for distortion vectors.
namespace cv
{
using Vec5d = Vec<double, 5>;
}

// Struct for camera model data.
// I have chosen to use both eigen and opencv matrices for K,
// since I frequently will need both.
struct CameraModel
{
  Eigen::Matrix3d K;
  cv::Matx33d K_cv;
  cv::Vec5d dist_coeffs_cv;

  Eigen::Vector2d principalPoint() const { return {K(0,2), K(1,2)}; }
  Eigen::Vector2d focalLengths() const { return {K(0,0), K(1,1)}; }
};


CameraModel setupCameraModel()
{
  // TODO 1: Calibrate your camera using the application "opencv_interactive-calibration".

  // TODO 1: Set K according to calibration.
  // Set calibration matrix K.
  Eigen::Matrix3d K;
  K << 6.6051081297156020e+02, 0.0,                    3.1810845757653777e+02, 
       0.0,                    6.6051081297156020e+02, 2.3995332228230293e+02, 
       0.0,                    0.0,                    1.0; 

  // Create an OpenCV-version for convenience.
  cv::Matx33d K_cv;
  cv::eigen2cv(K, K_cv);

  // TODO 1: Set dist_coeffs_cv according to the calibration.
  // Set distortion coefficients [k1, k2, 0, 0, k3].
  cv::Vec5d dist_coeffs_cv{0.0, 2.2202255011309072e-01, 0.0, 0.0, -5.0348071005413975e-01};

  return CameraModel{K, K_cv, dist_coeffs_cv};
}


models::PlaneWorldModel createWorldModel()
{
  // Read "world" image corresponding to the chosen paper size.
  //std::string image_path = "../world_A4.png";
  std::string image_path = "../data/world_A3.png";
  cv::Mat world_image = cv::imread(image_path);
  if (world_image.empty())
  {
    throw std::runtime_error{"Could not find: " + image_path};
  }

  // Physical world sizes in meters.
  // Choose the paper size you have used.
  //const cv::Size2d a4_size{0.297, 0.210};
  const cv::Size2d a3_size{0.420, 0.297};

  // Grid size in meters.
  // This will be the physical size of axes in the visualization.
  //const double a4_grid_size{0.025};
  const double a3_grid_size{0.040};

  // Create world model.
  models::PlaneWorldModel world(world_image, a3_size, a3_grid_size);

  return world;
}


void lab6()
{
  // TODO 1: Calibrate camera and set parameters in setupCameraModel().
  // Get camera model parameters.
  const CameraModel camera_model = setupCameraModel();

  // Construct plane world model.
  const models::PlaneWorldModel world = createWorldModel();

  // TODO 2-6: Implement HomographyPoseEstimator.
  // TODO 7: Implement MobaPoseEstimator by finishing CameraProjectionMeasurement.
  // Construct pose estimator.
  Eigen::Matrix2d measurement_noise; 
  measurement_noise << 1.0, 0.0, 
                       0.0, 1.0; 

  auto init_estimator = std::make_shared<estimators::HomographyPoseEstimator>(
    camera_model.K
  );
//  MobaPoseEstimator pose_estimator(
//    init_estimator,
//    camera_model.principalPoint(),
//    camera_model.focalLengths(),
//    measurement_noise
//  );
  // HomographyPoseEstimator pose_estimator(camera_model.K);
  // PnPPoseEstimator pnp_pose_estimator(camera_model.K);

  std::map<std::string, std::pair<std::shared_ptr<estimators::PoseEstimator>, matplot::color>> pose_estimators;
  // pose_estimators["PnPPoseEstimator"] = std::make_shared<PnPPoseEstimator>(camera_model.K);
  // pose_estimators["HomographyEstimator"] = std::make_shared<HomographyPoseEstimator>(camera_model.K);
  pose_estimators["impr"] = std::make_pair(
      std::make_shared<estimators::MobaPoseEstimator>(
          init_estimator,
          camera_model.principalPoint(),
          camera_model.focalLengths(),
          true
      ),
      matplot::color::blue
  );
  pose_estimators["orig"] = std::make_pair(
    std::make_shared<estimators::MobaPoseEstimator>(
      init_estimator,
      camera_model.principalPoint(),
      camera_model.focalLengths(),
      false
    ),
    matplot::color::green
  );
  /*
  std::make_shared<MobaPoseEstimator>(
      init_estimator,
      camera_model.principalPoint(),
      camera_model.focalLengths(),
      false);
  */
  std::map<std::string, std::map<std::string, std::map<std::string, std::vector<double>>>> estimates;

  // Construct AR visualizer.
  visualise::ARExample ar_example(world.gridSize());

  // Construct 3D visualizer.
  visualise::Scene3D scene_3D{world};

  // Setup camera stream.
  // constThis int camera_id = 1; // Should be 0 or 1 on the lab PCs.
  const std::string data_path = "../data/lab_6_cameradata.avi"; 
  cv::VideoCapture cap(data_path);

  if (!cap.isOpened())
  {
    throw std::runtime_error("Could not open camera " + data_path);
  }

  std::cout << "Starting optimization...\n";
  for (unsigned int i = 0; true; ++i)
  {
    // Read a frame from the camera.
    cv::Mat frame;
    cap >> frame;

    if (! (i % 100))
      std::cout << "Image: " << i << '\n';

    if (frame.empty())
    {
      std::cout << "Camera feed empty\n";
      break;
    }

    // Undistort the frame using the camera model.
    cv::Mat undistorted_frame;
    cv::undistort(frame, undistorted_frame, camera_model.K_cv, camera_model.dist_coeffs_cv);
    cv::Mat gray_frame;
    cv::cvtColor(undistorted_frame, gray_frame, cv::COLOR_BGR2GRAY);

    // Find the correspondences between the detected image points and the world points.
    // Measure how long the processing takes.
    auto start = Clock::now();
    std::vector<cv::Point2f> matched_image_points;
    std::vector<cv::Point3f> matched_world_points;
    std::vector<float> matched_distances;
    world.findCorrespondences(gray_frame, matched_image_points, matched_world_points, matched_distances);
    auto end = Clock::now();
    DurationInMs correspondence_matching_duration = end - start;

    // Update the pose estimate.
    // Measure how long the processing takes.
    start = Clock::now();
    // PoseEstimate estimate = pose_estimator.estimate(matched_image_points, matched_world_points);
    // PoseEstimate estimate = pose_estimators["impr"].first->estimate(matched_image_points, matched_world_points, matched_distances);
    for (const auto& it : pose_estimators)
    {
      estimators::PoseEstimate estimate = it.second.first->estimate(matched_image_points, matched_world_points, matched_distances);

      estimates[it.first]["t"]["t"].push_back(i);

      if (estimate.isFound())
      {
        if (estimate.cost.first <= 0)
          std::cout << "First cost less than zero: " << estimate.cost.first << '\n';
        else
        {
          estimates[it.first]["cost"]["init_cost"].push_back(estimate.cost.first);
          estimates[it.first]["cost"]["init_t"].push_back(i);
        }
        if (estimate.cost.second <= 0)
          std::cout << "First cost less than zero: " << estimate.cost.second << '\n';
        else
        {
          estimates[it.first]["cost"]["final_cost"].push_back(estimate.cost.second);
          estimates[it.first]["cost"]["final_t"].push_back(i);
        }

        estimates[it.first]["expected"]["x"].push_back(estimate.pose_W_C.translation().x());
        estimates[it.first]["expected"]["y"].push_back(estimate.pose_W_C.translation().y());
        estimates[it.first]["expected"]["z"].push_back(estimate.pose_W_C.translation().z());

        estimates[it.first]["expected"]["rot\\_x"].push_back(estimate.pose_W_C.angleX());
        estimates[it.first]["expected"]["rot\\_y"].push_back(estimate.pose_W_C.angleY());
        estimates[it.first]["expected"]["rot\\_z"].push_back(estimate.pose_W_C.angleZ());

        estimates[it.first]["std"]["+x"].push_back(estimate.pose_W_C.translation().x() + std::sqrt(estimate.poseCovariance(0, 0)));
        estimates[it.first]["std"]["+y"].push_back(estimate.pose_W_C.translation().y() + std::sqrt(estimate.poseCovariance(1, 1)));
        estimates[it.first]["std"]["+z"].push_back(estimate.pose_W_C.translation().z() + std::sqrt(estimate.poseCovariance(2, 2)));

        estimates[it.first]["std"]["+rot\\_x"].push_back(estimate.pose_W_C.angleX() + std::sqrt(estimate.poseCovariance(3, 3)));
        estimates[it.first]["std"]["+rot\\_y"].push_back(estimate.pose_W_C.angleY() + std::sqrt(estimate.poseCovariance(4, 4)));
        estimates[it.first]["std"]["+rot\\_z"].push_back(estimate.pose_W_C.angleZ() + std::sqrt(estimate.poseCovariance(5, 5)));

        estimates[it.first]["std"]["-x"].push_back(estimate.pose_W_C.translation().x() - std::sqrt(estimate.poseCovariance(0, 0)));
        estimates[it.first]["std"]["-y"].push_back(estimate.pose_W_C.translation().y() - std::sqrt(estimate.poseCovariance(1, 1)));
        estimates[it.first]["std"]["-z"].push_back(estimate.pose_W_C.translation().z() - std::sqrt(estimate.poseCovariance(2, 2)));

        estimates[it.first]["std"]["-rot\\_x"].push_back(estimate.pose_W_C.angleX() - std::sqrt(estimate.poseCovariance(3, 3)));
        estimates[it.first]["std"]["-rot\\_y"].push_back(estimate.pose_W_C.angleY() - std::sqrt(estimate.poseCovariance(4, 4)));
        estimates[it.first]["std"]["-rot\\_z"].push_back(estimate.pose_W_C.angleZ() - std::sqrt(estimate.poseCovariance(5, 5)));
      }
      else
      {
        if (i == 0)
        {
          estimates[it.first]["expected"]["x"].push_back(0);
          estimates[it.first]["expected"]["y"].push_back(0);
          estimates[it.first]["expected"]["z"].push_back(0);

          estimates[it.first]["expected"]["rot\\_x"].push_back(0);
          estimates[it.first]["expected"]["rot\\_y"].push_back(0);
          estimates[it.first]["expected"]["rot\\_z"].push_back(0);

          estimates[it.first]["std"]["+x"].push_back(0);
          estimates[it.first]["std"]["+y"].push_back(0);
          estimates[it.first]["std"]["+z"].push_back(0);

          estimates[it.first]["std"]["+rot\\_x"].push_back(0);
          estimates[it.first]["std"]["+rot\\_y"].push_back(0);
          estimates[it.first]["std"]["+rot\\_z"].push_back(0);

          estimates[it.first]["std"]["-x"].push_back(0);
          estimates[it.first]["std"]["-y"].push_back(0);
          estimates[it.first]["std"]["-z"].push_back(0);

          estimates[it.first]["std"]["-rot\\_x"].push_back(0);
          estimates[it.first]["std"]["-rot\\_y"].push_back(0);
          estimates[it.first]["std"]["-rot\\_z"].push_back(0);
        }
        else
        {
          estimates[it.first]["expected"]["x"].push_back(estimates[it.first]["expected"]["x"].back());
          estimates[it.first]["expected"]["y"].push_back(estimates[it.first]["expected"]["y"].back());
          estimates[it.first]["expected"]["z"].push_back(estimates[it.first]["expected"]["z"].back());

          estimates[it.first]["expected"]["rot\\_x"].push_back(estimates[it.first]["expected"]["rot\\_x"].back());
          estimates[it.first]["expected"]["rot\\_y"].push_back(estimates[it.first]["expected"]["rot\\_y"].back());
          estimates[it.first]["expected"]["rot\\_z"].push_back(estimates[it.first]["expected"]["rot\\_z"].back());

          estimates[it.first]["std"]["+x"].push_back(estimates[it.first]["std"]["+x"].back());
          estimates[it.first]["std"]["+y"].push_back(estimates[it.first]["std"]["+y"].back());
          estimates[it.first]["std"]["+z"].push_back(estimates[it.first]["std"]["+z"].back());

          estimates[it.first]["std"]["+rot\\_x"].push_back(estimates[it.first]["std"]["+rot\\_x"].back());
          estimates[it.first]["std"]["+rot\\_y"].push_back(estimates[it.first]["std"]["+rot\\_y"].back());
          estimates[it.first]["std"]["+rot\\_z"].push_back(estimates[it.first]["std"]["+rot\\_z"].back());

          estimates[it.first]["std"]["-x"].push_back(estimates[it.first]["std"]["-x"].back());
          estimates[it.first]["std"]["-y"].push_back(estimates[it.first]["std"]["-y"].back());
          estimates[it.first]["std"]["-z"].push_back(estimates[it.first]["std"]["-z"].back());

          estimates[it.first]["std"]["-rot\\_x"].push_back(estimates[it.first]["std"]["-rot\\_x"].back());
          estimates[it.first]["std"]["-rot\\_y"].push_back(estimates[it.first]["std"]["-rot\\_y"].back());
          estimates[it.first]["std"]["-rot\\_z"].push_back(estimates[it.first]["std"]["-rot\\_z"].back());
        }
      }

    }

    end = Clock::now();
    DurationInMs pose_estimation_duration = end - start;

    /*
    // Update Augmented Reality visualization.
    ar_example.update(undistorted_frame, estimate, camera_model.K,
                      correspondence_matching_duration.count(),
                      pose_estimation_duration.count());

    // Update 3D visualization.
    scene_3D.update(undistorted_frame, estimate, camera_model.K);

    if (cv::waitKey(1) >= 0)
    {
      break;
    }
    */
  }

  std::cout << "Plotting...\n";

  std::vector<std::string> s = {"x", "y", "z", "rot\\_x", "rot\\_y", "rot\\_z"};
  for (auto & state : s)
  {
    //matplot::subplot(2, 3, i);
    std::cout << state << '\n';
    auto figure = matplot::figure(true);
    auto figure2 = matplot::figure(true);

    figure->size(1920, 1080);
    figure2->size(1920, 1080);

    auto ax1 = figure->add_subplot(2, 1, 0);
    auto ax2 = figure->add_subplot(2, 1, 1);

    auto ax3 = figure2->add_subplot(2, 1, 0);
    auto ax4 = figure2->add_subplot(2, 1, 1);

    // auto ax = figure->add_axes();
    ax1->hold(true);
    ax2->hold(true);
    ax3->hold(true);
    ax4->hold(true);

    std::vector<std::string> l1, l2, l3, l4;

    std::set<std::string> error_set;
    for (const auto &it1 : estimates)
    {
      std::vector<double> t = it1.second.at("t").at("t");

      std::vector<double> x = it1.second.at("expected").at(state);
      std::vector<double> x_plus_std = it1.second.at("std").at("+" + state);
      std::vector<double> x_minus_std = it1.second.at("std").at("-" + state);

      ax1->plot(t, x)->color(pose_estimators[it1.first].second);
      ax1->plot(t, x_plus_std, "+")->color(pose_estimators[it1.first].second);
      ax1->plot(t, x_minus_std, "--")->color(pose_estimators[it1.first].second);

      ax3->plot(t, x)->color(pose_estimators[it1.first].second);

      l1.push_back(it1.first);
      l1.push_back(it1.first + " (+1 std)");
      l1.push_back(it1.first + " (-1 std)");

      l3.push_back(it1.first);

      for (const auto& it2 : estimates)
      {
        if (it1.first == it2.first || (error_set.find(it1.first) != error_set.end() && error_set.find(it2.first) != error_set.end()))
          continue;

        error_set.insert(it1.first);
        error_set.insert(it2.first);

        std::vector<double> t_2 = it2.second.at("t").at("t");

        std::vector<double> x_2 = it2.second.at("expected").at(state);
        std::vector<double> x_plus_std_2 = it2.second.at("std").at("+" + state);
        std::vector<double> x_minus_std_2 = it2.second.at("std").at("-" + state);

        for (unsigned int i = 0; i < t.size(); ++i)
        {
          double std_1 = x_plus_std[i] - x[i];
          double std_2 = x_plus_std_2[i] - x_2[i];
          double std_e = std::sqrt(std_1 * std_1 + std_2 * std_2);

          x_2[i] = x[i] - x_2[i];
          x_plus_std[i] = x_2[i] + std_e;
          x_minus_std[i] = x_2[i] - std_e;
        }

        ax2->plot(t, x_2)->color(pose_estimators[it1.first].second);
        ax2->plot(t, x_plus_std_2, "+")->color(pose_estimators[it1.first].second);
        ax2->plot(t, x_minus_std_2, "--")->color(pose_estimators[it1.first].second);

        ax4->plot(t, x_2)->color(pose_estimators[it1.first].second);

        l2.push_back("e: " + it1.first + "-" + it2.first);
        l2.push_back("e: " + it1.first + "-" + it2.first + " (+1 std)");
        l2.push_back("e: " + it1.first + "-" + it2.first + " (-1 std)");

        l4.push_back("e: " + it1.first + "-" + it2.first);
      }
    }

    ax1->hold(false);
    ax1->xlabel("Image");
    ax1->ylabel(state);
    ax1->title(state);
    ax1->legend(l1);

    ax2->hold(false);
    ax2->xlabel("Image");
    ax2->ylabel(state);
    ax2->title("Error: " + state);
    ax2->legend(l2);

    ax3->hold(false);
    ax3->xlabel("Image");
    ax3->ylabel(state);
    ax3->title(state);
    ax3->legend(l3);

    ax4->hold(false);
    ax4->xlabel("Image");
    ax4->ylabel(state);
    ax4->title("Error: " + state);
    ax4->legend(l4);

    figure->save("/home/martin/dev/lab_06/results/" + state + "_with_covariance.png");
    figure2->save("/home/martin/dev/lab_06/results/" + state + "_without_covariance.png");

    figure->save("/home/martin/dev/lab_06/results/" + state + "_with_covariance.svg");
    figure2->save("/home/martin/dev/lab_06/results/" + state + "_without_covariance.svg");
    // figure->show();
  }
  // matplot::show();

  auto fig = matplot::figure();
  auto fig2 = matplot::figure();
  fig->size(1920, 1080);
  fig2->size(1920, 1080);

  auto ax = fig->add_axes();
  ax->hold(true);

  auto ax2 = fig2->add_axes();
  ax2->hold(true);

  std::vector<std::string> l, l2;
  for (const auto &it1 : estimates)
  {
    ax->plot3(
      it1.second.at("expected").at("x"),
      it1.second.at("expected").at("y"),
      it1.second.at("expected").at("z")
    )->color(pose_estimators[it1.first].second);

    l.push_back(it1.first);

    std::vector<double> e;
    for (unsigned int i = 0; i < it1.second.at("cost").at("init_t").size(); ++i)
      e.push_back(std::abs(it1.second.at("cost").at("init_cost").at(i) - it1.second.at("cost").at("final_cost").at(i)));

    ax2->semilogy(it1.second.at("cost").at("init_t"),
        e)->color(pose_estimators[it1.first].second);
//    ax2->semilogy(
//      it1.second.at("cost").at("init_t"),
//      it1.second.at("cost").at("init_cost"),
//      "--"
//    )->color(pose_estimators[it1.first].second);
//
//    ax2->semilogy(
//      it1.second.at("cost").at("final_t"),
//      it1.second.at("cost").at("final_cost"),
//      "+"
//    )->color(pose_estimators[it1.first].second);

    l2.push_back(it1.first);
    // l2.push_back(it1.first + " init cost");
    // l2.push_back(it1.first + " final cost");
  }
  ax->hold(false);
  ax->title("Trajectories");
  ax->xlabel("x");
  ax->ylabel("y");
  ax->zlabel("z");
  ax->legend(l);

  ax2->hold(false);
  ax2->title("Change in cost");
  ax2->xlabel("Image");
  ax2->ylabel("Cost");
  ax2->legend(l2);

  fig->save("/home/martin/dev/lab_06/results/trajectories.png");
  fig2->save("/home/martin/dev/lab_06/results/cost.png");

  fig->save("/home/martin/dev/lab_06/results/trajectories.svg");
  fig2->save("/home/martin/dev/lab_06/results/cost.svg");

  std::cout << "Done\n";
}
