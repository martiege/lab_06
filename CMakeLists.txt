cmake_minimum_required( VERSION 3.14 )
set(CMAKE_CXX_STANDARD 17)

project(lab_6_pose)

# Compiler flags
if (MSVC)
  set(compiler_options /W4)
  add_definitions(/D_USE_MATH_DEFINES)
else ()
  set(compiler_options -Wall -Wextra -Wpedantic)
endif (MSVC)

# Add libraries.
find_package(OpenCV 4.0.1 CONFIG REQUIRED)
find_package(Eigen3 3.3.4 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Matplot++)

include_directories(
  include
)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/lab_6.cpp
  src/estimators/homography_pose_estimator.cpp
  src/estimators/moba_pose_estimator.cpp
  src/estimators/pnp_pose_estimator.cpp
  src/models/plane_world_model.cpp
  src/utilities/feature_utils.cpp
  src/utilities/camera_projection_measurement.cpp
  src/visualise/scene_3d.cpp
  src/visualise/ar_example.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
  Matplot++::matplot
)

target_compile_options(${PROJECT_NAME} PRIVATE ${compiler_options})
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_STANDARD 14
  )
