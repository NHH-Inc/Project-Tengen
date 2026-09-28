cmake_minimum_required(VERSION 3.14)
project(FRCAnalysisBackend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV decodes the video and prepares detector inputs. ONNX Runtime is optional for the Linux
# contract-only CI build; when vcpkg supplies it, the Windows binary includes RF-DETR inference.
option(FRC_BUILD_FUEL_TRACKER "Build the experimental YOLO + yellow fuel runtime" ON)
set(FRC_OPENCV_COMPONENTS core imgproc videoio)
if (FRC_BUILD_FUEL_TRACKER)
    list(APPEND FRC_OPENCV_COMPONENTS dnn highgui)
endif()
# OpenCV 5 moved findHomography from calib3d into geometry; the compatibility
# header keeps the source valid with both 4.x and 5.x.
find_package(OpenCV REQUIRED COMPONENTS ${FRC_OPENCV_COMPONENTS}
             OPTIONAL_COMPONENTS calib3d geometry)
if (NOT OpenCV_calib3d_FOUND AND NOT OpenCV_geometry_FOUND)
    message(FATAL_ERROR "OpenCV must provide calib3d (4.x) or geometry (5.x)")
endif()
find_package(onnxruntime CONFIG QUIET)

# vcpkg supplies this on Robert's Windows machine. Keep the FetchContent fallback so the
# Linux CI job remains self-contained and does not need a second package manager.
find_package(nlohmann_json CONFIG QUIET)
if (NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(json)
endif()

# Analysis executable
add_executable(analysis
    src/main.cpp
    src/ContractModels.h
    src/IoUTracker.cpp
    src/IoUTracker.h
    src/RobotDetector.cpp
    src/RobotDetector.h
    src/Homography.cpp
)

target_link_libraries(analysis PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})

# Tracker tests. The tracker is pure geometry, so it needs neither OpenCV nor a model -- which is
# what makes the awkward cases (crossing robots, occlusions, camera cuts) cheap to test directly.
add_executable(tracker_test
    tests/tracker_test.cpp
    src/IoUTracker.cpp
)
target_link_libraries(tracker_test PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})

# Detector post-processing tests. The arithmetic between a raw tensor and a box on a robot is
# pure, so it is tested here without a model or ONNX Runtime -- the ONNX session itself is behind
# FRC_HAVE_ONNXRUNTIME, which this target deliberately does not define.
add_executable(detector_test
    tests/detector_test.cpp
    src/RobotDetector.cpp
)
target_link_libraries(detector_test PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})

enable_testing()
add_test(NAME tracker_test COMMAND tracker_test)
add_test(NAME detector_test COMMAND detector_test)

if (FRC_BUILD_FUEL_TRACKER)
    add_executable(fuel_tracker src/fuel_tracker.cpp)
    target_link_libraries(fuel_tracker PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})
    set_target_properties(fuel_tracker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_executable(fuel_tracker_test tests/fuel_tracker_test.cpp)
    target_link_libraries(fuel_tracker_test PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})
    add_test(NAME fuel_tracker_test COMMAND fuel_tracker_test)
    add_test(NAME fuel_tracker_help COMMAND fuel_tracker --help)
endif()


if (TARGET onnxruntime::onnxruntime)
    target_link_libraries(analysis PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(analysis PRIVATE FRC_HAVE_ONNXRUNTIME=1)
endif()

# Installation/Output settings
set_target_properties(analysis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
