# Fruit-Tracking-with-Realsense-D435

Tracking fruits with realsense D435

Function
a. Get distance: get the distance of objects detected by yolo detector
    (a) ybias means the result text shown below the center of the object
b. Draw Point cloud: use GLFW to draw point cloud
c. Epipolar Geometry: draw epipolar line between two image using cv::findFundamentalMat
d. Camera Pose:
    (a) Refer to https://avisingh599.github.io/vision/monocular-vo/
    (b) Drawing visual odometry
    (c) Scale: space between continuous point in 3D
    (d) Frame: fps to calculate once
e. Tracking
    (a) Draw Tracking Line: Just for testing
    (b) Tracking: with bug
    (c) Faster Tracking: when the tracked point < 2000, re-detect the image feature points

.dll must include:
a. CUDA: cublas64_80.dll curand64_80.dll
b. GLFW: glfw3.dll opengl32sw.dll glfw3dll.exp (optional) 
c. openCV:
    opencv_core320.dll opencv_core320d.dll opencv_cudaarithm320.dll opencv_cudaarithm320d.dll
    opencv_cudev320.dll opencvcudev320d.dll opencv_features2d320.dll opencv_features2d320d.dll
    opencv_ffmpeg320_64.dll opencv_flann320.dll opencv_flann320d.dll opencv_imgproc320.dll opencv_improc320d.dll
    opencv_world320.dll opencv_world320d.dll opencv_xfeatures2d320.dll opencv_xfeatures2d320d.dll
d. YOLOv2: yolo_cpp_dll.dll
e. Realsense: realsense2.dll
f. Qt core: D3Dcompiler_47.dll libEGL.dll libGLESV2.dll Qt5Core.dll Qt5Gui.dll Qt5Svg.dll Qt5Widgets.dll
