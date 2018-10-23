#-------------------------------------------------
#
# Project created by QtCreator 2018-06-24T15:10:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Detect_3D
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += C:/intel_realsense_SDK2.0_2.10.4_modify/include

HEADERS += C:/intel_realsense_SDK2.0_2.10.4_modify/src/ \
           example_window.hpp \
    cvhelpers.hpp \
    feature_function.hpp \
    offline_tracking.hpp

LIBS += C:/intel_realsense_SDK2.0_2.10.4_modify/lib/realsense2.lib

LIBS += C:/intel_realsense_SDK2.0_2.10.4_modify/lib/glfw3dll.lib\
        C:/intel_realsense_SDK2.0_2.10.4_modify/lib/glfw3dll.exp\
        C:/intel_realsense_SDK2.0_2.10.4_modify/lib/glfw3.lib

#LIBS += -lglfw3
LIBS += -lopengl32
LIBS += -lglu32

#INCLUDEPATH += C:\\opencv3.2.0_x64_vc2015\\include \
#                C:\\opencv3.2.0_x64_vc2015\\include\\opencv \
#                C:\\opencv3.2.0_x64_vc2015\\include\\opencv2 \

#LIBS +=  C:\\opencv3.2.0_x64_vc2015\\lib\\opencv_world320.lib \
#         C:\\opencv3.2.0_x64_vc2015\\lib\\opencv_world320d.lib \

INCLUDEPATH += C:\\opencv3.2.0_x64_vc2015_contribute\\include \
                C:\\opencv3.2.0_x64_vc2015_contribute\\include\\opencv \
                C:\\opencv3.2.0_x64_vc2015_contribute\\include\\opencv2 \

LIBS +=  C:\\opencv3.2.0_x64_vc2015_contribute\\lib\\opencv_world320.lib \
         C:\\opencv3.2.0_x64_vc2015_contribute\\lib\\opencv_world320d.lib \
         C:\\opencv3.2.0_x64_vc2015_contribute\\lib\\opencv_ximgproc320.lib \
         C:\\opencv3.2.0_x64_vc2015_contribute\\lib\\opencv_xfeatures2d320.lib \
         C:\\opencv3.2.0_x64_vc2015_contribute\\lib\\opencv_xfeatures2d320d.lib \

INCLUDEPATH += C:\\yolov2_x64_vc2015\\src\

LIBS +=   C:\yolov2_x64_vc2015\lib_dll\\yolo_cpp_dll.lib \

QMAKE_CFLAGS += -std=c++11
QMAKE_CXXFLAGS = $$QMAKE_CFLAGS

CONFIG += link_pkgconfig

SOURCES += \
        main.cpp \
        detect_3d.cpp

HEADERS += \
        detect_3d.hpp

FORMS += \
        detect_3d.ui
