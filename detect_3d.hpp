#ifndef DETECT_3D_HPP
#define DETECT_3D_HPP

#define GLFW_INCLUDE_GLU
#define OPENCV

#include <QMainWindow>
#include <librealsense2/rs.hpp>
#include "opencv.hpp"
#include "opencv2/dnn.hpp"
#include <yolo_v2_class.hpp>
#include <GLFW/glfw3.h>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>

#include <QString>
#include <QDebug>
#include <QFile>
#include <QFileDialog>

struct bbox_t_history:bbox_t
{
    QList<unsigned int> history;
    QList<cv::Point2f> trajectory;
    unsigned int lost_frame;
    QList<cv::Mat> frame_mat;
    QList<cv::Point2d>  width_height;
    double maturity;
    int size;
};

struct global_coor{
    cv::Point2f global_point;
    unsigned int global_fruit_ID;
    double maturity;
    int size;
};

namespace Ui {
class detect_3d;
}

class detect_3d : public QMainWindow
{
    Q_OBJECT

public:
    explicit detect_3d(QWidget *parent = 0);
    ~detect_3d();

private slots:
    void on_save_stateChanged(int arg1);

    void on_distance_clicked();

    void on_point_cloud_clicked();

    void on_actionexit_triggered();


    void on_camera_pose_clicked();

    void on_save_pose_stateChanged(int arg1);

    void on_scale_valueChanged(int arg1);

    void on_frame_valueChanged(int arg1);

    void on_epipolar_clicked();

    void on_track_line_clicked();

    void on_tracking_clicked();

    void on_fast_track_clicked();

private:
    Ui::detect_3d *ui;
    bool save_dis;
    bool save_pose;
    QString dis_path;
    QString pose_path;
    int scale;
    int frame;

};

#endif // DETECT_3D_HPP
