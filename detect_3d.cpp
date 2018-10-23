#include "detect_3d.hpp"
#include "ui_detect_3d.h"

#include <example_window.hpp>
#include <cvhelpers.hpp>
#include <feature_function.hpp>
#include <offline_tracking.hpp>

using namespace rs2;
using namespace cv;

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vecd, std::vector<std::string> obj_namesd);
std::vector<std::string> objects_names_from_file(std::string const filename);
void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names);

detect_3d::detect_3d(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::detect_3d)
{
    ui->setupUi(this);
    save_dis = false;
    save_pose = false;
    dis_path = "./distance_result/distance.csv";
    pose_path = "./pose_result/pose.csv";
    ui->save_path->setText(dis_path);
    ui->pose_save->setText(pose_path);
    frame = ui->frame->value();
    scale = ui->scale->value();

}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id){
            std::cout << obj_names[i.obj_id] << " - ";
        }
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
                  << ", w = " << i.w << ", h = " << i.h
                  << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vecd, std::vector<std::string> obj_namesd) {
    for (auto &i : result_vecd) {
        cv::Scalar color(60, 160, 260);

        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
        if (obj_namesd.size() > i.obj_id) {
            std::string obj_name = obj_namesd[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 3, 0), std::max((int)i.y - 30, 0)),
                          cv::Point2f(std::min((int)i.x + max_width, mat_img.cols-1), std::min((int)i.y, mat_img.rows-1)),
                          color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y + 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
        }
    }
}

void drawlines(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Vec3f> lines, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2){
    cv::Scalar color(0, 0, 255);
    qDebug() << "line size: " << lines.size();
    qDebug() << "points1 size: " << points1.size();
    qDebug() << "points2 size: " << points2.size();
    for(int i = 0 ; i < points1.size() ; i++){
        cv::circle(img1, points1.at(i), 2, color, -1);
        cv::line(img1, cv::Point(0, -lines[i][2] / lines[i][1]), cv::Point(img1.cols, -(lines[i][2] + lines[i][0] * img1.cols) / lines[i][1]), color);
    }
    for(int i = 0 ; i < points2.size() ; i++){
        cv::circle(img2, points2.at(i), 2, color, -1);
    }
}



detect_3d::~detect_3d()
{
    delete ui;
}

void detect_3d::on_save_stateChanged(int arg1)
{
    if(arg1 == Qt::Checked) save_dis = true;
    else    save_dis = false;
}

void detect_3d::on_distance_clicked()
{
    std::string save = ui->save_path->text().toStdString();
    std::string cfg_file = "./model/yolo-voc.cfg";
    std::string weights_file = "./model/yolo-voc.weights";
    std::string names_file = "./model/voc.names";

    QString status = "Loaing detector....";
    ui->state->append(status);
    QApplication::processEvents();
    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);

    // Create a Pipeline, which serves as a top-level API for streaming and processing frames
    rs2::pipeline p;
    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;
    //        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    //        cfg.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
    //        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    qDebug() << "finish enable";

    // Configure and start the pipeline
    //        p.start(cfg);
    QFile coutresult(ui->save_path->text());
    QTextStream out(&coutresult);
    qDebug() << ui->save_path->text();
    qDebug() << save_dis;
    if(save_dis){
        if(coutresult.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
            out << "\n";
        }
    }
    coutresult.close();
    qDebug() << "Start";

    status = "Start streaming....";
    ui->state->append(status);
    QApplication::processEvents();
    auto config = p.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    cv::VideoWriter video_writer, video_writer2, video_writer3, video_writer4, video_writer5, video_writer44;
    video_writer.open("video/video_src.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));
    video_writer2.open("video/video_RGB.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));
    video_writer3.open("video/video_IR.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));
    video_writer4.open("video/video_depth.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));
    video_writer44.open("video/video_depth_src.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));
    video_writer5.open("video/video_distance.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(900, 640));




    cv::Mat draw_dis(640, 900, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::line(draw_dis, cv::Point(0, 128), cv::Point(5, 128), cv::Scalar(255, 200, 200));
    cv::putText(draw_dis, "4 (m)", cv::Point(7, 128), 0, 0.5, cv::Scalar(255, 200, 200));
    cv::line(draw_dis, cv::Point(0, 256), cv::Point(5, 256), cv::Scalar(255, 200, 200));
    cv::putText(draw_dis, "3 (m)", cv::Point(7, 256), 0, 0.5, cv::Scalar(255, 200, 200));
    cv::line(draw_dis, cv::Point(0, 384), cv::Point(5, 384), cv::Scalar(255, 200, 200));
    cv::putText(draw_dis, "2 (m)", cv::Point(7, 384), 0, 0.5, cv::Scalar(255, 200, 200));
    cv::line(draw_dis, cv::Point(0, 512), cv::Point(5, 512), cv::Scalar(255, 200, 200));
    cv::putText(draw_dis, "1 (m)", cv::Point(7, 512), 0, 0.5, cv::Scalar(255, 200, 200));

    int i = 0;
    while (1)
    {
        // Block program until frames arrive
        //            rs2::frameset frames = p.wait_for_frames();
        //            qDebug() << "get frame";

        auto data = p.wait_for_frames();
        data = align_to.process(data);
        auto color_frame = data.get_color_frame();
        rs2::depth_frame depth_frame = data.get_depth_frame();

        //        cv::Mat depth_mat = depth_frame_to_meters(p, depth_frame);
        //            auto color_mat = frame_to_mat(color_frame);
        //            auto depth_mat = frame_to_mat(depth_frame);

        //            cv::imshow("color", color_mat);
        //            cv::imshow("depth", depth_mat);
        //            cv::waitKey(33);

        //            rs2::frame color_frame = frames.get_color_frame();
        //            rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        //            rs2::frame depth_frame = frames.get_depth_frame();

        //            qDebug() << "get color frame";

        //            qDebug() << color_frame.get_data();
        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        //        cv::Mat ir(cv::Size(640, 480), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(640, 480), CV_16SC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        cv::cvtColor(color, color, CV_RGB2BGR);

        cv::Mat clone = color.clone();
        std::vector<bbox_t> result_vec = detector.detect(clone);

        double distance = depth_frame.get_distance(depth_frame.get_width() / 2, depth_frame.get_height() / 2);
        if(save_dis){
            if(coutresult.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                out << distance << ",";
            }
            coutresult.close();
        }

        status = "Distance: " + QString::number(distance);
        ui->state->append(status);
        QApplication::processEvents();
        //        qDebug() << "==========================================";
        qDebug() << "depth frame: " << depth.at<short>(depth.cols / 2, depth.rows / 2);
        //        qDebug() << "get_distance: " << depth_frame.get_distance(depth_frame.get_width() / 2, depth_frame.get_height() / 2);
        //        //        qDebug() << "depth to meter" << depth_mat.at<double>(960, 540);
        //        qDebug() << "==========================================";
        //        float distance = depth_frame.get_distance(10, 10);
        //        qDebug() << distance;
        //            qDebug() << result_vec.size();
        //        detector.free_image(clone);
        draw_boxes(clone, result_vec, obj_names);

        double y_bias = ui->y_bias->value();

        for(int i = 0 ; i < result_vec.size() ; i++){
            double obj_distance = depth_frame.get_distance(result_vec[i].x + result_vec[i].w / 2, result_vec[i].y + result_vec[i].h / 2);
            QString obj_name = QString::fromStdString(obj_names[result_vec[i].obj_id]);
            QString show_dis = "(" + QString::number(result_vec[i].x + result_vec[i].w / 2 ) + ", "
                    + QString::number(result_vec[i].y + result_vec[i].h / 2 ) + ", "
                    + QString::number(obj_distance) + "(meters))";
            cv::putText(clone, show_dis.toStdString(), cv::Point(result_vec[i].x, result_vec[i].y + y_bias), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 255), 1, CV_AA);

            //                        qDebug() << "==========================================";
            //            qDebug() << "depth frame" << depth.at<short>(result_vec[i].x + result_vec[i].w / 2, result_vec[i].y + result_vec[i].h / 2);
            //            qDebug() << "get_distance" << obj_name << " 's local x y z : ("
            //                     << result_vec[i].x + result_vec[i].w / 2 << " (pixel), "
            //                     << result_vec[i].y + result_vec[i].h / 2 << " (pixel), "
            //                     << distance << "(meters))";
            ////            qDebug() << "depth to meter" << depth_mat.at<double>(result_vec[i].x + result_vec[i].w / 2, result_vec[i].y + result_vec[i].h / 2);
            //            qDebug() << "==========================================";

        }

        //        qDebug() << "get data";

        //        qDebug() << depth.at<uchar>(22, 100);
        //        applyColorMap(ir, ir, COLORMAP_JET);
        //        draw_boxes(ir, result_vec, obj_names);
        cv::Mat scale_mat;
        depth.convertTo( scale_mat, CV_8U, -255.0 / 10000.0, 255.0 ); // 0-10000 -> 255(white)-0(black)
        //            cv::cvtColor(scale_mat, scale_mat, CV_RGB2BGR);

        //        qDebug() << depth.at<uchar>(22, 100);
        applyColorMap(scale_mat, scale_mat, COLORMAP_JET );
        //        qDebug() << scale_mat.rows << scale_mat.cols;
        video_writer44.write(scale_mat);
        draw_boxes(scale_mat, result_vec, obj_names);

        cv::circle(clone, cv::Point(clone.cols / 2, clone.rows / 2), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(color, cv::Point(color.cols / 2, color.rows / 2), 2, cv::Scalar(0, 0, 255), -1);
        //        cv::circle(ir, cv::Point(ir.cols / 2, ir.rows / 2), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(scale_mat, cv::Point(scale_mat.cols / 2, scale_mat.rows / 2), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(draw_dis, cv::Point(i, 640 - (640 * distance / 5)), 2, cv::Scalar(255, 255, 255), -1);


        cv::imshow("Yolo detect", clone);
        cv::imshow("Color Image", color);
        //        cv::imshow("IR Image", ir);
        cv::imshow("Depth Image", scale_mat);
        cv::imshow("Distance Image", draw_dis);
        //        cv::imshow("depth to meter", depth_mat);
        video_writer.write(color);
        video_writer2.write(clone);
        //        video_writer3.write(ir);
        video_writer4.write(scale_mat);
        video_writer5.write(draw_dis);
        i++;
        cv::waitKey(33);
    }
    p.stop();
}

void detect_3d::on_point_cloud_clicked()
{
    cv::VideoWriter write_depth;
    write_depth.open("video/video_depth_pc.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(1280, 720));
    show_window app(1280, 720, "RealSense Pointcloud Example");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    while (app) // Application still alive?
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        rs2::frame depth = frames.get_depth_frame();

        cv::Mat depth_img(cv::Size(1280, 720), CV_16SC1, (void*)depth.get_data(), Mat::AUTO_STEP);
        cv::Mat scale_mat;
        depth_img.convertTo( scale_mat, CV_8U, -255.0 / 10000.0, 255.0 ); // 0-10000 -> 255(white)-0(black)
        applyColorMap(scale_mat, scale_mat, COLORMAP_JET );
        //        write_depth.write(scale_mat);

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);
        auto vertices = points.get_vertices();
        //        qDebug() << "points.size: " << points.size();
        int count = 0;
        for (int i = 0; i < points.size(); i++){
            if (vertices[i].z){
                count++;
            }
        }
        qDebug() << "vertices x, y, z :" << vertices[921600/2].x << " " << vertices[921600/2].y << " " << vertices[921600/2].z;
        qDebug() << count;

        auto color = frames.get_color_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);


        // Upload the color frame to OpenGL
        app_state.tex.upload(color);

        // Draw the pointcloud
        draw_pointcloud(app.width(), app.height(), app_state, points);
        //        qDebug() << color.get_height() << color.get_width();
    }
    pipe.stop();

}

void detect_3d::on_actionexit_triggered()
{
    exit(1);
}

void detect_3d::on_camera_pose_clicked()
{
    rs2::pipeline p;
    rs2::align align_to(RS2_STREAM_COLOR);
    rs2::pipeline_profile profile = p.start();
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics_depth = depth_stream.get_intrinsics();
    auto intrinsics_color = color_stream.get_intrinsics();
    qDebug() << "depth focal length: " << intrinsics_depth.fx << intrinsics_depth.fy;   //642.931, 642.931
    qDebug() << "depth principal point: " << intrinsics_depth.ppx << intrinsics_depth.ppy;  //646.29, 365.498
    qDebug() << "color focal length: " << intrinsics_color.fx << intrinsics_color.fy;   //618.817, 617.885
    qDebug() << "color principal point: " << intrinsics_color.ppx << intrinsics_color.ppy;  //325.158, 229.738


    cv::VideoWriter video_writer;
    video_writer.open("pose_result/raw_video.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));

    int i = 0;
    cv::Mat frame0, frame1;
    while(1){
        auto data = p.wait_for_frames();
        data = align_to.process(data);
        auto color_frame = data.get_color_frame();
        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color, color, CV_RGB2BGR);
        if(i == 1)  frame0 = color;
        if(i == 2)  {frame1 = color; break;}
        cv::waitKey(10);
        i++;
    }
    /*    SIFT TEST    */
    //    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(100);
    //    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    //    detector->detect(frame0, keypoints_1);
    //    detector->detect(frame1, keypoints_2);

    //    cv::Mat img_keypoints_1;
    //    cv::Mat img_keypoints_2;

    //    cv::drawKeypoints(frame0, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    //    cv::drawKeypoints(frame1, keypoints_2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    //    cv::imshow("Keypoints 1", img_keypoints_1 );
    //    cv::imshow("Keypoints 2", img_keypoints_2 );


    cv::cvtColor(frame0, frame0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> point1, point2;
    featureDetection(frame0, point1, 0);
    std::vector<uchar> status;
    featureTracking(frame0, frame1, point1, point2, status);

    double focal = 618.817;
    cv::Point2d pp(325.158, 229.738);

    cv::Mat E, R, t, mask, R_f, t_f;
    E = cv::findEssentialMat(point2, point1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, point2, point1, R, t, focal, pp, mask);


    cv::Mat prevImage = frame1;
    cv::Mat currImage;
    std::vector<cv::Point2f> prevFeatures = point2;
    std::vector<cv::Point2f> currFeatures;

    R_f = R.clone();
    t_f = t.clone();
    cv::Mat trajectory_map = cv::Mat::zeros(600, 600, CV_8UC3);

    int count = 0;

    int width = 1280;
    int height = 720;
    show_window app(1280, 720, "Realsense camera pose");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    QList<cv::Point3f> pose_point;

    qDebug() << R_f.rows << R_f.cols;
    qDebug() << t_f.rows << t_f.cols;

    QFile coutresult(ui->pose_save->text());
    QTextStream out(&coutresult);
    qDebug() << ui->pose_save->text();
    qDebug() << save_pose;
    if(save_pose){
        if(coutresult.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
            out << "x, y, z\n";
        }
    }
    coutresult.close();
    scale = ui->scale->value();
    frame = ui->frame->value();
    qDebug() << scale << frame;


    while(app){
        //        if() break;
        auto data = p.wait_for_frames();
        data = align_to.process(data);
        auto color_frame = data.get_color_frame();
        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color, color, CV_RGB2BGR);

        cv::Mat src_frame = color.clone();

        cv::cvtColor(src_frame, currImage, cv::COLOR_BGR2GRAY);

        if(count % frame == 0){
            std::vector<uchar> status;
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
            E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
            cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

            //            double scale = change_scale;
            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;
            //            qDebug() << prevFeatures.size();
            int MIN_NUM_FEAT = 500;
            if (prevFeatures.size() < MIN_NUM_FEAT)	{
                featureDetection(prevImage, prevFeatures, count);
                //                qDebug() << "lower" << prevFeatures.size();
                featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
            }

            //            if (prevFeatures.size() > 0)	{
            prevImage = currImage.clone();
            prevFeatures = currFeatures;
            //            }

            //        int x = -int(t_f.at<double>(0)) + 300;
            //        int y = -int(t_f.at<double>(1)) + 300;
            int x = int(t_f.at<double>(0)) + 300;
            int y = int(t_f.at<double>(2)) + 300;
            cv::Point3f pose ;
            pose.x = -t_f.at<double>(0) / 1000.0;
            pose.y = -t_f.at<double>(1) / 1000.0;
            pose.z = -t_f.at<double>(2) / 1000.0;
            pose_point.append(pose);
            if(save_pose){
                if(coutresult.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                    out << pose.x << "," << pose.y << "," << pose.z << "\n";
                }
                coutresult.close();
            }
            //        pose_point.append();
            qDebug() << "=============================================================";
            qDebug() << pose.x << pose.y << pose.z;

            qDebug() << t_f.at<double>(0, 0) << t_f.at<double>(1, 0) << t_f.at<double>(2, 0);
            //            qDebug() << x << y;
            qDebug() << "=============================================================";
            cv::circle(trajectory_map, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), 2);
            cv::imshow("Raw video", src_frame);
            //        cv::imshow("Trajectory", trajectory_map);

        }
        video_writer.write(src_frame);
        draw_camera_pose(app.width(), app.height(), app_state, pose_point);
        cv::waitKey(33);
        count++;
    }
}


void detect_3d::on_save_pose_stateChanged(int arg1)
{
    if(arg1 == Qt::Checked) save_pose = true;
    else    save_pose = false;
}

void detect_3d::on_scale_valueChanged(int arg1)
{
    scale = arg1;
}

void detect_3d::on_frame_valueChanged(int arg1)
{
    frame = arg1;
}

void detect_3d::on_epipolar_clicked()
{
    //    QString F_img = QFileDialog::getOpenFileName(this, tr("Open First Image"));
    //    QString S_img = QFileDialog::getOpenFileName(this, tr("Open Second Image"));

    //    cv::Mat first_img = cv::imread(F_img.toStdString());
    //    cv::Mat second_img = cv::imread(S_img.toStdString());

    //        cv::Mat first_img = cv::imread("D://Fruit_harvest//Raw_data//20171226_JingYong//image//PC260117.JPG");
    //        cv::Mat second_img = cv::imread("D://Fruit_harvest//Raw_data//20171226_JingYong//image//PC260118.JPG");

    cv::Mat first_img = cv::imread("D://Qt_Tools//video2image//release//result//calibration_video.MOV-0.jpg");
    cv::Mat second_img = cv::imread("D://Qt_Tools//video2image//release//result//calibration_video.MOV-150.jpg");

    //        cv::Mat first_img = cv::imread("D://test//left.jpg");
    //        cv::Mat second_img = cv::imread("D://test//right.jpg");
    //    cv::Mat first_img = cv::imread("D://test//box.png");
    //    cv::Mat second_img = cv::imread("D://test//box_in_scene.png");

    //    cv::cvtColor(first_img, first_img, CV_BGR2GRAY);
    //    cv::cvtColor(second_img, second_img, CV_BGR2GRAY);

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(50);
    std::vector<cv::KeyPoint> keypoint1, keypoint2;
    surf->detect(first_img, keypoint1);
    surf->detect(second_img, keypoint2);

    cv::Mat descriptor1, descriptor2;

    surf->compute(first_img, keypoint1, descriptor1);
    surf->compute(second_img, keypoint2, descriptor2);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    double max_dist = 0, min_dist = 100;

    for( int i = 0; i < descriptor1.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> points1,points2;

    qDebug() << "rows: " << descriptor1.rows << "cols: " << descriptor1.cols;
    qDebug() << "matches.size(): " << matches.size();
    qDebug() << matches.at(3).queryIdx << matches.at(3).trainIdx;
    qDebug() << matches.at(10).queryIdx << matches.at(10).trainIdx;

    for( int i = 0; i < descriptor1.rows; i++ ){
        if( matches[i].distance <= max(2*min_dist, 0.02)){
            good_matches.push_back( matches[i]);
            points2.push_back( keypoint2[matches[i].trainIdx].pt );
            points1.push_back( keypoint1[matches[i].queryIdx].pt );
        }
    }

    qDebug() << "good_match: " << good_matches.size();


    cv::Mat img_matches;
    cv::drawMatches( first_img, keypoint1, second_img, keypoint2, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    imshow( "Good Matches", img_matches );

    for( int i = 0; i < (int)good_matches.size(); i++ )
    { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }



    cv::Mat FundamentalMat = cv::findFundamentalMat(points1, points2, cv::FM_LMEDS);

    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(points1, 1, FundamentalMat, lines2);
    cv::computeCorrespondEpilines(points2, 2, FundamentalMat, lines1);

    cv::Mat left_1 = first_img.clone();
    cv::Mat left_2 = second_img.clone();
    drawlines(left_1, left_2, lines1, points1, points2);

    cv::Mat right_1 = first_img.clone();
    cv::Mat right_2 = second_img.clone();
    drawlines(right_2, right_1, lines2, points2, points1);

    //    cv::Mat output;
    //    for(int i = 0 ; i < points1.size() ; i++){
    //        cv::line(second_img, cv::Point(0, -lines2[i][2] / lines2[i][1]),
    //                cv::Point(second_img.cols, -(lines2[i][2] + lines2[i][0] * second_img.cols) / lines2[i][1]),
    //                cv::Scalar(0, 0, 255));
    //        cv::circle(first_img, points1[i], 2, cv::Scalar(0, 0, 255));
    //    }
    cv::imshow("left_1", left_1);
    cv::imshow("right_2", right_2);
}

void detect_3d::on_track_line_clicked()
{
    //    rs2::pipeline pipe;
    //    rs2::pipeline_profile selection = pipe.start();
    //    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH)
    //                                 .as<rs2::video_stream_profile>();
    //    auto i = depth_stream.get_intrinsics();
    //        qDebug() << i.model;
    //        qDebug() << i.coeffs[0] << i.coeffs[1]
    //                 << i.coeffs[2] << i.coeffs[3];


    //    cv::Mat frame0 = cv::imread("D://Intel_realsense_D435_code//Detect_3D//release//src//raw_video.avi-10.jpg");
    //    cv::Mat frame1 = cv::imread("D://Intel_realsense_D435_code//Detect_3D//release//src//raw_video.avi-60.jpg");

    std::string cfg_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.cfg";
    std::string weights_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//model//strawberry_JingYong_1000.weights";
    std::string names_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.names";

    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);

    //    cv::Mat frame0 = cv::imread("D://Fruit_harvest//Raw_data//20171226_JingYong//image//PC260118.JPG");
    //    cv::Mat frame1 = cv::imread("D://Fruit_harvest//Raw_data//20171226_JingYong//image//PC260119.JPG");

    /*------------------------------------------ Great Result -----------------------------------------------*/
    //    cv::Mat frame0 = cv::imread("D://Qt_Tools//video2image//release//result//PC260038.MOV-520.jpg");
    //    cv::Mat frame1 = cv::imread("D://Qt_Tools//video2image//release//result//PC260038.MOV-530.jpg");
    /*-------------------------------------------------------------------------------------------------------*/

    cv::Mat frame0 = cv::imread("D://Qt_Tools//video2image//release//result//PC260047.MOV-190.jpg");
    cv::Mat frame1 = cv::imread("D://Qt_Tools//video2image//release//result//PC260047.MOV-200.jpg");
    //    cv::cvtColor(frame0, frame0, cv::COLOR_BGR2GRAY);
    //    cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);

    std::vector<bbox_t> result_vec0 = detector.detect(frame0);
    std::vector<bbox_t> result_vec1 = detector.detect(frame1);

    qDebug() << "--------" << result_vec0.size();

    cv::Mat point_0 = frame0.clone();
    cv::Mat point_1 = frame1.clone();
    qDebug() << point_0.rows << point_1.cols;

    QList<cv::Point> first_frame, second_frame;
    for(int i = 0 ; i < result_vec0.size() ; i++){
        cv::Point temp((result_vec0[i].x + result_vec0[i].w / 2), (result_vec0[i].y + result_vec0[i].h / 2));

        first_frame.append(temp);
        cv::circle(point_0, temp, 5, cv::Scalar(255, 0, 0), -1);
    }
    for(int i = 0 ; i < result_vec1.size() ; i++){
        cv::Point temp(result_vec1[i].x + result_vec1[i].w / 2, result_vec1[i].y + result_vec1[i].h / 2);
        second_frame.append(temp);
        cv::circle(point_1, temp, 5, cv::Scalar(255, 0, 0), -1);
    }

    std::vector<cv::Point2f> point1, point2;
    featureDetection(frame0, point1, 0);
    std::vector<uchar> status;
    featureTracking(frame0, frame1, point1, point2, status);

    cv::Mat translation = cv::findHomography(point1, point2);

    int threshold = 35;
    QList<cv::Point> result_0;
    for(int i = 0 ; i < first_frame.size() ; i++){
        cv::Point temp_point = get_tracked_point(translation, first_frame.at(i));
        result_0.append(temp_point);
        cv::line(point_0, first_frame.at(i), temp_point, cv::Scalar(0, 255, 0), 2);
        cv::circle(point_0, temp_point, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(point_0, temp_point, threshold, cv::Scalar(0, 0, 255), 2);
        putText(point_0, "prediction point", cv::Point2f(temp_point.x - 10, temp_point.y - 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 255), 1);
    }


    for(int i = 0 ; i < result_vec0.size() ; i++){
        result_vec0.at(i).track_id = i+1;
        cv::putText(point_0, "ID:" + QString::number(i + 1).toStdString(), cv::Point(first_frame.at(i).x, first_frame.at(i).y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
    }

    QList<bbox_t> total_fruit;
    for(int i = 0 ; i < result_vec0.size() ; i++){
        total_fruit.append(result_vec0.at(i));
    }

    qDebug() << total_fruit.size();
    bool previous_fruit = true;
    if(result_vec0.size() == 0) previous_fruit = false;
    //    set_ID(total_fruit, result_0, second_frame, result_vec0, result_vec1, threshold, previous_fruit);
    qDebug() << total_fruit.size();

    for(int i = 0 ; i < result_vec1.size() ; i++){
        cv::putText(point_1, "ID:" + QString::number(result_vec1.at(i).track_id).toStdString(), cv::Point(second_frame.at(i).x, second_frame.at(i).y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("point0", point_0);
    cv::imshow("point1", point_1);
    cv::imwrite("./point_0.jpg", point_0);
    cv::imwrite("./point_1.jpg", point_1);
}

void detect_3d::on_tracking_clicked()
{
    cv::Scalar fruit_point_color(0, 0, 255);
    cv::Scalar tracked_point_color(255, 0, 0);
    cv::Scalar boundary_color(255, 255, 0);
    int threshold = 35;     // Previous tracked fruit vs current fruit distance threshold
    double detect_threshold = 0.5;
    int boundary = 40;
    int lost_track_threshold = 40;
    int inactive_threshold = 150;

    std::string cfg_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.cfg";
    std::string weights_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//model//strawberry_JingYong_1000.weights";
    std::string names_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.names";
    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);

    //    QString video_file = QFileDialog::getOpenFileName(this, tr("Open Tracking Video"));
    QString video_file = "D://Fruit_harvest//Raw_data//20171226_JingYong//video//PC260055.MOV";
    cv::VideoWriter video_writer;
    video_writer.open("./tracking_video/tracking_PC260047.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(1280, 720));
    cv::VideoCapture video(video_file.toStdString());

    // Read Video First Frame
    cv::Mat input, frame0, draw_mat;
    int i = 0;
    while(i < 2){
        video >> input;
        draw_mat = input.clone();
        //        cv::imshow("src0", input);
        if(i == 1){  frame0 = input.clone();    break;}
        i++;
    }

    // Fruit detection in frame 0
    std::vector<bbox_t> result_vec0 = detector.detect(frame0, detect_threshold);
    for(int i = 0 ; i < result_vec0.size() ; i++){
        cv::Point2f temp((float)result_vec0[i].x + (float)result_vec0[i].w / 2, (float)result_vec0[i].y + (float)result_vec0[i].h / 2);
        cv::circle(draw_mat, temp, 5, fruit_point_color, -1);
    }

    // Kick fruit out of the boundary
    result_vec0.erase(std::remove_if(result_vec0.begin(), result_vec0.end(), [&](bbox_t &vector){
                          return ((vector.x + vector.w / 2) < boundary) ||
                          ((vector.x + vector.w / 2) > (draw_mat.cols - boundary)) ||
                          ((vector.y + vector.h / 2) < boundary) ||
                          ((vector.y + vector.h / 2) > (draw_mat.rows - boundary));
                      }), result_vec0.end());
    cv::rectangle(draw_mat, cv::Point2f(boundary, boundary), cv::Point2f(draw_mat.cols - boundary, draw_mat.rows - boundary),
                  boundary_color, 1);


    // Feature operation in frame 0
    std::vector<cv::Point2f> point0;
    featureDetection(frame0, point0, 0);
    //    featureDetection_GPU(frame0, point0, 0);
    //    qDebug() << "GPU SURF";
    std::vector<bbox_t_history> tranform = bbox_t2bbox_t_history(result_vec0);
    result_vec0.clear();
    std::vector<bbox_t_history> total_fruit;
    for(int i = 0 ; i < tranform.size() ; i++){
        tranform.at(i).track_id = i + 1;
        total_fruit.push_back(tranform.at(i));
        cv::Point2f trajectory((float)tranform[i].x + (float)tranform[i].w / 2, (float)tranform[i].y + (float)tranform[i].h / 2);
        total_fruit.at(i).trajectory.append(trajectory);
        total_fruit.at(i).history.append(2);
        cv::putText(draw_mat, "ID:" + std::to_string(tranform.at(i).track_id), cv::Point2f(tranform.at(i).x - 30, tranform.at(i).y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
    }
    cv::imwrite("./tracking_frame/1.jpg", draw_mat);

    // Define previous and current
    cv::Mat prev_mat, curr_mat;
    std::vector<cv::Point2f> prev_point, curr_point;
    std::vector<bbox_t_history> prev_vec, curr_vec;
    QList<cv::Mat>  Homo_history;

    prev_mat = curr_mat = frame0.clone();
    prev_point = curr_point = point0;
    prev_vec = curr_vec = tranform;

    //    qDebug() << total_fruit.size();

    // Tracking start with second frame in the video
    cv::VideoCapture video1(video_file.toStdString());
    int frame = 0;
    while(true){
        video1 >> input;
        cv::Mat draw_mat = input.clone();
        cv::Mat check_mat = input.clone();
        if(input.empty())   break;
        if(frame > 1){
            // Define current mat, point, vec
            curr_mat = input.clone();
            std::vector<bbox_t> temp = detector.detect(input, detect_threshold);
            curr_vec = bbox_t2bbox_t_history(temp);
            temp.clear();
            featureDetection(curr_mat, curr_point, frame);
            std::vector<uchar> status;
            featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);
            //            qDebug() << "done feature";

            if(prev_point.size() < 1000){
                featureDetection(prev_mat, prev_point, frame);
                featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);
            }

            // Current / Previous mat fruit
            QList<cv::Point2f> curr_fruit, prev_fruit;
            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::Point2f temp((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);
                cv::circle(draw_mat, temp, 5, fruit_point_color, -1);
            }
            for(int i = 0 ; i < prev_vec.size() ; i++){
                cv::Point2f temp((float)prev_vec[i].x + (float)prev_vec[i].w / 2, (float)prev_vec[i].y + (float)prev_vec[i].h / 2);
                prev_fruit.append(temp);
            }

            // Kick fruit out of the boundary
            curr_vec.erase(std::remove_if(curr_vec.begin(), curr_vec.end(), [&](bbox_t &vector){
                               return ((vector.x + vector.w / 2) < boundary) ||
                               ((vector.x + vector.w / 2) > (draw_mat.cols - boundary)) ||
                               ((vector.y + vector.h / 2) < boundary) ||
                               ((vector.y + vector.h / 2) > (draw_mat.rows - boundary));
                           }), curr_vec.end());
            cv::rectangle(draw_mat, cv::Point2f(boundary, boundary), cv::Point2f(draw_mat.cols - boundary, draw_mat.rows - boundary),
                          boundary_color, 1);

            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::Point2f temp((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);
                curr_fruit.append(temp);
            }
            qDebug() << "frame: " << frame;
            qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();

            // Calculate tracked previous point
            QList<cv::Point2f> prev_tracked_fruit;
            cv::Mat homo_matrix = cv::findHomography(prev_point, curr_point);

            qDebug() << homo_matrix.at<double>(0, 0) << homo_matrix.at<double>(0, 1) << homo_matrix.at<double>(0, 2);
            qDebug() << homo_matrix.at<double>(1, 0) << homo_matrix.at<double>(1, 1) << homo_matrix.at<double>(1, 2);
            qDebug() << homo_matrix.at<double>(2, 0) << homo_matrix.at<double>(2, 1) << homo_matrix.at<double>(2, 2);

            Homo_history.append(homo_matrix);
            for(int i = 0 ; i < prev_fruit.size() ; i++){
                cv::Point2f temp_point = get_tracked_point(homo_matrix, prev_fruit.at(i));
                qDebug() << "i: " << i;
                qDebug() << "origin x: " <<  prev_fruit.at(i).x << ", y: " << prev_fruit.at(i).y;
                qDebug() << "tracked x: " <<  temp_point.x << ", y: " << temp_point.y;
                prev_tracked_fruit.append(temp_point);
                cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);
                cv::circle(draw_mat, temp_point, threshold, tracked_point_color, 1);
            }

            // Set id with tracking result
            bool previous_fruit = true;
            if(prev_vec.size() == 0)    previous_fruit = false;
            // set_ID(total_fruit, prev_tracked_fruit, curr_fruit, prev_vec, curr_vec, threshold, previous_fruit);
            set_ID_new(total_fruit, prev_tracked_fruit, curr_fruit, prev_vec, curr_vec, Homo_history, threshold, previous_fruit, frame, lost_track_threshold, check_mat);



            QFile history("./tracking_frame/history.csv");
            QTextStream out(&history);
            if(history.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                out << "Frame : " << frame << "\n";
                for(int i = 0 ; i < total_fruit.size() ; i++){
                    out << "ID: " << total_fruit.at(i).track_id << "  ";
                    for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
                        out << total_fruit.at(i).history.at(j) << " ";
                    }
                    out << "\n";
                }
            }
            history.close();

            // Lost -> Inactive fruit
            if(frame > inactive_threshold){
                for(int i = 0 ; i < total_fruit.size() ; i++){
                    int inactive_time = 0;
                    for(int j = total_fruit.at(i).history.size() - 1 ; j >= 0 ; j--){
                        if(total_fruit.at(i).history.at(j) == 1){
                            inactive_time += 1;
                        }
                        else if(inactive_time >= inactive_threshold){
                            total_fruit.at(i).history.pop_back();
                            total_fruit.at(i).history.append(0);
                            break;
                        }
                        else break;
                    }
                }
            }


            // Draw ID
            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::putText(draw_mat, "ID:" + std::to_string(curr_vec.at(i).track_id), cv::Point2f(curr_fruit.at(i).x - 30, curr_fruit.at(i).y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
            }
            cv::putText(draw_mat, "Total Fruit : " + std::to_string(total_fruit.size()), cv::Point2f(0, draw_mat.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
            cv::imwrite("./tracking_frame/" + std::to_string(frame) + ".jpg", draw_mat);
            cv::imwrite("./tracking_frame/check_frame/" + std::to_string(frame) + ".jpg", check_mat);

            prev_mat = curr_mat.clone();
            prev_point = curr_point;
            prev_vec = curr_vec;

            curr_fruit.clear();
            curr_point.clear();
            curr_vec.clear();

            cv::imshow("Tracking", draw_mat);
            cv::imshow("Checking", check_mat);
            video_writer.write(draw_mat);
        }
        frame++;
        cv::waitKey(10);
    }

    qDebug() << "total_fruit.size(): " << total_fruit.size();

    // Calculate Fruit Size Histogram
    int INITIAL = 2000, END = 18000, INTERVAL = 200;
    int n_bar = (END - INITIAL) / INTERVAL;

    QFile histogram("./tracking_frame/fruit_histogram.csv");
    QTextStream out1(&histogram);
    QList<int> fruit_size;
    int histo[80];
    for(int i = 0 ; i < n_bar ; i++){
        histo[i] = 0;
    }
    int min = 100000, max = 0;
    int min_ID = 0, max_ID = 0;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int size = total_fruit.at(i).w * total_fruit.at(i).h;
        fruit_size.append(size);
        if(histogram.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
            out1 << "Fruit size histogram\n";
        }
        out1 << size << ", ";
        qDebug() << size;
        if(size < min){ min = size; min_ID = total_fruit.at(i).track_id;}
        if(size > max){ max = size; max_ID = total_fruit.at(i).track_id;}
        if(size > (INITIAL + n_bar*INTERVAL))   histo[n_bar-1]++;
        if(size < (INITIAL))   histo[0]++;
        for(int r = n_bar ; r > 0 ; r--){
            if((size < (INITIAL + r*INTERVAL)) && (size >= (INITIAL + (r-1)*INTERVAL))){
                histo[r-1]++;
                break;
            }
        }
    }

    out1 << "\n";
    out1 << "Interval, ";
    for(int i = 0 ; i < n_bar ; i++){
        out1 << INITIAL + i * INTERVAL << ", ";
    }
    out1 << "\n";
    out1 << "Number, ";
    for(int i = 0 ; i < n_bar ; i++){
        out1 << histo[i] << ", ";
    }
    out1 << "\nMax size: " << max << "\n";
    out1 << "Max ID: " << max_ID << "\n";
    out1 << "Min size: " << min << "\n";
    out1 << "Min ID: " << min_ID << "\n";
    histogram.close();

    qDebug() << "Max size: " << max;
    qDebug() << "Max ID: " << max_ID;
    qDebug() << "Min size " << min;
    qDebug() << "Min ID: " << min_ID;
}

void detect_3d::on_fast_track_clicked()
{
    cv::Scalar fruit_point_color(0, 0, 255);
    cv::Scalar tracked_point_color(255, 0, 0);
    cv::Scalar boundary_color(255, 255, 0);
    int threshold = 35;     // Previous tracked fruit vs current fruit distance threshold
    double detect_threshold = 0.5;
    int boundary = 40;
    int lost_track_threshold = 45;
    int inactive_threshold = 150;

    std::string cfg_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.cfg";
    std::string weights_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//model//strawberry_JingYong_1000.weights";
    std::string names_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.names";
    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);

    //    QString video_file = QFileDialog::getOpenFileName(this, tr("Open Tracking Video"));
    QString video_file = "D://Fruit_harvest//Raw_data//20171226_JingYong//video//PC260055.MOV";
    cv::VideoWriter video_writer;
    video_writer.open("./tracking_video/tracking_PC260055.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(1280, 720));
    cv::VideoCapture video(video_file.toStdString());

    // Read Video First Frame
    cv::Mat input, frame0, frame1, draw_mat;
    int i = 0;
    while(i < 2){
        video >> input;
        draw_mat = input.clone();
        //        cv::imshow("src0", input);
        if(i == 0){  frame0 = input.clone();    }
        if(i == 1){  frame1 = input.clone();    break;}
        i++;
    }

    // Fruit detection in frame 0, frame 1
    std::vector<bbox_t> result_vec0 = detector.detect(frame0, detect_threshold);
    std::vector<bbox_t> result_vec1 = detector.detect(frame1, detect_threshold);
    for(int i = 0 ; i < result_vec0.size() ; i++){
        cv::Point2f temp((float)result_vec0[i].x + (float)result_vec0[i].w / 2, (float)result_vec0[i].y + (float)result_vec0[i].h / 2);
        cv::circle(draw_mat, temp, 5, fruit_point_color, -1);
    }
    for(int i = 0 ; i < result_vec1.size() ; i++){
        cv::Point2f temp((float)result_vec1[i].x + (float)result_vec1[i].w / 2, (float)result_vec1[i].y + (float)result_vec1[i].h / 2);
        cv::circle(draw_mat, temp, 5, fruit_point_color, -1);
    }

    // Kick fruit out of the boundary
    result_vec0.erase(std::remove_if(result_vec0.begin(), result_vec0.end(), [&](bbox_t &vector){
                          return ((vector.x + vector.w / 2) < boundary) ||
                          ((vector.x + vector.w / 2) > (draw_mat.cols - boundary)) ||
                          ((vector.y + vector.h / 2) < boundary) ||
                          ((vector.y + vector.h / 2) > (draw_mat.rows - boundary));
                      }), result_vec0.end());
    result_vec1.erase(std::remove_if(result_vec1.begin(), result_vec1.end(), [&](bbox_t &vector){
                          return ((vector.x + vector.w / 2) < boundary) ||
                          ((vector.x + vector.w / 2) > (draw_mat.cols - boundary)) ||
                          ((vector.y + vector.h / 2) < boundary) ||
                          ((vector.y + vector.h / 2) > (draw_mat.rows - boundary));
                      }), result_vec1.end());
    cv::rectangle(draw_mat, cv::Point2f(boundary, boundary), cv::Point2f(draw_mat.cols - boundary, draw_mat.rows - boundary),
                  boundary_color, 1);

    std::vector<bbox_t_history> tranform0 = bbox_t2bbox_t_history(result_vec0);
    std::vector<bbox_t_history> tranform1 = bbox_t2bbox_t_history(result_vec1);
    result_vec0.clear();
    result_vec1.clear();
    std::vector<bbox_t_history> total_fruit;
    for(int i = 0 ; i < tranform0.size() ; i++){
        tranform0.at(i).track_id = i + 1;
        total_fruit.push_back(tranform0.at(i));
        cv::Point2f trajectory((float)tranform0[i].x + (float)tranform0[i].w / 2, (float)tranform0[i].y + (float)tranform0[i].h / 2);
        total_fruit.at(i).trajectory.append(trajectory);
        total_fruit.at(i).history.append(2);
        cv::putText(draw_mat, "ID:" + std::to_string(tranform0.at(i).track_id), cv::Point2f(tranform0.at(i).x - 30, tranform0.at(i).y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
    }

    // Current / Previous mat fruit
    QList<cv::Point2f> fruit0, fruit1;
    for(int i = 0 ; i < tranform0.size() ; i++){
        cv::Point2f temp((float)tranform0[i].x + (float)tranform0[i].w / 2, (float)tranform0[i].y + (float)tranform0[i].h / 2);
        fruit0.append(temp);
    }
    for(int i = 0 ; i < tranform1.size() ; i++){
        cv::Point2f temp((float)tranform1[i].x + (float)tranform1[i].w / 2, (float)tranform1[i].y + (float)tranform1[i].h / 2);
        fruit1.append(temp);
    }


    // Feature operation in frame 0
    std::vector<cv::Point2f> point0, point1;
    featureDetection(frame0, point0, 0);
    std::vector<uchar> status;
    featureTracking(frame0, frame1, point0, point1, status);

    // Calculate tracked previous point
    QList<cv::Point2f> prev_tracked_fruit;
    QList<cv::Mat>  Homo_history;
    cv::Mat homo_matrix = cv::findHomography(point0, point1);
    Homo_history.append(homo_matrix);
    for(int i = 0 ; i < fruit0.size() ; i++){
        cv::Point2f temp_point = get_tracked_point(homo_matrix, fruit0.at(i));
        prev_tracked_fruit.append(temp_point);
        cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);
        cv::circle(draw_mat, temp_point, threshold, tracked_point_color, 1);
    }
//    qDebug() <<

    // Set id with tracking result
    bool previous_fruit = true;
    if(tranform0.size() == 0)    previous_fruit = false;
    cv::Mat check_mat = input.clone();
    set_ID_fast(total_fruit, prev_tracked_fruit, fruit1, tranform0, tranform1, Homo_history, threshold, previous_fruit, 1, lost_track_threshold, check_mat, check_mat);

    cv::imwrite("./tracking_frame/1.jpg", draw_mat);

    // Define previous and current
    cv::Mat prev_mat, curr_mat;
    std::vector<cv::Point2f> prev_point, curr_point;
    std::vector<bbox_t_history> prev_vec, curr_vec;

    prev_mat = frame1.clone();
    prev_point = point1;
    prev_vec = tranform1;

    //    qDebug() << total_fruit.size();

    // Tracking start with second frame in the video
    cv::VideoCapture video1(video_file.toStdString());
    int frame = 0;
    while(true){
        video1 >> input;
        cv::Mat draw_mat = input.clone();
        cv::Mat maturity_mat = input.clone();
        if(input.empty())   break;
        if(frame > 1){
            // Define current mat, point, vec
            curr_mat = input.clone();
            std::vector<bbox_t> temp = detector.detect(input, detect_threshold);
            curr_vec = bbox_t2bbox_t_history(temp);
            temp.clear();
            //            featureDetection(curr_mat, curr_point, frame);
            std::vector<uchar> status;
            featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);
            //            qDebug() << "done feature";

            if(prev_point.size() < 2000){
                featureDetection(prev_mat, prev_point, frame);
                featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);
            }

            // Current / Previous mat fruit
            QList<cv::Point2f> curr_fruit, prev_fruit;
            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::Point2f temp((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);
                cv::circle(draw_mat, temp, 5, fruit_point_color, -1);
            }
            for(int i = 0 ; i < prev_vec.size() ; i++){
                cv::Point2f temp((float)prev_vec[i].x + (float)prev_vec[i].w / 2, (float)prev_vec[i].y + (float)prev_vec[i].h / 2);
                prev_fruit.append(temp);
            }

            // Kick fruit out of the boundary
            curr_vec.erase(std::remove_if(curr_vec.begin(), curr_vec.end(), [&](bbox_t_history &vector){
                               return ((vector.x + vector.w / 2) < boundary) ||
                               ((vector.x + vector.w / 2) > (draw_mat.cols - boundary)) ||
                               ((vector.y + vector.h / 2) < boundary) ||
                               ((vector.y + vector.h / 2) > (draw_mat.rows - boundary));
                           }), curr_vec.end());
            cv::rectangle(draw_mat, cv::Point2f(boundary, boundary), cv::Point2f(draw_mat.cols - boundary, draw_mat.rows - boundary),
                          boundary_color, 1);

            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::Point2f temp((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);
                curr_fruit.append(temp);
            }
            qDebug() << "frame: " << frame;
            qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();

            // Calculate tracked previous point
            QList<cv::Point2f> prev_tracked_fruit;
            cv::Mat homo_matrix = cv::findHomography(prev_point, curr_point);
//            qDebug() <
            Homo_history.append(homo_matrix);
            for(int i = 0 ; i < prev_fruit.size() ; i++){
                cv::Point2f temp_point = get_tracked_point(homo_matrix, prev_fruit.at(i));
                prev_tracked_fruit.append(temp_point);
                cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);
                cv::circle(draw_mat, temp_point, threshold, tracked_point_color, 1);
            }

            // Set id with tracking result
            bool previous_fruit = true;
            if(prev_vec.size() == 0)    previous_fruit = false;
            // set_ID(total_fruit, prev_tracked_fruit, curr_fruit, prev_vec, curr_vec, threshold, previous_fruit);
            set_ID_fast(total_fruit, prev_tracked_fruit, curr_fruit, prev_vec, curr_vec, Homo_history, threshold, previous_fruit, frame, lost_track_threshold, draw_mat, maturity_mat);

            QFile history("./tracking_frame/history.csv");
            QTextStream out(&history);
            if(history.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                out << "Frame : " << frame << "\n";
                for(int i = 0 ; i < total_fruit.size() ; i++){
                    out << "ID: " << total_fruit.at(i).track_id << "  ";
                    for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
                        out << total_fruit.at(i).history.at(j) << " ";
                    }
                    out << "\n";
                }
            }
            history.close();


            // Draw ID
            for(int i = 0 ; i < curr_vec.size() ; i++){
                cv::putText(draw_mat, "ID:" + std::to_string(curr_vec.at(i).track_id), cv::Point2f(curr_fruit.at(i).x - 30, curr_fruit.at(i).y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
            }
            cv::putText(draw_mat, "Total Fruit : " + std::to_string(total_fruit.size()), cv::Point2f(0, draw_mat.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
            cv::imwrite("./tracking_frame/" + std::to_string(frame) + ".jpg", draw_mat);
            //            cv::imwrite("./tracking_frame/check_frame/" + std::to_string(frame) + ".jpg", check_mat);

            prev_mat = curr_mat.clone();
            prev_point = curr_point;
            prev_vec = curr_vec;

            curr_fruit.clear();
            curr_point.clear();
            curr_vec.clear();

            cv::imshow("Tracking", draw_mat);
            //            cv::imshow("Checking", check_mat);
            video_writer.write(draw_mat);
        }
        frame++;
        cv::waitKey(10);
    }

    QFile track_result("./tracking_frame/tracking_result.csv");
    QTextStream out2(&track_result);
    if(track_result.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out2 << "Online-tracking\n";
        out2 << "total_fruit.size() = " << total_fruit.size();
        out2 << "\n\nOffline-tracking\nErase-ID\n";
    }
    track_result.close();
    qDebug() << "total_fruit.size(): " << total_fruit.size();

    // Eliminate false alarm
    int false_alarm_threshold = 2;  // if < 5 frame == Tracked --> false alarm
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int count = 0;
        for(int j = total_fruit.at(i).history.size() - 1 ; j >= 0 ; j--){
            if(total_fruit.at(i).history.at(j) == 2){
                count++;
            }
            if(count > false_alarm_threshold)   break;
        }
        if(count <= false_alarm_threshold){
            QFile track_result("./tracking_frame/tracking_result.csv");
            QTextStream out2(&track_result);
            if(track_result.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                out2 << total_fruit.at(i).track_id << ", ";
            }
            track_result.close();
            qDebug() << "Erase ID: " << total_fruit.at(i).track_id;
            total_fruit.erase(std::remove_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector){
                                  return (vector.track_id == total_fruit.at(i).track_id);
                              }), total_fruit.end());
            i--;
        }
    }
    if(track_result.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out2 << "\ntotal_fruit.size() = " << total_fruit.size();
    }
    track_result.close();
    qDebug() << "total_fruit.size(): " << total_fruit.size();


    // Calculate Fruit Size Histogram
    QFile histogram("./tracking_frame/fruit_histogram.csv");
    QTextStream out1(&histogram);
    if(histogram.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out1 << "Fruit size histogram\n";
    }
    int min = 100000, max = 0;
    int min_ID = 0, max_ID = 0;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int n_frame = 0, size = 0;
        for(int j = 0 ; j < total_fruit.at(i).width_height.size() ; j++){
            int curr_size = total_fruit.at(i).width_height.at(j).x * total_fruit.at(i).width_height.at(j).y;
            if(curr_size > size){
                size = curr_size;
                n_frame = j;
            }
        }
        total_fruit.at(i).size = size;
        out1 << size << ", ";
        if(size < min){ min = size; min_ID = total_fruit.at(i).track_id;}
        if(size > max){ max = size; max_ID = total_fruit.at(i).track_id;}
    }

    out1 << "\nMax size: " << max << "\n";
    out1 << "Max ID: " << max_ID << "\n";
    out1 << "Min size: " << min << "\n";
    out1 << "Min ID: " << min_ID << "\n";
    histogram.close();

    int bin_size = (max - min) / 5;
    int size_bin[6] = {min, min + bin_size, min + 2*bin_size, min + 3*bin_size, min + 4*bin_size, max};

    qDebug() << "Max size: " << max;
    qDebug() << "Max ID: " << max_ID;
    qDebug() << "Min size " << min;
    qDebug() << "Min ID: " << min_ID;

    // Calculate Fruit Ripening
    QFile Maturity("./tracking_frame/maturity.csv");
    QTextStream out3(&Maturity);
    if(Maturity.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out3 << "Fruit Ripening Stage\n";
    }
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int n_frame = 0, size = 0;
        for(int j = 0 ; j < total_fruit.at(i).width_height.size() ; j++){
            int curr_size = total_fruit.at(i).width_height.at(j).x * total_fruit.at(i).width_height.at(j).y;
            if(curr_size > size){
                size = curr_size;
                n_frame = j;
            }
        }
        out3 << "\nFruit ID: " << total_fruit.at(i).track_id;
        cv::Mat input = total_fruit.at(i).frame_mat.at(n_frame).clone();
        cv::Mat mask(input.rows, input.cols, CV_8U, cv::Scalar(0));
        cv::Mat bgModel, fgModel;
        int left_top_x = std::max(((int)total_fruit.at(i).trajectory.at(n_frame).x - (int)total_fruit.at(i).width_height.at(n_frame).x / 2 - 10), 0);
        int left_top_y = std::max(((int)total_fruit.at(i).trajectory.at(n_frame).y - (int)total_fruit.at(i).width_height.at(n_frame).y / 2 - 10), 0);
        int width = std::min((int)total_fruit.at(i).width_height.at(n_frame).x + 20, (int)input.cols);
        int height = std::min((int)total_fruit.at(i).width_height.at(n_frame).y + 20, (int)input.rows);
        cv::circle(mask, cv::Point(left_top_x + width / 2, left_top_y + height / 2), std::min(width / 2, height / 2) - 10, cv::Scalar(1), -1);
        // left-top_x, left-top_y, width, height
        cv::Rect rect(left_top_x, left_top_y, width, height);
        cv::grabCut(input, mask, rect, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);
        cv::Mat mask_result(input.rows, input.cols, CV_8UC3);
        int pixel_value[3] = {0};
        double average_value[3] = {0};
        int mask_pixel = 0, mature_pixel = 0;
        if(!mask.empty()){
            for(int i = 0 ; i < mask.rows ; i++){
                for(int j = 0 ; j < mask.cols ; j++){
                    if(mask.at<uchar>(i, j) == 1 || mask.at<uchar>(i, j) == 3){     // Foreground
                        pixel_value[0] += input.at<cv::Vec3b>(i, j)[0];
                        pixel_value[1] += input.at<cv::Vec3b>(i, j)[1];
                        pixel_value[2] += input.at<cv::Vec3b>(i, j)[2];
                        mask_pixel++;
                        out3 << "\n" << input.at<cv::Vec3b>(i, j)[0] << ", " << input.at<cv::Vec3b>(i, j)[1] << ", " << input.at<cv::Vec3b>(i, j)[2];
                        // Mature condition
                        if((input.at<cv::Vec3b>(i, j)[2] >= 200 && input.at<cv::Vec3b>(i, j)[1] < 200 && input.at<cv::Vec3b>(i, j)[0] < 160)
                                || (input.at<cv::Vec3b>(i, j)[2] >= 165 && input.at<cv::Vec3b>(i, j)[1] < 80 && input.at<cv::Vec3b>(i, j)[0] < 80)
                                || (input.at<cv::Vec3b>(i, j)[2] >= 100 && input.at<cv::Vec3b>(i, j)[1] < 40 && input.at<cv::Vec3b>(i, j)[0] < 40)){
                            mature_pixel++;
                            mask.at<uchar>(i, j) = 255;
                        }
                        else{
                            mask.at<uchar>(i, j) = 100;
                        }
                    }
                    else    {mask.at<uchar>(i, j) = 0;}                             // Background
                }
            }
            average_value[0] = (double)pixel_value[0] / (mask_pixel);
            average_value[1] = (double)pixel_value[1] / (mask_pixel);
            average_value[2] = (double)pixel_value[2] / (mask_pixel);
            double ripening_stage = (double)mature_pixel / (double)mask_pixel;
            out3 << "\n Average: " << average_value[0] << ", " << average_value[1] << ", " << average_value[2];
            out3 << "\n Ripening Stage: " << ripening_stage << "\n" ;
            if(mature_pixel == 0 || mask_pixel == 0)    ripening_stage = 0.0;
            qDebug() << ripening_stage;
            total_fruit.at(i).maturity = ripening_stage;
            qDebug() << total_fruit.at(i).maturity;
            for(int i = 0 ; i < mask.rows ; i++){
                for(int j = 0 ; j < mask.cols ; j++){
                    mask_result.at<cv::Vec3b>(i, j)[0] = input.at<cv::Vec3b>(i, j)[0] * 0.6 + mask.at<uchar>(i, j) * 0.4;
                    mask_result.at<cv::Vec3b>(i, j)[1] = input.at<cv::Vec3b>(i, j)[1] * 0.6 + mask.at<uchar>(i, j) * 0.4;
                    mask_result.at<cv::Vec3b>(i, j)[2] = input.at<cv::Vec3b>(i, j)[2] * 0.6 + mask.at<uchar>(i, j) * 0.4;
                }
            }
            cv::putText(mask_result, "Fruit ID : " + std::to_string(total_fruit.at(i).track_id), cv::Point2f(0, mask_result.rows - 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);
            cv::putText(mask_result, "Ripening Stage : " + std::to_string(ripening_stage), cv::Point2f(0, mask_result.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);

            cv::imwrite("./tracking_frame/maturity_mask/fruit_" + std::to_string(total_fruit.at(i).track_id) + ".png", mask_result);
        }
        else{
            qDebug() << "mask is empty";
            total_fruit.at(i).maturity = 0.0;
        }

    }

    // Calculate global coordinate
    QList<global_coor> global_coord;
    float max_x = 0.0, max_y = 0.0;
    float min_x = 10.0, min_y = 10.0;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
            if(total_fruit.at(i).history.at(j) == 2){   // First tracked point
                qDebug() << "ID:" << total_fruit.at(i).track_id << " at frame: " << j;
                cv::Point2f point = total_fruit.at(i).trajectory.at(0);
                if(j == 0){
                    global_coor coor;  // If tracked in the first frame -> No need to calculate
                    coor.global_point = point;
                    coor.global_fruit_ID = total_fruit.at(i).track_id;
                    coor.maturity = total_fruit.at(i).maturity;
                    coor.size = total_fruit.at(i).size;
                    global_coord.append(coor);
                }
                else{
                    for(int h = 0 ; h < j ; h++){
                        qDebug() << "point: " << point.x << ", " << point.y << " in " << h << " th transform";
                        cv::Point2f result = global_coordinate(Homo_history.at(h), point);
                        point.x = result.x;
                        point.y = result.y;
                    }
                    global_coor coor;
                    coor.global_point = point;
                    coor.global_fruit_ID = total_fruit.at(i).track_id;
                    coor.maturity = total_fruit.at(i).maturity;
                    coor.size = total_fruit.at(i).size;
                    global_coord.append(coor);
                }
                if(point.x > max_x) max_x = point.x;
                if(point.y > max_y) max_y = point.y;
                if(point.x < min_x) min_x = point.x;
                if(point.y < min_y) min_y = point.y;
                //                qDebug() << "global coordinate: " << global_coor.at(i).x << ", " << global_coor.at(i).y;
                break;
            }
        }
    }

    cv::Scalar Maturity_color[5] = {cv::Scalar(9, 113, 59)
                                    , cv::Scalar(104, 192, 252)
                                    , cv::Scalar(40, 74, 244)
                                    , cv::Scalar(29, 29, 255)
                                    , cv::Scalar(0, 0, 154)};
    qDebug() << "max x, y" << max_x << ", " << max_y << "min x, y" << min_x << ", " << min_y;
    if(min_x < 0)   max_x += abs(min_x);
    if(min_y < 0)   max_y += abs(min_y);
    cv::Mat Global_map((int)max_y + 1, (int)max_x + 1, CV_8UC3, cv::Scalar(255, 255, 255));
    qDebug() << Global_map.rows << " " << Global_map.cols;
    for(int i = 0 ; i < global_coord.size() ; i++){
        float global_x = global_coord.at(i).global_point.x;
        float global_y = global_coord.at(i).global_point.y;
        qDebug() << global_x << " " << global_y;
        if(min_x < 0)   global_x += abs(min_x);
        if(min_y < 0)   global_y += abs(min_y);
        qDebug() << global_x << " " << global_y;
        Global_map.at<cv::Vec3b>((int)global_y, (int)global_x)[0] = 0;
        Global_map.at<cv::Vec3b>((int)global_y, (int)global_x)[1] = 0;
        Global_map.at<cv::Vec3b>((int)global_y, (int)global_x)[2] = 0;

        // Set maturity Color
        cv::Scalar color;
        qDebug() << global_coord.at(i).maturity;
        if(global_coord.at(i).maturity < 0.25)  color = Maturity_color[0];
        else if(global_coord.at(i).maturity >= 0.25 && global_coord.at(i).maturity < 0.5)  color = Maturity_color[1];
        else if(global_coord.at(i).maturity >= 0.5 && global_coord.at(i).maturity < 0.75)  color = Maturity_color[2];
        else if(global_coord.at(i).maturity >= 0.75 && global_coord.at(i).maturity < 1.0)  color = Maturity_color[3];
        else if(global_coord.at(i).maturity >= 1.0)  color = Maturity_color[4];

        // Set size radius
        int radius;
        if(global_coord.at(i).size >= size_bin[0] && global_coord.at(i).size < size_bin[1])   radius = 4;
        else if(global_coord.at(i).size >= size_bin[1] && global_coord.at(i).size < size_bin[2])    radius = 8;
        else if(global_coord.at(i).size >= size_bin[2] && global_coord.at(i).size < size_bin[3])    radius = 12;
        else if(global_coord.at(i).size >= size_bin[3] && global_coord.at(i).size < size_bin[4])    radius = 16;
        else if(global_coord.at(i).size >= size_bin[4] && global_coord.at(i).size < size_bin[5])    radius = 20;

        cv::circle(Global_map, cv::Point(global_coord.at(i).global_point.x, global_coord.at(i).global_point.y), radius, color, -1);
        cv::putText(Global_map, "ID: " + std::to_string(global_coord.at(i).global_fruit_ID), cv::Point(global_coord.at(i).global_point.x - 80, global_coord.at(i).global_point.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("Global map", Global_map);
    cv::imwrite("./tracking_frame/Global_map.jpg", Global_map);

}
