#ifndef FEATURE_FUNCTION_HPP
#define FEATURE_FUNCTION_HPP

#define HAVE_OPENCV_XFEATURES2D

#include "opencv2/opencv_modules.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"

#include "opencv2/core/version.hpp"
#include "opencv2/videoio/videoio.hpp"

#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <yolo_v2_class.hpp>
#include <detect_3d.hpp>

#include <QString>

void featureDetection(cv::Mat img, std::vector<cv::Point2f>& point1, int count){

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(700);
    std::vector<cv::KeyPoint> keypoint1;
    detector->detect(img, keypoint1);

    //    std::vector<cv::KeyPoint> keypoint1;
    //    int fast_threshold = 20;
    //    bool nonmaxSuppression = true;
    //    cv::FAST(img, keypoint1, fast_threshold, nonmaxSuppression);
    cv::Mat img_keypoints_1;
    cv::drawKeypoints(img, keypoint1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::KeyPoint::convert(keypoint1, point1, std::vector<int>());
    //    cv::imshow("Keypoints 1", img_keypoints_1 );
    //    QString filename = "./pose_result/frame_" + QString::number(count) + ".png";
    //    cv::imwrite(filename.toStdString(), img_keypoints_1);
}

void featureDetection_GPU(cv::Mat img, std::vector<cv::Point2f>& point1, int count){
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(100);
    std::vector<cv::KeyPoint> keypoint1;
    detector->detect(img, keypoint1);
    cv::Mat img_keypoints_1;
    cv::drawKeypoints(img, keypoint1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::KeyPoint::convert(keypoint1, point1, std::vector<int>());
    //    cv::imshow("Keypoints 1", img_keypoints_1 );
    QString filename = "./pose_result/frame_" + QString::number(count) + ".png";
    //    cv::imwrite(filename.toStdString(), img_keypoints_1);
}

void featureTracking(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f>& point1, std::vector<cv::Point2f>& point2, std::vector<uchar>& status){
    std::vector<float> err;
    cv::Size winSize = cv::Size(50, 50);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img1, img2, point1, point2, status, err, winSize, 3, termcrit, 0, 0.001);
    int indexCorrection = 0;
    for(int i = 0 ; i < status.size() ; i++){
        cv::Point2f pt = point2.at(i - indexCorrection);
        if((status.at(i) == 0 ) || (pt.x < 0) || (pt.y < 0)){
            if((pt.x < 0) || (pt.y < 0)){
                status.at(i) = 0;
            }
            point1.erase(point1.begin() + i - indexCorrection);
            point2.erase(point2.begin() + i - indexCorrection);
            indexCorrection++;
        }
    }
}

cv::Point2f get_tracked_point(cv::Mat homography_matrix, cv::Point2f input_point){
    double M11 = homography_matrix.at<double>(0, 0);
    double M12 = homography_matrix.at<double>(0, 1);
    double M13 = homography_matrix.at<double>(0, 2);
    double M21 = homography_matrix.at<double>(1, 0);
    double M22 = homography_matrix.at<double>(1, 1);
    double M23 = homography_matrix.at<double>(1, 2);
    double M31 = homography_matrix.at<double>(2, 0);
    double M32 = homography_matrix.at<double>(2, 1);
    double M33 = homography_matrix.at<double>(2, 2);

    //    qDebug() << input_point.x << input_point.y;
    //    qDebug() << M11 << M12 << M13;
    //    qDebug() << M21 << M22 << M23;
    //    qDebug() << M31 << M32 << M33;

    float x = (float)input_point.x;
    float y = (float)input_point.y;
    cv::Point2f temp_point(((M11 * (float)x + M12 * (float)y + M13) / (M31 * (float)x + M32 * (float)y + M33))
                           , ((M21 * (float)x + M22 * (float)y + M23) / (M31 * (float)x + M32 * (float)y + M33)));
    //    qDebug() << temp_point.x << temp_point.y;

    return temp_point;
}

cv::Point2f global_coordinate(cv::Mat homography_matrix, cv::Point2f input_point){
    cv::Mat inverse_homo = homography_matrix.inv();
    double M11 = inverse_homo.at<double>(0, 0);
    double M12 = inverse_homo.at<double>(0, 1);
    double M13 = inverse_homo.at<double>(0, 2);
    double M21 = inverse_homo.at<double>(1, 0);
    double M22 = inverse_homo.at<double>(1, 1);
    double M23 = inverse_homo.at<double>(1, 2);
    double M31 = inverse_homo.at<double>(2, 0);
    double M32 = inverse_homo.at<double>(2, 1);
    double M33 = inverse_homo.at<double>(2, 2);

    float x = (float)input_point.x;
    float y = (float)input_point.y;

    cv::Point2f temp_point(((M11 * (float)x + M12 * (float)y + M13) / (M31 * (float)x + M32 * (float)y + M33))
                           , ((M21 * (float)x + M22 * (float)y + M23) / (M31 * (float)x + M32 * (float)y + M33)));
    return temp_point;
}

void set_ID(std::vector<bbox_t_history>& total_fruit, QList<cv::Point> prev_tracked_fruit, QList<cv::Point> curr_fruit, std::vector<bbox_t_history>& prev_vec, std::vector<bbox_t_history>& curr_vec, int threshold, bool prev_fruit){
    for(int i = 0 ; i < curr_fruit.size() ; i++){
        if(prev_fruit){
            double distance = cv::norm(prev_tracked_fruit.at(0) - curr_fruit.at(i));    // Distance between predict point and true point
            if(distance < threshold){
                curr_vec.at(i).track_id = prev_vec.at(0).track_id;
            }
            for(int j = 1 ; j < prev_tracked_fruit.size() ; j++){
                double temp = cv::norm(prev_tracked_fruit.at(j) - curr_fruit.at(i));
                if(temp < distance) {
                    distance = temp;
                    if(temp < threshold) curr_vec.at(i).track_id = prev_vec.at(j).track_id;
                    else curr_vec.at(i).track_id = 0;
                }
            }
            if(curr_vec.at(i).track_id == 0){   // New Fruit compare to last frame
                curr_vec.at(i).track_id = total_fruit.size() + 1;
                total_fruit.push_back(curr_vec.at(i));
            }
        }
        else{
            curr_vec.at(i).track_id = total_fruit.size() + 1;
            total_fruit.push_back(curr_vec.at(i));
        }
    }
}

double IOU(bbox_t_history prev_vec, bbox_t_history curr_vec, cv::Mat homo){

    cv::Point2f prev_start((float)prev_vec.x, (float)prev_vec.y);
    cv::Point2f curr_start((float)curr_vec.x, (float)curr_vec.y);
    cv::Point2f prev_end((float)prev_vec.x + (float)prev_vec.w, (float)prev_vec.y + (float)prev_vec.h);
    cv::Point2f curr_end((float)curr_vec.x + (float)curr_vec.w, (float)curr_vec.y + (float)curr_vec.h);

    cv::Point2f start1 = get_tracked_point(homo, prev_start);
    cv::Point2f end1 = get_tracked_point(homo, prev_end);
    cv::Point2f start2 = curr_start;
    cv::Point2f end2 = curr_end;

    double width = std::min(end1.x, end2.x) - std::max(start1.x, start2.x);
    double height = std::min(end1.y, end2.y) - std::max(start1.y, start2.y);

    double intersection_area = width * height;
    double IOU;
    if(width < 0 || height < 0) IOU = 0.0;
    else IOU = intersection_area / (((end1.x - start1.x) * (end1.y - start1.y)) + (curr_vec.w * curr_vec.h) - intersection_area);

    //    qDebug() << "IOU: " << IOU;

    return IOU;
}

double IOU(cv::Point2f prev_pt_LT, cv::Point2f curr_pt_LT, cv::Point2f prev_pt_RB, cv::Point2f curr_pt_RB){

    double curr_w = curr_pt_RB.x - curr_pt_LT.x;
    double curr_h = curr_pt_RB.y - curr_pt_LT.y;
    double prev_w = prev_pt_RB.x - prev_pt_LT.x;
    double prev_h = prev_pt_RB.y - prev_pt_LT.y;

    double width = std::min(prev_pt_RB.x, curr_pt_RB.x) - std::max(prev_pt_LT.x, curr_pt_LT.x);
    double height = std::min(prev_pt_RB.y, curr_pt_RB.y) - std::max(prev_pt_LT.y, curr_pt_LT.y);

    double intersection_area = width * height;
    double IOU;
    if(width < 0 || height < 0) IOU = 0.0;
    else IOU = intersection_area / (curr_w * curr_h + prev_w * prev_h - intersection_area);
    qDebug() << "2 IOU: " << IOU;
    return IOU;
}


void set_ID_fast(std::vector<bbox_t_history>& total_fruit
                 , std::vector<bbox_t_history>& prev_vec, std::vector<bbox_t_history>& curr_vec, QList<cv::Mat> Homo_history
                 , bool prev_fruit, int threshold, int lost_track_threshold, cv::Mat& check_mat, cv::Mat maturity_mat, bool save_IOU)
{
    cv::Mat save_frame = maturity_mat.clone();
    int curr_frame = Homo_history.size();

    QList<int> used_ids;
    int used_id = -1;
    for(int i = 0 ; i < curr_vec.size() ; i++){
        cv::Point2f trajectory((float)curr_vec.at(i).x + (float)curr_vec.at(i).w / 2, (float)curr_vec.at(i).y + (float)curr_vec.at(i).h / 2);
        cv::Point2d w_h(curr_vec.at(i).w, curr_vec.at(i).h);
        cv::Point2f curr_fruit((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);

        if(prev_fruit){
            double distance = 1000;    // Distance between predict point and true point
            for(int j = 0 ; j < prev_vec.size() ; j++){
                bool duplicate = false;
                for(int n = 0 ; n < used_ids.size() ; n++){
                    if(prev_vec.at(j).track_id == used_ids.at(n)){
                        duplicate = true;
                        break;
                    }
                }
                if(duplicate){
                    continue;
                }

                cv::Point2f p((float)prev_vec.at(j).x + (float)prev_vec.at(j).w / 2, (float)prev_vec.at(j).y + (float)prev_vec.at(j).h / 2);
                cv::Point2f prev_tracked_fruit = get_tracked_point(Homo_history.at(curr_frame - 1), p);
                double temp = cv::norm(prev_tracked_fruit - curr_fruit);
                double iou = IOU(prev_vec.at(j), curr_vec.at(i), Homo_history.at((curr_frame - 1)));

                QFile history;
                QTextStream out(&history);
                if(save_IOU){
                    history.setFileName("F://realsense_D435//info_bt_fruit.csv");
                    if(history.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                        out << "Frame: " << curr_frame << ", L2-norm: " << temp << ", IOU: " << iou << ", ";
                    }
                }
                if((temp < distance) && (iou >= 0.9)) {
                    distance = temp;
                    if(temp < threshold){
                        curr_vec.at(i).track_id = prev_vec.at(j).track_id;
                        used_id = prev_vec.at(j).track_id;
                        if(save_IOU)    out << "1, ID: " << used_id << "\n";
                        qDebug() << "first stage - tracked  ID:" << prev_vec.at(j).track_id;
                    }
                    else {
                        curr_vec.at(i).track_id = 0;
                        if(save_IOU)    out << "0\n";
                        qDebug() << "first stage - Lost2Track or New";}
                }
                else{
                    if(save_IOU)    out << "0\n";
                }
                if(save_IOU)    history.close();
            }
            //            qDebug() << used_id;
            if(used_id != -1)    used_ids.append(used_id);
            if(curr_vec.at(i).track_id != 0){   // Old Fruit compare to last frame
                auto it = std::find_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector)
                { return vector.track_id == curr_vec.at(i).track_id; });

                it->trajectory.append(trajectory);
                it->history.append(2);
                it->frame_mat.append(save_frame);
                it->width_height.append(w_h);
            }
        }
        else{
            curr_vec.at(i).track_id = total_fruit.size() + 1;
            for(int pp = 0 ; pp < curr_frame ; pp++){
                curr_vec.at(i).history.append(0);    // Inactive
            }
            curr_vec.at(i).history.append(2);   // First Tracked
            curr_vec.at(i).trajectory.append(trajectory);
            curr_vec.at(i).frame_mat.append(save_frame);
            curr_vec.at(i).width_height.append(w_h);
            total_fruit.push_back(curr_vec.at(i));
            qDebug() << "2. New Fruit (No Fruit in last frame)  ID: " << curr_vec.at(i).track_id;
        }
    }

    // Append history with lost and mark the lost frame
    for(int i = 0 ; i < total_fruit.size() ; i++){
        if(total_fruit.at(i).history.at(total_fruit.at(i).history.size() - 1) != 0){ // Not inactive
            if(total_fruit.at(i).history.size() < curr_frame + 1){
                total_fruit.at(i).history.append(1);    // Lost
                if(total_fruit.at(i).history.at(total_fruit.at(i).history.size() - 2) == 2){   // From tracked -> lost
                    total_fruit.at(i).lost_frame = curr_frame;
                }
            }
        }
    }
    for(int i = 0 ; i < curr_vec.size() ; i++){
        if(curr_vec.at(i).track_id == 0){   // 1. Lost -> Tracked  2. New fruit
            QList<double> dis;
            QList<double> IOU_r;
            QList<int> lost_index;
            cv::Point2f curr_point((float)curr_vec.at(i).x + (float)curr_vec.at(i).w / 2, (float)curr_vec.at(i).y + (float)curr_vec.at(i).h / 2);
            cv::Point2d w_h(curr_vec.at(i).w, curr_vec.at(i).h);
            qDebug() << "check : 1. Lost -> Tracked  2. New fruit";
            for(int j = 0 ; j < total_fruit.size() ; j++){
                if(total_fruit.at(j).history.at(total_fruit.at(j).history.size() - 1) == 1){    // Lost
                    double iou;
                    cv::Point2f lost_point = total_fruit.at(j).trajectory.at(total_fruit.at(j).trajectory.size() - 1);
                    cv::Point2f lost_wh = total_fruit.at(j).width_height.at(total_fruit.at(j).width_height.size() - 1);
                    cv::Point2f lost_left_top(lost_point.x - lost_wh.x / 2, lost_point.y - lost_wh.y / 2);
                    cv::Point2f lost_right_bottom(lost_point.x + lost_wh.x / 2, lost_point.y + lost_wh.y / 2);
                    cv::putText(check_mat, "ID : " + std::to_string(total_fruit.at(j).track_id), cv::Point2f(lost_point.x - 30, lost_point.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1.5);
                    qDebug() << "Search for Lost fruit, Lost ID: " << total_fruit.at(j).track_id
                             << ", lost frame: " << total_fruit.at(j).lost_frame
                             << ", lost point: " << lost_point.x << ", " << lost_point.y
                             << ", homo_size" << Homo_history.size() << " ===";
                    bool track_is_out = false;
                    for(int h = total_fruit.at(j).lost_frame - 1 ; h <= curr_frame - 1 ; h++){
                        cv::Point2f output = get_tracked_point(Homo_history.at(h), lost_point);
                        cv::Point2f output_lost_left_top = get_tracked_point(Homo_history.at(h), lost_left_top);
                        cv::Point2f output_lost_right_bottom = get_tracked_point(Homo_history.at(h), lost_right_bottom);
                        // If lost-tracked point is out of the img --> From Lost to Inactive && won't compare with others
                        if(output.x < 0 || output.x > check_mat.cols || output.y < 0 || output.y > check_mat.rows){
                            track_is_out = true;
                            total_fruit.at(j).history.pop_back();
                            total_fruit.at(j).history.append(0);
                            break;
                        }
                        lost_point = output;
                        lost_left_top = output_lost_left_top;
                        lost_right_bottom = output_lost_right_bottom;
                        cv::circle(check_mat, lost_point, 3, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), -1);
                        cv::rectangle(check_mat, lost_left_top, lost_right_bottom, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1);
                    }
                    if(track_is_out == false){      // If tracked point is inside the img --> put into compare
                        cv::circle(check_mat, lost_point, lost_track_threshold, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1);
                        double temp = cv::norm(lost_point - curr_point);
                        cv::Point2f curr_left_top(curr_vec.at(i).x, curr_vec.at(i).y);
                        cv::Point2f curr_right_bottom(curr_vec.at(i).x + curr_vec.at(i).w, curr_vec.at(i).y + curr_vec.at(i).h);
                        iou = IOU(lost_left_top, curr_left_top, lost_right_bottom, curr_right_bottom);
                        dis.append(temp);
                        qDebug() << "dis: " << temp;
                        IOU_r.append(iou);
                        lost_index.append(j);  // j means lost fruit in total_fruit with "INDEX" not "ID"

                    }
                    cv::putText(check_mat, "IOU : " + std::to_string(iou), cv::Point2f(lost_point.x - 30, lost_point.y + 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1.5);
                }
            }
            if(dis.size() != 0){    // There are Lost fruit in total fruit
                double min = dis.at(0);
                int index = lost_index.at(0);
                for(int k = 1 ; k < dis.size() ; k++){
                    if(dis.at(k) < min && IOU_r.at(k) >= 0.4){
                        min = dis.at(k);
                        index = lost_index.at(k);
                    }
                }
                qDebug() << "min dis" << min;
                if(min < lost_track_threshold){   // 1. Lost -> Tracked
                    curr_vec.at(i).track_id = total_fruit.at(index).track_id;
                    total_fruit.at(index).history.pop_back();
                    total_fruit.at(index).history.append(2);
                    total_fruit.at(index).trajectory.append(curr_point);
                    total_fruit.at(index).frame_mat.append(save_frame);
                    total_fruit.at(index).width_height.append(w_h);
                    qDebug() << "1. Lost -> Tracked" << " ID:" << total_fruit.at(index).track_id;
                }
                else{   // 2. New Fruit
                    curr_vec.at(i).track_id = total_fruit.size() + 1;
                    for(int pp = 0 ; pp < curr_frame ; pp++){
                        curr_vec.at(i).history.append(0);    // Inactive
                    }
                    curr_vec.at(i).history.append(2);   // First Tracked
                    curr_vec.at(i).trajectory.append(curr_point);
                    curr_vec.at(i).frame_mat.append(save_frame);
                    curr_vec.at(i).width_height.append(w_h);
                    total_fruit.push_back(curr_vec.at(i));
                    qDebug() << "2. New Fruit" << " ID:" << curr_vec.at(i).track_id;
                }
            }
            else{   // There is no Lost fruit in total fruit --> New Fruit
                curr_vec.at(i).track_id = total_fruit.size() + 1;
                for(int pp = 0 ; pp < curr_frame ; pp++){
                    curr_vec.at(i).history.append(0);    // Inactive
                }
                curr_vec.at(i).history.append(2);   // First Tracked
                curr_vec.at(i).trajectory.append(curr_point);
                curr_vec.at(i).frame_mat.append(save_frame);
                curr_vec.at(i).width_height.append(w_h);
                total_fruit.push_back(curr_vec.at(i));
                qDebug() << "2. New Fruit (No Fruit Lost in total fruit)" << " ID:" << curr_vec.at(i).track_id;
            }
        }
    }
}

std::vector<bbox_t_history> bbox_t2bbox_t_history(std::vector<bbox_t> input){
    std::vector<bbox_t_history> output;
    for(int i = 0 ; i < input.size() ; i++){
        bbox_t_history temp;
        temp.h = input.at(i).h;
        temp.obj_id = input.at(i).obj_id;
        temp.prob = input.at(i).prob;
        temp.track_id = input.at(i).track_id;
        temp.w = input.at(i).w;
        temp.x = input.at(i).x;
        temp.y = input.at(i).y;
        output.push_back(temp);
    }
    return output;
}

#endif // FEATURE_FUNCTION_HPP
