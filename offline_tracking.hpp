#ifndef OFFLINE_TRACKING_HPP
#define OFFLINE_TRACKING_HPP

#include "detect_3d.hpp"

void save_track_result(QString track_result_path, int online, QList<int> false_alarm){

    QFile track_result(track_result_path);
    QTextStream out2(&track_result);
    if(track_result.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out2 << "Online-tracking\n";
        out2 << "total_fruit.size() = " << online;
        out2 << "\n\nOffline-tracking\nErase-ID\n";
        for(int i = 0 ; i < false_alarm.size() ; i++){
            out2 << false_alarm.at(i) << ", ";
        }
        out2 << "\ntotal_fruit.size() = " << online - false_alarm.size();
    }
    track_result.close();
}

void save_histogram(QString save_path, QList<int> histogram_list, std::vector<std::pair<int, int>> max_min){
    QFile histogram(save_path);
    QTextStream out1(&histogram);
    if(histogram.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out1 << "Fruit size histogram\n";
        for(int i = 0 ; i < histogram_list.size() ; i++){
            out1 << histogram_list.at(i) << ", ";
        }
    }
    out1 << "\nMax ID, size: " << max_min.at(0).first << ", " << max_min.at(0).second << "\n";
    out1 << "\nMax ID, size: " << max_min.at(1).first << ", " << max_min.at(1).second << "\n";
    histogram.close();
}

std::pair<int, int> max_fruit_frame_size(QList<cv::Point2d> total_fruit_width_height){
    int frame = 0, max = 0;
    for(int i = 0 ; i < total_fruit_width_height.size() ; i++){
        int size = total_fruit_width_height.at(i).x * total_fruit_width_height.at(i).y;
        if(size > max){
            max = size;
            frame = i;
        }
    }
    return std::make_pair(frame, max);
}

QList<int> Eliminate_false_alarm(std::vector<bbox_t_history>& total_fruit, int threshold){
    QList<int> erase_id;     // if < threshold frame == Tracked --> false alarm
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int count = 0;
        for(int j = total_fruit.at(i).history.size() - 1 ; j >= 0 ; j--){
            if(total_fruit.at(i).history.at(j) == 2)    count++;
            if(count > threshold)   break;
        }
        if(count <= threshold){
            erase_id.append(total_fruit.at(i).track_id);
            total_fruit.erase(std::remove_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector){
                                  return (vector.track_id == total_fruit.at(i).track_id);
                              }), total_fruit.end());
            i--;
        }
    }
    return erase_id;
}

std::pair<QList<int>, std::vector<std::pair<int, int>>> Fruit_size_histogram(std::vector<bbox_t_history>& total_fruit){
    QList<int> histogram;
    int min = 100000, max = 0;
    int min_ID = 0, max_ID = 0;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        std::pair<int, int> max_frame_size = max_fruit_frame_size(total_fruit.at(i).width_height);
        int size = max_frame_size.second;

        total_fruit.at(i).size = size;
        histogram.append(size);
        if(size < min){ min = size; min_ID = total_fruit.at(i).track_id;}
        if(size > max){ max = size; max_ID = total_fruit.at(i).track_id;}
    }
    std::vector<std::pair<int, int>> max_min;
    max_min.push_back(std::make_pair(max_ID, max));
    max_min.push_back(std::make_pair(min_ID, min));

    qDebug() << "Max ID, size: " << max_ID << ", " << max;
    qDebug() << "Min ID, size: " << min_ID << ", " << min;

    return std::make_pair(histogram, max_min);
}

double maturity(cv::Mat& mask, cv::Mat input){
    double maturity;
    int mask_pixel = 0, mature_pixel = 0;
    if(!mask.empty()){
        for(int i = 0 ; i < mask.rows ; i++){
            for(int j = 0 ; j < mask.cols ; j++){
                if(mask.at<uchar>(i, j) == 1 || mask.at<uchar>(i, j) == 3){     // Foreground
                    mask_pixel++;
                    // Mature condition
                    if((input.at<cv::Vec3b>(i, j)[2] >= 200 && input.at<cv::Vec3b>(i, j)[1] < 200 && input.at<cv::Vec3b>(i, j)[0] < 160)
                            || (input.at<cv::Vec3b>(i, j)[2] >= 165 && input.at<cv::Vec3b>(i, j)[1] < 80 && input.at<cv::Vec3b>(i, j)[0] < 80)
                            || (input.at<cv::Vec3b>(i, j)[2] >= 100 && input.at<cv::Vec3b>(i, j)[1] < 40 && input.at<cv::Vec3b>(i, j)[0] < 40)){
                        mature_pixel++;
                        mask.at<uchar>(i, j) = 255;
                    }
                    else{mask.at<uchar>(i, j) = 100;}
                }
                else    {mask.at<uchar>(i, j) = 0;}                             // Background
            }
        }
        if(mature_pixel != 0 && mask_pixel != 0){   maturity = (double)mature_pixel / (double)mask_pixel;}
        else                                        maturity = 0;
    }
    else{   maturity = 0.0;}

    return maturity;
}

void save_ripen_img(cv::Mat input, cv::Mat mask, std::string save_path, double stage, int ID){
    cv::Mat mask_result(input.rows, input.cols, CV_8UC3);
    for(int i = 0 ; i < mask.rows ; i++){
        for(int j = 0 ; j < mask.cols ; j++){
            mask_result.at<cv::Vec3b>(i, j)[0] = input.at<cv::Vec3b>(i, j)[0] * 0.6 + mask.at<uchar>(i, j) * 0.4;
            mask_result.at<cv::Vec3b>(i, j)[1] = input.at<cv::Vec3b>(i, j)[1] * 0.6 + mask.at<uchar>(i, j) * 0.4;
            mask_result.at<cv::Vec3b>(i, j)[2] = input.at<cv::Vec3b>(i, j)[2] * 0.6 + mask.at<uchar>(i, j) * 0.4;
        }
    }
    cv::putText(mask_result, "Fruit ID : " + std::to_string(ID), cv::Point2f(0, mask_result.rows - 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);
    cv::putText(mask_result, "Ripening Stage : " + std::to_string(stage), cv::Point2f(0, mask_result.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);

    cv::imwrite(save_path, mask_result);
}

double ripening_stage(cv::Mat input, cv::Point2f trajectory, cv::Point2f width_height, int ID){

    cv::Mat mask(input.rows, input.cols, CV_8U, cv::Scalar(0));
    cv::Mat bgModel, fgModel;
    int left_top_x = std::max((int)trajectory.x - (int)width_height.x / 2 - 10, 0);
    int left_top_y = std::max((int)trajectory.y - (int)width_height.y / 2 - 10, 0);
    int width = std::min((int)width_height.x + 20, (int)input.cols);
    int height = std::min((int)width_height.y + 20, (int)input.rows);
    cv::circle(mask, cv::Point(left_top_x + width / 2, left_top_y + height / 2), std::min(width / 2, height / 2) - 10, cv::Scalar(1), -1);
    cv::Rect rect(left_top_x, left_top_y, width, height);
    cv::grabCut(input, mask, rect, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);

    double maturity_per = maturity(mask, input);

    std::string save_path = "./tracking_frame/maturity_mask/fruit_" + std::to_string(ID) + ".png";
    save_ripen_img(input, mask, save_path, maturity_per, ID);

    return maturity_per;
}

QList<double> Fruit_ripening_stage(std::vector<bbox_t_history>& total_fruit){
    QList<double>   fruit_ripening_stage;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        std::pair<int, int> max_frame_size = max_fruit_frame_size(total_fruit.at(i).width_height);
        int n_frame = max_frame_size.first;

        cv::Mat input = total_fruit.at(i).frame_mat.at(n_frame).clone();
        double stage = ripening_stage(input, total_fruit.at(i).trajectory.at(n_frame), total_fruit.at(i).width_height.at(n_frame), total_fruit.at(i).track_id);
        total_fruit.at(i).maturity = stage;
        fruit_ripening_stage.append(stage);
    }
    return fruit_ripening_stage;
}

global_coor set_coordinate(bbox_t_history total_fruit, cv::Point2f point){

    global_coor coor;

    coor.global_point = point;
    coor.global_fruit_ID = total_fruit.track_id;
    coor.maturity = total_fruit.maturity;
    coor.size = total_fruit.size;

    return coor;
}

QList<global_coor> Calculate_global_coordinate(std::vector<bbox_t_history> total_fruit, cv::Point2f& max_global, cv::Point2f& min_global, QList<cv::Mat> Homo_history){
    QList<global_coor> global_coord;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
            if(total_fruit.at(i).history.at(j) == 2){   // First tracked point
                cv::Point2f point = total_fruit.at(i).trajectory.at(0);
                global_coor coor;
                if(j == 0){         // If tracked in the first frame -> No need to calculate
                    coor = set_coordinate(total_fruit.at(i), point);
                }
                else{
                    for(int h = 0 ; h < j ; h++){
                        cv::Point2f result = global_coordinate(Homo_history.at(h), point);
                        point.x = result.x;
                        point.y = result.y;
                    }
                    coor = set_coordinate(total_fruit.at(i), point);
                }
                global_coord.append(coor);
                if(point.x > max_global.x) max_global.x = point.x;
                if(point.y > max_global.y) max_global.y = point.y;
                if(point.x < min_global.x) min_global.x = point.x;
                if(point.y < min_global.y) min_global.y = point.y;
                break;
            }
        }
    }
    return global_coord;
}

cv::Scalar set_maturity_color(double maturity){
    cv::Scalar color;
    cv::Scalar Maturity_color[5] = {cv::Scalar(9, 113, 59)
                                    , cv::Scalar(104, 192, 252)
                                    , cv::Scalar(40, 74, 244)
                                    , cv::Scalar(29, 29, 255)
                                    , cv::Scalar(0, 0, 154)};

    if(maturity < 0.25)  color = Maturity_color[0];
    else if(maturity >= 0.25 && maturity < 0.5)  color = Maturity_color[1];
    else if(maturity >= 0.5 && maturity < 0.75)  color = Maturity_color[2];
    else if(maturity >= 0.75 && maturity < 1.0)  color = Maturity_color[3];
    else if(maturity >= 1.0)  color = Maturity_color[4];
    return color;
}

int set_radius(int size, int size_bin[]){
    int radius;
    if(size >= size_bin[0] && size < size_bin[1])   radius = 4;
    else if(size >= size_bin[1] && size < size_bin[2])    radius = 8;
    else if(size >= size_bin[2] && size < size_bin[3])    radius = 12;
    else if(size >= size_bin[3] && size < size_bin[4])    radius = 16;
    else if(size >= size_bin[4] && size < size_bin[5])    radius = 20;
    return radius;
}
#endif // OFFLINE_TRACKING_HPP
