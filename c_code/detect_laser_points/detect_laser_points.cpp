// Full pipeline for laser point extraction in C++
// Dependencies: OpenCV, nlohmann::json

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>
#include <chrono>

#define MAX_WORKERS 6

#include "json.hpp"
using json = nlohmann::json;

namespace fs = std::filesystem;

std::mutex io_mutex;

cv::Point2f green_laser_finder_faster(const cv::Mat& frame, int threshold) {
    cv::Mat green_channel, binary;
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);
    green_channel = channels[1];

    cv::threshold(green_channel, binary, threshold, 255, cv::THRESH_BINARY);
    cv::Moments m = cv::moments(binary);

    if (m.m00 != 0) {
        return cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
    } else {
        return cv::Point2f(NAN, NAN);
    }
}

// void extract_laser_points(int dataset_idx, int cam_idx, const std::string& root_dir, const std::string& dataset_name,
                        //   const std::string& cam_name, int frame_start, int frame_end, const std::string& output_path)
void extract_laser_points(int dataset_idx, int cam_idx, const std::string& root_dir, const std::string& dataset_name, 
                          const std::string& cam_name, int frame_start, int frame_end, const std::string& output_path,
                            bool use_roi) 
   {
    
    std::string video_file = root_dir + "/" + dataset_name + "/" + cam_name + ".mp4";
    std::string output_file = output_path + "/" + cam_name + "_centroids.csv";

    cv::VideoCapture vr(video_file, cv::CAP_FFMPEG);
    if (!vr.isOpened()) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Failed to open video file: " << video_file << std::endl;
        return;
    }

    vr.set(cv::CAP_PROP_POS_FRAMES, frame_start);
    std::ofstream out(output_file);
    out << "frame,x,y\n";

    int valid_count = 0;   
    cv::Point2f previous(-1, -1);
    int roi_half_width = 75;

    for (int i = frame_start; i < frame_end; ++i) {
        
        // read full frame
        cv::Mat frame;
        if (!vr.read(frame)) break;

        cv::Mat roi_frame = frame;
        bool valid_roi = false;

        if (use_roi && previous.x >= 0 && previous.y >= 0) {
            int x0 = std::max(int(previous.x) - roi_half_width, 0);
            int y0 = std::max(int(previous.y) - roi_half_width, 0);
            int x1 = std::min(x0 + 2 * roi_half_width, frame.cols - 1);
            int y1 = std::min(y0 + 2 * roi_half_width, frame.rows - 1);
            roi_frame = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            valid_roi = true;
        }

        cv::Point2f pt = green_laser_finder_faster(roi_frame, 50);

        // // if (1) using ROI mode && you do not find in roi, try in full frame
        // if (use_roi && valid_roi && std::isnan(pt.x) && !std::isnan)
        // {
        //     cv::Point2f pt = green_laser_finder_faster(frame, 50);  
        //     valid_roi = false;  
        // }

        if (!std::isnan(pt.x)) {
            if (valid_roi) pt += cv::Point2f(previous.x - roi_half_width, previous.y - roi_half_width);            
            previous = pt;
            out << i << "," << pt.x << "," << pt.y << "\n";
            ++valid_count;
        } else {
            previous = cv::Point2f(-1, -1);
        }
    }


    {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << cam_name << ": " << valid_count << " valid frames" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    
    auto start_time = std::chrono::high_resolution_clock::now();


    if (argc < 3) {
        std::cerr << "Usage: ./laser_extract -c <config_path> [-i <dataset_idx>] --use_roi" << std::endl;
        return 1;
    }

    // parse arguments
    std::string config_dir;
    int dataset_idx = -1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_dir = argv[++i];
        } else if ((arg == "-i" || arg == "--dataset_idx") && i + 1 < argc) {
            dataset_idx = std::stoi(argv[++i]);
        }
    }

    bool use_roi = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_dir = argv[++i];
        } else if ((arg == "-i" || arg == "--dataset_idx") && i + 1 < argc) {
            dataset_idx = std::stoi(argv[++i]);
        } else if (arg == "--use_roi") {
            use_roi = true;
            std::cout << " using roi search method" << std::endl;
        }
    }


    // check for config json
    std::ifstream file(config_dir + "/config.json");
    if (!file) {
        std::cerr << "Failed to open config.json" << std::endl;
        return 1;
    }

    // extract metadata
    json calib_config;
    file >> calib_config;

    std::string root_dir = calib_config["root_dir"];
    std::vector<std::string> laser_datasets = calib_config["lasers"];
    std::vector<std::string> cam_serials = calib_config["cam_serials"];
    std::vector<std::string> cam_names;
    for (const auto& serial : cam_serials) {
        cam_names.push_back("Cam" + serial);
    }

    std::string results_dir = config_dir + "/results";
    fs::create_directories(results_dir);

    auto process_dataset = [&](int ds_idx) {
        std::string dataset_name = laser_datasets[ds_idx];
        int frame_start = calib_config["frames"][ds_idx][0];
        int frame_end = calib_config["frames"][ds_idx][1];

        std::string output_path = results_dir + "/" + dataset_name;
        fs::create_directories(output_path);

        const size_t total_cams = cam_names.size();

        std::vector<std::future<void>> workers;
        for (size_t cam_idx = 0; cam_idx < cam_names.size(); ++cam_idx) {
            workers.push_back(std::async(std::launch::async, extract_laser_points,
                                         ds_idx, cam_idx, root_dir, dataset_name,
                                         cam_names[cam_idx], frame_start, frame_end, output_path,use_roi));
        }
        for (auto& w : workers) w.get();
    };

    // auto process_dataset = [&](int ds_idx) {
    //     std::string dataset_name = laser_datasets[ds_idx];
    //     int frame_start = calib_config["frames"][ds_idx][0];
    //     int frame_end = calib_config["frames"][ds_idx][1];
    
    //     std::string output_path = results_dir + "/" + dataset_name;
    //     fs::create_directories(output_path);
    
    //     const size_t total_cams = cam_names.size();
    //     // const unsigned int max_workers = std::min<unsigned int>(
    //     //     std::thread::hardware_concurrency(), total_cams);
    //     const unsigned int max_workers = 2;
    //     for (size_t i = 0; i < total_cams; i += max_workers) {
    //         std::vector<std::future<void>> workers;
    
    //         size_t batch_end = std::min(i + max_workers, total_cams);
    //         for (size_t cam_idx = i; cam_idx < batch_end; ++cam_idx) {
    //             workers.push_back(std::async(std::launch::async, extract_laser_points,
    //                                          ds_idx, cam_idx, root_dir, dataset_name,
    //                                          cam_names[cam_idx], frame_start, frame_end,
    //                                          output_path, use_roi));
    //         }
    
    //         // Wait for all in batch
    //         for (auto& w : workers) w.get();
    //     }
    // };
    

    // process one or more datasets if needed
    if (dataset_idx != -1) {
        process_dataset(dataset_idx);
    } 
    else 
    {
        for (int i = 0; i < laser_datasets.size(); ++i) {
            process_dataset(i);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            std::cout << "Total execution time for dataset: "<< i << " is " << elapsed.count() << " seconds" << std::endl;
        }
    }

   


    return 0;
}
