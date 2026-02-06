#include "AickTensorrt.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace DeepLearningFuncs;

namespace {
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <image_path> [get_num]\n"
              << "Notes:\n"
              << "  - 10Xconfig.txt must be in current working directory.\n"
              << "  - This test calls AickTensorrt pipeline (yolo + impurity + regression).\n";
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string image_path = argv[1];
    int get_num = 10;
    if (argc >= 3) {
        get_num = std::atoi(argv[2]);
        if (get_num <= 0) get_num = 10;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    const int target_w = 1600;
    const int target_h = 1600;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    std::vector<float> stageInfo = {0.0f, 0.0f, 0.0f, 1000.0f, 1000.0f};

    std::vector<std::vector<float>> params;
    params.push_back({0.0f, 0.0f, 0.0f}); 
    params.push_back({0.0f});            

    AickTensorrtStarter* api = AickTensorrtStarter::get_instance();
    api->InitParam(params);

    if (api->InintModel() != 0) {
        std::cerr << "InintModel failed. Check 10Xconfig.txt and engine paths.\n";
        return 1;
    }

    // Warmup to stabilize TensorRT latency
    const int warmup_iters = 3;
    for (int i = 0; i < warmup_iters; ++i) {
        (void)api->AickTensorrt(reinterpret_cast<char*>(resized.data), resized.cols, resized.rows, stageInfo);
    }

    // Timed average inference
    const int measure_iters = 10;
    double total_ms = 0.0;
    int ret = 0;
    for (int i = 0; i < measure_iters; ++i) {
        auto t_start = std::chrono::steady_clock::now();
        ret = api->AickTensorrt(reinterpret_cast<char*>(resized.data), resized.cols, resized.rows, stageInfo);
        auto t_end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
    double avg_ms = total_ms / measure_iters;
    std::cout << "Average inference time (" << measure_iters << " runs): " << avg_ms << " ms" << std::endl;
    if (ret < 0) {
        std::cerr << "AickTensorrt pipeline failed, ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "AickTensorrt pipeline completed. count=" << ret << std::endl;

    std::vector<std::vector<float>> shootingPoint;
    std::vector<std::string> shootingNames;
    api->GetShootingPoint(shootingPoint, shootingNames, get_num);

    std::cout << "Detected points: " << shootingPoint.size() << std::endl;
    for (size_t i = 0; i < shootingPoint.size(); ++i) {
        std::cout << "Point[" << i << "]: ";
        for (size_t j = 0; j < shootingPoint[i].size(); ++j) {
            std::cout << shootingPoint[i][j] << (j + 1 == shootingPoint[i].size() ? "" : ", ");
        }
        if (i < shootingNames.size()) {
            std::cout << " | name=" << shootingNames[i];
        }
        std::cout << "\n";
    }

    return 0;
}
