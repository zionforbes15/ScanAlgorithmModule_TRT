#pragma once
#include "./ImgHandle.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>


#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class CTestModel
{
public:
    CTestModel();
    virtual ~CTestModel();
    // --- 核心初始化 ---
    int initModel(std::string model_path, int class_num, float conf_thres, float iou_thres, std::tuple<int, int, int> input_size);
    bool loadEngine(const std::string& engine_path);
    bool initYoloEngine(const std::string& engine_path);
    void ReadConfig_Det(std::map<std::string, std::string> &config);
    int getIndexOfMax(std::vector<float>& v);
    void test_new(std::string str);
    std::vector<std::vector<det_box>> getOutput(char* image, int w, int h);
    std::vector<det_box> getMcnOutput(char* image, int w, int h);
    std::vector<instance_seg> getCellFishOutput(char* image, int w, int h);
    std::vector<det_box> getCellOutput(char* image, int w, int h);
    std::vector<cv::Mat> clarity_getOutput(cv::Mat& inputImg);
    std::vector<float> getOutput(cv::Mat& inputImg); 
    std::vector<float> getRegressionOutput(cv::Mat& inputImg, int version);
    std::vector<det_box> getFungus40XOutput(cv::Mat img, int w, int h, float conf_threshold, float nms_threshold);
    std::vector<det_box> getFungusOverallOutput(cv::Mat img, int w, int h, float conf_threshold, float nms_threshold);

    // --- 清晰度算法 ---
    cv::Mat extractChromosomeROI(const cv::Mat& src);
    void detectChromosomeCorners(const cv::Mat& roi, const cv::Mat& mask,
        std::vector<cv::KeyPoint>& corners,
        std::vector<float>& responses,
        float& chromosome_area);
    double calculateSharpnessScoreFast(const cv::Mat& image);
    double calculateSharpnessScore(const cv::Mat& src);
    std::vector<cv::Mat> batchInferBoxes(const std::vector<cv::Mat>& frames, std::vector<std::vector<cv::Rect>>& all_detect_boxes);

protected:
    int getChannels_Det(cv::Mat &img);
    int PointLength(const float* data, int max_size); 
    void encryptDecrypt(const std::string& toEncrypt, const std::string& key, std::string& output);
    void checkModelInfo();
    void sigmoid_threshold(const float* input, int length, float threshold, std::vector<uint8_t>& result);
    double myfunction(double num);
    int MatPLength_Det(const char* length);
    std::tuple<int, int> m_input_w_h ;
    int m_input_w = 0; 
    int m_input_h = 0;
    int m_input_c = 0;
    int m_input_b = 0;      
    int m_classNum = 3;
    float m_confThres = 0.25f;
    float m_iouThres = 0.45f;
    bool m_isEngineLoaded = false;  

    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    Logger m_logger;
    cudaStream_t m_stream = nullptr;

    struct TensorBinding {
        std::string name;
        nvinfer1::TensorIOMode mode;
        nvinfer1::Dims dims;
        size_t count = 0;
        void* device = nullptr;
        std::vector<float> host; // output host buffer
    };

    std::vector<TensorBinding> m_bindings;
    int m_input_index = -1;
    std::vector<int> m_output_indices;

    int m_output_det_index = 0;
    int m_output_mask_index = -1;

    void clearResources();

    // helpers
    bool bindAllTensors();
    bool copyOutputsToHost();

private:
    const int GAUSSIAN_KERNEL = 5;
    const int FAST_THRESHOLD = 30;
    const int EDGE_PADDING = 5;
    const double MIN_CHROMOSOME_AREA = 400.0;
};
