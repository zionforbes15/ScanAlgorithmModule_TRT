#include "DetModel.h"
#include "AickTensorrt.h"
#include <math.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <iterator>
#include <string>
#include <numeric>  
#include <cstring>
#include <cctype>
#include <cuda_runtime_api.h>


CTestModel::CTestModel()
{
    std::cout << "CTestModel Initializing for TensorRT..." << std::endl;

    m_runtime = nullptr;
    m_engine = nullptr;
    m_context = nullptr;
    m_stream = nullptr;
    m_isEngineLoaded = false;

    // 读取配置
    std::map<std::string, std::string> config;
    ReadConfig_Det(config);
    
    if(config.count("conf_thres")) m_confThres = std::stof(config["conf_thres"]);
    if(config.count("iou_thres")) m_iouThres = std::stof(config["iou_thres"]);

    std::cout << "CTestModel Config Loaded. Conf: " << m_confThres << std::endl;
}

// --- 析构函数 ---
CTestModel::~CTestModel()
{
    std::cout << "Releasing TensorRT resources..." << std::endl;
    clearResources();
    if (m_context) { delete m_context; m_context = nullptr; }
    if (m_engine) { delete m_engine; m_engine = nullptr; }
    if (m_runtime) { delete m_runtime; m_runtime = nullptr; }
}

void CTestModel::clearResources() {
    for (auto& b : m_bindings) {
        if (b.device) {
            cudaFree(b.device);
            b.device = nullptr;
        }
        b.host.clear();
    }
    m_bindings.clear();
    m_output_indices.clear();
    m_input_index = -1;
    m_output_det_index = 0;
    m_output_mask_index = -1;

    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

bool CTestModel::bindAllTensors() {
    if (!m_context) return false;
    for (const auto& b : m_bindings) {
        if (!b.device) return false;
        if (!m_context->setTensorAddress(b.name.c_str(), b.device)) {
            std::cerr << "[TensorRT] ERROR: Failed to set address for tensor: " << b.name << std::endl;
            return false;
        }
    }
    return true;
}

bool CTestModel::copyOutputsToHost() {
    if (m_output_indices.empty()) return false;
    for (int idx : m_output_indices) {
        auto& b = m_bindings[idx];
        if (b.host.size() != b.count) b.host.resize(b.count);
        cudaMemcpyAsync(b.host.data(), b.device, b.count * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    }
    cudaStreamSynchronize(m_stream);
    return true;
}
int CTestModel::initModel(std::string model_path, int class_num, float conf_thres, float iou_thres, std::tuple<int, int, int> input_size) {
    //防御性检查：确保路径不为空
    if (model_path.empty()) {
        std::cerr << "[TensorRT] ERROR: Model path is empty! Check your config file." << std::endl;
        return -1;
    }

    m_confThres = conf_thres;
    m_iouThres = iou_thres;
    m_classNum = class_num; 
    m_input_h = std::get<0>(input_size);
    m_input_w = std::get<1>(input_size);
    m_input_c = std::get<2>(input_size);
    m_input_b = 1; 
    m_input_w_h = std::make_tuple(m_input_w, m_input_h);

    std::cout << "[TensorRT] Initializing model: " << model_path << std::endl;
    std::cout << "[TensorRT] Configured Dims: " << m_input_w << "x" << m_input_h << "x" << m_input_c << std::endl;

    // 调用加载引擎.engine
    if (!loadEngine(model_path)) {
        std::cerr << "[TensorRT] Model initialization FAILED at loadEngine!" << std::endl;
        m_isEngineLoaded = false;
        return -1; 
    }

    if (!bindAllTensors()) {
        std::cerr << "[TensorRT] ERROR: Failed to bind tensor addresses!" << std::endl;
        return -1;
    }

    m_isEngineLoaded = true;
    std::cout << "[TensorRT] Model initialized and tensors bound successfully!" << std::endl;
    return 0; 
}

void CTestModel::ReadConfig_Det(std::map<std::string, std::string> &config) {
    std::ifstream fp("10Xconfig.txt"); 
    if (!fp.is_open()) {
        std::cerr << "Cannot open file!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(fp, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        
        if (line.empty()) continue; 
        size_t pos = line.find('|');
        if (pos != std::string::npos) {
            std::string temp_key = line.substr(0, pos);
            std::string temp_value = line.substr(pos + 1);
            
            config[temp_key] = temp_value; 
            if (temp_key == "conf_thres") m_confThres = std::stof(temp_value);
            if (temp_key == "iou_thres") m_iouThres = std::stof(temp_value);
        }
    }
}

void CTestModel::encryptDecrypt(const std::string& toEncrypt, const std::string& key, std::string& output) {
    output = toEncrypt;
    int keyLength = key.length();
    for (size_t i = 0; i < toEncrypt.length(); i++) {
        output[i] = toEncrypt[i] ^ key[i % keyLength];
    }
}

double CTestModel::myfunction(double num) {
    return exp(num);
}

int CTestModel::MatPLength_Det(const char* length) {
    if (length == nullptr) return 0;
    return static_cast<int>(strlen(length));
}

int CTestModel::getChannels_Det(cv::Mat& img) {
    if (img.empty()) return 0;
    return img.channels(); 
}

template <typename T>
void softmax(const std::vector<T> &v, std::vector<T> &s) {
    double sum = 0.0;
    std::transform(v.begin(), v.end(), s.begin(), [](T num) { 
        return static_cast<T>(std::exp(static_cast<double>(num)));
    });
    sum = std::accumulate(s.begin(), s.end(), 0.0);
    for (size_t i = 0; i < s.size(); ++i) {
        s.at(i) /= (T)sum;
    }
}

int CTestModel::getIndexOfMax(std::vector<float>& v) {
    if (v.empty()) {
        std::cerr << "Vector is empty, cannot find max index." << std::endl;
        return -1; 
    }

    size_t maxIndex = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[maxIndex] < v[i]) {
            maxIndex = i;
        }
    }
    return static_cast<int>(maxIndex);
}

void CTestModel::test_new(std::string str) {
    std::ofstream ofs;
    ofs.open("10X_result.txt", std::ios::app);
    if (ofs.is_open()) {
        ofs << str << std::endl;
        ofs.close();
    }
}

bool hasFileExtension(const std::string& filename, const std::string& extension) {
    size_t dotPosition = filename.find_last_of('.');
    if (dotPosition == std::string::npos || dotPosition == filename.length() - 1) {
        return false;
    }
    std::string fileExtension = filename.substr(dotPosition + 1);
    return fileExtension == extension;
}

bool CTestModel::loadEngine(const std::string& engine_path) {
    try {
        std::cout << "Loading engine file: " << engine_path << std::endl;

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Engine file does not exist: " << engine_path << std::endl;
            return false;
        }

        std::vector<uint8_t> engine_data;
        if (engine_path.find(".engine") != std::string::npos) {
            engine_data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        } else {
            // 加密处理逻辑保留
            engine_data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            std::vector<uint8_t> key = { 0x01, 0x02, 0x03 };
            for (size_t i = 0; i < engine_data.size(); i++) {
                engine_data[i] ^= key[i % key.size()];
            }
            std::cout << "Engine decrypted successfully." << std::endl;
        }
        file.close();

        clearResources();
        if (m_context) { delete m_context; m_context = nullptr; }
        if (m_engine) { delete m_engine; m_engine = nullptr; }
        if (m_runtime) { delete m_runtime; m_runtime = nullptr; }

        // 反序列化
        m_runtime = nvinfer1::createInferRuntime(m_logger);
        m_engine = m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!m_engine) return false;

        m_context = m_engine->createExecutionContext();
        cudaStreamCreate(&m_stream);

        // 动态绑定所有输入输出
        int num_tensors = m_engine->getNbIOTensors();
        m_bindings.resize(num_tensors);
        m_output_indices.clear();
        m_input_index = -1;
        m_output_det_index = 0;
        m_output_mask_index = -1;

        for (int i = 0; i < num_tensors; ++i) {
            const char* name = m_engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
            nvinfer1::Dims dims = m_engine->getTensorShape(name);
            
            size_t count = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                // 处理动态 batch 情况，如果 dim < 0 则设为 1
                count *= (dims.d[j] > 0) ? dims.d[j] : 1;
            }

            auto& b = m_bindings[i];
            b.name = name;
            b.mode = mode;
            b.dims = dims;
            b.count = count;

            // 为每个 Tensor 分配显存
            cudaMalloc(&b.device, count * sizeof(float));
            
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (m_input_index < 0) m_input_index = i;
                std::cout << "[TensorRT] Input Found: " << name << " (Index: " << i << ")" << std::endl;
            } else {
                m_output_indices.push_back(i);
                b.host.resize(count);
                std::cout << "[TensorRT] Output Found: " << name << " (Index: " << i << "), Size: " << count << std::endl;
            }
        }

        // 识别 mask 输出
        m_output_det_index = 0;
        m_output_mask_index = -1;
        auto to_lower = [](const std::string& s) {
            std::string out = s;
            for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return out;
        };
        for (size_t oi = 0; oi < m_output_indices.size(); ++oi) {
            const auto& b = m_bindings[m_output_indices[oi]];
            std::string lower = to_lower(b.name);
            if (lower.find("mask") != std::string::npos || lower.find("proto") != std::string::npos || lower.find("seg") != std::string::npos) {
                m_output_mask_index = static_cast<int>(oi);
                break;
            }
        }
        if (m_output_mask_index < 0 && m_output_indices.size() >= 2) {
            m_output_mask_index = 1;
        }

        m_isEngineLoaded = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in loadEngine: " << e.what() << std::endl;
        return false;
    }
}

void CTestModel::checkModelInfo() {
    if (!m_engine) return;

    std::cout << "\n=== TensorRT Model Inspection ===" << std::endl;

    int nbIO = m_engine->getNbIOTensors();
    
    for (int i = 0; i < nbIO; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);

        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
        std::string type = (mode == nvinfer1::TensorIOMode::kINPUT) ? "Input" : "Output";

        std::cout << type << " [" << i << "]: " << name << " | Shape: (";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j == dims.nbDims - 1 ? "" : ", ");
        }
        std::cout << ")" << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT && dims.nbDims == 4) {
            m_input_b = dims.d[0]; // Batch
            m_input_c = dims.d[1]; // Channels
            m_input_h = dims.d[2]; // Height
            m_input_w = dims.d[3]; // Width
            
            if (m_input_b < 0) m_input_b = 1;
        }
    }
    std::cout << "==================================\n" << std::endl;
}

int CTestModel::PointLength(const float* data, int max_size) {
    if (data == nullptr) return 0;
    return max_size; 
}

void CTestModel::sigmoid_threshold(const float* input, int length, float threshold, std::vector<uint8_t>& result) {
    result.clear();
    result.resize(length);

    for (int i = 0; i < length; i++) {
        float sigmoid_v = 1.0f / (1.0f + std::exp(-input[i]));
        
        // 调试打印
        // std::cout << "sigmoid " << input[i] << " -> " << sigmoid_v << std::endl;

        if (sigmoid_v > threshold) {
            result[i] = 1;
        } else {
            result[i] = 0;
        }
    }
}

std::vector<cv::Mat> CTestModel::clarity_getOutput(cv::Mat& inputImg) {
    cv::Mat resized = letterbox_image_v3(inputImg, std::make_tuple(m_input_w, m_input_h));
    
    double rgb_means = 0.7696278 * 255;
    double rgb_std = 1 / (0.20933049 * 255);

    resized.convertTo(resized, CV_32FC3, 1.0);
    cv::Mat one_sub = cv::Mat::ones(resized.rows, resized.cols, CV_32FC1) * rgb_means;
    
    // RGB -> GRAY -> 减均值 -> 乘方差 -> 变回 RGB
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_RGB2GRAY);
    gray = gray - one_sub;
    gray = gray * rgb_std;
    cv::cvtColor(gray, resized, cv::COLOR_GRAY2RGB);

    cv::Mat preprocessedImage;
    cv::dnn::blobFromImage(resized, preprocessedImage); 

    if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return {};
    auto& in = m_bindings[m_input_index];
    size_t input_mem_size = in.count * sizeof(float);
    cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

    // 执行 TensorRT 推理 
    m_context->enqueueV3(m_stream);
    copyOutputsToHost();

    if (m_output_indices.empty()) return {};
    const auto& out = m_bindings[m_output_indices[0]];
    const float* out_ptr = out.host.data();

    // 后处理 (Sigmoid 与 Mask 生成)
    std::vector<cv::Mat> Semantic;
    float threshold = m_confThres;
    
    // 这里的维度从类的成员变量中直接获取，不再依赖 node_dims
    int out_h = m_input_h; 
    int out_w = m_input_w; 
    int out_c = m_input_c; 

    for (int k = 0; k < out_c; k++) {
        cv::Mat mask = cv::Mat::zeros(out_h, out_w, CV_8UC1);
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                int index = i * out_w + j + k * out_h * out_w;
                float temp_v = out_ptr[index];
                
                // Sigmoid 激活
                float sigmoid_v = 1.0f / (1.0f + std::exp(-temp_v));
                
                mask.at<uchar>(i, j) = (sigmoid_v > threshold) ? 1 : 0;
            }
        }
        Semantic.push_back(mask);
    }

    return Semantic;
}

std::vector<std::vector<det_box>> CTestModel::getOutput(char* image, int w, int h) {
    std::vector<std::vector<det_box>> det_boxes;
    cv::Mat inputImg = cv::Mat(h, w, CV_8UC3, image);
    try {
        //图像载入
        cv::Mat inputImg = cv::Mat(h, w, CV_8UC3, image);
        test_p("getOutput ================inputImg---------------rows: " + std::to_string(inputImg.rows) + 
               "---cols: " + std::to_string(inputImg.cols) + "---c: " + std::to_string(inputImg.channels()));

        //预处理与缩放日志
        cv::Mat resized = letterbox_image_v2(inputImg, m_input_w_h);
        test_p("getOutput ================resized---------------rows: " + std::to_string(resized.rows) + 
               "---cols: " + std::to_string(resized.cols) + "---c: " + std::to_string(resized.channels()));

        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        
        cv::Mat imgRGBFloat;
        resized.convertTo(imgRGBFloat, CV_32F, 1.0 / 255);
        test_p("getOutput ================1.0 / 255 (归一化完成)");

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(imgRGBFloat, preprocessedImage); // HWC -> CHW
        test_p("getOutput ================HWC->CHW---------------rows: " + std::to_string(preprocessedImage.rows));

        std::cout << "===== TensorRT Config Dimensions =====" << std::endl;
        std::cout << "m_input_b: " << m_input_b << " | m_input_c: " << m_input_c 
                  << " | m_input_h: " << m_input_h << " | m_input_w: " << m_input_w << std::endl;

        test_p("============= 开始 TensorRT 显存拷贝与推理 Run：");
        
        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) {
            test_p("============= ERROR: Invalid input binding index");
            return det_boxes;
        }
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

        m_context->enqueueV3(m_stream);
        copyOutputsToHost();
        test_p("============= TensorRT 推理已完成，结果已拷贝回 Host");

        test_p("============= 开始非极大值抑制 NMS (TRT 版)");
        const int stride5 = 5 + m_classNum;
        const int stride4 = 4 + m_classNum;

        std::vector<float> merged5; // row-major [N, stride5]
        size_t totalN5 = 0;
        std::vector<float> merged4; // C-first [stride4, N]
        size_t totalN4 = 0;

        auto append_rows = [&](const float* src, size_t rows, int stride, std::vector<float>& dst) {
            size_t base = dst.size();
            dst.resize(base + rows * stride);
            std::memcpy(dst.data() + base, src, rows * stride * sizeof(float));
        };
        auto append_CN_to_rowmajor = [&](const float* src, size_t N, int C, std::vector<float>& dst) {
            size_t base = dst.size();
            dst.resize(base + N * C);
            for (int c = 0; c < C; ++c) {
                for (size_t n = 0; n < N; ++n) {
                    dst[base + n * C + c] = src[c * N + n];
                }
            }
        };
        auto append_to_Cfirst = [&](const float* src_CN, size_t N, int C, std::vector<float>& dst, size_t& totalN) {
            if (dst.empty()) dst.resize(C * N, 0.0f);
            else dst.resize(C * (totalN + N), 0.0f);
            for (int c = 0; c < C; ++c) {
                std::memcpy(dst.data() + c * (totalN + N) + totalN, src_CN + c * N, N * sizeof(float));
            }
            totalN += N;
        };
        auto append_NC_to_Cfirst = [&](const float* src_NC, size_t N, int C, std::vector<float>& dst, size_t& totalN) {
            if (dst.empty()) dst.resize(C * N, 0.0f);
            else dst.resize(C * (totalN + N), 0.0f);
            for (size_t n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    dst[c * (totalN + N) + (totalN + n)] = src_NC[n * C + c];
                }
            }
            totalN += N;
        };

        for (int out_idx : m_output_indices) {
            const auto& b = m_bindings[out_idx];
            const float* ptr = b.host.data();
            int nd = b.dims.nbDims;
            if (nd == 3) {
                int d1 = b.dims.d[1];
                int d2 = b.dims.d[2];
                if (d1 == stride5) { // [1, C, N] -> row-major
                    append_CN_to_rowmajor(ptr, d2, d1, merged5);
                    totalN5 += d2;
                } else if (d2 == stride5) { // [1, N, C]
                    append_rows(ptr, d1, d2, merged5);
                    totalN5 += d1;
                } else if (d1 == stride4) {
                    append_to_Cfirst(ptr, d2, d1, merged4, totalN4);
                } else if (d2 == stride4) {
                    append_NC_to_Cfirst(ptr, d1, d2, merged4, totalN4);
                }
            } else if (nd == 2) {
                int d0 = b.dims.d[0];
                int d1 = b.dims.d[1];
                if (d1 == stride5) {
                    append_rows(ptr, d0, d1, merged5);
                    totalN5 += d0;
                } else if (d0 == stride5) {
                    append_CN_to_rowmajor(ptr, d1, d0, merged5);
                    totalN5 += d1;
                } else if (d1 == stride4) {
                    append_NC_to_Cfirst(ptr, d0, d1, merged4, totalN4);
                } else if (d0 == stride4) {
                    append_to_Cfirst(ptr, d1, d0, merged4, totalN4);
                }
            }
        }

        if (!merged4.empty() && totalN4 > 0) {
            det_boxes = non_max_suppression_trt_yolov8(merged4.data(), (int)totalN4, m_confThres, m_iouThres, m_classNum);
        } else if (!merged5.empty() && totalN5 > 0) {
            det_boxes = non_max_suppression_trt(merged5.data(), (int)totalN5, m_confThres, m_iouThres, m_classNum);
        } else {
            test_p("============= WARNING: No usable output tensor layout matched for NMS");
        }
        
        test_p("============= 10X检测推理完成，det_boxes 数量: " + std::to_string(det_boxes.size()));

    } catch (const std::exception& e) {
        std::cerr << "Exception caught in TRT: " << e.what() << std::endl;
        test_p("============= 开始10X检测推理 exception: " + std::string(e.what()));
    }

    return det_boxes;
}

std::vector<det_box> CTestModel::getMcnOutput(char* image, int w, int h) {
    test_p("=============开始微核推理 (TensorRT 版) getMcnOutput");
    std::vector<std::vector<det_box>> det_boxes_all;
    std::vector<det_box> res_boxs;

    try {
        cv::Mat inputImg = cv::Mat(h, w, CV_8UC3, image);
        cv::Mat gray_img;
        cv::cvtColor(inputImg, gray_img, cv::COLOR_BGR2GRAY);

        // 合并为3通道灰度图
        cv::Mat gray3Ch;
        std::vector<cv::Mat> channels_img = {gray_img, gray_img, gray_img};
        cv::merge(channels_img, gray3Ch);

        // 减去最小值
        double minVal;
        cv::minMaxLoc(gray3Ch, &minVal, NULL);
        gray3Ch = gray3Ch - minVal;

        test_p("getMcnOutput ================gray3Ch---------------rows: " + std::to_string(gray3Ch.rows));

        // 缩放与归一化
        cv::Mat resized = letterbox_image_v2(gray3Ch, m_input_w_h);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        
        cv::Mat imgRGBFLoat;
        resized.convertTo(imgRGBFLoat, CV_32F, 1.0 / 255);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(imgRGBFLoat, preprocessedImage); // HWC -> CHW

        test_p("============= 开始显存拷贝与 TRT 推理");
        
        // 上传数据
        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return res_boxs;
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

        // 执行推理
        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        int output_dim = m_engine->getNbIOTensors() - 1; 

        if (output_dim == 1 && !m_output_indices.empty()) {
            test_p("============= 开始 NMS (YOLOv8-TRT 模式)");
            // 使用之前定义的针对 TRT float* 的 NMS
            int obj_count = 8400; // 根据 m_output_size 动态调整
            const auto& out = m_bindings[m_output_indices[0]];
            det_boxes_all = non_max_suppression_trt_yolov8(out.host.data(), obj_count, m_confThres, m_iouThres, m_classNum);
            
            // 坐标还原
            std::tuple<int, int> size_1024 = std::make_tuple(1024, 1024);
            scale_coords_v4(inputImg, size_1024, det_boxes_all);
        } else {
            test_p("Warning: 多输出(分割)逻辑在板端需配合具体 Tensor 名称解析");
        }

        // 将二维 vector 展平为一维返回
        for (auto& cls_boxes : det_boxes_all) {
            for (auto& box : cls_boxes) {
                res_boxs.push_back(box);
            }
        }

    } catch (const std::exception& e) {
        std::string errorMessage = e.what();
        test_p("============= getMcnOutput exception: " + errorMessage);
    }

    return res_boxs;
}

std::vector<instance_seg> CTestModel::getCellFishOutput(char* image, int w, int h) {
    test_p("=============开始FISH细胞推理 (TensorRT) getCellFishOutput");
    std::vector<instance_seg> res_boxs;

    try {
        cv::Mat inputImg = cv::Mat(h, w, CV_8UC3, image);
        cv::Mat gray_img;
        cv::cvtColor(inputImg, gray_img, cv::COLOR_BGR2GRAY);
        
        // 合并为三通道灰度图
        cv::Mat gray3Ch(gray_img.rows, gray_img.cols, CV_8UC3);
        std::vector<cv::Mat> channels_img = {gray_img, gray_img, gray_img};
        cv::merge(channels_img, gray3Ch);

        // 核心：减去最小值偏移
        double minVal;
        cv::minMaxLoc(gray3Ch, &minVal, NULL);
        gray3Ch = gray3Ch - minVal;
        
        test_p("getCellFishOutput ================gray3Ch---------------rows: " + std::to_string(gray3Ch.rows));

        // 缩放与归一化
        cv::Mat resized = letterbox_image_v2(gray3Ch, m_input_w_h);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        test_p("getCellFishOutput ================cvtColor---------------rows: " + std::to_string(resized.rows));

        cv::Mat imgRGBFLoat;
        resized.convertTo(imgRGBFLoat, CV_32F, 1.0 / 255);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(imgRGBFLoat, preprocessedImage); // HWC -> CHW

        // TensorRT 推理执行
        test_p("============= 开始显存上载与双输出推理执行");
        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return res_boxs;
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        if (m_output_indices.empty()) return res_boxs;
        int det_order = m_output_det_index;
        int mask_order = m_output_mask_index;
        if (det_order < 0 || det_order >= (int)m_output_indices.size()) det_order = 0;
        if (mask_order < 0 || mask_order >= (int)m_output_indices.size()) mask_order = (m_output_indices.size() > 1 ? 1 : -1);
        const auto& det_out = m_bindings[m_output_indices[det_order]];
        const auto* mask_out = (mask_order >= 0) ? &m_bindings[m_output_indices[mask_order]] : nullptr;
        test_p("============= 推理完成，显存回传内存完毕");

        int _segChannels = 32;
        int obj_count = 8400; 
        int dims = m_classNum + 4 + _segChannels;
        int obj_counts = obj_count * dims;

        test_p("getCellFishOutput ================ 开始输出张量转置");
        std::vector<float> transposed_prob0(obj_counts);
        for (int i = 0; i < dims; i++) {
            for (int j = 0; j < obj_count; j++) {
                // TensorRT [1, 116, 8400] -> [1, 8400, 116]
                transposed_prob0[j * dims + i] = det_out.host[i * obj_count + j];
            }
        }

        std::vector<int> outputTensorShape = {1, std::get<0>(m_input_w_h), obj_count};
        std::vector<int> outputMaskTensorShape = {1, 32, std::get<0>(m_input_w_h)/4, std::get<1>(m_input_w_h)/4};


        std::vector<int> ids;
        std::vector<float> confs;
        std::vector<cv::Rect> rects;
        std::vector<cv::Mat> mks;

        test_p("============= 开始 NMS 与 Mask 生成逻辑");
        if (!mask_out) return res_boxs;
        non_max_suppression_trt_yolov8_seg(inputImg, transposed_prob0.data(), const_cast<float*>(mask_out->host.data()), 
                                           outputTensorShape, outputMaskTensorShape, 
                                           ids, confs, rects, mks, m_classNum);

        for (size_t i = 0; i < ids.size(); i++) {
            instance_seg temp;
            temp.mask = mks[i];
            temp.rect = rects[i];
            temp.conf = confs[i];
            temp.cls_idx = ids[i];
            temp.x1 = rects[i].x;
            temp.y1 = rects[i].y;
            temp.w = rects[i].width;
            temp.h = rects[i].height;
            res_boxs.push_back(temp);
        }
        test_p("getCellFishOutput ================ 处理完成，总目标数: " + std::to_string(res_boxs.size()));

    } catch (const std::exception& e) {
        std::string errorMessage = e.what();
        test_p("============= getCellFishOutput Exception: " + errorMessage);
    }

    return res_boxs;
}

struct MaskParams {
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

std::vector<det_box> CTestModel::getCellOutput(char* image, int w, int h) {
    test_p("=============开始原位推理 (TensorRT) getCellOutput");
    std::vector<det_box> res_boxs;
    try {
        cv::Mat inputImg = cv::Mat(h, w, CV_8UC3, image);
        cv::Mat gray_img;
        cv::cvtColor(inputImg, gray_img, cv::COLOR_BGR2GRAY);
        
        // 构造三通道灰度图
        cv::Mat gray3Ch(gray_img.rows, gray_img.cols, CV_8UC3);
        std::vector<cv::Mat> channels_img = {gray_img, gray_img, gray_img};
        cv::merge(channels_img, gray3Ch);

        double minVal;
        cv::minMaxLoc(gray3Ch, &minVal, NULL);
        gray3Ch = gray3Ch - minVal;
        
        test_p("getCellOutput ================gray3Ch---------------rows: " + std::to_string(gray3Ch.rows));

        // Letterbox 与 归一化
        cv::Mat resized = letterbox_image_v2(gray3Ch, m_input_w_h);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        
        cv::Mat imgRGBFLoat;
        resized.convertTo(imgRGBFLoat, CV_32F, 1.0 / 255);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(imgRGBFLoat, preprocessedImage); // HWC -> CHW
        test_p("getCellOutput ================ 开始上载显存并推理");
        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return res_boxs;
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);

        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);
        m_context->enqueueV3(m_stream);
        copyOutputsToHost();
        test_p("getCellOutput ================ 推理完成，数据已回传");


        std::tuple<int, int> net_size = std::tuple<int, int>(1024, 1024);
        int obj_count = 8400; 
    
        if (m_output_indices.empty()) return res_boxs;
        const auto& out = m_bindings[m_output_indices[0]];
        std::vector<std::vector<det_box>> det_boxes_all = non_max_suppression_trt_yolov8(
            out.host.data(), 
            obj_count, 
            m_confThres, 
            m_iouThres, 
            m_classNum
        );

        test_p("============= 原位推理坐标还原");
        scale_coords_v4(inputImg, net_size, det_boxes_all);

        for (const auto& cls_vec : det_boxes_all) {
            for (const auto& box : cls_vec) {
                res_boxs.push_back(box);
            }
        }
        
        test_p("getCellOutput ================ 处理结束，目标总数: " + std::to_string(res_boxs.size()));

    } catch (const std::exception& e) {
        test_p("============= 原位推理 Exception: " + std::string(e.what()));
    }

    return res_boxs;
}

inline cv::Mat letterbox_image_ambitus(cv::Mat image_src, std::tuple<int, int>& size) {
    cv::Mat grayImg;
    cv::cvtColor(image_src, grayImg, cv::COLOR_RGB2GRAY);
    
    int h = grayImg.rows;
    int w = grayImg.cols;

    // 获取四个顶点的像素值
    double a1 = (int)grayImg.at<uchar>(0, 0);
    double a2 = (int)grayImg.at<uchar>(0, w - 1);
    double a3 = (int)grayImg.at<uchar>(h - 1, 0);
    double a4 = (int)grayImg.at<uchar>(h - 1, w - 1);
    
    // 计算平均填充背景值
    float padding_number = (a1 + a2 + a3 + a4) / 4.0f;

    // 如果是 224 输入的模型，强制填充白色
    if (std::get<0>(size) == 224)
        padding_number = 255;

    // test_p("panding_value: " + std::to_string(padding_number));

    float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<0>(size)) / image_src.rows);
    int nw = (int)(image_src.cols * scale);
    int nh = (int)(image_src.rows * scale);

    cv::Mat image;
    cv::resize(image_src, image, cv::Size(nw, nh));

    int pad_w = std::get<0>(size) - nw;
    int pad_h = std::get<0>(size) - nh;

    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;

    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, 
                      cv::Scalar(padding_number, padding_number, padding_number));
    return image;
}

std::vector<float> CTestModel::getOutput(cv::Mat &inputImg) {
    test_p("============= 开始分类推理 (TensorRT) getOutput");
    std::vector<float> output_x;

    try {
        cv::Mat resized = letterbox_image_ambitus(inputImg, m_input_w_h);
        
        float rgb_means = 0.7696278f * 255.0f;
        float rgb_std = 1.0f / (0.20933049f * 255.0f);

        // 核心数学转换
        resized.convertTo(resized, CV_32FC3, 1.0);
        
        // 构造均值减法矩阵
        cv::Mat one_sub = cv::Mat::ones(resized.rows, resized.cols, CV_32FC1) * rgb_means;
        
        // 转换为灰度图
        cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY); 
        
        // 减去均值并乘以标准差
        resized = resized - one_sub;
        resized = resized * rgb_std;
        
        cv::cvtColor(resized, resized, cv::COLOR_GRAY2RGB);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(resized, preprocessedImage);

        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return {};
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);
        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        if (m_output_indices.empty()) return {};
        const auto& out = m_bindings[m_output_indices[0]];
        std::vector<float> prob_vec(out.host.begin(), out.host.end());

        // 后处理,取出有效类别数的数据
        std::vector<float> input_x;
        for (int i = 0; i < m_classNum; i++) {
            input_x.push_back(prob_vec[i]);
        }

        // 执行 Softmax 归一化
        output_x.resize(input_x.size());
        softmax<float>(input_x, output_x);

        test_p("getOutput 推理完成，类别0得分: " + std::to_string(output_x[0]));

    } catch (const std::exception& e) {
        test_p("============= getOutput Exception: " + std::string(e.what()));
    }

    return output_x;
}

std::vector<float> CTestModel::getRegressionOutput(cv::Mat &inputImg, int version) {
    test_p("============= 开始回归推理 (TensorRT) getRegressionOutput, version: " + std::to_string(version));
    std::vector<float> output_x;

    try {
        cv::Mat resized = letterbox_image_ambitus(inputImg, m_input_w_h);
        
        double rgb_means = 0.0;//0.7696278*255；
        double rgb_std = 1.0 / 255.0;// 1 /（0.20933049*255）；

        resized.convertTo(resized, CV_32FC3, 1.0);
        cv::Mat one_sub = cv::Mat::ones(resized.rows, resized.cols, CV_32FC1) * (float)rgb_means;
        cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
        resized = resized - one_sub;
        resized = resized * (float)rgb_std;

        cv::cvtColor(resized, resized, cv::COLOR_GRAY2RGB);

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(resized, preprocessedImage);

        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return {};
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);
        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        if (m_output_indices.empty()) return {};
        const auto& out = m_bindings[m_output_indices[0]];
        std::vector<float> prob_vec(out.host.begin(), out.host.end());

        if (version == 0) {
            for (int i = 0; i < m_classNum; i++) {
                output_x.push_back(prob_vec[i]);
            }
        }
        else {
            for (int i = 0; i < 2; i++) {
                output_x.push_back(prob_vec[i]);
            }
            
            std::vector<float> temp_vector;
            for (int i = 2; i < m_classNum; i++) {
                temp_vector.push_back(prob_vec[i]);
            }
            
            int max_index = getIndexOfMax(temp_vector);
            test_p("********************************* prob max_index: " + std::to_string(max_index));
            output_x.push_back((float)max_index);
        }

    } catch (const std::exception& e) {
        test_p("============= getRegressionOutput Exception: " + std::string(e.what()));
    }

    return output_x;
}

std::vector<det_box> CTestModel::getFungus40XOutput(cv::Mat img, int w, int h, float conf_threshold, float nms_threshold) 
{   
    test_p("============= 开始真菌40X推理 (TensorRT/YOLOv11) getFungus40XOutput");
    std::vector<det_box> result;
    cv::Size original_size = img.size();

    try {
        // 预处理：BGR -> RGB -> Resize -> Float32 -> Blob
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
        
        cv::Mat resized;
        cv::resize(rgbImg, resized, cv::Size(m_input_w, m_input_h));

        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(resized, preprocessedImage, 1.0/255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return result;
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        
        // 数据搬运：Host -> Device
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        if (m_output_indices.empty()) return result;
        const auto& out = m_bindings[m_output_indices[0]];

        int obj_count = (int)(out.count / (4 + m_classNum)); 
    
        std::vector<std::vector<det_box>> det_boxes_all = non_max_suppression_trt_yolov8(
            out.host.data(), 
            obj_count, 
            conf_threshold, 
            nms_threshold, 
            m_classNum
        );

        // 坐标还原：只针对 NMS 后的最终结果
        float scale_x = static_cast<float>(original_size.width) / m_input_w;
        float scale_y = static_cast<float>(original_size.height) / m_input_h;

        for (auto& cls_vec : det_boxes_all) {
            for (auto& box : cls_vec) {
                // 此时 box.x1/y1/w/h 还是网络输入尺寸(如640)下的坐标，需还原至原图
                box.x1 *= scale_x;
                box.y1 *= scale_y;
                box.w  *= scale_x;
                box.h  *= scale_y;

                box.x1 = std::max(0.0f, std::min(box.x1, (float)original_size.width - 1));
                box.y1 = std::max(0.0f, std::min(box.y1, (float)original_size.height - 1));
                box.w  = std::max(0.0f, std::min(box.w,  (float)original_size.width - box.x1));
                box.h  = std::max(0.0f, std::min(box.h,  (float)original_size.height - box.y1));

                result.push_back(box);
            }
        }
        
        test_p("Fungus40X 推理及后处理完成，目标数: " + std::to_string(result.size()));

    } catch (const std::exception& e) {
        test_p("============= getFungus40XOutput 异常: " + std::string(e.what()));
    }

    return result;
}

std::vector<det_box> CTestModel::getFungusOverallOutput(cv::Mat img, int w, int h, float conf_threshold, float nms_threshold)
{
    test_p("============= 开始真菌全景推理 (TensorRT/YOLOv11) getFungusOverallOutput");
    std::vector<det_box> result;
    cv::Size original_size = img.size();

    try {
        // BGR -> RGB -> Resize -> Float32 -> Blob
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
        
        cv::Mat resized;
        cv::resize(rgbImg, resized, cv::Size(m_input_w, m_input_h));

        // 归一化并将 [H, W, C] 转换为 [C, H, W]
        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(resized, preprocessedImage, 1.0/255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

        if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return result;
        auto& in = m_bindings[m_input_index];
        size_t input_mem_size = in.count * sizeof(float);
        cudaMemcpyAsync(in.device, preprocessedImage.data, input_mem_size, cudaMemcpyHostToDevice, m_stream);

        m_context->enqueueV3(m_stream);
        copyOutputsToHost();

        if (m_output_indices.empty()) return result;
        const auto& out = m_bindings[m_output_indices[0]];

        int obj_count = (int)(out.count / (4 + m_classNum)); 
        
        std::vector<std::vector<det_box>> det_boxes_all = non_max_suppression_trt_yolov8(
            out.host.data(), 
            obj_count, 
            conf_threshold, 
            nms_threshold, 
            m_classNum
        );

        // 坐标还原：将 640 尺寸下的坐标映射回原图尺寸
        float scale_x = static_cast<float>(original_size.width) / m_input_w;
        float scale_y = static_cast<float>(original_size.height) / m_input_h;

        for (auto& cls_vec : det_boxes_all) {
            for (auto& box : cls_vec) {
                // 还原并计算
                box.x1 *= scale_x;
                box.y1 *= scale_y;
                box.w  *= scale_x;
                box.h  *= scale_y;

                // 安全裁剪，防止绘制越界
                box.x1 = std::max(0.0f, std::min(box.x1, (float)original_size.width - 1));
                box.y1 = std::max(0.0f, std::min(box.y1, (float)original_size.height - 1));
                box.w  = std::max(0.0f, std::min(box.w,  (float)original_size.width - box.x1));
                box.h  = std::max(0.0f, std::min(box.h,  (float)original_size.height - box.y1));

                result.push_back(box);
            }
        }
        
        test_p("Overall 推理及后处理完成，目标数: " + std::to_string(result.size()));

    } catch (const std::exception& e) {
        test_p("============= getFungusOverallOutput Error: " + std::string(e.what()));
    }

    return result;
}

double varianceOfLaplacian(const cv::Mat& image) {
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_32F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    return stddev.val[0] * stddev.val[0];
}

double CTestModel::calculateSharpnessScoreFast(const cv::Mat& image) {
    if (image.empty()) return 0.0;

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // 快速缩放
    cv::Mat resized;
    double fx = 0.5;
    double fy = 0.5;
    cv::resize(gray, resized, cv::Size(), fx, fy, cv::INTER_LINEAR);

    double laplacianVar = varianceOfLaplacian(resized);

    return laplacianVar;
}

double CTestModel::calculateSharpnessScore(const cv::Mat& src) {
    if (src.empty()) return 0.0;

    cv::Mat roi = extractChromosomeROI(src);
    if (roi.empty()) {
        return calculateSharpnessScoreFast(src);
    }

    // Use Laplacian variance on ROI as the high-precision score.
    return varianceOfLaplacian(roi);
}

cv::Mat CTestModel::extractChromosomeROI(const cv::Mat& src) {
    if (src.empty()) return cv::Mat();

    cv::Mat gray, denoised, thresh, roi;

    // 颜色空间转换
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 降噪：使用高斯模糊平滑细节，减少二值化时的噪点
    cv::GaussianBlur(gray, denoised, cv::Size(GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0.9);

    cv::adaptiveThreshold(denoised, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY_INV, 21, 3);

    // 形态学操作：先腐蚀去小噪点，再膨胀连接染色体主体
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
    cv::erode(thresh, thresh, kernel_small);
    cv::dilate(thresh, thresh, kernel_large);

    // 轮廓提取与面积过滤
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建全黑掩膜
    cv::Mat mask = cv::Mat::zeros(thresh.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area >= MIN_CHROMOSOME_AREA) {
            cv::drawContours(mask, contours, (int)i, cv::Scalar(255), -1);
        }
    }
    // 结果提取：将掩膜应用到去噪后的灰度图上
    roi = cv::Mat::zeros(denoised.size(), denoised.type());
    denoised.copyTo(roi, mask);

    return roi;
}

void CTestModel::detectChromosomeCorners(const cv::Mat& roi, const cv::Mat& mask,
	std::vector<cv::KeyPoint>& corners,
	std::vector<float>& responses,
	float& chromosome_area) {
    
	//统计染色体实际像素面积 (非零像素点)
	chromosome_area = static_cast<float>(cv::countNonZero(mask));
    
	// 面积太小则认为是噪声，不进行检测
	if (chromosome_area < MIN_CHROMOSOME_AREA) {
		corners.clear();
		responses.clear();
		return;
	}

	// 初始化并运行 FAST 特征检测器
	cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(
		FAST_THRESHOLD, true, cv::FastFeatureDetector::TYPE_9_16
	);
	fast->detect(roi, corners);

	// 提取角点响应强度
	responses.resize(corners.size());
	for (size_t i = 0; i < corners.size(); ++i) {
		responses[i] = corners[i].response;
	}

	// 精细过滤
	std::vector<cv::KeyPoint> filtered_corners;
	std::vector<float> filtered_responses;
	for (size_t i = 0; i < corners.size(); ++i) {
		cv::KeyPoint kp = corners[i];
		float resp = responses[i];
		int x = static_cast<int>(kp.pt.x);
		int y = static_cast<int>(kp.pt.y);

		// 掩膜过滤：确保角点落在 extractChromosomeROI 提取出的染色体区域内
		if (x < 0 || x >= mask.cols || y < 0 || y >= mask.rows ||
			mask.at<uchar>(y, x) != 255) {
			continue;
		}

		// 边缘抑制：剔除距离图像边缘过近的点 (EDGE_PADDING 建议值 5-10)
		if (kp.pt.x < EDGE_PADDING || kp.pt.x > static_cast<float>(roi.cols - EDGE_PADDING) ||
			kp.pt.y < EDGE_PADDING || kp.pt.y > static_cast<float>(roi.rows - EDGE_PADDING)) {
			continue;
		}

		filtered_corners.push_back(kp);
		filtered_responses.push_back(resp);
	}

	corners.swap(filtered_corners);
	responses.swap(filtered_responses);
}

bool CTestModel::initYoloEngine(const std::string& engine_path)
{
    if (m_isEngineLoaded) {
        test_p("【提示】模型已初始化，无需重复加载！");
        return true;
    }
    test_p("加载 TensorRT 引擎路径: " + engine_path);

    if (!loadEngine(engine_path)) {
        test_p("【错误】反序列化引擎失败！");
        return false;
    }
    if (!bindAllTensors()) {
        test_p("【错误】绑定 Tensor 地址失败！");
        return false;
    }

    checkModelInfo();
    m_input_w_h = std::make_tuple(m_input_w, m_input_h);
    m_isEngineLoaded = true;
    test_p("【提示】TensorRT 模型初始化成功！");
    test_p("模型实际输入尺寸确认: W=" + std::to_string(m_input_w) + ", H=" + std::to_string(m_input_h));
    return true;
}

std::vector<cv::Mat> CTestModel::batchInferBoxes(
    const std::vector<cv::Mat>& frames,
    std::vector<std::vector<cv::Rect>>& all_detect_boxes)
{
    all_detect_boxes.clear();
    if (!m_isEngineLoaded) {
        test_p("【错误】模型未初始化！");
        return {};
    }

    size_t B = frames.size();
    if (B == 0) return {};
    if (B > (size_t)m_input_b) {
        test_p("【警告】输入 Batch 大于模型上限，将只处理前 " + std::to_string(m_input_b) + " 帧");
        B = m_input_b;
    }

    all_detect_boxes.resize(B);
    std::vector<cv::Mat> results(B);

    std::vector<float> x_factors(B), y_factors(B);
    
    // 准备一个连续的 CPU 缓冲区，用于一次性拷贝到显存
    size_t single_frame_floats = 3 * m_input_h * m_input_w;
    std::vector<float> batch_input_buffer(B * single_frame_floats);

    for (size_t i = 0; i < B; ++i) {
        const cv::Mat& frame = frames[i];
        int w = frame.cols, h = frame.rows;
        int _max = std::max(h, w);
        
        // Letterbox 逻辑：保持长宽比，填充黑色背景
        cv::Mat canvas = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
        frame.copyTo(canvas(cv::Rect(0, 0, w, h)));

        x_factors[i] = static_cast<float>(_max) / m_input_w;
        y_factors[i] = static_cast<float>(_max) / m_input_h;
        cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0f / 255.0f, cv::Size(m_input_w, m_input_h), cv::Scalar(0, 0, 0), true, false);
        
        // 拷贝到批量缓冲区的对应位置
        memcpy(batch_input_buffer.data() + i * single_frame_floats, blob.ptr<float>(), single_frame_floats * sizeof(float));
    }

    if (m_input_index < 0 || m_input_index >= (int)m_bindings.size()) return {};
    auto& in = m_bindings[m_input_index];
    cudaMemcpyAsync(in.device, batch_input_buffer.data(), batch_input_buffer.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream);

    m_context->enqueueV3(m_stream);
    copyOutputsToHost();

    if (m_output_indices.empty()) return {};
    auto& out = m_bindings[m_output_indices[0]];

    int obj_count = 8400; 
    int dims = m_input_c; 
    for (size_t i = 0; i < B; ++i) {
        // 指向第 i 个样本的起始地址
        float* sample_ptr = out.host.data() + i * dims * obj_count;
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        for (int j = 0; j < obj_count; ++j) {
            // 找到类别得分最高的项
            float max_score = 0;
            for (int c = 0; c < m_classNum; ++c) {
                float score = sample_ptr[(4 + c) * obj_count + j];
                if (score > max_score) {
                    max_score = score;
                }
            }

            if (max_score > m_confThres) {
                float cx = sample_ptr[0 * obj_count + j];
                float cy = sample_ptr[1 * obj_count + j];
                float ow = sample_ptr[2 * obj_count + j];
                float oh = sample_ptr[3 * obj_count + j];

                int x = static_cast<int>((cx - 0.5f * ow) * x_factors[i]);
                int y = static_cast<int>((cy - 0.5f * oh) * y_factors[i]);
                int width = static_cast<int>(ow * x_factors[i]);
                int height = static_cast<int>(oh * y_factors[i]);

                boxes.emplace_back(x, y, width, height);
                confidences.push_back(max_score);
            }
        }

        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, m_confThres, m_iouThres, indices);
        }

        bool has_box = false;
        if (!indices.empty()) {
            std::vector<cv::Point> pts;
            for (int idx : indices) {
                const auto& b = boxes[idx];
                pts.push_back(b.tl());
                pts.push_back(b.br());
            }
            cv::Rect big_box = cv::boundingRect(pts);
            // 裁剪至原图范围内
            big_box = big_box & cv::Rect(0, 0, frames[i].cols, frames[i].rows);

            if (big_box.area() > 0) {
                all_detect_boxes[i] = { big_box };
                results[i] = frames[i](big_box).clone();
                has_box = true;
            }
        }

        if (!has_box) {
            results[i] = cv::Mat(frames[i].size(), CV_8UC3, cv::Scalar(255, 255, 255));
        }
    }

    return results;
}
