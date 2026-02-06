#include "AickTensorrt.h"
#include <ctime>
#include <fstream>
#include <exception>  
#include <map>
#include <vector>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <algorithm> 
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <iomanip>
#include <tuple>   

#include "./dbscan.h"
#include "./DetModel.h"
#include "./MemoryState.h" 

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"

#define MINIMUM_POINTS 4     
#define EPSILON (0.76)     

using namespace cv;
using namespace std;

vector<vector<float>> score_boxs;
vector<string> img_names;
float num10XImage = 0.0f; 
vector<vector<float>> IParam;
bool isCultivation = false;
std::vector<std::vector<float>> CellDataMe;

struct zx_box {
	float zx_score;
	string cls_name;
};

string& replace_all(string& src, const string& old_value, const string& new_value) {
    for (string::size_type pos(0); pos != string::npos; pos += new_value.length()) {
        if ((pos = src.find(old_value, pos)) != string::npos) {
            src.replace(pos, old_value.length(), new_value);
        }
        else break;
    }
    return src;
}

void ReadConfig(std::map<std::string, std::string> &config) {
    std::ifstream infile("10Xconfig.txt");
    if (!infile.is_open()) {
        std::cerr << "[Config] Error: Cannot open 10Xconfig.txt!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }

        if (line.empty() || line[0] == '#') continue; 

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '|') && std::getline(iss, value)) {
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            config[key] = value;
        }
    }

    infile.close();
    std::cout << "[Config] Read " << config.size() << " items from config file." << std::endl;
}

void ModifyConfig(std::map<std::string, std::string> &config, const std::string &key, const std::string &newValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        it->second = newValue;
    }
    else {
        std::cerr << "[Config] Key not found: " << key << std::endl;
    }
}

void WriteConfig(const std::map<std::string, std::string> &config) {
    std::ofstream outfile("10Xconfig.txt");
    if (!outfile.is_open()) {
        std::cerr << "[Config] Error: Cannot open file to write!" << std::endl;
        return;
    }

    for (const auto &pair : config) {
        outfile << pair.first << "|" << pair.second << std::endl;
    }

    outfile.close();
    sync(); 
}

vector<Point> Vec2dToVec(const std::vector<std::vector<Point>>& vec) {
    vector<Point> new_vector;
    size_t total_size = 0;
    for (const auto& v : vec) total_size += v.size();
    new_vector.reserve(total_size);

    for (const auto& v : vec) {
        new_vector.insert(new_vector.end(), v.begin(), v.end());
    }

    return new_vector;
}

static std::string get_date_time(bool onlynum = false) {
    struct timeval tv;
    gettimeofday(&tv, NULL); 
    char buffer[64];
    if (onlynum) {
        snprintf(buffer, sizeof(buffer), "%ld%03ld", tv.tv_sec, tv.tv_usec / 1000);
    }
    else {
        struct tm *info = localtime(&tv.tv_sec);
        // 格式化日期：YYYY/MM/DD HH:MM:SS
        char date_buf[32];
        strftime(date_buf, sizeof(date_buf), "%Y/%m/%d %H:%M:%S", info);
        snprintf(buffer, sizeof(buffer), "%s.%03ld", date_buf, tv.tv_usec / 1000);
    }

    return std::string(buffer);
}


void test_p(std::string str)
{
    if (access("./Log", 0) == -1) {
        mkdir("./Log", 0777); 
    }
    if (access("./Log/10X", 0) == -1) {
        mkdir("./Log/10X", 0777); 
    }

    std::string datatime_str = get_date_time();
    str = datatime_str + " " + str; 
    //提取日期作为文件名 (例如 20231027)
    std::string data_str_ = str.substr(0, 10);
    std::string data_str = replace_all(data_str_, "/", "");
    std::ofstream ofs;
    std::string log_path = "./Log/10X/10X_" + data_str + ".txt";
    
    ofs.open(log_path, std::ios::app);
    if (ofs.is_open()) {
        ofs << str << std::endl;
        ofs.close();
    } else {
        std::cerr << "[Log Error] Failed to write: " << str << std::endl;
    }
}

float resultDataProcessing(std::vector<float> &inf_impurities, std::vector<float> &inf_reg, int vec_w, int vec_h, std::map<std::string, std::string> config10X) {

    // 初始化权重向量
    std::vector<float> inf_reg_weight = { 1.0f, 1.0f, 1.0f };

    // 预测值截断：锁定在 [0, 1] 之间
    // 注意：在板端计算时，确保 inf_reg 至少有 3 个元素以防越界
    for (size_t i = 0; i < inf_reg.size(); i++) {
        if (inf_reg[i] < 0.0f) inf_reg[i] = 0.0f;
        if (inf_reg[i] > 1.0f) inf_reg[i] = 1.0f;
    }

    // 异常值逻辑处理（根据配置文件阈值进行降权）
    try {
        if (inf_reg[0] > std::stof(config10X["lenTop"])) {
            inf_reg_weight[0] = 0.0f;
            test_p("----------- 数量分数过高（长度维度），则将长度分数置为0");
        }
        if (inf_reg[1] > std::stof(config10X["dispTop"])) {
            inf_reg_weight[1] = 0.0f;
            test_p("----------- 分散度分数过高，则将分散度分数置为0");
        }
        if (inf_reg[2] > std::stof(config10X["numTop"])) {
            inf_reg_weight[2] = 0.0f;
            test_p("----------- 数量分数过高，则将数量分数置为0");
        }

        // 输出当前预测的置信度日志
        test_p("-----reg conf_len: " + std::to_string(inf_reg[0]) + 
               "-----reg conf_disp: " + std::to_string(inf_reg[1]) + 
               "-----reg conf_num: " + std::to_string(inf_reg[2]));

        // 筛图评分逻辑计算
        double central_point = std::stof(config10X["centralPoint"]);
        double central_d = std::stof(config10X["centralDistance"]);
        double mean_score = 0;

        // 计算数量得分
        double num_score = (1.0 - (std::min(std::abs(inf_reg[2] - central_point), central_d) / central_d));

        // 综合加权评分
        mean_score = std::stof(config10X["lenWeight"])  * inf_reg[0] * inf_reg_weight[0] + 
                     std::stof(config10X["dispWeight"]) * inf_reg[1] * inf_reg_weight[1] + 
                     std::stof(config10X["numWeight"])  * num_score  * inf_reg_weight[2];

        test_p("-----mean_score: " + std::to_string(mean_score));

        return static_cast<float>(mean_score);

    } catch (const std::exception& e) {
        // 如果配置文件中缺少某个 Key，stof 会抛出异常，这里进行保护
        test_p("Error in resultDataProcessing: " + std::string(e.what()));
        std::cerr << "[Critical] Config mapping error: " << e.what() << std::endl;
        return 0.0f;
    }
}

bool compare_intro(float a, float b, float c) {
    return a < b && b < c;
}

float resultDataProcessing_new(std::vector<float> &inf_impurities, std::vector<float> &inf_reg, int vec_w, int vec_h, std::map<std::string, std::string> &config10X) {
    // 原 Linux 端改动保留为注释，按 Windows 端逻辑执行
    // 方案B：三项都 clamp，避免 num 负数/过大拉低总分
    // for (size_t i = 0; i < inf_reg.size(); i++) {
    //     if (inf_reg[i] < 0.0f) inf_reg[i] = 0.0f;
    //     if (inf_reg[i] > 1.0f) inf_reg[i] = 1.0f;
    // }
    //
    // 方案B：num 参与总分（先加，再做软惩罚）
    // double hw_disp_weight = std::stof(config10X["hw_disperse"]);
    // double mean_score = hw_disp_weight * inf_reg[0] + (1.0 - hw_disp_weight) * inf_reg[1];
    // if (inf_reg.size() > 2) {
    //     double num_weight = std::stof(config10X["numWeight"]);
    //     mean_score += num_weight * inf_reg[2];
    // }
    //
    // // 软惩罚：num 过低时降低得分，而不是直接归零
    // if (inf_reg.size() > 2 && inf_reg[2] < std::stof(config10X["numBottom"])) {
    //     mean_score *= 0.8;
    // }

    try {
        if (inf_reg.size() > 1) {
            for (size_t i = 0; i < inf_reg.size() - 1; i++) {
                if (inf_reg[i] < 0.0f) inf_reg[i] = 0.0f;
                if (inf_reg[i] > 1.0f) inf_reg[i] = 1.0f;
            }
        }

        test_p("-----resultDataProcessing_new step 1: ");

        double hw_disp_weight = std::stof(config10X["hw_disperse"]);
        double mean_score = hw_disp_weight * inf_reg[0] + (1.0 - hw_disp_weight) * inf_reg[1];

        std::string mode = config10X["mode"];
        if (mode == "0") {
            if (inf_reg.size() > 2 && inf_reg[2] != 1.0f) {
                mean_score = 0.01;
            }
        }
        else {
            if (inf_reg.size() > 2 && inf_reg[2] == 2.0f) {
                mean_score += 2.0;
                test_p("-----mean_score: " + std::to_string(mean_score));
                return static_cast<float>(mean_score);
            }
        }

        test_p("-----resultDataProcessing_new step 2: ");

        float score_th = std::stof(config10X["meanScore_th"]);
        if (mean_score > score_th) {
            float Aspect = static_cast<float>(vec_w) / static_cast<float>(vec_h);

            bool cond1 = compare_intro(std::stof(config10X["aspectBottom"]), Aspect, std::stof(config10X["aspectTop"]));
            bool cond2 = compare_intro(std::stof(config10X["heightBottom"]), (float)vec_h, std::stof(config10X["heightTop"]));
            bool cond3 = compare_intro(std::stof(config10X["widthBottom"]), (float)vec_w, std::stof(config10X["widthTop"]));
            bool cond4 = compare_intro(std::stof(config10X["disperseBottom"]), inf_reg[1], std::stof(config10X["disperseTop"]));
            bool cond5 = compare_intro(std::stof(config10X["ChromoaspectBottom"]), inf_reg[0], std::stof(config10X["ChromoaspectTop"]));

            if (cond1 && cond2 && cond3 && cond4 && cond5) {
                mean_score += 1.0;
            }
        }

        test_p("-----mean_score: " + std::to_string(mean_score));
        return static_cast<float>(mean_score);

    } catch (const std::exception& e) {
        test_p("Critical Error in resultDataProcessing_new: " + std::string(e.what()));
        return 0.0f;
    }
}

// 保留几位小数并转为字符串
auto formatDobleValue(double val, int fixed) {
    std::string str = std::to_string(val);
    size_t pos = str.find(".");
    if (pos == std::string::npos) {
        return str;
    }
    return str.substr(0, std::min(str.length(), pos + fixed + 1));
}

//浮点数转字符串
std::string Convert(float Num)
{
    std::ostringstream oss;
    oss << Num;
    return oss.str();
}

int MatPLength(const char* length)
{
    if (length == nullptr) return 0;
    return static_cast<int>(std::strlen(length));
}

int getChannels(char *image, int W, int H, size_t data_size) {
    int channels = 0;
    if (image != nullptr && W > 0 && H > 0) {
        int bytesPerPixel = static_cast<int>(data_size / (W * H));
        
        if (bytesPerPixel == 1) channels = 1;      
        else if (bytesPerPixel == 3) channels = 3; 
        else if (bytesPerPixel == 4) channels = 4; 
    }
    return channels;
}

// 保存char * image流
void save_to_txt(char *char_stream, const char *filename, size_t stream_size = 0) {
    FILE *fp = fopen(filename, "wb"); 
    if (fp == nullptr) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    if (stream_size > 0) {
        fwrite(char_stream, 1, stream_size, fp);
    } else {
        fprintf(fp, "%s", char_stream);
    }

    fclose(fp);
}

float ImgEntropy(cv::Mat image) {
    if (image.empty()) return 0.0f;
    cv::Mat gray;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    int hist[256] = { 0 };
    size_t total_pixels = static_cast<size_t>(gray.rows) * static_cast<size_t>(gray.cols);
    if (gray.isContinuous()) {
        const uchar* p = gray.ptr<uchar>(0);
        for (size_t i = 0; i < total_pixels; ++i) {
            hist[p[i]]++;
        }
    } else {
        size_t rows = static_cast<size_t>(gray.rows);
        size_t cols = static_cast<size_t>(gray.cols);
        for (size_t i = 0; i < rows; i++) {
            const uchar* p = gray.ptr<uchar>(static_cast<int>(i));
            for (size_t j = 0; j < cols; j++) {
                hist[p[j]]++;
            }
        }
    }

    double entropy = 0;
    double inv_total = 1.0 / static_cast<double>(total_pixels); 
    for (size_t i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            double prob = (double)hist[i] * inv_total;
            entropy += -prob * log2(prob);
        }
    }
    return static_cast<float>(entropy);
}

std::string matToString(cv::Mat mat) {
    if (mat.empty()) return "";

    std::vector<unsigned char> buff;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 90 }; 
    cv::imencode(".jpg", mat, buff, params);

    return std::string(buff.begin(), buff.end());
}

/**
 * @brief 根据聚类结果计算每个簇的质心（中心点）
 * @param points 原始点集
 * @param cls_point 用于存储分类后的点集（输出参数）
 * @return vector<PointZX> 所有簇的中心点集合
 */
vector<PointZX> calculateCenterPoint(vector<PointZX>& points, vector<vector<PointZX>>& cls_point) {
    vector<int> cls;

    for (const auto& point : points) {
        bool flag = false;
        int index = 0;
        // 检查当前点是否已存在于类别中
        for (size_t i = 0; i < cls.size(); ++i) {
            if (point.clusterID == cls[i]) {
                flag = true;
                index = i;
                break;
            }
        }
        
        if (!flag) {
            cls.push_back(point.clusterID);
            vector<PointZX> p;
            p.push_back(point);
            cls_point.push_back(p);
        } else {
            // 已有类别：直接放入对应的堆中
            cls_point[index].push_back(point);
        }
    }

    vector<PointZX> CenterPoint;
    for (const auto& single_cls : cls_point) {
        if (single_cls.empty()) continue; 

        float x_sum = 0, y_sum = 0;
        int clusterid = 0;
        int count = 0;

        for (const auto& p : single_cls) {
            x_sum += p.x;
            y_sum += p.y;
            clusterid = p.clusterID;
            count++;
        }

        if (count > 0) {
            PointZX c_point;
            c_point.x = x_sum / static_cast<float>(count);
            c_point.y = y_sum / static_cast<float>(count); 
            c_point.clusterID = clusterid;
            CenterPoint.push_back(c_point);
        }
    }

    return CenterPoint;
}

namespace DeepLearningFuncs
{
	string AickTensorrtStarter::valid_mask = ""; 
	bool AickTensorrtStarter::leap_over = false;
	AickTensorrtStarter::AickTensorrtStarter() : test_model(nullptr)
	{
		std::cout << "Aick start !!!!!" << std::endl;
		
		// 第一次启动时初始化 valid_mask
		if (valid_mask.empty()) {
			cv::Mat mat = cv::Mat::ones(3300, 2200, CV_8UC1);
			valid_mask = matToString(mat);
			std::cout << "valid_mask initialized in constructor." << std::endl;
		}

		test_model = new CTestModel();
		impurities_score = new CTestModel();
		reg_score = new CTestModel();
		c_score = new CTestModel();

		//真菌
		Fungus40XDet = new CTestModel();
		FungusOverDet = new CTestModel();

		test_p(get_date_time() + "End AickTensorrtStarter()");
	}


	AickTensorrtStarter* AickTensorrtStarter::get_instance()
    {
        static AickTensorrtStarter instance;
        return &instance;
    }

	int AickTensorrtStarter::InitFungusModel()
    {
        test_p("============ 开始初始化真菌模型 (TensorRT 版) !!!");
        
        std::map<std::string, std::string> config;
        ReadConfig(config);

        string str_fungus_overall_path = config["fungus_overall_det"];
        string str_fungus40X_path = config["fungus_40X_Detect"];
        
        if (!str_fungus_overall_path.empty()) replace_all(str_fungus_overall_path, "\\", "/");
        if (!str_fungus40X_path.empty()) replace_all(str_fungus40X_path, "\\", "/");

        float overall_th = 0.2f; // 默认值
        try {
            string str_overall_th = config["fungus_overall_det_threshold"];
            if (!str_overall_th.empty()) {
                overall_th = std::stof(str_overall_th);
            }
        } catch (...) {
            test_p("Warning: fungus_overall_det_threshold invalid, using 0.2");
        }

        std::tuple<int, int, int> fungus_overall_tup = std::make_tuple(640, 1440, 3);
        std::tuple<int, int, int> fungus_tup = std::make_tuple(2448, 2048, 3);
        
        //同时初始化两个大模型，观察显存（使用 tegrastats 命令）
        if (!str_fungus_overall_path.empty() && FungusOverDet != nullptr)
        { 
            test_p("Loading Overall Model: " + str_fungus_overall_path);
            std::tuple<int, int, int> fungus_overall_tup = std::make_tuple(640, 1440, 3);
			FungusOverDet->initModel(str_fungus_overall_path, 1, overall_th, 0.2, fungus_overall_tup);

        }

        if (!str_fungus40X_path.empty() && Fungus40XDet != nullptr)
        {
            test_p("Loading 40X Model: " + str_fungus40X_path);
			std::tuple<int, int, int> fungus_tup = std::make_tuple(2448, 2048, 3);
            Fungus40XDet->initModel(str_fungus40X_path, 2, 0.1, 0.2, fungus_tup);
        }

        test_p("============ 完成真菌初始化模型 !!!");
        return 0;
    }


	int AickTensorrtStarter::InitFungusSeg() {
        return 0;
    }

    int AickTensorrtStarter::InintModel(std::string model_path, int class_num, float conf_thres, float iou_thres) {
        test_p("============ 开始初始化 AickTensorrt 10X 模型 !!!");
        
        num10XImage = 0;
        std::map<std::string, std::string> config;
        ReadConfig(config);

        int show_inf = 0;
        int operation_model = 0;
        try {
            if (!config["showInf"].empty()) show_inf = std::stoi(config["showInf"]);
            if (!config["operationModel"].empty()) operation_model = std::stoi(config["operationModel"]);
        } catch (...) {
            test_p("Config Error: showInf or operationModel invalid, using default 0");
        }

        if (show_inf) {
            UseCondition();
        }

        string static_path = config["10X_detection"];  // 2cls_fp16.engine
        string str_impurities_score = config["Impurity_score"];// ch_impurities_v2_fp16.engine
        string str_reg_score = config["Chromosome_score"];// regression_fp16.engine
        string str_clarity = config["Foreground_seg"];//resnet34_960_960_mc_fp16.engine

        float clarity_th = 0.5f;
        if (!config["Foreground_seg_threshold"].empty()) {
            clarity_th = std::stof(config["Foreground_seg_threshold"]);
        }

        std::tuple<int, int, int> score_tup = std::make_tuple(224, 224, 3);
        std::tuple<int, int, int> clarity_tup = std::make_tuple(960, 960, 3);
        std::tuple<int, int, int> tup = std::make_tuple(1600, 1600, 3);

        test_p("Loading Impurity Score: " + str_impurities_score);
        impurities_score->initModel(str_impurities_score, 2, conf_thres, iou_thres, score_tup);
        
        test_p("Loading Reg Score: " + str_reg_score);
        reg_score->initModel(str_reg_score, 3, conf_thres, iou_thres, score_tup);
        
        test_p("Loading Main 10X Detector: " + static_path);
        test_model->initModel(static_path, 1, conf_thres, iou_thres, tup);

        if (operation_model == 0) {
            test_p("Loading Clarity Seg: " + str_clarity);
            c_score->initModel(str_clarity, 3, clarity_th, iou_thres, clarity_tup);
        } else {
            test_p("Skip Clarity Seg (operationModel=1)");
        }

        test_p("============ 完成 10X 初始化模型 !!!");
        return 0;
    }

	int AickTensorrtStarter::InitParam(vector<vector<float>> fParam) {
		IParam = fParam; 
		return 0;
	}

	// 类卷积操作，将卷积核区域处理做方差或者取最大值
	cv::Mat imageSplitByKer(const cv::Mat& img, int ksize, string flag) {
		if (img.empty()) return cv::Mat();

		int w = img.rows;
		int h = img.cols;
		int kstride = (ksize - 1) / 2;

		// 扩充边界
		cv::Mat img_res;
		// 使用 OpenCV 优化过的边界扩充，代替手动 copyTo，Scalar(255)
		cv::copyMakeBorder(img, img_res, kstride, kstride, kstride, kstride, cv::BORDER_CONSTANT, cv::Scalar(255));

		// 类型转换
		if (img_res.type() != CV_32F) {
			img_res.convertTo(img_res, CV_32F);
		}
		
		cv::Mat canvas = cv::Mat::zeros(w, h, CV_32F);
		if (flag == "var") {
			cv::Mat mean_img, mean_sq_img, sq_img;
			cv::boxFilter(img_res, mean_img, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_ISOLATED);

			cv::multiply(img_res, img_res, sq_img);
			cv::boxFilter(sq_img, mean_sq_img, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_ISOLATED);

			cv::Mat mean_img_sq;
			cv::multiply(mean_img, mean_img, mean_img_sq);
			canvas = mean_sq_img - mean_img_sq;
			
			// 裁剪回到原图大小
			canvas = canvas(cv::Rect(kstride, kstride, h, w)).clone();
		}
		else if (flag == "max") {
			// 使用形态学膨胀操作计算局部最大值
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
			cv::dilate(img_res, canvas, kernel);
			canvas = canvas(cv::Rect(kstride, kstride, h, w)).clone();
		}

		cv::Mat varLogImage;
		cv::log(canvas + 1e-6f, varLogImage);

		test_p("imageSplitByKer processing completed with flag: " + flag);
		return varLogImage;
	}


	void saveImageAsPPM(const char* image, int w, int h, const std::string& filename) {
		if (image == nullptr) return;

		std::ofstream file(filename, std::ios::out | std::ios::binary);

		if (file.is_open()) {
			file << "P6\n" << w << " " << h << "\n255\n"; 
			size_t total_bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
			for (size_t i = 0; i < total_bytes; i += 3) {
				file.put(image[i + 2]); 
				file.put(image[i + 1]); 
				file.put(image[i]);     
			}

			file.close();
			std::cout << "图像保存成功：" << filename << std::endl;
		}
		else {
			std::cerr << "无法打开文件" << filename << std::endl;
		}
	}

	enum ImageDepth {
		DEPTH_UNKNOWN,
		DEPTH_8U,
		DEPTH_16U
	};

	cv::Mat createMatFromImage(const char* image, int w, int h) {
		if (image == nullptr) return cv::Mat();
		ImageDepth depth = DEPTH_UNKNOWN;

		// 找出最大值和最小值
		unsigned short maxVal = std::numeric_limits<unsigned short>::min();
		unsigned short minVal = std::numeric_limits<unsigned short>::max();

		long long total_bytes = static_cast<long long>(w) * h * 3;

		for (long long i = 0; i < total_bytes; ++i) {
			unsigned short val = static_cast<unsigned char>(image[i]);
			if (val > maxVal) maxVal = val;
			if (val < minVal) minVal = val;
		}

		// 根据值范围确定位深度 
		if (maxVal <= 255 && minVal >= 0) {
			test_p("======---==DEPTH_8U (Detected by range)");
			depth = DEPTH_8U;
		}
		else if (maxVal <= 65535 && minVal >= 0) {
			test_p("======---==DEPTH_16U (Detected by range)");
			depth = DEPTH_16U;
		}
		else {
			test_p("======---==无法确定图像数据类型");
			return cv::Mat();
		}

		try {
			if (depth == DEPTH_8U) {
				test_p("Creating CV_8UC3 Mat for 10X image: " + std::to_string(w) + "x" + std::to_string(h));
				return cv::Mat(h, w, CV_8UC3, const_cast<char*>(image)).clone(); 
			}
			else if (depth == DEPTH_16U) {
				test_p("Creating CV_16UC3 Mat for 10X image");
				return cv::Mat(h, w, CV_16UC3, const_cast<char*>(image)).clone();
			}
		} catch (const std::exception& e) {
			test_p("Exception in createMatFromImage: " + std::string(e.what()));
		}

		return cv::Mat();
	}

	bool isCharArrayContiguous(const char* arr, size_t size) 
	{
		return (arr != nullptr); 
	}

	//按照轮廓面积大小降序排列
	bool AickTensorrtStarter::compareContourAreas(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
		double area1 = std::abs(cv::contourArea(contour1));
		double area2 = std::abs(cv::contourArea(contour2));
		return area1 > area2;
	}

	int AickTensorrtStarter::CellDetector(int nSliceID, char* image, int w, int h, int detType, std::string savePath, std::vector<std::vector<float>>& CellData, std::string model_path, int class_num, float conf_thres, float iou_thres) 
{
    class_num = 2;  
    cv::Mat imgf = cv::imread(savePath + "/zoom.jpg");
    cv::Mat img;
    if (imgf.empty()) {
        test_p("Error: Cannot read zoom.jpg from " + savePath);
        return -1;
    }
    cv::flip(imgf, img, 0);  // 参数0表示垂直镜像，1表示水平镜像，-1表示水平和垂直镜像
    cv::imwrite(savePath + "/zoom.jpg", img);

	//DetType = 0 表示正常玻片
	//DetType = 1 表示原位玻片
    if (detType == 1) 
    {
        map<string, string> config10X;
        ReadConfig(config10X);

        float inSituType = 0.0f;
        try {
            if (!config10X["inSituType"].empty()) inSituType = std::stof(config10X["inSituType"]);
        } catch (...) { inSituType = 0.0f; }

        if (inSituType == 0) 
        {
            try {
                conf_thres = std::stof(config10X["confThres"]);
                iou_thres = std::stof(config10X["iouThres"]);
            } catch (...) {
                conf_thres = 0.3f; iou_thres = 0.45f; 
            }

            string str_McnPath = config10X["CellDet"];//"CellDet.engine"
            replace_all(str_McnPath, "\\", "/");

            if (CellDet != nullptr) { delete CellDet; CellDet = nullptr; }
            CellDet = new CTestModel();
            test_p("=============开始初始化原位模型 --- conf: " + to_string(conf_thres) + "===iou_thres: " + to_string(iou_thres) + "=====class_num: " + to_string(class_num) + "=====detType: " + to_string(detType));
            std::tuple<int, int, int> tup = std::make_tuple(1024, 1024, 3);
            CellDet->initModel(str_McnPath, class_num, conf_thres, iou_thres, tup);
			test_p("==============完成原位模型初始化: " + str_McnPath); 
			
            auto now = std::chrono::high_resolution_clock::now();
            long long tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
            string img_name = std::to_string(tmp) + "_.png";
			
			test_p("=============开始model原位检测");
            cv::Mat imgClone = img.isContinuous() ? img : img.clone();
            char* image_n = reinterpret_cast<char*>(imgClone.data);
            
			
            std::vector<det_box> CellResult = CellDet->getCellOutput(image_n, w, h);
			test_p("=============结束推理处理返回结果 " + to_string(CellResult.size()));

            std::sort(CellResult.begin(), CellResult.end(), [](const det_box &box1, const det_box &box2) {
                return (box1.w * box1.h) > (box2.w * box2.h);
            });

            string txtPath = savePath + "/zoom.txt";
            ofstream ofs(txtPath, ios::out | ios::app); 
            if (ofs.is_open()) {
                ofs << nSliceID << "\n";
                if (IParam.size() >= 6) {
                    for (const auto& vec : IParam) {
                        for (size_t i = 0; i < vec.size(); ++i) {
                            ofs << vec[i] << (i < vec.size() - 1 ? "," : "");
                        }
                        ofs << "\n";
                    }
                }
                for (size_t i = 0; i < CellResult.size(); i++) {
                    float rx = std::max(0.0f, CellResult[i].x1);
                    float ry = std::max(0.0f, CellResult[i].y1);
                    float rw = ((CellResult[i].w + rx) < (float)w) ? CellResult[i].w : ((float)w - rx);
                    float rh = ((CellResult[i].h + ry) < (float)h) ? CellResult[i].h : ((float)h - ry);

                    ofs << (i + 1) << "\n" << rx << "," << ry << "," << rw << "," << rh << "\n";
                    ofs << rx << "," << ry << "," << rx+rw << "," << ry << "," 
                        << rx+rw << "," << ry+rh << "," << rx << "," << ry+rh << "\n";

                    float stageX = rx * IParam[2][0] / 1000.0f + IParam[3][0];
                    float stageY = -ry * IParam[2][1] / 1000.0f + IParam[3][1] + IParam[5][nSliceID - 1] - IParam[5][0];
                    float stageX2 = (rx + rw) * IParam[2][0] / 1000.0f + IParam[3][0];
                    float stageY2 = -(ry + rh) * IParam[2][1] / 1000.0f + IParam[3][1] + IParam[5][nSliceID - 1] - IParam[5][0];

                    CellData.push_back({rx, ry, rw, rh, (float)CellResult[i].cls_idx, CellResult[i].score});
                    CellDataMe.push_back({ stageX, stageY, (stageX2 - stageX), (stageY - stageY2), (float)(i + 1) });
                }
                ofs.close();
            }
            isCultivation = true;
			test_p("=============结束model原位检测 共检测到: " + to_string(CellData.size()));
			test_p("=============开始delete原位模型");
            delete CellDet;
            CellDet = nullptr;
			test_p("=============完成delete原位模型");
            return 0;
        } 
        else 
        {
            test_p("============= 开始 image 原位检测");
			test_p("===============img: " + to_string(img.isContinuous()));
            cv::Mat image_G;
            cv::cvtColor(img, image_G, cv::COLOR_BGR2GRAY);

            cv::Mat varLogImage = imageSplitByKer(image_G, 15, "var");
            cv::Mat sumLogImage = imageSplitByKer(varLogImage + 1, 5, "max");

            cv::threshold(sumLogImage, sumLogImage, 1, 0, cv::THRESH_TOZERO);
            cv::threshold(sumLogImage, sumLogImage, 2, 0, cv::THRESH_TOZERO_INV);
            cv::threshold(sumLogImage, sumLogImage, 0, 1, cv::THRESH_BINARY);
			test_p("===============sumLogImage: " + to_string(sumLogImage.isContinuous()));

            cv::Mat eroKer = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
            cv::erode(sumLogImage, sumLogImage, eroKer, cv::Point(-1, -1), 6);
            sumLogImage.convertTo(sumLogImage, CV_8U);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(sumLogImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            cv::Mat SeMask = cv::Mat::zeros(sumLogImage.size(), CV_8U);
            float cntRateThres = 0.5f;
            if(!config10X["cntRate"].empty()) cntRateThres = std::stof(config10X["cntRate"]);

            for (const auto& cnt : contours) {
                cv::Mat tmpMask = cv::Mat::zeros(sumLogImage.size(), CV_8U);
                cv::drawContours(tmpMask, std::vector<std::vector<cv::Point>>{cnt}, 0, 1, -1);
                double cntSize = cv::sum(tmpMask)[0];
                if ((cntSize / (double)cnt.size()) < cntRateThres) continue;
                cv::drawContours(SeMask, std::vector<std::vector<cv::Point>>{cnt}, 0, 1, -1);
            }

            std::vector<std::vector<cv::Point>> contoursRes;
            cv::findContours(SeMask, contoursRes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
			test_p("=-====---===原位薄片尺寸SeMask： ---cols: " + to_string(SeMask.cols) + "----rows: " + to_string(SeMask.rows));
            
            std::sort(contoursRes.begin(), contoursRes.end(), [this](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
                return this->compareContourAreas(a, b);
            });

            string outPath = savePath + "/zoom.txt"; 
            ofstream ofs(outPath, ios::out | ios::ate); 
            if (ofs.is_open()) {
                ofs << nSliceID << "\n";
                for (const auto& vec : IParam) {
                    for (size_t i = 0; i < vec.size(); ++i) ofs << vec[i] << (i < vec.size() - 1 ? "," : "");
                    ofs << "\n";
                }

                int cls_num = 0;
                for (const auto& cnt : contoursRes) {
					cv::Rect rect = cv::boundingRect(cnt);
					// 保存类别索引
					ofs << to_string(cls_num + 1) << endl;

					// 处理“框框”字符串 (ssBoxs)
					std::stringstream ssBox;
					ssBox << rect.x << "," << rect.y << "," << rect.width << "," << rect.height;
					std::string ssBoxs = ssBox.str();
					
					test_p("=-=-=-=-=-=-=-=-原位群落坐标：" + ssBoxs); 
					ofs << ssBoxs << endl;

					// 处理“轮廓”字符串 (ssCnts)
					std::stringstream ssCnt;
					for (size_t i = 0; i < cnt.size(); ++i) {
						ssCnt << cnt[i].x << "," << cnt[i].y;
						if (i < cnt.size() - 1) {
							ssCnt << ", "; 
						}
					}
					std::string ssCnts = ssCnt.str();
					ofs << ssCnts << endl;

                    float stageX = rect.x * IParam[2][0] / 1000.0f + IParam[3][0];
                    float stageY = -rect.y * IParam[2][1] / 1000.0f + IParam[3][1] + IParam[5][nSliceID - 1] - IParam[5][0];
                    float stageX2 = (rect.x + rect.width) * IParam[2][0] / 1000.0f + IParam[3][0];
                    float stageY2 = -(rect.y + rect.height) * IParam[2][1] / 1000.0f + IParam[3][1] + IParam[5][nSliceID - 1] - IParam[5][0];
                    
					float stageW = stageX2 - stageX;  // 第0个为10x的放大率
					float stageH = stageY - stageY2;  // 第0个为10x的放大率
                    CellData.push_back({ (float)rect.x, (float)rect.y, (float)rect.width, (float)rect.height, (float)(cls_num + 1) });
                    CellDataMe.push_back({ stageX, stageY, (stageX2 - stageX), (stageY - stageY2), (float)(cls_num + 1) });
                    test_p("==========原位群落号: " + to_string(cls_num + 1) + "----x: " + to_string(stageX) + "----y: " + to_string(stageY) + "----w: " + to_string(stageW) + "----h: " + to_string(stageH));
					cls_num++;
                }
                ofs.close();
            }
            isCultivation = true;
            return 0;
        }
    } 
    return 0; 
}

	int AickTensorrtStarter::InintModelMcn(std::string model_path, int class_num, float conf_thres, float iou_thres) {
		if (McnDet != nullptr) {
			test_p("警告: 微核模型已存在，正在重新初始化...");
			delete McnDet;
			McnDet = nullptr;
		}

		McnDet = new CTestModel();
		test_p("=============开始初始化微核模型");

		std::map<std::string, std::string> config;
		ReadConfig(config);
		string str_McnPath = config["McnDet"]; 
		replace_all(str_McnPath, "\\", "/"); 

		std::tuple<int, int, int> tup = std::make_tuple(1024, 1024, 3);
		
		try {
			int status = McnDet->initModel(str_McnPath, class_num, conf_thres, iou_thres, tup);
			if (status != 0) {
				test_p("错误: 微核模型 initModel 失败，返回码: " + std::to_string(status));
				return -1;
			}
		} catch (const std::exception& e) {
			test_p("异常: 模型初始化过程中发生错误: " + std::string(e.what()));
			return -1;
		}

		test_p("==============完成微核模型初始化: " + str_McnPath);
		return 0;
	}

	int AickTensorrtStarter::DelMcnDet() {
		test_p("=============开始delete微核模型");
		if (McnDet != nullptr) {
			delete McnDet;
			McnDet = nullptr;
			test_p("=============完成delete微核模型");
		} else {
			test_p("提示: 微核模型本来就是空的，无需 delete");
		}
		return 0;
	}

	int AickTensorrtStarter::McnDetector(char* image, int w, int h, std::vector<std::vector<float>>& McnData) {
		cv::Mat img(h, w, CV_8UC3, image);
		auto now = std::chrono::high_resolution_clock::now();
		long long tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
		string img_name = std::to_string(tmp) + "_.png";

		map<string, string> config10X;
		ReadConfig(config10X);

		if (config10X["saveMcnPath"] != "0") {
			string base_path = config10X["saveMcnPath"];
			replace_all(base_path, "\\", "/"); 

			if (access(base_path.c_str(), F_OK) == -1) {
				if (mkdir(base_path.c_str(), 0777) == -1) {
					test_p("警告: 无法创建微核图保存目录: " + base_path);
				}
			}

			if (!base_path.empty() && base_path.back() != '/') {
				base_path += "/";
			}

			string img_save_path = base_path + img_name;
			test_p("------------微核图保存路径: " + img_save_path);
			
			// 执行保存
			if (!img.empty()) {
				cv::imwrite(img_save_path, img);
				test_p("------------微核图保存完成 !!!");
			}
		}

		test_p("=============开始微核检测");
		if (McnDet == nullptr) {
			test_p("错误: McnDet 模型未初始化，请先调用 InintModelMcn");
			return -1;
		}
		std::vector<det_box> McnResult = McnDet->getMcnOutput(image, w, h);
		test_p("=============微核检测完成");

		for (size_t i = 0; i < McnResult.size(); i++) {
			std::vector<float> bbox;

			float x1 = std::max(0.0f, McnResult[i].x1);
			float y1 = std::max(0.0f, McnResult[i].y1);
			
			float boxW = ((McnResult[i].w + x1) < (float)w) ? McnResult[i].w : ((float)w - x1);
			float boxH = ((McnResult[i].h + y1) < (float)h) ? McnResult[i].h : ((float)h - y1);

			bbox.push_back(x1);
			bbox.push_back(y1);
			bbox.push_back(boxW);
			bbox.push_back(boxH);
			bbox.push_back((float)McnResult[i].cls_idx);
			bbox.push_back(McnResult[i].score);

			McnData.push_back(bbox);
		}

		test_p("=============结束微核检测 共检测到: " + std::to_string(McnData.size()));
		return 0;
	}


	int AickTensorrtStarter::InintModelCellFish(std::string model_path, int class_num, float conf_thres, float iou_thres) {
		if (CellFishDet != nullptr) {
			test_p("提示: FISH细胞模型已初始化，正在释放旧模型以重新加载...");
			delete CellFishDet;
			CellFishDet = nullptr;
		}

		CellFishDet = new CTestModel();
		test_p("=============开始初始化FISH细胞模型");
		std::map<std::string, std::string> config;
		ReadConfig(config);
		string str_CellFishPath = config.count("CellFishDet") ? config["CellFishDet"] : model_path;

		replace_all(str_CellFishPath, "\\", "/");

		std::tuple<int, int, int> tup = std::make_tuple(1280, 1280, 3);
		
		try {
			int status = CellFishDet->initModel(str_CellFishPath, class_num, conf_thres, iou_thres, tup);
			if (status != 0) {
				test_p("错误: FISH细胞模型初始化失败，错误码: " + std::to_string(status));
				delete CellFishDet;
				CellFishDet = nullptr;
				return -1;
			}
		} catch (const std::exception& e) {
			test_p("异常: 加载 FISH 模型时捕获到异常: " + std::string(e.what()));
			return -1;
		}

		test_p("==============完成FISH细胞模型初始化: " + str_CellFishPath);
		return 0;
	}

	int AickTensorrtStarter::DelCellFishDet() {
		test_p("=============开始 delete FISH 细胞模型");
		if (CellFishDet != nullptr) {
			delete CellFishDet;
			CellFishDet = nullptr;
			test_p("=============完成 delete FISH 细胞模型");
		} else {
			test_p("提示: FISH 细胞模型为空，无需释放");
		}
		return 0;
	}

	int AickTensorrtStarter::CellFishDetector(char* image, int w, int h, std::vector<std::vector<float>> &CellFish_box) {
		cv::Mat img(h, w, CV_8UC3, image);

		auto now = std::chrono::high_resolution_clock::now();
		long long tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
		string img_name = std::to_string(tmp) + "_.png";

		map<string, string> config10X;
		ReadConfig(config10X);

		if (config10X["saveCellFishPath"] != "0") {
			string base_path = config10X["saveCellFishPath"];
			replace_all(base_path, "\\", "/"); 

			if (access(base_path.c_str(), F_OK) == -1) {
				if (mkdir(base_path.c_str(), 0777) == -1) {
					test_p("警告: 无法创建保存目录: " + base_path);
				}
			}

			if (!base_path.empty() && base_path.back() != '/') {
				base_path += "/";
			}

			string img_save_path = base_path + img_name;
			test_p("------------细胞图保存路径: " + img_save_path);
			
			if (!img.empty()) {
				cv::imwrite(img_save_path, img);
				test_p("------------细胞图保存完成 !!!");
			}
		}

		test_p("=============开始FISH细胞检测");
		if (CellFishDet == nullptr) {
			test_p("错误: CellFishDet 模型未初始化！");
			return -1;
		}
		
		// 获取实例分割输出
		std::vector<instance_seg> CellFishResult = CellFishDet->getCellFishOutput(image, w, h);
		test_p("=============FISH细胞检测完成");

		for (size_t i = 0; i < CellFishResult.size(); i++) {
			std::vector<float> bbox;

			// 确保框不超出图像
			float rx = std::max(0.0f, (float)CellFishResult[i].rect.x);
			float ry = std::max(0.0f, (float)CellFishResult[i].rect.y);
			
			float rw = ((CellFishResult[i].rect.width + rx) < (float)w) ? (float)CellFishResult[i].rect.width : ((float)w - rx);
			float rh = ((CellFishResult[i].rect.height + ry) < (float)h) ? (float)CellFishResult[i].rect.height : ((float)h - ry);

			bbox.push_back(rx);
			bbox.push_back(ry);
			bbox.push_back(rw);
			bbox.push_back(rh);
			bbox.push_back((float)CellFishResult[i].cls_idx);
			bbox.push_back(CellFishResult[i].conf);

			std::cout << rx << "  " << ry << "  " << rw << "  " << rh << "  " << CellFishResult[i].cls_idx << std::endl;
			
			CellFish_box.push_back(bbox);
		}

		test_p("=============结束FISH细胞检测 共检测到: " + std::to_string(CellFishResult.size()));
		return 0;
	}

	int AickTensorrtStarter::Clarity_evaluation(std::vector<char*> image_stream, int w, int h, int& index, std::vector<double>& offset) {
		std::tuple<int, int> size = std::make_tuple(960, 960);
		cv::Rect boundRect;
		std::vector<std::vector<cv::Point>> contour_tmp;
		std::vector<cv::Point> contour;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Mat> unet_result;
		std::vector<cv::Vec4i> hierarchy;
		int status = 0;
		int Rect_area = 0;
		cv::Mat concat = cv::Mat::zeros(h, w, CV_8UC1);
		
		std::map<std::string, std::string> config;
		ReadConfig(config);

		for (auto iter = config.begin(); iter != config.end(); ++iter) {
			string strKey = iter->first;
			string strValue = iter->second;
		}

		test_p("开始清晰度评价流程");

		if (stoi(config["showInf"])) {
			UseCondition();
		}

		cv::Mat sharp_mask;
        if (leap_over) {
            int idx = image_stream.size() / 2;
            cv::Mat t_img = cv::Mat(h, w, CV_8UC3, image_stream[idx]);
            unet_result = c_score->clarity_getOutput(t_img);
            status = scale_coords_v5(t_img, size, unet_result);

            for (auto it = unet_result.begin(); it != unet_result.end(); ++it) {
                cv::add(concat, *it, concat);
            }

            cv::threshold(concat, concat, 1, 1, cv::THRESH_BINARY);
            cv::findContours(concat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            
            if (!contours.empty()) {
                contour = Vec2dToVec(contours);
                contour_tmp.clear(); 
                contour_tmp.push_back(contour);
                
                boundRect = cv::boundingRect(contour_tmp[0]);
                Rect_area = boundRect.width * boundRect.height;
                test_p("ROI 区域面积: " + std::to_string(Rect_area));
            }

            cv::Mat structe_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(concat, sharp_mask, structe_element);
            cv::imwrite("/sharp_mask.png", sharp_mask * 255);
        }

        else {
            sharp_mask = cv::Mat::ones(h, w, CV_8UC1);
        }
        
        test_p(get_date_time() + " --- sharp_mask 处理完成");

		cv::Mat sharp_img, past_img;
        float tolerance = 0.0f, max_cast = 0.0f;
        std::vector<float> cast_list;
        int record = 0;
        bool Channel_N = true;

		for (auto it = image_stream.begin(); it != image_stream.end(); ++it) {

            test_p(get_date_time() + " --- image_stream iterator Artifact: " + std::to_string(record));
            cv::Mat scrImg(h, w, CV_8UC3, *it);
            cv::Mat grayImg;
            cv::cvtColor(scrImg, grayImg, cv::COLOR_BGR2GRAY);

            // 伪影/干扰排除逻辑 
            if (std::stoi(config["Artifact"]) != 0) {
                if (Channel_N == true) {
                    if ((it + 1) != image_stream.end()) {
                        cv::Mat next_img(h, w, CV_8UC3, *(it + 1));
                        cv::Mat next_gray;
                        cv::cvtColor(next_img, next_gray, cv::COLOR_BGR2GRAY);
                        tolerance = (float)cv::mean(grayImg - next_gray)[0];
                        past_img = next_gray.clone(); 
                    }
                    Channel_N = false;
                }
                else {
                    tolerance = (float)cv::mean(grayImg - past_img)[0];
                    past_img = grayImg.clone();
                }
            }

            float gaussian = 0.0f;
            cv::Mat blurredImage;

            if (leap_over) {
                
                if (std::stoi(config["operationModel"]) == 0 && Rect_area > 1000) {
                    cv::Mat img_roi = grayImg(boundRect); 
                    
                    cv::GaussianBlur(img_roi, blurredImage, cv::Size(15, 15), 0);
                    gaussian = (float)cv::mean(blurredImage - img_roi)[0];
                }
                else {
                    cv::GaussianBlur(grayImg, blurredImage, cv::Size(15, 15), 0);
                    gaussian = (float)cv::mean(blurredImage - grayImg)[0];
                }
            }
            else {
                cv::Mat lightImg;
                cv::resize(grayImg, lightImg, cv::Size(), 0.5, 0.5);

                cv::GaussianBlur(lightImg, blurredImage, cv::Size(15, 15), 0);
                gaussian = (float)cv::mean(blurredImage - lightImg)[0];
            }

            if (gaussian > max_cast && tolerance < 1.2f) {
                sharp_img = grayImg.clone();
                max_cast = gaussian;
                index = record;
            }

            record++;
        }

		test_p(get_date_time() + "Clarity index Get");

		if (leap_over) {
			leap_over = false;
			return 0;
		}
		leap_over = true;

		if (std::stoi(config["operationModel"]) != 0) {
            cv::Mat full_ones = cv::Mat::ones(h, w, CV_8UC1);
            valid_mask = matToString(full_ones); 
            
            offset.push_back(-200.0);
            offset.push_back(-200.0);
            return 0;
        }

		cv::Mat img;
		cvtColor(sharp_img, img, cv::COLOR_GRAY2BGR);

		unet_result = c_score->clarity_getOutput(img);
		status = scale_coords_v5(img, size, unet_result);

		cv::findContours(concat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        
        if (!contours.empty()) {
            contour = Vec2dToVec(contours); // 使用函数开头定义的 contour
            boundRect = cv::boundingRect(contour); // 使用函数开头定义的 boundRect

            if (boundRect.width * boundRect.height < 100) {
                offset.push_back(0.0);
                offset.push_back(0.0);
            }
            else {
                float off_x = -(float)(w / 2 - boundRect.x - boundRect.width / 2);
                float off_y = -(float)(h / 2 - boundRect.y - boundRect.height / 2);
                offset.push_back(off_x);
                offset.push_back(off_y);
            }
        }

        test_p(get_date_time() + " --- 清晰度评价计算结束，最优帧索引: " + std::to_string(index));

        if (std::stoi(config["showInf"])) {
            std::cout << "--- 偏移计算详情 ---" << std::endl;
            std::cout << "目标矩形: [x:" << boundRect.x << " y:" << boundRect.y 
                      << " w:" << boundRect.width << " h:" << boundRect.height << "]" << std::endl;
            std::cout << "图像中心: [" << w / 2 << ", " << h / 2 << "]" << std::endl;
            std::cout << "目标中心: [" << boundRect.x + boundRect.width / 2 << ", " 
                      << boundRect.y + boundRect.height / 2 << "]" << std::endl;
            std::cout << "最终偏移: X=" << offset[0] << " Y=" << offset[1] << std::endl;
            
            UseCondition(); 
        }

        return 0;
	}

	int AickTensorrtStarter::Center_alignment(char* image_stream, int w, int h, std::vector<double>& offset) {
		test_p(get_date_time() + " --- Center_alignment 开始执行");
		
		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);
		float eps = std::stof(config10X["eps"]);
		int minPts = std::stoi(config10X["minPts"]);

		cv::Rect boundRect;
		std::vector<std::vector<cv::Point>> contour_tmp, all_contour;
		std::vector<std::vector<std::vector<cv::Point>>> class_contour;
		std::vector<cv::Point> contour;
		std::vector<cv::Mat> unet_result;
		std::vector<cv::Vec4i> hierarchy;
		std::vector<PointZX> chromo_points, CenterPoint;
		std::vector<std::vector<PointZX>> cls_point;
		std::vector<int> area_list, num_list;
		int status, num = 0;
		cv::Mat concat = cv::Mat::zeros(h, w, CV_8UC1);
		std::tuple<int, int> size = std::make_tuple(960, 960);

		try {
			test_p(get_date_time() + " --- Center_alignment: 开始模型推理");
			cv::Mat ori_img(h, w, CV_8UC3, image_stream);
			
			unet_result = c_score->clarity_getOutput(ori_img);
			status = scale_coords_v5(ori_img, size, unet_result);

			for (auto it = unet_result.begin(); it != unet_result.end(); ++it) {
				cv::add(concat, *it, concat);
				
				std::vector<std::vector<cv::Point>> temp_contours;
				std::vector<cv::Vec4i> temp_hierarchy;
				cv::findContours(*it, temp_contours, temp_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
				
				for (auto& temp_contour : temp_contours) {
					boundRect = cv::boundingRect(temp_contour);
					PointZX tp;
					tp.x = (double)boundRect.x;
					tp.y = (double)boundRect.y;
					tp.z = num;           // 存入索引，方便后面根据聚类结果找回轮廓
					tp.clusterID = -1;   
					
					all_contour.push_back(temp_contour);
					num++;
					chromo_points.push_back(tp);
				}
			}

			if (chromo_points.empty()) {
				test_p("警告: 未检测到任何目标点，对齐终止");
				return 1;
			}

			// 执行 DBSCAN 聚类
			test_p(get_date_time() + " --- Center_alignment: 开始 DBSCAN 聚类, 点数: " + std::to_string(chromo_points.size()));
			DBSCAN ds(minPts, eps, chromo_points);
			ds.run();
			CenterPoint = calculateCenterPoint(ds.m_points, cls_point);

			size_t mark = 0, max_index = 0;
			double max_area = 0.0;
			
			test_p(get_date_time() + " --- Center_alignment: 正在计算各簇面积");
			for (auto& n_point : cls_point) {
				double temp_area = 0;
				std::vector<std::vector<cv::Point>> temp_c;
				
				for (auto& s_point : n_point) {
					temp_area += cv::contourArea(all_contour[s_point.z]);
					temp_c.push_back(all_contour[s_point.z]);
				}
				
				if (max_area < temp_area) {
					max_area = temp_area;
					max_index = mark;
				}
				mark++;
				area_list.push_back((int)temp_area);
				num_list.push_back(n_point.size());
				class_contour.push_back(temp_c);
			}

			if (!class_contour.empty() && max_index < class_contour.size()) {
				contour = Vec2dToVec(class_contour[max_index]);
				boundRect = cv::boundingRect(contour);

				if (boundRect.width * boundRect.height < 100) {
					offset.push_back(0.0);
					offset.push_back(0.0);
				} else {
					offset.push_back(-(double)(w / 2 - boundRect.x - boundRect.width / 2));
					offset.push_back(-(double)(h / 2 - boundRect.y - boundRect.height / 2));
				}
			}

			test_p(get_date_time() + " --- 对齐完成，偏移量: " + std::to_string(offset[0]) + " | " + std::to_string(offset[1]));
			return 0;
		}
		catch (const std::exception& e) {
			test_p("错误: Center_alignment 发生异常: " + std::string(e.what()));
			return 1;
		}
	}

	void CoordinatesPixelToPhysical(int nWidth, int nHeight, std::vector<float> m_dFocusDiff10_100, std::vector<float> score_boxs, std::vector<float> stageInfo, std::vector<float>& simpleScanPoint, std::string Solution) {
		float rect_x = score_boxs[0];
		float rect_y = score_boxs[1];
		float rect_width = score_boxs[2];
		float rect_height = score_boxs[3];

		//提取当前载物台物理状态 (X, Y, Z, 以及单视野对应的物理尺寸)
		float dStagePosX = stageInfo[0];   // 当前中心 X (mm)
		float dStagePosY = stageInfo[1];   // 当前中心 Y (mm)
		float dStagePosZ = stageInfo[2];   // 当前 Z 焦面 (mm)
		float dStageWidth = stageInfo[3];  // 10X 单张图对应的物理宽度 (um)
		float dStageHeight = stageInfo[4]; // 10X 单张图对应的物理高度 (um)

		test_p("坐标转换输入: PosX=" + std::to_string(dStagePosX) + 
			", PosY=" + std::to_string(dStagePosY) + 
			", Width(um)=" + std::to_string(dStageWidth));

		int pPosX = (int)(rect_x + rect_width / 2.0f);
		int pPosY = nHeight - (int)(rect_y + rect_height / 2.0f); // 翻转Y轴以对齐显微镜坐标系

		int dpX = pPosX - nWidth / 2;
		int dpY = pPosY - nHeight / 2;
		float dx = (dStageWidth / (float)nWidth) * (float)dpX / 1000.0f;
		float dy = (dStageHeight / (float)nHeight) * (float)dpY / 1000.0f;
		float sPosX = dStagePosX + dx;
		float sPosY = dStagePosY - dy; // 显微镜 Y 轴物理移动通常与像素相反

		if (Solution == "zx_scenario") {
			if (IParam.size() >= 2 && IParam[0].size() >= 1 && IParam[1].size() >= 1) {
				float zx_x = stageInfo[0] + (rect_x + rect_width / 2.0f) * IParam[1][0] + IParam[0][0] - (stageInfo[2] / 2.0f * IParam[1][0]);
				float zx_y = (stageInfo[1] - stageInfo[3] * IParam[1][0]) + (stageInfo[3] - rect_y - rect_height / 2.0f) * IParam[1][0] + IParam[0][0] - (stageInfo[3] / 2.0f * IParam[1][0]);
				
				simpleScanPoint.push_back(zx_x);
				simpleScanPoint.push_back(zx_y);
			} else {
				test_p("错误: IParam 未初始化或尺寸不足，无法执行 zx_scenario");
			}
		}
		else {
			//常规 10X -> 100X 转换：叠加 10/100 倍率间的机械差 (FocusDiff)
			simpleScanPoint.push_back(sPosX + m_dFocusDiff10_100[0]); // X 偏移
			simpleScanPoint.push_back(sPosY + m_dFocusDiff10_100[1]); // Y 偏移
			simpleScanPoint.push_back(dStagePosZ + m_dFocusDiff10_100[2]); // Z 偏移 (10X到100X的焦距差)
		}

		test_p("转换结果 -> 物理坐标: " + std::to_string(simpleScanPoint[0]) + ", " + std::to_string(simpleScanPoint[1]));
	}

	//int AickTensorrtStarter::test_10X(char* image, int w, int h) {
	//	int Channel = getChannels(image, w, h);
	//	cv::Mat img;

	//	test_p("================待检测10X小图 W: " + std::to_string(w) + "\t H:" + std::to_string(h) + "\t C:" + std::to_string(Channel));

	//	img = cv::Mat(h, w, CV_8UC3, image);

	//	test_p("================开始10X小图检测 ---------------");
	//	map<string, string> config10X;
	//	ReadConfig(config10X);

	//	test_p("=-=-=-=-=-=-开始预测是否为杂质 !!!");
	//	std::vector<float> impurities_result = impurities_score->getOutput(img);
	//	test_p("=-=-=-=-=-=-预测为杂质可能性为: " + to_string(impurities_result[1]) + "-----判断为杂质的阈值: " + config10X["impurityThreshold"]);
	//	std::vector<float>::iterator biggest = std::max_element(std::begin(impurities_result), std::end(impurities_result));
	//	string sub_class = to_string(std::distance(std::begin(impurities_result), biggest));
	//	float score = 0.0;
	//	std::string out_class = "";
	//	std::vector<float> reg_result = { 0, 0, 0 };
	//	// 杂质分类
	//	/*std::cout << "biggest : " << *biggest << "sub_class : " << sub_class << std::endl;*/

	//	//if (sub_class != "1") {|
	//	if (impurities_result[1] < stof(config10X["impurityThreshold"])) {
	//		test_p("------------开始预测分裂相评分!!!");
	//		reg_result = reg_score->getRegressionOutput(img, 1);
	//		test_p("------------分裂相评分结束---长度分数为: " + to_string(reg_result[0]) + "---分散度分数为: " + to_string(reg_result[1]) + "---数量分数为: " + to_string(reg_result[2]));

	//		if (reg_result[0] < stof(config10X["lenBottom"])) {
	//			sub_class = "1";
	//			test_p("-----------长度分数过低, 当作杂质处理");
	//		}
	//		else if (reg_result[1] < stof(config10X["dispBottom"])) {
	//			sub_class = "1";
	//			test_p("----------- 分散分数过低, 当作杂质处理");
	//		}
	//		else if (reg_result[2] < stof(config10X["numBottom"])) {
	//			sub_class = "1";
	//			test_p("----------- 数量分数过低, 当作杂质处理ttom");
	//		}
	//		else {
	//			test_p("------------开始为分裂相计算综合得分-----------");
	//			score = resultDataProcessing_new(impurities_result, reg_result, w, h, config10X);
	//			test_p("------------结束为分裂相计算综合得分-----------综合得分为: " + to_string(score));
	//		}

	//	}
	//	std::cout << " impurities_result: " << impurities_result[1] << " score : " << formatDobleValue(score * 100, 2) << std::endl;
	//	test_p("=-=-=-=-=-=-=-=杂质判定结果: " + sub_class + "------综合评分为: " + formatDobleValue(score * 100, 2));
	//

	//}
		// FISH染色体模型初始化
	int AickTensorrtStarter::InintModelChromosomeFish(std::string model_path, int class_num, float conf_thres, float iou_thres) {
		ChromosomeFishDet = new CTestModel();
		test_p("============= 开始初始化FISH染色体模型");

		std::map<std::string, std::string> config;
		ReadConfig(config);
		string str_ChromosomeFishPath = config["ChromosomeFishDet"];//"McnDet.onnx"

		std::tuple<int, int, int> tup = std::tuple<int, int, int>(1280, 1280, 3);
		ChromosomeFishDet->initModel(str_ChromosomeFishPath, class_num, conf_thres, iou_thres, tup);
		test_p("==============完成FISH染色体模型初始化: " + str_ChromosomeFishPath);
		return 0;
	}

	int AickTensorrtStarter::DelChromosomeFishDet() {
		test_p("=============开始deleteFISH染色体模型");
		delete ChromosomeFishDet;
		ChromosomeFishDet = nullptr;
		test_p("=============完成deleteFISH染色体模型");
		return 0;
	}

	int AickTensorrtStarter::AickTensorrt_fish(char* image, int w, int h, std::vector<float> stageInfo) {
		if (IParam.empty()) {
			test_p("错误: IParam 未初始化，无法获取倍率切换参数");
			return -1;
		}
		std::vector<float> m_dFocusDiff10_100 = IParam[0]; 
		num10XImage++;

		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);

		////是否保存Fish图
		//if (config10X["saveCellFishPath"] != "0") {
		//	test_p("------------config10X[saveCellFishPath]: " + config10X["saveCellFishPath"]);
		//	//cv::rectangle(temp_save, cv::Rect(vec_x, vec_y, vec_w, vec_h), cv::Scalar(0, 0, 255), 1, 1, 0);
		//	//cv::imwrite("temp_test.png", temp_save);
		//	// test_p(to_string(tmp));
		//	//string img_save_path = "E:/10XDetect100X/" + img_name;
		//	if (access(config10X["saveCellFishPath"].c_str(), 0) == -1) {
		//		mkdir(config10X["saveCellFishPath"].c_str());
		//	}
		//	string img_save_path = config10X["saveCellFishPath"] + img_name;
		//	test_p("------------细胞图保存路径: " + img_save_path);
		//	cv::imwrite(img_save_path, img);
		//	test_p("------------细胞图保存完成 !!!");
		//}


		test_p("============= 开始 FISH 染色体检测");
		if (ChromosomeFishDet == nullptr) {
			test_p("错误: ChromosomeFishDet 模型未加载！");
			return -1;
		}
		std::vector<instance_seg> ChromosomeFishResult = ChromosomeFishDet->getCellFishOutput(image, w, h);
		test_p("============= FISH 染色体检测完成");

		for (size_t i = 0; i < ChromosomeFishResult.size(); i++) {
			std::vector<float> bbox;

			float rx = std::max(0.0f, (float)ChromosomeFishResult[i].rect.x);
			float ry = std::max(0.0f, (float)ChromosomeFishResult[i].rect.y);
			float rw = ((ChromosomeFishResult[i].rect.width + rx) < (float)w) ? (float)ChromosomeFishResult[i].rect.width : ((float)w - rx);
			float rh = ((ChromosomeFishResult[i].rect.height + ry) < (float)h) ? (float)ChromosomeFishResult[i].rect.height : ((float)h - ry);

			bbox.push_back(rx);
			bbox.push_back(ry);
			bbox.push_back(rw);
			bbox.push_back(rh);

			std::vector<float> simpleScanPoint; 
			CoordinatesPixelToPhysical(w, h, m_dFocusDiff10_100, bbox, stageInfo, simpleScanPoint, config10X["Solution"]);

			simpleScanPoint.push_back(-1.0f);
			simpleScanPoint.push_back(ChromosomeFishResult[i].conf);

			auto now_curr = std::chrono::high_resolution_clock::now();
			long long tmp_curr = std::chrono::duration_cast<std::chrono::nanoseconds>(now_curr.time_since_epoch()).count();
			
			std::string img_name = std::to_string(tmp_curr) + "_point_" + 
								formatDobleValue(simpleScanPoint[0], 5) + "_" + 
								formatDobleValue(simpleScanPoint[1], 5) + "_" + 
								formatDobleValue(simpleScanPoint[2], 5) + "_num10Ximage_" + 
								formatDobleValue((double)num10XImage, 2) + "_.png";
			
			if (config10X["EquipmentManufacturer"] == "sunny") {
				score_boxs.push_back(simpleScanPoint);
			}
			else {
				std::vector<float> bboxs_c;
				bboxs_c.push_back(bbox[0] - bbox[2] / 2.0f); // 相对中心X
				bboxs_c.push_back(bbox[1] - bbox[3] / 2.0f); // 相对中心Y
				bboxs_c.push_back(0.0f);
				bboxs_c.push_back(-1.0f);
				bboxs_c.push_back(ChromosomeFishResult[i].conf);
				// 合并当前载物台信息
				bboxs_c.insert(bboxs_c.end(), stageInfo.begin(), stageInfo.end());
				score_boxs.push_back(bboxs_c);
			}
			
			img_names.push_back(img_name);
		}

		test_p("============= 结束 FISH 染色体检测 共检测到: " + std::to_string(ChromosomeFishResult.size()));
		return 0;
	}

	int AickTensorrtStarter::AickTensorrt(char* image, int w, int h, vector<float> stageInfo) {
		if (IParam.empty()) {
			test_p("错误: IParam 未初始化，无法获取倍率校准数据");
			return -1;
		}
		std::vector<float> m_dFocusDiff10_100 = IParam[0]; 
		num10XImage++;
		size_t data_size = w * h * 3; 
		int Channel = getChannels(image, w, h, data_size);

		test_p("================待检测10X大图 W: " + std::to_string(w) + "\t H:" + std::to_string(h) + "\t C:" + std::to_string(Channel));

		cv::Mat img = cv::Mat(h, w, CV_8UC3, image);

		test_p("================开始10X大图检测分裂相---------------");

		if (test_model == nullptr) {
			test_p("错误: test_model (分裂相模型) 未初始化！");
			return -1;
		}
		std::vector<std::vector<det_box>> result = test_model->getOutput(image, w, h);

		std::tuple<int, int> size = std::make_tuple(1600, 1600);
		scale_coords_v4(img, size, result); 

		test_p("================ 结束 10X 大图检测 ------ 检测到分裂相数量: " + std::to_string(result[0].size()));

		cv::Mat temp_save = img.clone();

		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);

		if (config10X["ModelOverload"] == "True") {
            InintModel(); 
            config10X["ModelOverload"] = "False";
            ModifyConfig(config10X, "ModelOverload", "False");
            WriteConfig(config10X); 
        }

        int count = 0;

		// 避免什么都不拍，放一个杂质进去
		if (img_names.empty() && std::stof(config10X["reservedImpuritie"]) == 1.0f) {
            float score = 0.0f;
            std::vector<float> bboxs;
            bboxs.push_back((float)w / 2.0f); 
            bboxs.push_back((float)h / 2.0f); 
            bboxs.push_back(20.0f);           
            bboxs.push_back(20.0f);           

            std::vector<float> simpleScanPoint_t;  // 返回结果
            CoordinatesPixelToPhysical(w, h, m_dFocusDiff10_100, bboxs, stageInfo, simpleScanPoint_t, config10X["Solution"]);
            auto now = std::chrono::high_resolution_clock::now();
            long long tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
            test_p("======== 首次添加一个杂质");
            string img_name = to_string(tmp) + "_point_" + formatDobleValue(simpleScanPoint_t[0], 5) + "_" + formatDobleValue(simpleScanPoint_t[1], 5) + "_" + formatDobleValue(simpleScanPoint_t[2], 5) + "_num10Ximage_" + formatDobleValue(num10XImage, 2) + "_.png";

            if (config10X["EquipmentManufacturer"] == "sunny") {
                simpleScanPoint_t.push_back(-1.0f); 
                simpleScanPoint_t.push_back(score); 
                score_boxs.push_back(simpleScanPoint_t);
            }
            else {
                std::vector<float> bboxs_c;
                bboxs_c.push_back(bboxs[0] - bboxs[2] / 2.0f); 
                bboxs_c.push_back(bboxs[1] - bboxs[3] / 2.0f); 
                bboxs_c.push_back(0.0f);                       
                bboxs_c.push_back(-1.0f);                      
                bboxs_c.push_back(score);                      
                bboxs_c.insert(bboxs_c.end(), stageInfo.begin(), stageInfo.end());
            }

            img_names.push_back(img_name);
            count++;
        }

		// Debug counters for filtering analysis
		int dbg_total = 0;
		int dbg_filtered_impurity = 0;
		int dbg_filtered_len = 0;
		int dbg_filtered_disp = 0;
		int dbg_filtered_num = 0;
		int dbg_filtered_score = 0;
		int dbg_passed = 0;

		for (auto it = result.begin(); it != result.end(); ++it)
		{
			for (auto itit = it->begin(); itit != it->end(); ++itit)
			{
				dbg_total++;
				// std::cout << itit->x1 << "\t" << itit->y1 << "\t" << itit->w << "\t" << itit->h << "\t" << itit->score << std::endl;
				int vec_x = (itit->x1 < 0) ? 0 : (int)itit->x1;
				int vec_y = (itit->y1 < 0) ? 0 : (int)itit->y1;
				int vec_w = ((itit->w + vec_x) < w) ? (int)itit->w : (w - vec_x);
				int vec_h = ((itit->h + vec_y) < h) ? (int)itit->h : (h - vec_y);

				std::cout << vec_x << "\t" << vec_y << "\t" << vec_w << "\t" << vec_h << std::endl;
				cv::Mat img_temp = img(cv::Rect(vec_x, vec_y, vec_w, vec_h));

				//杂质预测
				test_p("=-=-=-=-=-=-开始预测是否为杂质 !!!");
				std::vector<float> impurities_result = impurities_score->getOutput(img_temp);
				cout << "impurities_result info: size is: " << impurities_result.size() << " " << impurities_result[0] << " " << impurities_result[1] << " " << impurities_result[2] << endl;
				test_p("=-=-=-=-=-=-预测为杂质可能性为: " + to_string(impurities_result[1]) + "-----判断为杂质的阈值: " + config10X["impurityThreshold"]);
				auto biggest_it = std::max_element(impurities_result.begin(), impurities_result.end());
				int sub_class_idx = std::distance(impurities_result.begin(), biggest_it);
				std::string sub_class = std::to_string(sub_class_idx);
				float score = 0.0f;
				std::vector<float> reg_result = { 0, 0, 0 };

				std::cout << "biggest : " << *biggest_it << "sub_class : " << sub_class << std::endl;

				cout << "to_string(impurities_result[1]) " << to_string(impurities_result[1]) << endl;
				if (impurities_result[1] < std::stof(config10X["impurityThreshold"])) {
					test_p("------------开始预测分裂相评分!!!");
					
					// 执行回归评分模型
					reg_result = reg_score->getRegressionOutput(img_temp, 0);
					for (auto& v : reg_result) {
						if (v < 0.0f) v = 0.0f;
						if (v > 1.0f) v = 1.0f;
					}
					test_p("------------分裂相评分结束---长度分数为: " + to_string(reg_result[0]) + "---分散度分数为: " + to_string(reg_result[1]) + "---数量分数为: " + to_string(reg_result[2]));
					cout << "------------分裂相评分结束---长度分数为: " << to_string(reg_result[0]) << "---分散度分数为: " << to_string(reg_result[1]) << "---数量分数为: "<< to_string(reg_result[2]) << endl;

					// 评分门槛过滤：长度、分散度、数量必须达标
					if (reg_result[0] < std::stof(config10X["lenBottom"])) {
						dbg_filtered_len++;
						sub_class = "1";
						test_p("-----------长度分数过低, 当作杂质处理");
					} 
					else if (reg_result[1] < std::stof(config10X["dispBottom"])) {
						dbg_filtered_disp++;
						sub_class = "1";
						test_p("----------- 分散分数过低, 当作杂质处理");
					} 
					else if (reg_result[2] < std::stof(config10X["numBottom"])) {
						dbg_filtered_num++;
						sub_class = "1";
						test_p("----------- 数量分数过低, 当作杂质处理");
					} 
					else {
						test_p("------------开始为分裂相计算综合得分-----------");
						score = resultDataProcessing_new(impurities_result, reg_result, vec_w, vec_h, config10X);
						test_p("------------结束为分裂相计算综合得分-----------综合得分为: " + to_string(score));
					}
				} else {
					dbg_filtered_impurity++;
				}

				test_p("=-=-=-=-=-=-=-=杂质判定结果: " + sub_class + "------综合评分为: " + formatDobleValue(score * 100, 2));
				test_p("\t" + to_string(vec_x) + "\t" + to_string(vec_y) + "\t" + to_string(vec_w) + "\t" + to_string(vec_h) + "\t" + to_string(score));

				std::vector<float> bboxs = {(float)vec_x, (float)vec_y, (float)vec_w, (float)vec_h};
				std::vector<float> simpleScanPoint;
				CoordinatesPixelToPhysical(w, h, m_dFocusDiff10_100, bboxs, stageInfo, simpleScanPoint, config10X["Solution"]);

				auto now = std::chrono::high_resolution_clock::now();
				long long tmp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
				
				cout << "score is : " << score << endl;

				std::string img_name = formatDobleValue(score * 100, 2) + "_" + std::to_string(tmp_time) + 
									"_point_" + formatDobleValue(simpleScanPoint[0], 5) + "_" + 
									formatDobleValue(simpleScanPoint[1], 5) + "_" + 
									formatDobleValue(simpleScanPoint[2], 5) + 
									"_num10Ximage_" + formatDobleValue((double)num10XImage, 2) + 
									"_subClass_" + formatDobleValue(impurities_result[1] * 100, 2) + 
									"_score_" + formatDobleValue(reg_result[0] * 100, 2) + "_" + 
									formatDobleValue(reg_result[1] * 100, 2) + "_" + 
									formatDobleValue(reg_result[2] * 100, 2) + "_" + 
									formatDobleValue(score * 100, 2) + "_.png";

				// 分数过滤：总分高于 scoreFilter 则记录
				if (score > std::stof(config10X["scoreFilter"])) {
					if (config10X["EquipmentManufacturer"] == "sunny") {
						simpleScanPoint.push_back(-1); // 类别占位
						simpleScanPoint.push_back(score);
						score_boxs.push_back(simpleScanPoint);
					} else {
						std::vector<float> bboxs_c;
						bboxs_c.push_back(bboxs[0] - bboxs[2] / 2.0f);
						bboxs_c.push_back(bboxs[1] - bboxs[3] / 2.0f);
						bboxs_c.push_back(0.0f);
						bboxs_c.push_back(-1.0f);
						bboxs_c.push_back(score);
						bboxs_c.insert(bboxs_c.end(), stageInfo.begin(), stageInfo.end());
						score_boxs.push_back(bboxs_c);
					}
					img_names.push_back(img_name);
					count++;
					dbg_passed++;
				} else {
					dbg_filtered_score++;
				}

				if (std::stoi(config10X["isSaveImg"])) {
					cout << "=========================----" << vec_x << "--" << vec_y << "--" << vec_w << "--" << vec_h << endl;
					std::string save_dir = config10X["saveImgPath"];
					if (save_dir.empty()) {
						save_dir = "./";
					}
					if (save_dir.back() != '/' && save_dir.back() != '\\') {
						save_dir.push_back('/');
					}
					if (access(save_dir.c_str(), 0) == -1) {
						#ifdef _WIN32
							mkdir(save_dir.c_str());
						#else
							mkdir(save_dir.c_str(), 0777);
						#endif
					}
					string img_save_path = save_dir + img_name;
					test_p("------------分裂相保存路径: " + img_save_path);
					cv::imwrite(img_save_path, img_temp);
					test_p("------------分裂相保存完成 !!!");
				}
				test_p("=-=-=-=-=-=-=-分裂相处理结束End");
				test_p("-     ");
			}
		}
		test_p("DEBUG_FILTER: total=" + std::to_string(dbg_total) + 
		       " pass=" + std::to_string(dbg_passed) +
		       " imp=" + std::to_string(dbg_filtered_impurity) + 
		       " len=" + std::to_string(dbg_filtered_len) + 
		       " disp=" + std::to_string(dbg_filtered_disp) + 
		       " num=" + std::to_string(dbg_filtered_num) + 
		       " score=" + std::to_string(dbg_filtered_score));
		test_p("-=     ");
		return count; // count1 + count2;

	}
	// 手动实现 iota 函数
	template <typename T>
	void iota(T first, T last, int value) {
		while (first != last) {
			*first = value;
			++first;
			++value;
		}
	}
	// Lambda 表达式作为参数，接受 score_boxs 和 img_names 作为参数
	template <typename T, typename U>
	void sortScoreBoxesAndNames(T first, T last, U names, const std::vector<std::vector<float>>& boxes) {
		std::sort(first, last, [&boxes, &names](const size_t& a, const size_t& b) {
			return boxes[a][4] > boxes[b][4];
			});
	}

	int AickTensorrtStarter::GetShootingPoint(std::vector<std::vector<float>> &shootingPoint, std::vector<std::string> &shootingNames, int getNum) {
		// 创建索引数组并排序
		std::vector<size_t> indices(score_boxs.size());
		iota(indices.begin(), indices.end(), 0); 

		// 根据分数从高到低排序索引 (score_boxs[idx][4] 是分数)
		sortScoreBoxesAndNames(indices.begin(), indices.end(), img_names.begin(), score_boxs);

		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);
		
		//  距离去重逻辑 (防止 100X 在同一个地方重复拍摄)
		float repeatDistance = std::stof(config10X["repeatDistance"]);
		std::vector<std::vector<float>> zxPoints;
		std::vector<std::string> zxImgNames;
		std::vector<bool> removed(score_boxs.size(), false); 

		for (size_t i = 0; i < score_boxs.size(); ++i) {
			if (!removed[i]) {
				// indices[i] 是按分数排序后的索引
				zxPoints.push_back(score_boxs[indices[i]]);
				zxImgNames.push_back(img_names[indices[i]]);
				
				for (size_t j = i + 1; j < score_boxs.size(); ++j) {
					if (!removed[j]) {
						float diffX = std::abs(score_boxs[indices[i]][0] - score_boxs[indices[j]][0]);
						float diffY = std::abs(score_boxs[indices[i]][1] - score_boxs[indices[j]][1]);
						
						if (diffX < repeatDistance && diffY < repeatDistance) {
							removed[j] = true;
							test_p("靠太近的分裂相去除: " + std::to_string(score_boxs[indices[j]][0]) + " | " + std::to_string(score_boxs[indices[j]][1]));
						}
					}
				}
			}
		}
		test_p("玻片扫描分裂相总数: " + std::to_string(score_boxs.size()) + " | 去重后分散数: " + std::to_string(zxPoints.size()));

		if (isCultivation) {
			test_p("======== 原位模式开启: isCultivation=true, getNum=" + std::to_string(getNum));
			
			for (size_t i = 0; i < zxPoints.size(); i++) {
				std::vector<float>& zx_point = zxPoints[i];
				float zx_cls = -1; // 默认不属于任何群落
				
				// 判定分裂相坐标是否落在已检测到的群落 (CellDataMe) 范围内
				for (size_t j = 0; j < CellDataMe.size(); j++) {
					std::vector<float>& zx_box = CellDataMe[j]; // [x, y, w, h, cls]
					// 物理坐标系判定逻辑
					if (zx_point[0] > zx_box[0] && zx_point[0] < (zx_box[0] + zx_box[2]) && 
						zx_point[1] > (zx_box[1] - zx_box[3]) && zx_point[1] < zx_box[1]) {
						zx_cls = zx_box[4]; // 记录所属群落号
						test_p("分裂相落在群落: " + std::to_string(zx_cls));
					}
				}
				zxPoints[i][3] = zx_cls; // 更新类别索引
			}
			std::unordered_map<float, int> categoryCount;    // 每个类别有多少个
			std::unordered_map<float, int> getCategoryCount; // 每个类别已取多少个
			for (const auto& box : zxPoints) {
				float category = box[3];
				categoryCount[category]++;
				getCategoryCount[category] = 0;
			}

			int totalCategories = static_cast<int>(categoryCount.size());
			float c_num = (totalCategories > 0) ? (float)getNum / totalCategories : 0;
			test_p("群落种类: " + std::to_string(totalCategories) + " | 每种平均取: " + std::to_string(c_num));

			for (size_t i = 0; i < zxPoints.size(); i++) {
				float category = zxPoints[i][3];
				if (getCategoryCount[category] < c_num && getCategoryCount[category] < categoryCount[category]) {
					shootingPoint.push_back(zxPoints[i]);
					shootingNames.push_back(zxImgNames[i]);
					getCategoryCount[category]++;
				}
			}

			int remainingCount = getNum - static_cast<int>(shootingPoint.size());
			int rcount = 0;
			for (size_t i = 0; i < zxPoints.size(); i++) {
				if (rcount >= remainingCount) break;

				auto it = std::find(shootingNames.begin(), shootingNames.end(), zxImgNames[i]);
				if (it == shootingNames.end()) {
					shootingPoint.push_back(zxPoints[i]);
					shootingNames.push_back(zxImgNames[i]);
					rcount++;
				}
			}
		}
		else if (config10X["mode"] == "1") {
			// 根据配置的比例计算多倍体应取数量
			int polyploid_num = getNum * std::stof(config10X["polyploid_rate"]);
			test_p(">>> 模式1: 固定比例采样. 计划多倍体数: " + std::to_string(polyploid_num));

			// 第一遍循环：优先抓取多倍体 (分数 >= 2)
			for (size_t i = 0; i < zxPoints.size(); i++) {
				std::vector<float> s_point = zxPoints[i];
				if (s_point[4] < 2.0f) continue; // 过滤非多倍体

				if (shootingPoint.size() < static_cast<size_t>(polyploid_num)) {
					s_point[3] = -1.0f; // 统一类别标识
					std::string s_name = zxImgNames[i];
					test_p("======== 多倍体优先取出: " + s_name + " 分数: " + std::to_string(s_point[4]));
					shootingPoint.push_back(s_point);
					shootingNames.push_back(s_name);
				} else {
					break;
				}
			}

			// 第二遍循环：补充单倍体/普通分裂相 (0 < 分数 < 2)
			for (size_t j = 0; j < zxPoints.size(); j++) {
				std::vector<float> s_point = zxPoints[j];
				if (s_point[4] >= 2.0f || s_point[4] == 0.0f) continue;

				if (shootingPoint.size() < static_cast<size_t>(getNum)) {
					s_point[3] = -1.0f;
					std::string s_name = zxImgNames[j];
					test_p("======== 常规分裂相取出: " + s_name + " 分数: " + std::to_string(s_point[4]));
					shootingPoint.push_back(s_point);
					shootingNames.push_back(s_name);
				} else {
					break;
				}
			}
		}  //stof(config10X["reservedImpuritie"]) == 1

		else if (config10X["mode"] == "-1") {
			if (!zxPoints.empty()) {
				int polyploid_sum = 0, haploid_sum = 0;
				for (const auto& p : zxPoints) {
					if (p[4] < 2.0f && p[4] > 0.0f) haploid_sum++;
					else if(p[4] >= 2.0f) polyploid_sum++;
				}
				test_p(">>> 模式-1: 动态比例采样. 统计多倍体:" + std::to_string(polyploid_sum) + " 单倍体:" + std::to_string(haploid_sum));

				int polyploid_num, haploid_num;
				if ((int)zxPoints.size() < getNum) {
					polyploid_num = polyploid_sum;
					haploid_num = haploid_sum;
				} else {
					// 使用 std::ceil 向上取整，确保多倍体优先权
					polyploid_num = (int)std::ceil((float)getNum * polyploid_sum / zxPoints.size());
					haploid_num = getNum - polyploid_num;
				}
				test_p("配额分配 -> 多倍体:" + std::to_string(polyploid_num) + " 单倍体:" + std::to_string(haploid_num));

				for (size_t i = 0; i < zxPoints.size(); i++) {
					vector<float> s_point = zxPoints[i];
					if (s_point[4] < 2)
						continue;
					if (shootingPoint.size() < static_cast<size_t>(polyploid_num)) {

						s_point[3] = -1; // 从分裂相类别1，2，改为群落坐类别-1
						//cout << "20231219===============" << s_point[3] << endl;
						string s_name = zxImgNames[i];
						test_p("========多倍体常规取出待拍摄x:" + to_string(s_point[0]) + "---y: " + to_string(s_point[1]) + "---z: " + to_string(s_point[2]) + "---群落号: " + to_string(s_point[3]) + "---总分: " + to_string(s_point[4]) + +"-----box_name:" + s_name);
						shootingPoint.push_back(s_point);
						shootingNames.push_back(s_name);
					}
					else {
						break;
					}
				}
				for (size_t j = 0; j < zxPoints.size(); j++) {
					vector<float> s_point = zxPoints[j];
					if (s_point[4] >= 2 || s_point[4] == 0)
						continue;
					if (shootingPoint.size() < static_cast<size_t>(getNum)) {

						s_point[3] = -1; // 从分裂相类别1，2，改为群落坐类别-1
						//cout << "20231219===============" << s_point[3] << endl;
						string s_name = zxImgNames[j];
						test_p("========常规取出待拍摄x:" + to_string(s_point[0]) + "---y: " + to_string(s_point[1]) + "---z: " + to_string(s_point[2]) + "---群落号: " + to_string(s_point[3]) + "---总分: " + to_string(s_point[4]) + +"-----box_name:" + s_name);
						shootingPoint.push_back(s_point);
						shootingNames.push_back(s_name);
					}
					else {
						break;
					}
				}
			
			}
		}
		else {
			test_p("======== isCultivation: " + std::to_string(isCultivation));
			test_p("======== score_boxs size: " + std::to_string(score_boxs.size()));

			for (size_t i = 0; i < zxPoints.size(); i++) {
				if (i < static_cast<size_t>(getNum)) {
					std::vector<float> s_point = zxPoints[i];

					if (s_point[4] == 0.0f) continue;

					s_point[3] = -1.0f; // 类别转换：从检测类别转换为统一的坐标点类别
					std::string s_name = zxImgNames[i];

					test_p("======== 常规取出待拍摄 x:" + std::to_string(s_point[0]) + 
						" --- y: " + std::to_string(s_point[1]) + 
						" --- 总分: " + std::to_string(s_point[4]) + 
						" --- box_name: " + s_name);

					shootingPoint.push_back(s_point);
					shootingNames.push_back(s_name);
				} else {
					break; 
				}
			}
		}

		if (std::stof(config10X["reservedImpuritie"]) == 1.0f && shootingPoint.empty()) {
			test_p("======== 此玻片没有找到高质量分裂相，强制添加杂质点进行拍摄");
			for (size_t i = 0; i < zxPoints.size(); i++) {
				shootingPoint.push_back(zxPoints[i]);
				shootingNames.push_back(zxImgNames[i]);
			}
		}

		score_boxs.clear();
		img_names.clear();
		num10XImage = 0;   // 重置当前玻片计图器
		isCultivation = false; 

		test_p("======== GetShootingPoint 任务结束，内部缓存已清空");

		return 0;
	}

	int AickTensorrtStarter::FungusDetector(cv::Mat img, int w, int h, std::vector<std::vector<float>>& roiRects)
	{
		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);

		// 从配置读取阈值，Linux 下 stof 需确保字符串非空
		float conf_threshold = std::stof(config10X["conf_threshold"]);
		float nms_threshold = std::stof(config10X["nms_threshold"]);

		std::vector<det_box> result = Fungus40XDet->getFungus40XOutput(img, w, h, conf_threshold, nms_threshold);

		for (const auto& box : result) {
			std::vector<float> tmp;
			tmp.push_back(box.x1);
			tmp.push_back(box.y1);
			tmp.push_back(box.w);
			tmp.push_back(box.h);
			tmp.push_back(box.score);
			tmp.push_back((float)box.cls_idx);

			roiRects.push_back(tmp);
		}

		return 0;
	}

	int AickTensorrtStarter::FungusOverallDetector(cv::Mat img, int w, int h, std::vector<std::vector<float>>& roiRects)
	{
		std::map<std::string, std::string> config10X;
		ReadConfig(config10X);

		float conf_threshold = std::stof(config10X["fungus_overall_det_threshold"]);
		
		// 全局检测通常使用较小的 NMS 阈值 (如 0.1) 来过滤重叠
		std::vector<det_box> result = FungusOverDet->getFungusOverallOutput(img, w, h, conf_threshold, 0.1f);

		for (const auto& box : result) {
			std::vector<float> tmp = {
				box.x1, 
				box.y1, 
				box.w, 
				box.h, 
				box.score, 
				(float)box.cls_idx
			};
			roiRects.push_back(tmp);
		}

		return 0;
	}


	// 清晰度算法相关
	int AickTensorrtStarter::InintModelSharpness() {
		// 实例化模型对象
		SharpnessDet = new CTestModel();
		test_p("============= 开始初始化清晰度算法模型");

		// 加载配置
		std::map<std::string, std::string> config;
		ReadConfig(config); 

		// 获取模型路径
		std::string str_SharpnessPath = config["SharpnessDet"];
		if (str_SharpnessPath.empty()) {
			test_p("错误: 配置文件中 SharpnessDet 路径为空！");
			return -2;
		}

		// TensorRT 模型初始化
		SharpnessDet->initYoloEngine(str_SharpnessPath);
		
		test_p("============== 完成清晰度算法模型初始化: " + str_SharpnessPath);
		return 0;
	}

	struct ScoreWithMat {
		int index;
		cv::Mat img;
		double score;

		// 用于 std::sort：按 score 降序（分数高的排前面）
		bool operator<(const ScoreWithMat& other) const {
			return score > other.score; 
		}
	};

	int AickTensorrtStarter::SharpnessDetector(const std::vector<cv::Mat>& frames, std::vector<float>& scores) {
		using namespace std::chrono;
		if (frames.empty()) {
			scores.clear();
			std::cout << "[Error] 输入帧为空。" << std::endl;
			return -1;
		}

		auto t_start = high_resolution_clock::now();

		// 第一轮：全局快速评分
		scores.resize(frames.size());
		std::vector<ScoreWithMat> all_images;
		all_images.reserve(frames.size()); // 预分配内存提高 Linux 下的效率

		for (size_t i = 0; i < frames.size(); ++i) {
			double score = SharpnessDet->calculateSharpnessScoreFast(frames[i]);
			scores[i] = static_cast<float>(score);
			all_images.push_back({ static_cast<int>(i), frames[i].clone(), score });
		}


		if (all_images.empty()) return -1;

		// 排序选出前 Top K
		std::sort(all_images.begin(), all_images.end());

		size_t top_k = std::min<size_t>(5, all_images.size());
		std::vector<cv::Mat> top5_images;
		std::vector<int> top5_indices;

		for (size_t i = 0; i < top_k; ++i) {
			top5_images.push_back(all_images[i].img);
			top5_indices.push_back(all_images[i].index);
		}

		// 批量推理获取 ROI
		// 使用 TensorRT 的 Batch 能力并行处理 5 张图，极大提升效率
		std::vector<std::vector<cv::Rect>> all_boxes;
		std::vector<cv::Mat> results = SharpnessDet->batchInferBoxes(top5_images, all_boxes);

		// 第二轮：针对 ROI 区域进行高精度重评分
		for (size_t i = 0; i < results.size(); ++i) {
			if (results[i].empty()) continue; 
			double roi_score = SharpnessDet->calculateSharpnessScore(results[i]);
			int original_index = top5_indices[i];
			
			// 用高精度分数覆盖初始分数
			scores[original_index] = static_cast<float>(roi_score); 
		}

		auto t_end = high_resolution_clock::now();
		// test_p("清晰度检测总耗时: " + std::to_string(duration_cast<milliseconds>(t_end - t_start).count()) + " ms");

		return 0;
	}

	int AickTensorrtStarter::DelSharpnessDet() {
		test_p("============= 开始释放清晰度算法模型");
		if (SharpnessDet != nullptr) {
			delete SharpnessDet;
			SharpnessDet = nullptr;
		}
		test_p("============= 完成释放清晰度算法模型");
		return 0;
	}
}
