#pragma once
#include <string>
#include <vector>
#include <string>
#include <vector>
#include <memory> 
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

#ifndef DLL_EXP
    #ifdef _WIN32
        #define DLL_EXP __declspec(dllexport)
    #else
        #define DLL_EXP __attribute__((visibility("default")))
    #endif
#endif

typedef struct Point_zx
{
    float x, y, z;
    int clusterID;
} PointZX;

void test_p(std::string str);
string Convert(float Num);
class CTestModel;

// 静态全局变量
extern vector<vector<float>> score_boxs;
extern vector<string> img_names;
extern float num10XImage;
//static vector<float> dst_roots;

extern vector<vector<float>> IParam;
extern bool isCultivation;
extern std::vector<std::vector<float>> CellDataMe;

namespace DeepLearningFuncs
{
    class DLL_EXP AickTensorrtStarter
    {
    public:
        int InintModel(std::string model_path = "./2cls.engine", int class_num = 1, float conf_thres = 0.25f, float iou_thres = 0.45f);
        int InintModelMcn(std::string model_path = "./McnDet.onnx", int class_num = 2, float conf_thres = 0.25f, float iou_thres = 0.25f);
		int InitFungusModel();
		int McnDetector(char* image, int w, int h, vector<std::vector<float>> &McnData);
		int DelMcnDet();
		int CellDetector(int nSliceID, char* image, int w, int h, int detType, std::string savePath, vector<std::vector<float>> &CellData, std::string model_path = "./CellDet.onnx", int class_num = 1, float conf_thres = 0.05f, float iou_thres = 0.75f);
		
		bool compareContourAreas(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2);
		//int DelCellDet();
		int AickTensorrt(char* image, int w, int h, vector<float> stageInfo);
		int AickTensorrt_fish(char* image, int w, int h, vector<float> stageInfo);
		int GetShootingPoint(vector<vector<float>> &shootingPoint, vector<string> &shootingNames, int getNum);
		int InitParam(vector<vector<float>> fParam);
		int Clarity_evaluation(std::vector<char*> image_stream, int w, int h, int& index, std::vector<double>& offset);  // , char* mask
		int Center_alignment(char* image_stream, int w, int h, std::vector<double>& offset);
		//int CoordinatesPixelToPhysical(int nWidth, int nHeight, vector<float> m_dFocusDiff10_100, vector<float> score_boxs, vector<float> stageInfo, vector<float>& simpleScanPoint);
		//int test_10X(char* image, int w, int h);  

		int InintModelCellFish(std::string model_path = "./McnDet.onnx", int class_num = 1, float conf_thres = 0.25f, float iou_thres = 0.5f);
		int CellFishDetector(char* image, int w, int h, vector<std::vector<float>> &CellFish_box);
		int DelCellFishDet();

		int InintModelChromosomeFish(std::string model_path = "./McnDet.onnx", int class_num = 1, float conf_thres = 0.25f, float iou_thres = 0.5f);
		int DelChromosomeFishDet();
		
		int FungusDetector(cv::Mat img, int w, int h, std::vector<std::vector<float>>& roiRects);
		int FungusOverallDetector(cv::Mat img, int w, int h, std::vector<std::vector<float>>& roiRects);
		int InitFungusSeg();
		static AickTensorrtStarter* get_instance();

		// 清晰度算法
		int InintModelSharpness();
		int SharpnessDetector(const std::vector<cv::Mat>& frames, std::vector<float>& scores);
		int DelSharpnessDet();

    private:
        AickTensorrtStarter();
        CTestModel *test_model = nullptr;
        CTestModel *impurities_score = nullptr;
		CTestModel *reg_score = nullptr;
		CTestModel *c_score = nullptr;

		CTestModel *McnDet = nullptr;
		CTestModel *ChromosomeFishDet = nullptr;
		CTestModel *CellFishDet = nullptr;
		CTestModel *CellDet = nullptr;

		// 新增真菌40背景下检测rect模型
		CTestModel *Fungus40XDet = nullptr;
		CTestModel* FungusOverDet = nullptr;
		// 分割模型
		CTestModel* FungusOverSeg = nullptr;

		// 清晰度模型
		CTestModel* SharpnessDet = nullptr;


		static string valid_mask;
		static bool leap_over;
    };
}
