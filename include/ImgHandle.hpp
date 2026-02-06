#pragma once
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifndef IMGHANDLE_HPP
#define IMGHANDLE_HPP

struct det_box {
	float  x1, y1, w, h;
	float score;
	int cls_idx;
};

struct instance_seg {
	float  x1, y1, w, h;
	float score;
	cv::Mat mask;
	cv::Rect rect;
	float conf;
	int cls_idx;
};

struct BboxInfo {
	int x_left;//左上角x坐标
	int y_left;//左上角y坐标
	int x_right;//右上角x坐标
	int y_right;//右上角y坐标
};

struct DetectorInfo {
	struct BaseClassInfo {
		int input_w = 640;
		int input_h = 480;
		int channel = 3;
		float conf = 0.4;
		float nms = 0.5;
		int classNum = 2;
		std::vector<std::string>  className;
		std::vector<cv::Scalar>  classColor;
		std::string modelPath = "";
	}baseClassInfo;
};

template <typename T>
inline T VectorProduct(const std::vector<T>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

inline cv::Mat letterbox_image_v2(cv::Mat image_src, std::tuple<int, int > & size) {

	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);

	cv::Mat image;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	cv::resize(image_src, image, cv::Size(nw, nh));


	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;
	cv::copyMakeBorder(image, image, top, bottom, left, right, CV_HAL_BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	return image;
}

inline cv::Mat letterbox_image_v3(cv::Mat image_src,const std::tuple<int, int > & size) {

	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);

	cv::Mat image;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	cv::resize(image_src, image, cv::Size(nw, nh));


	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;
	cv::copyMakeBorder(image, image, top, bottom, left, right, CV_HAL_BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	return image;
}

inline void  scale_coords_v2(cv::Mat image_src, std::tuple<int, int > & size, std::vector<std::vector<det_box>> & det_boxes) {
	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);
	cv::Mat image;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	//cv::resize(image_src,image,cv::Size(nw,nh));
	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;

	for (size_t i = 0; i < det_boxes.size(); i++) {
		for (size_t j = 0; j < det_boxes[i].size(); j++) {
			det_boxes[i][j].x1 -= left;
			det_boxes[i][j].y1 -= top;
			det_boxes[i][j].x1 /= scale;
			det_boxes[i][j].y1 /= scale;
			det_boxes[i][j].w /= scale;
			det_boxes[i][j].h /= scale;
		}

	}
}



/**
 * 返回根据网络输入进行pad,scale后的图片 并返回总的目标数量
 * @param image_src
 * @param size
 * @return
 */
inline int  scale_coords_v3(cv::Mat image_src, std::tuple<int, int > & size, std::vector<std::vector<det_box>> & det_boxes) {
	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);
	//cv::Mat image;
	std::cout << "scale " << scale << std::endl;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	//cv::resize(image_src,image,cv::Size(nw,nh));
	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;


	std::cout << "top " << top << std::endl;
	std::cout << "bottom " << bottom << std::endl;
	std::cout << "left " << left << std::endl;
	std::cout << "right " << right << std::endl;


	int obj_num = 0;
	for (size_t i = 0; i < det_boxes.size(); i++) {
		for (size_t j = 0; j < det_boxes[i].size(); j++) {
			det_boxes[i][j].x1 -= left;
			det_boxes[i][j].y1 -= top;
			det_boxes[i][j].x1 /= scale;
			det_boxes[i][j].y1 /= scale;
			det_boxes[i][j].w /= scale;
			det_boxes[i][j].h /= scale;
			obj_num++;
		}

	}

	std::cout << "obj_num  " << obj_num << std::endl;
	return obj_num;

}


inline int  scale_coords_v4(cv::Mat image_src, std::tuple<int, int > & size, std::vector<std::vector<det_box>> & det_boxes) {
	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);
	//cv::Mat image;repo:NVIDIA/cuda-samples 
	//std::cout<<"scale "<<scale<<std::endl;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	//cv::resize(image_src,image,cv::Size(nw,nh));
	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;


	std::cout << "top " << top << std::endl;
	std::cout << "bottom " << bottom << std::endl;
	std::cout << "left " << left << std::endl;
	std::cout << "right " << right << std::endl;


	int obj_num = 0;
	for (size_t i = 0; i < det_boxes.size(); i++) {
		for (size_t j = 0; j < det_boxes[i].size(); j++) {
			det_boxes[i][j].x1 -= left;
			det_boxes[i][j].y1 -= top;
			det_boxes[i][j].x1 /= scale;
			det_boxes[i][j].y1 /= scale;
			det_boxes[i][j].w /= scale;
			det_boxes[i][j].h /= scale;
			obj_num++;
		}

	}

	std::cout << "obj_num  " << obj_num << std::endl;
	return obj_num;

}


inline int  scale_coords_v4_single(cv::Mat image_src, std::tuple<int, int > & size, std::vector<cv::Rect> & det_boxes) {
	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);
	//cv::Mat image;
	//std::cout<<"scale "<<scale<<std::endl;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	//cv::resize(image_src,image,cv::Size(nw,nh));
	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;


	std::cout << "top " << top << std::endl;
	std::cout << "bottom " << bottom << std::endl;
	std::cout << "left " << left << std::endl;
	std::cout << "right " << right << std::endl;


	int obj_num = 0;
	for (size_t i = 0; i < det_boxes.size(); i++) {
	
		det_boxes[i].x -= left;
		det_boxes[i].y -= top;
		det_boxes[i].x /= scale;
		det_boxes[i].y /= scale;
		det_boxes[i].width /= scale;
		det_boxes[i].height /= scale;

		det_boxes[i].x = (det_boxes[i].x > image_src.cols) ? image_src.cols-2 : det_boxes[i].x;
		det_boxes[i].y = (det_boxes[i].y > image_src.rows) ? image_src.rows-2 : det_boxes[i].y;
		det_boxes[i].x = (det_boxes[i].x < 0) ? 0 : det_boxes[i].x;
		det_boxes[i].y = (det_boxes[i].y < 0) ? 0 : det_boxes[i].y;
		det_boxes[i].width = (det_boxes[i].width < 0) ? 2 : det_boxes[i].width;
		det_boxes[i].height = (det_boxes[i].height < 0) ? 2 : det_boxes[i].height;
		det_boxes[i].width = (det_boxes[i].width + det_boxes[i].x < image_src.cols) ? det_boxes[i].width : image_src.cols - det_boxes[i].x;
		det_boxes[i].height = (det_boxes[i].height + det_boxes[i].y < image_src.rows) ? det_boxes[i].height : image_src.rows - det_boxes[i].y;

		obj_num++;
		
	}

	std::cout << "obj_num  " << obj_num << std::endl;
	return obj_num;

}

inline int  scale_coords_v5(cv::Mat image_src, std::tuple<int, int > & size, std::vector<cv::Mat> &result) {
	float scale = std::min((float)(std::get<0>(size)) / image_src.cols, (float)(std::get<1>(size)) / image_src.rows);
	//cv::Mat image;
	std::cout << "scale " << scale << std::endl;
	int nw = (int)(image_src.cols*scale);
	int nh = (int)(image_src.rows*scale);

	//cv::resize(image_src,image,cv::Size(nw,nh));
	int pad_w = std::get<0>(size) - nw;
	int pad_h = std::get<1>(size) - nh;

	int top = pad_h / 2;
	int bottom = pad_h - top;
	int left = pad_w / 2;
	int right = pad_w - left;


	std::cout << "top " << top << std::endl;
	std::cout << "bottom " << bottom << std::endl;
	std::cout << "left " << left << std::endl;
	std::cout << "right " << right << std::endl;

	std::vector<cv::Mat> result_temp;

	for (std::vector<cv::Mat>::iterator it = result.begin(); it != result.end(); it++) {
		cv::Mat tmp_m = (*it)(cv::Rect(left, top, nw, nh));
		cv::resize(tmp_m, tmp_m, cv::Size(image_src.cols, image_src.rows));
		result_temp.push_back(tmp_m);
	}
	result = result_temp;
	return 1;
}
inline std::vector<std::vector<det_box>> non_max_suppression_trt(const float* prob, int obj_count, float conf_thres = 0.25, float iou_thres = 0.45, int class_num = 1) {
    int max_wh = 4096;
    std::vector<cv::Rect> boxes_vec;
    std::vector<int> clsIdx_vec;
    std::vector<float> scores_vec;
    std::vector<int> boxIdx_vec;

    int stride = 5 + class_num; 

    for (size_t i = 0; i < obj_count; i++) {
        const float* current_row = prob + (i * stride);
        float obj_conf = current_row[4]; // 第 5 位是置信度

        if (obj_conf < conf_thres) continue;

        for (size_t cls_idx = 0; cls_idx < class_num; cls_idx++) {
            float cls_score = current_row[5 + cls_idx];
            float mix_conf = obj_conf * cls_score;

            if (mix_conf > conf_thres) {
                float cx = current_row[0];
                float cy = current_row[1];
                float w  = current_row[2];
                float h  = current_row[3];

                boxes_vec.push_back(cv::Rect(cx - w/2 + cls_idx * max_wh,
                                           cy - h/2 + cls_idx * max_wh,
                                           w, h));
                scores_vec.push_back(mix_conf);
                clsIdx_vec.push_back(cls_idx);
            }
        }
    }

    cv::dnn::NMSBoxes(boxes_vec, scores_vec, conf_thres, iou_thres, boxIdx_vec);

    std::vector<std::vector<det_box>> det_boxes(class_num);
    for (size_t idx : boxIdx_vec) {
        det_box tmp;
        tmp.x1 = boxes_vec[idx].x - clsIdx_vec[idx] * max_wh;
        tmp.y1 = boxes_vec[idx].y - clsIdx_vec[idx] * max_wh;
        tmp.w = boxes_vec[idx].width;
        tmp.h = boxes_vec[idx].height;
        tmp.score = scores_vec[idx];
        tmp.cls_idx = clsIdx_vec[idx];
        det_boxes[tmp.cls_idx].push_back(tmp);
    }
    return det_boxes;
}

inline std::vector<std::vector<det_box>> non_max_suppression_trt_yolov8(const float* prob, int obj_count, float conf_thres = 0.25, float iou_thres = 0.45, int class_num = 1) {
    int max_wh = 4096;
    
    std::vector<cv::Rect> boxes_vec;
    std::vector<int> clsIdx_vec;
    std::vector<float> scores_vec;
    std::vector<int> boxIdx_vec;

    // 直接遍历 8400 个 anchor，不需要先转置整个矩阵
    for (size_t j = 0; j < obj_count; j++) {
        
        // 1. 直接从原始排布中计算分值
        float max_score = 0;
        int max_cls_idx = -1;
        for (size_t cls_idx = 0; cls_idx < class_num; cls_idx++) {
            // 通过偏移量 [ (4 + 类别索引) * 8400 + 框索引 ] 获取得分
            float score = prob[(4 + cls_idx) * obj_count + j]; 
            if (score > max_score) {
                max_score = score;
                max_cls_idx = cls_idx;
            }
        }

        // 只有得分达标的才去取坐标
        if (max_score > conf_thres) {
            float cx = prob[0 * obj_count + j];
            float cy = prob[1 * obj_count + j];
            float w  = prob[2 * obj_count + j];
            float h  = prob[3 * obj_count + j];

            boxes_vec.push_back(cv::Rect(cx - w/2 + max_cls_idx * max_wh,
                                       cy - h/2 + max_cls_idx * max_wh,
                                       w, h));
            scores_vec.push_back(max_score);
            clsIdx_vec.push_back(max_cls_idx);
        }
    }

    if(boxes_vec.empty()) return std::vector<std::vector<det_box>>(class_num);

    cv::dnn::NMSBoxes(boxes_vec, scores_vec, 0.0f, iou_thres, boxIdx_vec);

    std::vector<std::vector<det_box>> det_boxes(class_num);
    for (size_t idx : boxIdx_vec) {
        det_box tmp;
        tmp.x1 = boxes_vec[idx].x - clsIdx_vec[idx] * max_wh;
        tmp.y1 = boxes_vec[idx].y - clsIdx_vec[idx] * max_wh;
        tmp.w = boxes_vec[idx].width;
        tmp.h = boxes_vec[idx].height;
        tmp.score = scores_vec[idx];
        tmp.cls_idx = clsIdx_vec[idx];
        det_boxes[tmp.cls_idx].push_back(tmp);
    }
    return det_boxes;
}

/**
 * Sigmoid 函数：用于将 Mask 逻辑回归值映射到 0-1
 */
inline float sigmoid_x(float x) {
	return 1.0f / (1.0f + exp(-x));
}

/**
 * 解析实例分割掩码 (Get Mask)
 * 逻辑：将检测分支输出的 32 位 Mask 权重与 Mask 原型分支进行矩阵乘法，并还原回原图尺寸
 */
inline void get_mask(const cv::Mat& mask_info, const cv::Mat& mask_protos, std::vector<int>& intput_size, cv::Mat& mask, cv::Rect& rect) {
	//获取检测框在 160x160 特征图上的对应位置 (通常是网络输入尺寸的 1/4)
	int width = mask_protos.size[3];   // 160
	int height = mask_protos.size[2];  // 160
	
	// 对应 letterbox 后的缩放比例
	float scale_x = (float)width / intput_size[0];
	float scale_y = (float)height / intput_size[1];

	int x = rect.x * scale_x;
	int y = rect.y * scale_y;
	int w = rect.width * scale_x;
	int h = rect.height * scale_y;

	cv::Mat res = mask_info * mask_protos.reshape(1, 32);
	res = res.reshape(1, height); // 得到 160x160 的单通道掩码

	for (size_t i = 0; i < res.rows; i++) {
		for (size_t j = 0; j < res.cols; j++) {
			res.at<float>(i, j) = sigmoid_x(res.at<float>(i, j));
		}
	}

	// 限制 Rect 边界防止溢出
	x = std::max(0, x);
	y = std::max(0, y);
	w = std::min(width - x, w);
	h = std::min(height - y, h);

	// 提取框内掩码并二值化
	cv::Mat mask_roi = res(cv::Rect(x, y, w, h)) > 0.5f;
	mask = mask_roi;
}

inline void non_max_suppression_trt_yolov8_seg(cv::Mat& img, float* prob0, float* prob1, 
    std::vector<int>& outputTensorShape, std::vector<int>& outputMaskTensorShape, 
    std::vector<int>& ids, std::vector<float>& confs, std::vector<cv::Rect>& rects, 
    std::vector<cv::Mat>& masks, int class_num, float conf_thres = 0.25, float iou_thres = 0.45) {

    // outputTensorShape 通常为 [1, 4 + class_num + 32, 8400]
    int obj_count = outputTensorShape[2]; 
    int seg_channels = 32;
    int dims_total = 4 + class_num + seg_channels; 
    int max_wh = 4096; // 偏移增量

    std::vector<cv::Rect2d> boxes_vec; // 使用 Rect2f 防止 NMS 内部面积计算溢出
    std::vector<float> scores_vec;
    std::vector<int> class_ids;
    std::vector<cv::Mat> mask_weights;

    // 遍历检测分支数据 (按 TensorRT 的 CHW 布局直接读取)
    for (size_t j = 0; j < obj_count; j++) {
        
        // 寻找当前 anchor 中概率最大的类别
        float max_score = 0;
        int max_cls_id = -1;
        for (size_t c = 0; c < class_num; c++) {
            // 索引计算：(4 + 类别索引) * 8400 + anchor索引
            float score = prob0[(4 + c) * obj_count + j];
            if (score > max_score) {
                max_score = score;
                max_cls_id = c;
            }
        }

        // 阈值过滤
        if (max_score > conf_thres) {
            // 读取坐标：0,1,2,3 对应 cx, cy, w, h
            float cx = prob0[0 * obj_count + j];
            float cy = prob0[1 * obj_count + j];
            float w  = prob0[2 * obj_count + j];
            float h  = prob0[3 * obj_count + j];

            // 应用类别偏移，实现多类别独立 NMS
            float offset_x = (float)max_cls_id * max_wh;
            float offset_y = (float)max_cls_id * max_wh;

            boxes_vec.push_back(cv::Rect2d(cx - w/2.0f + offset_x, 
                                          cy - h/2.0f + offset_y, 
                                          w, h));
            scores_vec.push_back(max_score);
            class_ids.push_back(max_cls_id);

            // 提取 32 通道的掩码权重 (Mask Weights)
            cv::Mat temp_mask(1, seg_channels, CV_32F);
            float* mask_ptr = (float*)temp_mask.data;
            for (size_t k = 0; k < seg_channels; k++) {
                // 索引：(4 + class_num + 掩码通道索引) * 8400 + anchor索引
                mask_ptr[k] = prob0[(4 + class_num + k) * obj_count + j];
            }
            mask_weights.push_back(temp_mask);
        }
    }

    if (boxes_vec.empty()) return;

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_vec, scores_vec, 0.0f, iou_thres, indices);

    // 假设网络输入尺寸是正方形，取 outputTensorShape 中的维度
    std::vector<int> net_size = {outputTensorShape[1], outputTensorShape[1]}; 
    std::tuple<int, int> net_size_tuple = {net_size[0], net_size[1]};

    cv::Mat mask_protos(4, outputMaskTensorShape.data(), CV_32F, prob1);

    for (size_t idx : indices) {
        cv::Rect2f raw_rect = boxes_vec[idx];
        raw_rect.x -= (float)class_ids[idx] * max_wh;
        raw_rect.y -= (float)class_ids[idx] * max_wh;

        // 转为整数矩形用于 get_mask
        cv::Rect res_rect = cv::Rect(
            (int)std::round(raw_rect.x), (int)std::round(raw_rect.y),
            (int)std::round(raw_rect.width), (int)std::round(raw_rect.height)
        );

        // 生成掩码 (内部通常涉及线性加权和 Sigmoid)
        cv::Mat mask;
        get_mask(mask_weights[idx], mask_protos, net_size, mask, res_rect);

        // 将坐标从网络输入尺寸(如640)还原到原图尺寸
        std::vector<cv::Rect> temp_rects = { res_rect };
        scale_coords_v4_single(img, net_size_tuple, temp_rects);

        // 存入结果
        ids.push_back(class_ids[idx]);
        confs.push_back(scores_vec[idx]);
        rects.push_back(temp_rects[0]);
        masks.push_back(mask);
    }
}

inline std::vector<std::vector<det_box>> non_max_suppression_trt_yolov8_seg_back(
    const float* prob,      // TensorRT 输出指针 (prob0)
    int obj_count,          // 通常为 8400
    int class_num,          // 类别数
    float conf_thres = 0.25, 
    float iou_thres = 0.45) 
{
    int max_wh = 4096;
    int dims = 4 + class_num + 32; 

    std::vector<cv::Rect> boxes_vec;
    std::vector<int> clsIdx_vec;
    std::vector<float> scores_vec;
    std::vector<int> boxIdx_vec;

    for (size_t j = 0; j < obj_count; j++) {
        for (size_t cls_idx = 0; cls_idx < class_num; cls_idx++) {
            float mix_conf = prob[(4 + cls_idx) * obj_count + j];

            if (mix_conf > conf_thres) {
                // 读取坐标索引: 0,1,2,3 分别对应 cx, cy, w, h
                float cx = prob[0 * obj_count + j];
                float cy = prob[1 * obj_count + j];
                float w  = prob[2 * obj_count + j];
                float h  = prob[3 * obj_count + j];

                float x1 = cx - w / 2.0f;
                float y1 = cy - h / 2.0f;

                boxes_vec.push_back(cv::Rect(
                    static_cast<int>(x1 + cls_idx * max_wh),
                    static_cast<int>(y1 + cls_idx * max_wh),
                    static_cast<int>(w),
                    static_cast<int>(h)
                ));
                scores_vec.push_back(mix_conf);
                clsIdx_vec.push_back(cls_idx);
            }
        }
    }

    if (boxes_vec.empty()) return std::vector<std::vector<det_box>>(class_num);

    // 执行 NMS
    cv::dnn::NMSBoxes(boxes_vec, scores_vec, 0.0f, iou_thres, boxIdx_vec);

    std::vector<std::vector<det_box>> det_boxes(class_num);
    for (size_t idx : boxIdx_vec) {
        det_box det_box_tmp;
        det_box_tmp.x1 = static_cast<float>(boxes_vec[idx].x - clsIdx_vec[idx] * max_wh);
        det_box_tmp.y1 = static_cast<float>(boxes_vec[idx].y - clsIdx_vec[idx] * max_wh);
        det_box_tmp.w = static_cast<float>(boxes_vec[idx].width);
        det_box_tmp.h = static_cast<float>(boxes_vec[idx].height);
        det_box_tmp.score = scores_vec[idx];
        det_box_tmp.cls_idx = clsIdx_vec[idx];

        det_boxes[det_box_tmp.cls_idx].push_back(det_box_tmp);
    }

    return det_boxes;
}

//字符串转宽字符字符串
inline std::wstring String2WideString(const std::string& s)
{
    if (s.empty()) return L"";
    size_t _Dsize = mbstowcs(NULL, s.c_str(), 0);
    if (_Dsize == (size_t)-1) return L""; 

    std::vector<wchar_t> _Dest(_Dsize + 1);
    mbstowcs(_Dest.data(), s.c_str(), _Dsize + 1);
    
    return std::wstring(_Dest.data());
}

#endif 