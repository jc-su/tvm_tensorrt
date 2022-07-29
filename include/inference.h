#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/core/mat.hpp>
#include <optional>
#include <string_view>
#include <vector>
#include "NvInfer.h"

std::vector<unsigned char> load_file(std::string_view file);

std::optional<int> inference(std::string_view engine_file, std::string_view input_file);

std::optional<cv::Mat> resize_normalize_image(cv::Mat image, float* gpu_input, const nvinfer1::Dims& dims);

#endif// INFERENCE_H