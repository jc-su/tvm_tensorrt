#include "engine.h"
#include "macro.h"
#include <array>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>


std::vector<unsigned char> load_file(std::string_view file) {
    std::ifstream in(file.data(), std::ios::in | std::ios::binary);
    if (!in.is_open()) return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char *) &data[0], length);
    }
    in.close();
    return data;
}

std::optional<int> inference(std::string_view engine_file, std::string_view input_file) {
    TRTLogger logger;
    auto engine_data = load_file(engine_file);

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())};
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};

    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream));
    // auto stream = std::make_shared<cudaStream_t>();
    // CHECK(cudaStreamCreate((void **) &stream));

    constexpr int IMAGE_WIDTH = 224;
    constexpr int IMAGE_HEIGHT = 224;
    constexpr int IMAGE_CHANNELS = 3;
    constexpr int BATCH_SIZE = 1;
    constexpr int INPUT_SIZE = BATCH_SIZE * IMAGE_CHANNELS * IMAGE_WIDTH * IMAGE_HEIGHT;

    // std::shared_ptr<float> input_data_host = std::make_shared<float>(INPUT_SIZE);
    // std::shared_ptr<float> input_data_device = std::make_shared<float>(INPUT_SIZE);

    // std::shared_ptr<float> output_data_host = std::make_shared<float>(INPUT_SIZE);
    // std::shared_ptr<float> output_data_device = std::make_shared<float>(INPUT_SIZE);

    // CHECK(cudaMallocHost((void **) &input_data_host, INPUT_SIZE * sizeof(float)));
    // CHECK(cudaMalloc((void **) &input_data_device, INPUT_SIZE * sizeof(float)));
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;

    CHECK(cudaMallocHost(&input_data_host, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&input_data_device, INPUT_SIZE * sizeof(float)));

    std::array<float, 3> mean{0.406, 0.456, 0.485};
    std::array<float, 3> std{0.225, 0.224, 0.229};

    auto image = cv::imread(input_file.data());
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return std::nullopt;
    }
    cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //归一化/除均值减标准差
    // #pragma omp parallel for
    //     for (int i = 0; i < IMAGE_HEIGHT; i++) {
    //         for (int j = 0; j < IMAGE_WIDTH; j++) {
    //             for (int k = 0; k < IMAGE_CHANNELS; k++) {
    //                 input_data_host[i * IMAGE_WIDTH * IMAGE_CHANNELS + j * IMAGE_CHANNELS + k] =
    //                         (image.data[i * IMAGE_WIDTH * IMAGE_CHANNELS + j * IMAGE_CHANNELS + k] - mean[k]) / std[k];
    //             }
    //         }
    //     }
    int image_area = image.cols * image.rows;
    unsigned char *pimage = image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
#pragma omp parallel for
    for (int i = 0; i < image_area; ++i, pimage += 3) {
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    CHECK(cudaMemcpyAsync(input_data_device, input_data_host, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    const int num_classes = 1000;
    float output_data_host[num_classes];
    float *output_data_device = nullptr;
    CHECK(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = BATCH_SIZE;

    context->setBindingDimensions(0, input_dims);
    // 用一个指针数组bindings指定input和output在gpu中的指针。
    float *bindings[] = {input_data_device, output_data_device};
    // benchmark start
    std::vector<std::chrono::time_point<std::chrono::system_clock>> start_time;
    std::vector<std::chrono::time_point<std::chrono::system_clock>> end_time;

    //warm up
    bool success = context->enqueueV2((void **) bindings, stream, nullptr);
        // synchronize the enqueueV2
    cudaStreamSynchronize(stream);
    if (!success) {
        std::cerr << "Failed to enqueueV2" << std::endl;
        return std::nullopt;
    }

    for (int i = 0; i < 600; ++i) {
        start_time.push_back(std::chrono::system_clock::now());
        context->enqueueV2((void **) bindings, stream, nullptr);
        // synchronize the enqueueV2
        cudaStreamSynchronize(stream);
        end_time.push_back(std::chrono::system_clock::now());
    }
    // average time
    std::vector<double> time_cost;
    for (int i = 0; i < 600; ++i) {
        time_cost.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end_time[i] - start_time[i]).count());
    }
    double total_time = 0;
    for (const auto &t : time_cost) {
        total_time += t;
    }
    double average_time = total_time / time_cost.size();
    std::cout << "average time: " << average_time/1000 << " ms" << std::endl;

    CHECK(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float *prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    float conf = prob[predict_label];
    std::cout << "predict_label: " << predict_label << ", conf: " << conf << std::endl;

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFreeHost(input_data_host));
    CHECK(cudaFree(input_data_device));
    CHECK(cudaFree(output_data_device));

    return predict_label;
}