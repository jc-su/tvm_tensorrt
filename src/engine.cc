#include "engine.h"
#include "NvOnnxParser.h"
#include <NvInfer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>

namespace fs = std::filesystem;

std::optional<bool> build_engine(std::string_view onnx_file, std::string_view engine_file) {
    if (!fs::exists(onnx_file)) {
        std::cerr << "File " << onnx_file << " does not exist" << std::endl;
        return std::nullopt;
    }
    std::cout << "Engine not found, generating..." << std::endl;

    TRTLogger logger;
    auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        std::cerr << "Could not create builder" << std::endl;
        return std::nullopt;
    }
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicit_batch)};
    if (!network) {
        std::cerr << "Could not create network" << std::endl;
        return std::nullopt;
    }
    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    if (!parser) {
        std::cerr << "Could not create parser" << std::endl;
        return std::nullopt;
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        std::cerr << "Could not create builder config" << std::endl;
        return std::nullopt;
    }

    if (!parser->parseFromFile(reinterpret_cast<const char *>(onnx_file.data()), static_cast<int>(onnx_file.size()))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return std::nullopt;
    }

    config->setMaxWorkspaceSize(Options::max_workspace_size);
    
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    input_dims.d[0] = Options::max_batch_size;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);
    // config->setCalibrationProfile(profile);

    if (Options::FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (Options::INT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // config->setInt8Calibrator(nullptr);

        // int batch_count = 4096;
        // config->setInt8Calibrator(std::make_shared<nvinfer1::IInt8Calibrator>(batch_count).get());
    } else {
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    std::unique_ptr<nvinfer1::IHostMemory> engine_data(builder->buildSerializedNetwork(*network, *config));
    if (!engine_data) {
        std::cerr << "Failed to build engine" << std::endl;
        return std::nullopt;
    }

    std::ofstream outfile(engine_file.data(), std::ios::binary);
    outfile.write(reinterpret_cast<const char *>(engine_data->data()), engine_data->size());
    std::cout << "Saved engine to " << engine_file << std::endl;

    return true;
}
