#include "NvInfer.h"
#include "argparse/argparse.hpp"
#include "engine.h"
#include "inference.h"
#include "labels.h"
#include "macro.h"
#include <array>
#include <filesystem>
#include <string_view>

int main(int argc, char *argv[]) {
    namespace fs = std::filesystem;
    argparse::ArgumentParser program("resnet_trt");
    program.add_argument("--engine", "-e")
            .help("Path to the engine file")
            .default_value(std::string{"models/resnet-18-fp32-cpp.engine"});
    program.add_argument("--input", "-i")
            .help("Path to the input image")
            .default_value(std::string{"resources/imgs/snake.jpg"});
    program.add_argument("--onnx", "-o")
            .help("Path to the ONNX model")
            .default_value(std::string{"models/resnet101-v2-7.onnx"});
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    auto engine_file = program.get<std::string>("--engine");
    auto input_file = program.get<std::string>("--input");
    auto onnx_file = program.get<std::string>("--onnx");
    if (!fs::exists(engine_file)) {
        std::optional<bool> success = build_engine(onnx_file, engine_file);
        if (!success) {
            std::cout << "Failed to build engine" << std::endl;
            return 1;
        }
    }

    std::optional<int> result = inference(engine_file, input_file);
    if (!result) {
        std::cout << "Failed to inference" << std::endl;
        return 1;
    }
    std::cout << "Result: " << imagenet_labels[*result] << std::endl;
    return 0;
}