// header guard
#ifndef ENGINE_H
#define ENGINE_H

#include "NvInfer.h"

#include <iostream>
#include <optional>
#include <string_view>


struct Options {
    inline static constexpr size_t max_workspace_size = 1 << 30;
    inline static constexpr size_t max_batch_size = 64;
    inline static constexpr bool FP16 = false;
    inline static constexpr bool INT8 = false;
    inline static constexpr int deviceIndex = 0;
};

class TRTLogger : public nvinfer1::ILogger {
public:
    inline virtual void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

std::optional<bool> build_engine(std::string_view onnx_file, std::string_view engine_file);

#endif// ENGINE_H