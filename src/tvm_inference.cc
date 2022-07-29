#include "dlpack/dlpack.h"
#include "tvm/runtime/container/base.h"
#include "tvm/runtime/logging.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

#include "argparse/argparse.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>

namespace fs = std::filesystem;

void inference(tvm::runtime::Module &gmod, const std::vector<std::string> &input_paths, DLDevice dev) {
    tvm::runtime::PackedFunc get_input_info = gmod.GetFunction("get_input_info");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
    tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    auto input_info = static_cast<tvm::Map<tvm::String, tvm::ObjectRef>>(get_input_info());
    auto shape_info = tvm::Downcast<tvm::Map<tvm::String, tvm::ShapeTuple>>(input_info["shape"]);
    auto dtype_info = tvm::Downcast<tvm::Map<tvm::String, tvm::String>>(input_info["dtype"]);
    std::unordered_map<std::string, DLDataType> dtype_map = {
            {"float32", DLDataType{kDLFloat, 32, 1}},
            {"float16", DLDataType{kDLFloat, 16, 1}},
            {"int8", DLDataType{kDLInt, 8, 1}},
            {"uint8", DLDataType{kDLUInt, 8, 1}},
            {"int16", DLDataType{kDLInt, 16, 1}},
            {"uint16", DLDataType{kDLUInt, 16, 1}},
    };

    // if input size not match model
    if (input_paths.size() != shape_info["data"][0]) {
        LOG(ERROR) << "input size not match model";
        return;
    }
    std::vector<cv::Mat> batch_images;
    auto x = tvm::runtime::NDArray::Empty(shape_info["data"], dtype_map[dtype_info["data"]], dev);
    auto y = tvm::runtime::NDArray::Empty(shape_info["softmax_label"], dtype_map[dtype_info["softmax_label"]], dev);

    for (const auto &path: input_paths) {
        if (!fs::exists(path)) {
            LOG(ERROR) << "file not exist: " << path;
            return;
        }

        auto img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            LOG(ERROR) << "read image failed: " << path;
            return;
        }
        // resize image to fit the model input size
        if (img.size() != cv::Size(shape_info["data"][2], shape_info["data"][3])) {
            cv::resize(img, img, cv::Size(shape_info["data"][2], shape_info["data"][3]));
        }
        // convert to chw format
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        batch_images.emplace_back(img);
    }
    // blob to ndarray
    // auto blob = cv::dnn::blobFromImages(batch_images, 1.0, cv::Size(shape_info["data"][2], shape_info["data"][3]), cv::Scalar(), true, false);
    // copy blob to ndarray
    // auto blob_shape = tvm::Array<tvm::Expr>{tvm::Integer(blob.rows), tvm::Integer(blob.cols), tvm::Integer(blob.channels), tvm::Integer(blob.depth)};


    // TODO: support batch inference

    LOG(INFO) << "Running graph executor...";

    // // run the code
    // run();
    // // get the output
    // tvm::runtime::NDArray output = get_output(0);
}

void evaluate(tvm::runtime::Module &gmod, DLDevice dev) {
    tvm::runtime::PackedFunc get_input_info = gmod.GetFunction("get_input_info");
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
    tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    tvm::runtime::PackedFunc time_eval =
            tvm::runtime::Registry::Get("runtime.RPCTimeEvaluator")->operator()(gmod, "run", static_cast<int>(dev.device_type), dev.device_id, 100, 10, 0, 0, 1, "");


    int num_inputs = get_num_inputs();
    int num_outputs = get_num_outputs();
    int num_flat_args = num_inputs + num_outputs;

    std::unique_ptr<TVMValue> values(new TVMValue[num_flat_args]);
    std::unique_ptr<int> type_codes(new int[num_flat_args]);
    tvm::runtime::TVMArgsSetter setter(values.get(), type_codes.get());
    int offs = 0;

    for (int i = 0; i < num_inputs; i++) {
        DLTensor *arg = const_cast<DLTensor *>(static_cast<tvm::runtime::NDArray>(get_input(i)).operator->());
        setter(offs, arg);
        offs++;
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
        DLTensor *arg = const_cast<DLTensor *>(static_cast<tvm::runtime::NDArray>(get_output(i)).operator->());
        setter(offs, arg);
        offs++;
    }

    tvm::runtime::TVMRetValue rv;
    time_eval.CallPacked(tvm::runtime::TVMArgs(values.get(), type_codes.get(), num_flat_args), &rv);
    std::string results = rv.operator std::string();
    const double *results_arr = reinterpret_cast<const double *>(results.data());
    std::cout << "mean (ms)   median (ms)   max (ms)   min (ms)   std (ms)" << std::endl;
    std::cout << results_arr[0] * 1000 << "    " << results_arr[1] * 1000 << "    " << results_arr[2] * 1000 << "    " << results_arr[4] * 1000 << "    " << results_arr[5] * 1000 << std::endl;
}

void DeployGraphExecutor(const std::string &module_path, const std::vector<std::string> &input_paths) {
    // load in the library
    DLDevice dev{kDLCUDA, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(module_path);
    // create the graph executor module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);

    if (input_paths.empty()) {
        evaluate(gmod, dev);
    } else {
        inference(gmod, input_paths, dev);
    }
}

int main(int argc, char *argv[]) {
    namespace fs = std::filesystem;
    argparse::ArgumentParser program("resnet_tvm");
    program.add_argument("--module", "-m")
            .help("Path to the module file")
            .default_value(std::string{"modules/resnet18-naive.so"})
            .required();
    program.add_argument("--input", "-i")
            .help("Path to the input files")
            .default_value(std::vector<std::string>{})
            .append();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto module_path = program.get<std::string>("module");
    if (!fs::exists(module_path)) {
        std::cerr << "Module file " << module_path << " does not exist" << std::endl;
        std::exit(1);
    }

    auto input_paths = program.get<std::vector<std::string>>("input");

    DeployGraphExecutor(module_path, input_paths);
    return 0;
}
