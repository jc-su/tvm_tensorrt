import argparse
from tvm import auto_scheduler

import tvm
import onnx
import tvm.relay.testing
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

from tvm import relay, autotvm

import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

def load_onnx(model_file, shape_dict, dtype="float32"):
    model = onnx.load(model_file)
    mod, params = relay.frontend.from_onnx(model, shape_dict, dtype)
    return mod, params


def trt_integration(mod, params):
    return partition_for_tensorrt(mod, params)


def export_lib(log_file, mod, params, target, lib_path, tuned, schedule=False, config=None):
    if tuned:
        print(log_file)
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3, config=config):
                lib = relay.build(mod, target=target, params=params)
    elif schedule:
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3, config=config):
            lib = relay.build(mod, target=target, params=params)
    lib.export_library(lib_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str,
                        default="resnet")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lib_path", type=str,
                        default="modules/resnet.tune.so")

    parser.add_argument("--tuned", type=bool, default=True)
    parser.add_argument("--schedule", type=bool, default=False)

    parser.add_argument("--is_trt", type=bool, default=False)
    parser.add_argument("--FP16", type=bool, default=False)

    args = parser.parse_args()
    log_file  = "resources/tuning_log/resnet101-tune-{}.log"
    lib_path = "modules/resnet101-tune-{}.so"
    target = tvm.target.Target(tvm.target.cuda(arch="sm_75"))

    model_file = "models/resnet101-v2-7.onnx"

    shape_dict = {'data': (args.batch_size, 3, 224, 224)}
    mod, params = load_onnx(model_file, shape_dict)

    if args.is_trt:
        mod, config = trt_integration(mod, params)
        trt_config = {'relay.ext.tensorrt.options': config}
        export_lib(log_file, mod, params, target,
                   args.lib_path, args.tuned, trt_config)
    elif args.tuned:
        for i in [2048]:
            export_lib(log_file.format(i), mod, params, target,
                        lib_path.format(i) , args.tuned)
    elif args.schedule:
        for i in [32768]:
            export_lib("resources/resnet_101-B1-cuda.json", mod, params, target,
                       "modules/resnet_101-sch.so", args.tuned, args.schedule)
