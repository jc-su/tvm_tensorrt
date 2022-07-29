import tvm
import tvm.relay.testing

import numpy as np
import argparse

from tvm import testing
from tvm.contrib import graph_executor

# testing.utils.install_request_hook(depth=3)


def load_lib(lib_path):
    lib = tvm.runtime.load_module(lib_path)
    return lib


def evaluate_network(lib, target, input_shape, dtype="float32"):
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array(
        (np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # evaluate
    print(module.benchmark(dev, number=100, repeat=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_path", type=str,
                        default="modules/resnet18-naive.so")
    args = parser.parse_args()

    target = tvm.target.Target(tvm.target.cuda(arch="sm_75"))
    lib = load_lib(args.lib_path)
    evaluate_network(lib, target, input_shape=(1, 3, 224, 224))
