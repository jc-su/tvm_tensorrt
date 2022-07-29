from tvm import relay, auto_scheduler
import tvm
import onnx
import argparse


def print_task(index, task):
    print("=" * 60)
    print(f"Index: {index}")
    print(f"flop_ct: {task.compute_dag.flop_ct}")
    print(f"workload_key: {task.workload_key}")
    print("Compute DAG:")
    print(task.compute_dag)


def load_onnx(model_file, shape_dict, dtype="float32"):
    model = onnx.load(model_file)
    mod, params = relay.frontend.from_onnx(model, shape_dict, dtype)
    return mod, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    parser.add_argument("--file", type=str, default="models/resnet101-v2-7.onnx")
    args = parser.parse_args()

    dtype = "float32"
    target = tvm.target.Target(tvm.target.cuda(arch="sm_75"))
    model_file = args.file
    shape_dict = {'data': (1, 3, 224, 224)}

    mod, params = load_onnx(model_file, shape_dict, dtype)

    print("Load tasks...")
    scheduler_tasks, scheduler_task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target)

    if args.idx is None:
        for i, t in enumerate(scheduler_tasks):
            print_task(i, t)
    else:
        print_task(scheduler_tasks, scheduler_tasks[args.idx])