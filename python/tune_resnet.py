import os
import time
import onnx
import tvm
from tvm import relay, autotvm, auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import argparse


def load_onnx(model_file, shape_dict, dtype="float32"):
    model = onnx.load(model_file)
    mod, params = relay.frontend.from_onnx(model, shape_dict, dtype)
    return mod, params


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_graph(graph, records, opt_sch_file, use_DP=True, shape_dict=None, min_exec_num=1):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, shape_dict, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=min_exec_num)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


def tune_scheduler(tasks, task_weights, n_trials, log_fn):
    # cost_model = auto_scheduler.XGBModel()
    # cost_model.update_from_file(log_fn)
    # search_policy = auto_scheduler.SketchPolicy(
    #     task, cost_model, init_search_callbacks=[
    #         auto_scheduler.PreloadMeasuredStates(log_fn)]
    # )

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(
        tasks, task_weights, load_log_file=log_fn)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_fn)],
    )
    tuner.tune(tune_option)


def resume_search(task, log_file, n_trials):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[
            auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials, measure_callbacks=[
            auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tune resnet18 on imagenet"
    )
    parser.add_argument(
        "--trial", type=int, help="number of trials for tuning", default=100
    )
    parser.add_argument(
        "--scheduler_trial", type=int, help="number of trials for scheduler", default=100
    )
    args = parser.parse_args()

    trial_num = args.trial
    scheduler_trial = args.scheduler_trial

    log_file = 'resources/tuning_log/resnet101-tune-{}.log'.format(trial_num)
    graph_opt_sch_file = 'resources/tuning_log/resnet101-scheduler-{}.log'.format(
        scheduler_trial)

    dtype = "float32"

    target = tvm.target.Target(tvm.target.cuda(
        arch="sm_75"))
    model_file = "models/resnet101-v2-7.onnx"
    shape_dict = {'data': (1, 3, 224, 224)}

    mod, params = load_onnx(model_file, shape_dict, dtype)

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": trial_num,
        "early_stopping": max(1, int(0.3*trial_num)),
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(
                number=20, repeat=3, timeout=4, min_repeat_ms=150),
        )
    }
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(
            relay.op.get("nn.conv2d"),)
    )

    # scheduler_tasks, scheduler_task_weights = auto_scheduler.extract_tasks(
    #     mod["main"], params, target)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 
    # tune_graph(mod["main"], "resources/tuning_log/resnet18-tune-2048.log", graph_opt_sch_file, use_DP=True, shape_dict=shape_dict, min_exec_num=min_exec_num)
    # for idx, task in enumerate(scheduler_tasks):
    #     print("========== Task %d  (workload key: %s) ==========" %
    #           (idx, task.workload_key))
    #     print(task.compute_dag)
    # tune_scheduler(scheduler_tasks, scheduler_task_weights,
    #                scheduler_trial, graph_opt_sch_file)
    # resume_search(scheduler_tasks, graph_opt_sch_file, scheduler_trial)
    tune_tasks(tasks, **tuning_option)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
