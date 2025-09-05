import aiter
import pandas as pd
import torch
import argparse
import multiprocessing
import logging
from rich.progress import (
    Progress,
    TextColumn,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from tqdm import tqdm

from aiter import dtypes
from aiter.test_common import run_perftest
from gemm_a8w8_blockscale_tune import (
    checkClose,
    run_torch,
    get_untuned_gemm_list,
    get_tuned_gemm_list,
)
from gen_instances_bruteforce import get_kernel_instances_info

_, default_kernels_list = get_kernel_instances_info("aiter/configs/a8w8_blockscale_kernel_instances_info.pkl")
block_shape = (128, 128)


def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0, num_warmup=1, num_iters=2):
    return run_perftest(aiter.gemm_a8w8_blockscale_tune_bruteforce,
                        x, weight, x_scale, w_scale, out, kernel_id, splitK,
                        num_warmup=num_warmup, num_iters=num_iters,
                        testGraph=False,
                        )


def tune_topk(x, weight, x_scale, w_scale, out, ref_out, kernels_list=default_kernels_list, candidates=[], num_warmup=1, num_iters=2, topk=1, progress_queue=None, tune_log_file=None):
    """
    Tune the top-k candidates based on their execution time.
    """
    if not candidates or len(candidates) == 0:
        return []

    m, k = x.shape
    n = weight.shape[0]

    # Update child progress bar to indicate the start of tuning
    if progress_queue is not None:
        progress_queue.put((torch.cuda.current_device(), m, n, k, len(candidates)))

    # Initialize a DataFrame to log tuning results
    tune_log_df = pd.DataFrame(
        columns=["cu_num", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
    )

    # Run performance test for each candidate
    for candidate in candidates:
        kernel_id = candidate["kernelId"]
        splitK = candidate["splitK"]
        try:
            (out), avg_t = kernel_instance_test(
                x, weight, x_scale, w_scale, out, kernel_id, splitK,
                num_warmup=num_warmup, num_iters=num_iters
            )
            isClosed = checkClose(ref_out, out, rtol=1e-2, atol=0.1)
        except RuntimeError:
            isClosed = False

        if isClosed:
            candidate["time"] = avg_t
        else:
            candidate["time"] = float('inf')

        # Log the tuning result
        if tune_log_file is not None:
            kernel = kernels_list[kernel_id]
            temp = pd.DataFrame(
                {
                    "cu_num": [torch.cuda.current_device()],
                    "M": [m],
                    "N": [n],
                    "K": [k],
                    "kernelId": [kernel_id],
                    "splitK": [splitK],
                    "us": [round(avg_t, 4)],
                    "kernelName": [kernel.name],
                    "valid": [isClosed],
                }
            )
            tune_log_df = temp if tune_log_df.empty else pd.concat([tune_log_df, temp], ignore_index=True)

        # Update child progress bar to indicate progress
        if progress_queue is not None:
            progress_queue.put((m, n, k))

    # Save the tuning log to the specified file
    if tune_log_file is not None:
        tune_log_df.to_csv(tune_log_file, mode='a', header=False, index=False)

    # Update child progress bar to indicate completion
    if progress_queue is not None:
        progress_queue.put((m, n, k, -1))

    # Sort candidates by time and select the top-k
    candidates.sort(key=lambda x: x["time"])
    return candidates[:topk]


def tune_gemm(m, n, k, useSplitK=False, kernels_list=default_kernels_list, progress_queue=None, tune_log_file=None):
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    out = torch.empty(m, n, dtype=dtypes.bf16, device="cuda")

    ref_out = run_torch(x, weight, x_scale, w_scale)

    candidates = []
    for i in range(len(kernels_list)):
        kernel = kernels_list[i]
        maxsplitK = (
            aiter.compute_gemm_SplitK(
                m, n, k, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
            )
            if useSplitK
            else 0
        )
        for splitK in range(maxsplitK + 1):
            candidates.append({
                "kernelId": i,
                "splitK": splitK,
                "time": float('inf'),
            })

    for num_warmup, num_iters, topk in [(2, 3, (len(candidates)+99)//100), (10, 101, 1)]:
        candidates = tune_topk(x, weight, x_scale, w_scale, out, ref_out,
                            kernels_list=kernels_list, candidates=candidates,
                            num_warmup=num_warmup, num_iters=num_iters, topk=topk,
                            progress_queue=progress_queue, tune_log_file=tune_log_file)

    return candidates[0].values()


def worker(pid, gemm_counter, lock_tune_file, progress_queue, output_queue, untunedf, tunedf, useSplitK, kernels_list, tune_file):

    # Set the device for the current thread
    gpu = torch.device(f"cuda:{pid}")
    torch.cuda.set_device(gpu)

    # Get the number of CU for the current GPU
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    # Main loop to tune GEMM
    while True:
        with gemm_counter.get_lock():
            if gemm_counter.value >= len(untunedf):
                break
            i = gemm_counter.value
            gemm_counter.value += 1
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        if tunedf[
            (tunedf["M"] == M) & (tunedf["N"] == N) & (tunedf["K"] == K) & (tunedf["cu_num"] == cu_num)
        ].empty:
            tune_log_file = tune_file.replace(".csv", f"_{pid}.csv.log") if tune_file else None
            kernelId, splitK, time = tune_gemm(M, N, K, useSplitK, kernels_list, progress_queue, tune_log_file)
            kernelName = "None" if kernelId == -1 else kernels_list[kernelId].name
            temp = pd.DataFrame(
                {
                    "cu_num": [cu_num],
                    "M": [M],
                    "N": [N],
                    "K": [K],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [round(time, 4)],
                    "kernelName": [kernelName],
                }
            )
            output_queue.put(temp)
            # Update the tune file if provided
            if tune_file is not None:
                with lock_tune_file:
                    temp.to_csv(tune_file, mode='a', header=False, index=False)
        # Update main progress bar
        if progress_queue is not None:
            progress_queue.put(None)


def rich_progress_updater(progress_queue, untunedf):
    # Initialize progress bar
    pbar = Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    pbar.start()
    main_task = pbar.add_task(description=f"Tuning GEMM with {torch.cuda.device_count()} GPUs", total=len(untunedf))
    child_task = {}
    while True:
        data = progress_queue.get()
        if data is None:
            # Update the main progress bar
            pbar.advance(main_task)
            if pbar.finished:
                break
        elif len(data) == 5:
            # Create the child progress bar
            pid, m, n, k, total_kernels = data
            child_task[(m,n,k)] = pbar.add_task(
                description=f"Tuning {m}x{n}x{k} on gpu {pid}", total=total_kernels
            )
        elif len(data) == 4:
            # Remove the child progress bar when tuning is finished
            m, n, k, _ = data
            pbar.remove_task(child_task[(m, n, k)])
            del child_task[(m, n, k)]
        elif len(data) == 3:
            # Update the child progress bar
            m, n, k = data
            pbar.advance(child_task[(m, n, k)])
        else:
            pass  # Ignore other messages
    pbar.stop()


def tqdm_progress_updater(progress_queue, untunedf):
    main_pbar = tqdm(total=len(untunedf), desc="Tuning GEMM")
    child_pbar = {}
    while True:
        data = progress_queue.get()
        if data is None:
            # Update the main progress bar
            main_pbar.update(1)
            if main_pbar.n >= main_pbar.total:
                break
        elif len(data) == 5:
            # Create the child progress bar
            pid, m, n, k, total_kernels = data
            child_pbar[(m, n, k)] = tqdm(total=total_kernels, desc=f"Tuning {m}x{n}x{k} on gpu {pid}", leave=False)
        elif len(data) == 4:
            # Remove the child progress bar when tuning is finished
            m, n, k, _ = data
            child_pbar[(m, n, k)].close()
            del child_pbar[(m, n, k)]
        elif len(data) == 3:
            # Update the child progress bar
            m, n, k = data
            child_pbar[(m, n, k)].update(1)
        else:
            pass  # Ignore other messages
    for p in child_pbar.values():
        p.close()


def tune_gemm_list(untunedf, tunedf, useSplitK=False, kernels_list=default_kernels_list, tune_file=None):

    # Set the start method for multiprocessing to "spawn"
    # This is necessary for compatibility with CUDA in some environments
    multiprocessing.set_start_method("spawn")

    # Disable aiter logger
    logger = logging.getLogger("aiter")
    logger.disabled = True

    gemm_counter = multiprocessing.Value('i', 0)
    lock_tune_file = multiprocessing.Lock()
    progress_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    processes = []
    for pid in range(torch.cuda.device_count()):
        p = multiprocessing.Process(target=worker, args=(pid, gemm_counter, lock_tune_file, progress_queue,
                                    output_queue, untunedf, tunedf, useSplitK, kernels_list, tune_file,))
        processes.append(p)
        p.start()
    processes.append(multiprocessing.Process(target=rich_progress_updater, args=(progress_queue, untunedf,)))
    # processes.append(multiprocessing.Process(target=tqdm_progress_updater, args=(progress_queue, untunedf,)))
    processes[-1].start()
    for p in processes:
        p.join()

    # Collect results from the output queue
    while not output_queue.empty():
        temp = output_queue.get()
        if isinstance(temp, pd.DataFrame):
            tunedf = temp if tunedf.empty else pd.concat([tunedf, temp], ignore_index=True)

    # Re-enable aiter logger
    logger.disabled = False

    return tunedf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/a8w8_blockscale_untuned_gemm_bruteforce.csv",
        required=False,
        help="input",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_blockscale_tuned_gemm_bruteforce.csv",
        required=False,
        help="output: tuning result store this file",
    )

    parser.add_argument(
        "-k",
        "--splitK",
        action="store_true",
        required=False,
        default=True,
        help="Use splitK kernels"
    )

    parser.add_argument(
        "--sort",
        action="store_true",
        required=False,
        default=True,
        help="Arranged according to the M N K size",
    )

    # Dummy run to make sure gemm_a8w8_blockscale_tune_bruteforce is compiled
    tmp_out = torch.empty((1, 128), dtype=dtypes.bf16, device="cuda")
    kernel_instance_test(
        (torch.rand((1, 128), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8),
        (torch.rand((128, 128), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8),
        torch.rand((1, 1), dtype=dtypes.fp32, device="cuda"),
        torch.rand((1, 1), dtype=dtypes.fp32, device="cuda"),
        tmp_out,
        0,
        0,
    )

    args = parser.parse_args()
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(untunedf, tunedf, args.splitK, default_kernels_list, args.tune_file)

    # Sort and save the tuning results if requested
    if args.sort:
        tunedf = tunedf.sort_values(by=["cu_num", "M", "N", "K"])
        tunedf.to_csv(args.tune_file, index=False)
        print(f"Tuning result saved to {args.tune_file}")
