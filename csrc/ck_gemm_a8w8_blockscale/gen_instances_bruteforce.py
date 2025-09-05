import os
import pandas as pd
import argparse
import torch
import pickle
import threading
import logging
from tqdm import tqdm

from gen_instances import kernelInstance, gemm_a8w8_blockscale_codegen, get_tune_dict
from gemm_a8w8_blockscale_common import kernels_list as default_kernels_list
from aiter.jit.core import (
    rm_module,
    clear_build,
    build_module,
    AITER_CSRC_DIR,
    AITER_ROOT_DIR,
    bd_dir,
)


def get_all_kernel_instances(Scale_Block_M=1, Scale_Block_N=128, Scale_Block_K=128):
    """
    Generate all possible kernel instances for the given scale block sizes.
    """
    kernel_instances_list = []
    for block_size in [256, 512, 1024]:
        AB_Block_Transfer_list = [[i, block_size // i, 1] for i in range(1, block_size + 1) if block_size % i == 0 and (i in [16, 32] or block_size // i in [16, 32])]
        C_Block_Transfer_list = [[1, i, 1, block_size // i] for i in range(1, block_size + 1) if block_size % i == 0 and (i in [16, 32] or block_size // i in [16, 32])]
        num_warp = block_size // 64
        for M_per_Block in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            for N_per_Block in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                for M_per_XDL, N_per_XDL in [(16, 16), (32, 32)]:
                    XDL_per_Block = (M_per_Block // M_per_XDL) * (N_per_Block // N_per_XDL)
                    if XDL_per_Block < num_warp or XDL_per_Block % num_warp != 0:
                        continue
                    XDL_per_Wave = XDL_per_Block // num_warp
                    for M_XDL_per_Wave, N_XDL_per_Wave in [(i, XDL_per_Wave // i) for i in range(1, XDL_per_Wave + 1)]:
                        if M_per_Block % (M_XDL_per_Wave * M_per_XDL) != 0:
                            continue
                        if N_per_Block % (N_XDL_per_Wave * N_per_XDL) != 0:
                            continue
                        for A_Block_Transfer in AB_Block_Transfer_list:  # K0_M_K1
                            for B_Block_Transfer in AB_Block_Transfer_list: # K0_N_K1
                                for C_Block_Transfer in C_Block_Transfer_list: # MBlock_MXdlPerWave_MWaveMPerXdl NBlock_NXdlPerWave_NWaveNPerXdl
                                    if M_per_Block % A_Block_Transfer[1] != 0:
                                        continue
                                    if N_per_Block % B_Block_Transfer[1] != 0:
                                        continue
                                    if M_per_Block % C_Block_Transfer[1] != 0:
                                        continue
                                    if N_per_Block % C_Block_Transfer[3] != 0:
                                        continue
                                    for K_per_Block in [64, 128, 256, 512, 1024, 2048]:
                                        shared_mem_used = min(M_per_Block * M_XDL_per_Wave, N_per_Block * N_XDL_per_Wave) * K_per_Block
                                        if shared_mem_used > 2**16: # MI308X and MI300X have 64KB shared memory
                                            continue
                                        for AK1 in [8, 16, 32]:
                                            for BK1 in [8, 16, 32]:
                                                shared_mem_used = (M_per_Block + N_per_Block) * K_per_Block
                                                if shared_mem_used > 2**16: # MI308X and MI300X have 64KB shared memory
                                                    continue
                                                AK0 = K_per_Block // AK1
                                                BK0 = K_per_Block // BK1
                                                if AK0 == 0 or BK0 == 0:
                                                    continue
                                                if AK0 % A_Block_Transfer[0] != 0:
                                                    continue
                                                if BK0 % B_Block_Transfer[0] != 0:
                                                    continue
                                                for CShuffle_M_XDL_per_Wave_per_Shuffle in [1, 2, 4]:
                                                    for CShuffle_N_XDL_per_Wave_per_Shuffle in [1, 2, 4]:
                                                        shared_mem_used = CShuffle_M_XDL_per_Wave_per_Shuffle * CShuffle_N_XDL_per_Wave_per_Shuffle * (M_per_Block / M_XDL_per_Wave) * (N_per_Block / N_XDL_per_Wave)
                                                        if shared_mem_used > 2**16: # MI308X and MI300X have 64KB shared memory
                                                            continue
                                                        if M_XDL_per_Wave % CShuffle_M_XDL_per_Wave_per_Shuffle != 0:
                                                            continue
                                                        if N_XDL_per_Wave % CShuffle_N_XDL_per_Wave_per_Shuffle != 0:
                                                            continue
                                                        if CShuffle_N_XDL_per_Wave_per_Shuffle > N_XDL_per_Wave:
                                                            continue
                                                        C_Block_Transfer_Scaler_Per_Vector = N_per_Block // C_Block_Transfer[3] // (N_XDL_per_Wave // CShuffle_N_XDL_per_Wave_per_Shuffle)
                                                        if C_Block_Transfer_Scaler_Per_Vector < 4:
                                                            continue
                                                        C_Block_Transfer_Scaler_Per_Vector = min(C_Block_Transfer_Scaler_Per_Vector, 8)
                                                        C_Block_Transfer_Scaler_Per_Vector = [C_Block_Transfer_Scaler_Per_Vector]
                                                        for Pipeline_ver in [1, 3]:
                                                            if Pipeline_ver == 3 and K_per_Block != Scale_Block_K:
                                                                continue
                                                            instance = kernelInstance(
                                                                block_size,
                                                                Scale_Block_M,
                                                                Scale_Block_N,
                                                                Scale_Block_K,
                                                                M_per_Block,
                                                                N_per_Block,
                                                                K_per_Block,
                                                                AK1,
                                                                BK1,
                                                                M_per_XDL,
                                                                N_per_XDL,
                                                                M_XDL_per_Wave,
                                                                N_XDL_per_Wave,
                                                                A_Block_Transfer,
                                                                B_Block_Transfer,
                                                                CShuffle_M_XDL_per_Wave_per_Shuffle,
                                                                CShuffle_N_XDL_per_Wave_per_Shuffle,
                                                                C_Block_Transfer,
                                                                C_Block_Transfer_Scaler_Per_Vector,
                                                                "Intrawave",
                                                                Pipeline_ver
                                                            )
                                                            kernel_instances_list.append(instance)
    for key, kernel in default_kernels_list.items():
        if kernel not in kernel_instances_list:
            print(f"Warning: {key}: {kernel} not found in instances list.")
    return kernel_instances_list


def try_compile_kernel_instaces(kernel_instances_list, md_name, tmp_ck_dir_name="ck", use_print=True):
    """
    Generate a single instance of the kernel.
    """
    try:
        # Generate instances
        os.system(f"rm -rf {bd_dir}/{md_name}/blob/*")
        os.system(f"mkdir -p {bd_dir}/{md_name}/blob")
        codegen = gemm_a8w8_blockscale_codegen(working_path=f"{bd_dir}/{md_name}/blob", istune=True)
        codegen.gen_instances({(i): kernel_instances_list[i] for i in range(len(kernel_instances_list))})

        # Build the module
        build_module(
            md_name=md_name,
            srcs=[
                f'{AITER_CSRC_DIR}/pybind/gemm_a8w8_blockscale_tune_pybind.cu',
                f'{AITER_CSRC_DIR}/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.cu',
                f'{AITER_CSRC_DIR}/ck_gemm_a8w8_blockscale/include',
            ],
            flags_extra_cc=[],
            flags_extra_hip=[],
            blob_gen_cmd="-c pass", # Dummy command to trigger copy from blob to srcs directory
            extra_include=[],
            extra_ldflags=None,
            verbose=False,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=False,
            hipify=True,
            ck_dir=f"{bd_dir}/{tmp_ck_dir_name}",
        )

        if use_print:
            if len(kernel_instances_list) > 1:
                print(f"Success generated and built {len(kernel_instances_list)} instances in {md_name}")
            else:
                print(f"Success generated and built instance: {kernel_instances_list[0]}")

        return True

    except RuntimeError as e:
        if use_print:
            if len(kernel_instances_list) > 1:
                print(f"Error generating {len(kernel_instances_list)} instances in {md_name}")
            else:
                print(f"Error generating instance {kernel_instances_list[0]}")
        return False

    except Exception as e:
        if len(kernel_instances_list) > 1:
            print(f"Unexpected error generating {len(kernel_instances_list)} instances in {md_name}: {type(e).__name__}")
        else:
            print(f"Unexpected error generating instance {kernel_instances_list[0]}: {type(e).__name__}")
        raise e


def try_compile_kernel_instances_alone(kernel_instances_list):
    """
    Try to compile all kernel instances in the list.
    This function is used to check if the kernel instances can be compiled successfully.
    """
    # Disable aiter logger
    logger = logging.getLogger("aiter")
    logger.disabled = True

    compilable_ids = []

    instance_counter = [0]
    lock_i = threading.Lock()
    lock_compilable_ids = threading.Lock()
    lock_pbar = threading.Lock()

    def worker(tid, instance_counter, pbar):

        # Set ck directory for this thread
        tmp_ck_dir_name = f"ck_{tid}"

        # Main loop to compile kernel instances
        while True:
            with lock_i:
                if instance_counter[0] >= len(kernel_instances_list):
                    break
                i = instance_counter[0]
                instance_counter[0] += 1
            md_name = f"gemm_a8w8_blockscale_single_kernel_instance_{i}"
            compile_ok = try_compile_kernel_instaces(
                kernel_instances_list=kernel_instances_list[i:i + 1],
                md_name=md_name,
                tmp_ck_dir_name=tmp_ck_dir_name,
                use_print=False,  # Disable print in this thread to avoid cluttering output
                # use_print=True,  # Enable print in this thread to see progress
            )
            clear_build(md_name) # Clean up the build directory to release resources
            rm_module(md_name) # Remove the module to release resources
            if compile_ok:
                with lock_compilable_ids:
                    compilable_ids.append(i)
            with lock_pbar:
                pbar.update(1)
        os.system(f"rm -rf {bd_dir}/{tmp_ck_dir_name}")  # Clean up temporary ck directory

    pbar = tqdm(total=len(kernel_instances_list), desc="Searching for compilable kernel instances", unit="instance")
    threads = []
    for tid in range(min(os.cpu_count() // 2, len(kernel_instances_list))):
        t = threading.Thread(target=worker, args=(tid, instance_counter, pbar,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    pbar.close()

    # Enable aiter logger back
    logger.disabled = False

    compilable_ids.sort()
    compilable_kernel_instances_list = [kernel_instances_list[i] for i in compilable_ids]

    print(f"Total {len(kernel_instances_list)} kernel instances found, {len(compilable_kernel_instances_list)} compilable kernel instances found.")
    print("Compilable kernel instances:")
    print(compilable_ids)

    return compilable_kernel_instances_list


def get_kernel_instances_info(kernel_instances_info_file):
    """
    Load the kernel instances info from the file.
    """
    if not os.path.exists(kernel_instances_info_file):
        print(f"Kernel instances info file {kernel_instances_info_file} does not exist.")
        return [], []
    else:
        print(f"Loading kernel instances info from {kernel_instances_info_file}")
    with open(kernel_instances_info_file, "rb") as fp:
        try:
            kernel_instances_list = pickle.load(fp)
            compilable_ids = pickle.load(fp)
        except EOFError:
            print(f"File {kernel_instances_info_file} is empty or corrupted.")
        except Exception as e:
            print(f"Error loading compilable kernel instance ids: {e}")
    compilable_kernel_instances_list = [kernel_instances_list[i] for i in compilable_ids]
    return kernel_instances_list, compilable_kernel_instances_list


def save_kernel_instances_info(kernel_instances_info_file, kernel_instances_list, compilable_kernel_instances_list):
    """
    Save the kernel instances info to the file.
    """
    compilable_ids = [kernel_instances_list.index(kernel) for kernel in compilable_kernel_instances_list]
    compilable_ids.sort()
    with open(kernel_instances_info_file, "wb") as fp:
        pickle.dump(kernel_instances_list, fp)
        pickle.dump(compilable_ids, fp)
    print(f"Kernel instances info saved to {kernel_instances_info_file}")


def get_tune_dict(tune_dict_csv, kernels_list):
    tune_dict = {}
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            tune_df = tune_df[tune_df["cu_num"] == cu_num].reset_index()
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
            tune_dict[(M, N, K)] = kernels_list[kid]
    return tune_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Generate API for CK a8w8 blockscale gemm.",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default=f"{AITER_ROOT_DIR}",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default=f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_tuned_gemm_bruteforce.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
    )

    parser.add_argument(
        "-i",
        "--info_file",
        default=f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_kernel_instances_info.pkl",
        required=False,
        help="info file to store all compilable kernel instances",
    )

    args = parser.parse_args()

    kernel_instances_list = get_all_kernel_instances()
    saved_kernel_instances_list, saved_compilable_kernel_instances_list = get_kernel_instances_info(args.info_file)
    if kernel_instances_list != saved_kernel_instances_list:
        print(f"Base kernel instances list has changed, rechecking kernel instances compibility...")
        shorten_kernel_instances_list = list(set(kernel_instances_list).difference(set(saved_kernel_instances_list)))
        compilable_kernel_instances_list = try_compile_kernel_instances_alone(shorten_kernel_instances_list)
        compilable_kernel_instances_list += list(set(kernel_instances_list).intersection(set(saved_compilable_kernel_instances_list)))
        save_kernel_instances_info(args.info_file, kernel_instances_list, compilable_kernel_instances_list)
    else:
        print(f"Base kernel instances list is the same as saved, using saved compilable ids.")
        compilable_kernel_instances_list = saved_compilable_kernel_instances_list

    kernel_list = {i: compilable_kernel_instances_list[i] for i in range(len(compilable_kernel_instances_list))}

    # Generate the codegen instance
    codegen = gemm_a8w8_blockscale_codegen(args.working_path, args.tune)
    if args.tune:
        codegen.gen_instances(kernel_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file, kernel_list))
