# CK gemm a8w8 blockscale tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_blockscale_untuned_gemm.csv`
    |**M**|**N**|**K**|
    |-----|-----|-----|
    |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_blockscale_tune via jit:
`python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv`
You can find the results of the tuning in `aiter/configs/a8w8_blockscale_tuned_gemm.csv`.
    |**cu_num**|**M**|**N**|**K**|**kernelId**|**splitK**|**us**|**kernelName**|
    |----------|-----|-----|-----|------------|----------|------|--------------|
    |80        |128  |1536 |7168 |23          |0         |32.99 |xxxxxxxx      |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a8w8_blockscale.py` and run it, please wait a few minutes as it will build gemm_a8w8_blockscale tuned kernels in `aiter/configs/a8w8_blockscale_tuned_gemm.csv` via jit:
`python3 op_tests/test_gemm_a8w8_blockscale.py`
If you have built gemm_a8w8 kernels brefore tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py`. It will rebuild kernels from `aiter/configs/a8w8_blockscale_tuned_gemm.csv`.

## More
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 kernels in tuned gemm csv by default. If you want to use the new result of gemm_a8w8_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-intall aiter after finishing tune. This can take a lot of time and is not recommended.

---

# Bruteforce
Bruteforce tuning needs `module_gemm_a8w8_blockscale_v2` module. So, you need to follow the steps
below.

1. Build `module_gemm_a8w8_blockscale_v2` : `AITER_GEMM_A8W8_BLOCKSCALE_BF=1 python csrc/ck_gemm_a8w8_blockscale/manual_build_module.py`
2. Run bruteforce tuning : `AITER_GEMM_A8W8_BLOCKSCALE_BF=1 python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune_bruteforce.py`

**Caution) If you have to build bruteforce tuning module, it takes about 4 hours. So we strongly suggest copy the module and build files from our storage.**

Run aiter gemm_a8w8_blockscale API with `AITER_GEMM_A8W8_BLOCKSCALE_BF=1` if you want to use bruteforce-tuned kernel. On the other hand, you can basic tuned kernel by using `AITER_GEMM_A8W8_BLOCKSCALE_BF=0 (default)`.
