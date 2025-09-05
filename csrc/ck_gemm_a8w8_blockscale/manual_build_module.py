from aiter.jit.core import build_module, get_args_of_build

args = get_args_of_build("module_gemm_a8w8_blockscale_v2")

build_module(
    md_name="module_gemm_a8w8_blockscale_v2",
    srcs=args["srcs"],
    flags_extra_cc=args["flags_extra_cc"],
    flags_extra_hip=args["flags_extra_hip"],
    blob_gen_cmd=args["blob_gen_cmd"],
    extra_include=args["extra_include"],
    extra_ldflags=args["extra_ldflags"],
    verbose=True,
    is_python_module=args["is_python_module"],
    is_standalone=args["is_standalone"],
    torch_exclude=args["torch_exclude"]
)
