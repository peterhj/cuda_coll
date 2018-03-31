extern crate bindgen;

use std::env;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  let cuda_dir = PathBuf::from(match env::var("CUDA_HOME") {
    Ok(path) => path,
    Err(_) => "/usr/local/cuda".to_owned(),
  });

  println!("cargo:rustc-link-lib=nccl");

  let nccl_bindings = bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrap.h")
    .whitelist_recursively(false)
    .whitelist_type("ncclResult_t")
    .whitelist_type("ncclComm")
    .whitelist_type("ncclComm_t")
    .whitelist_type("ncclUniqueId")
    .whitelist_type("ncclRedOp_t")
    .whitelist_type("ncclDataType_t")
    .whitelist_function("ncclGetErrorString")
    .whitelist_function("ncclGetUniqueId")
    .whitelist_function("ncclCommInitRank")
    .whitelist_function("ncclCommInitAll")
    .whitelist_function("ncclCommDestroy")
    .whitelist_function("ncclCommCount")
    .whitelist_function("ncclCommCuDevice")
    .whitelist_function("ncclCommUserRank")
    .whitelist_function("ncclReduce")
    .whitelist_function("ncclBcast")
    .whitelist_function("ncclAllReduce")
    .whitelist_function("ncclReduceScatter")
    .whitelist_function("ncclAllGather")
    .whitelist_function("ncclGroupStart")
    .whitelist_function("ncclGroupEnd")
    .generate()
    .expect("bindgen failed to generate nccl bindings");
  nccl_bindings
    .write_to_file(out_dir.join("nccl_bind.rs"))
    .expect("bindgen failed to write nccl bindings");
}
