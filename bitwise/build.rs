use std::{path::PathBuf, vec};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_files = vec!["src/bitwise.cu"];
    for lib_file in lib_files.iter() {
        println!("cargo:rerun-if-changed={lib_file}");
    }
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(lib_files)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose");
    let bindings = builder.build_ptx().unwrap();
    let _ = bindings.write("src/kernels.rs");
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
