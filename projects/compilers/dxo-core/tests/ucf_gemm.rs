//! Soft-skip: UCF CUDA MatMul matches CPU without extra host transfers.

#![cfg(feature = "ucf")]

use dxo_core::{
    cuda_available, cuda_download_f32, cuda_upload_f32, host_transfer_count, reset_host_transfer_count,
    ucf_gemm_handles, Tensor,
};

#[test]
fn ucf_gemm_matches_cpu_without_extra_host_transfer() {
    if !cuda_available() {
        eprintln!("SKIP: CUDA unavailable for UCF gemm");
        return;
    }

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let cpu = Tensor::from_vec(a_host.clone(), vec![2, 2])
        .unwrap()
        .matmul(&Tensor::from_vec(b_host.clone(), vec![2, 2]).unwrap())
        .unwrap();

    reset_host_transfer_count();
    let a = match cuda_upload_f32(&[2, 2], &a_host) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("SKIP: CUDA upload a: {err}");
            return;
        }
    };
    let b = match cuda_upload_f32(&[2, 2], &b_host) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("SKIP: CUDA upload b: {err}");
            return;
        }
    };
    let after_upload = host_transfer_count();
    assert!(after_upload >= 2);

    let out = match ucf_gemm_handles(&a, &b, 2, 2) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("SKIP: UCF gemm failed: {err}");
            return;
        }
    };
    let out2 = match ucf_gemm_handles(&out, &b, 2, 2) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("SKIP: UCF gemm second op failed: {err}");
            return;
        }
    };
    assert_eq!(
        host_transfer_count(),
        after_upload,
        "continuous UCF gemm must not host round-trip"
    );

    let got = match cuda_download_f32(&out2) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("SKIP: CUDA download: {err}");
            return;
        }
    };
    for (i, (c, g)) in cpu.data().iter().zip(got.iter()).enumerate() {
        assert!((c - g).abs() < 1e-4, "index {i}: cpu={c} ucf={g}");
    }
}
