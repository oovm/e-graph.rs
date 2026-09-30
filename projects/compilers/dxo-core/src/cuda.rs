//! CUDA adapter — Titan HAL session + runtime ops with device-resident handles.
//!
//! DXO must not compile kernels. Host transfers are counted so residency verifies
//! can assert continuous device ops do not round-trip through host memory.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::{BackendDriver, DeviceSession};
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor, TensorHandle};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

use crate::tensor::TensorError;

/// Shared Titan CUDA session retained for DXO ops and optional UCF binding.
pub(crate) struct CudaState {
    runtime: Mutex<Runtime>,
    device: Device,
    /// Shared Titan CUDA session (UCF path reuses this primary context).
    pub(crate) session: Arc<dyn DeviceSession>,
}

static CUDA_STATE: OnceLock<Result<CudaState, String>> = OnceLock::new();
static HOST_TRANSFERS: AtomicU64 = AtomicU64::new(0);

fn dxo_source() -> SourceSpan {
    SourceSpan { file: "dxo-core".into(), line: 0, column: 0 }
}

fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(shape[i + 1] as i64);
    }
    strides
}

fn op_request(operator: &str, inputs: Vec<TensorHandle>, output_shape: Vec<usize>, attrs: AttrMap) -> OpRequest {
    let shape_u64: Vec<u64> = output_shape.iter().map(|&d| d as u64).collect();
    OpRequest {
        operator: OperatorId(operator.into()),
        inputs,
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(contiguous_strides(&output_shape)),
            shape: Shape(shape_u64),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: dxo_source(),
    }
}

pub(crate) fn cuda_state() -> Result<&'static CudaState, TensorError> {
    let init = CUDA_STATE.get_or_init(|| {
        let driver = CudaDriver::open().map_err(|e| e.to_string())?;
        let devices = driver.enumerate().map_err(|e| e.to_string())?;
        let fingerprint = devices.first().ok_or_else(|| "no CUDA devices enumerated".to_string())?;
        let session = driver.open(fingerprint.device).map_err(|e| e.to_string())?;
        let device = Device::from_session(session.clone());
        let cache_path = std::env::temp_dir().join("dxo-titan-runtime-cache");
        let runtime = Runtime::open(cache_path);
        Ok(CudaState { runtime: Mutex::new(runtime), device, session })
    });
    init.as_ref().map_err(|msg| {
        TensorError::from_diagnostic(
            crate::diagnostic::Diagnostic::error("DXO_TITAN_BACKEND_UNAVAILABLE", format!("CUDA unavailable: {msg}"))
                .with_arg("requested", "cuda")
                .with_arg("available", "cpu")
                .with_detail("debug", msg.clone())
                .with_backend("cuda")
                .with_operation("open"),
        )
    })
}

/// Whether a Titan CUDA session can be opened on this machine.
pub fn is_available() -> bool {
    cuda_state().is_ok()
}

/// Host↔device transfer count since process start or last [`reset_host_transfer_count`].
pub fn host_transfer_count() -> u64 {
    HOST_TRANSFERS.load(Ordering::Relaxed)
}

/// Reset the residency instrumentation counter (tests only).
pub fn reset_host_transfer_count() {
    HOST_TRANSFERS.store(0, Ordering::Relaxed);
}

fn note_host_transfer() {
    HOST_TRANSFERS.fetch_add(1, Ordering::Relaxed);
}

/// Capability fingerprint string for GPU CI manifests.
pub fn capability_fingerprint() -> Result<String, TensorError> {
    let state = cuda_state()?;
    let fp = state.session.fingerprint();
    Ok(format!(
        "backend={:?} ordinal={} model={} driver={} capability={}",
        fp.device.backend, fp.device.ordinal, fp.model, fp.driver, fp.capability_revision
    ))
}

/// Exercise Titan HAL `wait_event` on the **CUDA** session (upload → compute dependency).
pub fn probe_event_dep() -> Result<(), TensorError> {
    let state = cuda_state()?;
    let session = state.session.clone();
    let upload_stream = session.create_stream().map_err(|e| TensorError::from_hal(e, "cuda"))?;
    let compute_stream = session.create_stream().map_err(|e| TensorError::from_hal(e, "cuda"))?;
    let buf = session.allocate(16, 4).map_err(|e| TensorError::from_hal(e, "cuda"))?;
    let bytes = 1.0f32.to_le_bytes();
    let upload_event =
        session.upload(upload_stream.as_ref(), buf.as_ref(), &bytes).map_err(|e| TensorError::from_hal(e, "cuda"))?;
    note_host_transfer();
    session.wait_event(compute_stream.as_ref(), upload_event.as_ref()).map_err(|e| TensorError::from_hal(e, "cuda"))?;
    session.wait(upload_event.as_ref()).map_err(|e| TensorError::from_hal(e, "cuda"))?;
    Ok(())
}

/// Upload host f32 into an opaque Titan device handle (counts as one host transfer).
pub fn upload_f32(shape: &[usize], data: &[f32]) -> Result<TensorHandle, TensorError> {
    let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).unwrap_or(0);
    if data.len() != expected {
        return Err(TensorError::invalid_shape(format!("CUDA upload length mismatch: got {} expect {}", data.len(), expected)));
    }
    let state = cuda_state()?;
    let handle = TensorHandle::from_f32_vec(state.session.clone(), shape.to_vec(), data).map_err(|e| {
        TensorError::from_diagnostic(
            crate::diagnostic::Diagnostic::error("DXO_TITAN_UPLOAD_FAILED", format!("CUDA upload: {e}"))
                .with_detail("debug", e.to_string())
                .with_backend("cuda")
                .with_operation("upload"),
        )
    })?;
    note_host_transfer();
    Ok(handle)
}

/// Explicit device→host readback (counts as one host transfer).
pub fn download_f32(handle: &TensorHandle) -> Result<Vec<f32>, TensorError> {
    let out = handle.to_vec_f32().map_err(|e| {
        TensorError::from_diagnostic(
            crate::diagnostic::Diagnostic::error("DXO_TITAN_READBACK_FAILED", format!("CUDA readback: {e}"))
                .with_detail("debug", e.to_string())
                .with_backend("cuda")
                .with_operation("readback"),
        )
    })?;
    note_host_transfer();
    Ok(out)
}

fn execute_keeping_device(request: OpRequest) -> Result<TensorHandle, TensorError> {
    let state = cuda_state()?;
    let mut runtime = state.runtime.lock().map_err(|_| TensorError::Device("CUDA runtime lock poisoned".into()))?;
    let finished = runtime
        .execute(request)
        .map_err(|e| TensorError::Device(format!("CUDA dispatch: {e}")))?
        .wait()
        .map_err(|e| TensorError::Device(format!("CUDA completion: {e}")))?;
    finished.outputs.into_iter().next().ok_or_else(|| TensorError::Device("CUDA op returned no outputs".into()))
}

/// Contiguous row-major f32 GEMM on CUDA, **keeping** the result on device.
pub fn gemm_handles(lhs: &TensorHandle, rhs: &TensorHandle, m: usize, n: usize) -> Result<TensorHandle, TensorError> {
    execute_keeping_device(op_request("gemm", vec![lhs.clone(), rhs.clone()], vec![m, n], AttrMap::new()))
}

/// Legacy host-in / host-out GEMM (counts upload×2 + readback). Prefer [`gemm_handles`].
#[allow(dead_code)]
pub fn gemm_f32(lhs: &[f32], m: usize, k: usize, rhs: &[f32], n: usize) -> Result<Vec<f32>, TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::Shape("CUDA gemm requires non-zero M, K, N".into()));
    }
    if lhs.len() != m * k || rhs.len() != k * n {
        return Err(TensorError::Shape(format!(
            "CUDA gemm length mismatch: lhs={} (expect {}), rhs={} (expect {})",
            lhs.len(),
            m * k,
            rhs.len(),
            k * n
        )));
    }
    let a = upload_f32(&[m, k], lhs)?;
    let b = upload_f32(&[k, n], rhs)?;
    let out = gemm_handles(&a, &b, m, n)?;
    download_f32(&out)
}

/// Element-wise add (same shape) on device.
pub fn add_handles(lhs: &TensorHandle, rhs: &TensorHandle, shape: &[usize]) -> Result<TensorHandle, TensorError> {
    execute_keeping_device(op_request("elementwise.add.f32", vec![lhs.clone(), rhs.clone()], shape.to_vec(), AttrMap::new()))
}

/// Broadcast add on device.
pub fn broadcast_add_handles(lhs: &TensorHandle, rhs: &TensorHandle, out_shape: &[usize]) -> Result<TensorHandle, TensorError> {
    execute_keeping_device(op_request("broadcast.add", vec![lhs.clone(), rhs.clone()], out_shape.to_vec(), AttrMap::new()))
}

/// Softmax along last axis on device.
pub fn softmax_handle(input: &TensorHandle, shape: &[usize]) -> Result<TensorHandle, TensorError> {
    let mut attrs = AttrMap::new();
    attrs.insert("axis".into(), AttrValue::Int((shape.len().saturating_sub(1)) as i64));
    execute_keeping_device(op_request("softmax", vec![input.clone()], shape.to_vec(), attrs))
}

/// Conv2d NCHW via Titan (`weight` OIHW). Stride/padding as attrs when supported.
#[allow(dead_code)]
pub fn conv2d_handles(
    input: &TensorHandle,
    weight: &TensorHandle,
    out_shape: &[usize],
    stride: usize,
    padding: usize,
) -> Result<TensorHandle, TensorError> {
    let mut attrs = AttrMap::new();
    attrs.insert("stride".into(), AttrValue::Int(stride as i64));
    attrs.insert("padding".into(), AttrValue::Int(padding as i64));
    execute_keeping_device(op_request("conv2d", vec![input.clone(), weight.clone()], out_shape.to_vec(), attrs))
}

/// Scaled dot-product attention on device (q/k/v same last dims).
#[allow(dead_code)]
pub fn attention_handles(
    q: &TensorHandle,
    k: &TensorHandle,
    v: &TensorHandle,
    out_shape: &[usize],
) -> Result<TensorHandle, TensorError> {
    execute_keeping_device(op_request(
        "scaled_dot_product_attention",
        vec![q.clone(), k.clone(), v.clone()],
        out_shape.to_vec(),
        AttrMap::new(),
    ))
}

/// Upload via typed Titan tensor (used by older gemm path helpers).
#[allow(dead_code)]
pub fn device_ref() -> Result<&'static Device, TensorError> {
    Ok(&cuda_state()?.device)
}

/// Fixed-rank upload helper retained for Titan `Tensor::from_slice` call sites.
#[allow(dead_code)]
pub fn upload_tensor_2d(data: &[f32], rows: usize, cols: usize) -> Result<Tensor<f32, 2>, TensorError> {
    let state = cuda_state()?;
    let t = Tensor::<f32, 2>::from_slice(&state.device, [rows, cols], data)
        .map_err(|e| TensorError::Device(format!("CUDA upload lhs: {e}")))?;
    note_host_transfer();
    Ok(t)
}
