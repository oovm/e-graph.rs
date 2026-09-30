//! Optional UCF CUDA execution path for device-resident MatMul.
//!
//! Enabled with `--features ucf`. User-facing backend labels stay DXO (`cuda`);
//! UCF is an internal execution substrate.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use titan_backend_cuda::{CudaBuffer, CudaStream};
use titan_hal::{Buffer, Stream};
use titan_tensor::TensorHandle;
use ucf_backend_cuda::{CudaBackend, CudaExecStream, CudaExternalBuffer};
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_runtime::Runtime;
use ucf_scheduler::ExecutionBindings;
use ucf_types::ResourceId;

use crate::cuda::cuda_state;
use crate::tensor::TensorError;

/// Process-wide UCF runtime sharing Titan's primary context CUDA backend.
struct UcfCudaState {
    runtime: Mutex<Runtime>,
}

static UCF_CUDA: OnceLock<Result<UcfCudaState, String>> = OnceLock::new();
static LAST_EVENT_KINDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn ucf_backend() -> Result<&'static UcfCudaState, TensorError> {
    // Ensure Titan has retained the primary context first.
    let _ = cuda_state()?;
    let init = UCF_CUDA.get_or_init(|| {
        let mut runtime = Runtime::new();
        let backend = CudaBackend::new_primary(0).map_err(|e| e.to_string())?;
        runtime.register_backend(Box::new(backend));
        Ok(UcfCudaState {
            runtime: Mutex::new(runtime),
        })
    });
    init.as_ref().map_err(|msg| {
        TensorError::from_diagnostic(
            crate::diagnostic::Diagnostic::error(
                "DXO_UCF_BACKEND_UNAVAILABLE",
                format!("UCF CUDA unavailable: {msg}"),
            )
            .with_arg("requested", "cuda")
            .with_arg("available", "cpu")
            .with_detail("debug", msg.clone())
            .with_detail("executor", "ucf")
            .with_backend("cuda")
            .with_operation("open"),
        )
    })
}

fn record_runtime_diagnostics(rt: &Runtime) {
    let kinds = rt
        .diagnostics()
        .events()
        .iter()
        .map(|e| e.kind.clone())
        .collect();
    if let Ok(mut last) = LAST_EVENT_KINDS.lock() {
        *last = kinds;
    }
}

/// Stable UCF execution event kinds from the most recent [`gemm_handles`] call (inspect hook).
pub fn last_execution_event_kinds() -> Vec<String> {
    LAST_EVENT_KINDS
        .lock()
        .map(|v| v.clone())
        .unwrap_or_default()
}

fn cuda_device_ptr(handle: &TensorHandle, role: &str) -> Result<(u64, u64), TensorError> {
    let buf = handle.buffer().ok_or_else(|| {
        TensorError::Device(format!("UCF gemm {role} handle has no device buffer"))
    })?;
    let cuda = buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| {
        TensorError::Device(format!(
            "UCF gemm {role} buffer is not a Titan CUDA allocation"
        ))
    })?;
    Ok((cuda.device_ptr(), cuda.byte_len() as u64))
}

fn matmul_graph(m: usize, n: usize, k: usize) -> Graph {
    let a_bytes = (m * k * 4) as u64;
    let b_bytes = (k * n * 4) as u64;
    let out_bytes = (m * n * 4) as u64;
    fn buffer(id: u64, bytes: u64) -> ResourceNode {
        ResourceNode {
            id: ResourceId(id),
            kind: ResourceKind::Buffer,
            domain: Domain::Vram,
            access: Access::ReadWrite,
            byte_size: Some(bytes),
        }
    }
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, a_bytes),
                buffer(2, b_bytes),
                buffer(3, out_bytes),
            ],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::MatMul,
                shader: ShaderId(10),
                params: BTreeMap::from([
                    ("a".into(), ParamValue::I64(1)),
                    ("b".into(), ParamValue::I64(2)),
                    ("out".into(), ParamValue::I64(3)),
                    ("m".into(), ParamValue::I64(m as i64)),
                    ("n".into(), ParamValue::I64(n as i64)),
                    ("k".into(), ParamValue::I64(k as i64)),
                ]),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(2)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

/// Contiguous row-major f32 GEMM via UCF CUDA, keeping the result on device.
///
/// Does not increment DXO host-transfer counters (no upload/readback inside).
pub fn gemm_handles(
    lhs: &TensorHandle,
    rhs: &TensorHandle,
    m: usize,
    n: usize,
) -> Result<TensorHandle, TensorError> {
    if m == 0 || n == 0 {
        return Err(TensorError::Shape("UCF gemm requires non-zero M and N".into()));
    }
    let k = lhs.shape().get(1).copied().unwrap_or(0);
    if lhs.shape() != [m, k].as_slice() || rhs.shape() != [k, n].as_slice() {
        return Err(TensorError::Shape(format!(
            "UCF gemm shape mismatch: lhs={:?} rhs={:?} expect [{m},{k}] x [{k},{n}]",
            lhs.shape(),
            rhs.shape()
        )));
    }

    let state = cuda_state()?;
    let (a_ptr, a_bytes) = cuda_device_ptr(lhs, "lhs")?;
    let (b_ptr, b_bytes) = cuda_device_ptr(rhs, "rhs")?;
    let expect_a = (m * k * 4) as u64;
    let expect_b = (k * n * 4) as u64;
    if a_bytes != expect_a || b_bytes != expect_b {
        return Err(TensorError::Shape(format!(
            "UCF gemm byte size mismatch: a={a_bytes} (expect {expect_a}) b={b_bytes} (expect {expect_b})"
        )));
    }
    let out_handle = TensorHandle::allocate_f32(state.session.clone(), vec![m, n]).map_err(|e| {
        TensorError::Device(format!("UCF gemm output allocate: {e}"))
    })?;
    let (out_ptr, out_bytes) = cuda_device_ptr(&out_handle, "out")?;
    let expect_out = (m * n * 4) as u64;
    if out_bytes != expect_out {
        return Err(TensorError::Shape(format!(
            "UCF gemm output bytes {out_bytes} != {expect_out}"
        )));
    }

    let stream = state
        .session
        .create_stream()
        .map_err(|e| TensorError::from_hal(e, "cuda"))?;
    let cuda_stream = Stream::as_any(stream.as_ref())
        .downcast_ref::<CudaStream>()
        .ok_or_else(|| TensorError::Device("UCF gemm stream is not a Titan CUDA stream".into()))?;
    let ucf_stream = CudaExecStream::from_raw(cuda_stream.raw());

    let mut bindings = ExecutionBindings::new();
    bindings.bind_buffer(ResourceId(1), CudaExternalBuffer::from_device_ptr(a_ptr, expect_a));
    bindings.bind_buffer(ResourceId(2), CudaExternalBuffer::from_device_ptr(b_ptr, expect_b));
    bindings.bind_buffer(
        ResourceId(3),
        CudaExternalBuffer::from_device_ptr(out_ptr, expect_out),
    );
    bindings.set_stream(ucf_stream);

    let graph = matmul_graph(m, n, k);
    let ucf = ucf_backend()?;
    let mut runtime = ucf
        .runtime
        .lock()
        .map_err(|_| TensorError::Device("UCF CUDA lock poisoned".into()))?;
    runtime.clear_diagnostics();
    runtime.set_bindings(bindings);
    runtime
        .prepare(&graph)
        .map_err(|e| TensorError::Device(format!("UCF prepare: {e}")))?;
    runtime
        .run_prepared(&graph)
        .map_err(|e| TensorError::Device(format!("UCF run: {e}")))?;
    runtime
        .flush()
        .map_err(|e| TensorError::Device(format!("UCF flush: {e}")))?;
    record_runtime_diagnostics(&runtime);

    Ok(out_handle)
}
