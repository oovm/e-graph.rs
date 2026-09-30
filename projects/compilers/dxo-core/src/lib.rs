//! DXO dxo-core engine — Titan CPU tensor, eager autograd, and CUDA matmul facade.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod autograd;
mod broadcast;
mod conv;
mod cuda;
#[cfg(feature = "ucf")]
mod ucf_exec;
mod diagnostic;
mod dtype;
mod engine;
mod image_buffer;
mod ops;
mod optim;
mod precision;
mod safetensors;
mod shape;
mod storage;
mod tensor;
mod transformer;

pub use autograd::{is_grad_enabled, set_grad_enabled, without_grad};
pub use cuda::{
    capability_fingerprint as cuda_capability_fingerprint, host_transfer_count, is_available as cuda_available,
    reset_host_transfer_count,
};
#[cfg(feature = "ucf")]
pub use ucf_exec::{gemm_handles as ucf_gemm_handles, last_execution_event_kinds as ucf_last_execution_event_kinds};
#[cfg(feature = "ucf")]
pub use cuda::{download_f32 as cuda_download_f32, upload_f32 as cuda_upload_f32};
pub use diagnostic::{Diagnostic, DiagnosticValue, Severity, from_hal_error, from_titan_error, titan_kind_to_code};
pub use dtype::DType;
pub use engine::{backend_label, cpu_session, probe_event_dep, probe_event_dep_cuda};
pub use image_buffer::{AlphaMode, ColorSpace, HostImageBuffer, ImageDtype, ImageLayout};
pub use optim::{AdamState, adam_step, backward_sgd_step, sgd_step, zero_grads};
pub use precision::{PrecisionCapabilities, precision_capabilities, resolve_weight_dtype};
pub use safetensors::{
    SafetensorBufferSlice, SafetensorSlice, decode_safetensors, encode_safetensors, encode_safetensors_buffers,
};
pub use shape::{Shape, Strides, contiguous_strides};
pub use storage::Storage;
pub use tensor::{DeviceKind, Tensor, TensorError};

/// Crate version string (mirrors npm `@dxo/core` semver during preview).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
