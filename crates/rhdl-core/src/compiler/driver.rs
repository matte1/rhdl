use std::any::type_name;

use crate::{DigitalFn, RHDLError, kernel::KernelFnKind};

use super::stage1::CompilationMode;
use anyhow::anyhow;

pub fn compile_design_stage1<K: DigitalFn>(
    mode: CompilationMode,
) -> Result<crate::rhif::Object, RHDLError> {
    let Some(kernel) = K::kernel_fn() else {
        return Err(anyhow!("Missing kernel function provided for {}", type_name::<K>()).into());
    };
    compile_kernel_stage1(kernel, mode)
}

pub fn compile_design_stage2(
    object: &crate::rhif::Object,
) -> Result<crate::rtl::Object, RHDLError> {
    super::stage2::compile(object)
}

pub fn compile_design<K: DigitalFn>(
    mode: CompilationMode,
) -> Result<crate::rtl::Object, RHDLError> {
    let rhif = compile_design_stage1::<K>(mode)?;
    compile_design_stage2(&rhif)
}

/// Compile a runtime-constructed kernel to RHIF. Used by generators that
/// build the kernel AST programmatically rather than via the `#[kernel]`
/// attribute macro.
pub fn compile_kernel_stage1(
    kernel: KernelFnKind,
    mode: CompilationMode,
) -> Result<crate::rhif::Object, RHDLError> {
    let KernelFnKind::AstKernel(kernel) = kernel else {
        return Err(anyhow!("Kernel must be an AstKernel for compilation").into());
    };
    super::stage1::compile(&kernel, mode)
}

/// Compile a runtime-constructed kernel all the way to RTL.
pub fn compile_kernel(
    kernel: KernelFnKind,
    mode: CompilationMode,
) -> Result<crate::rtl::Object, RHDLError> {
    let rhif = compile_kernel_stage1(kernel, mode)?;
    compile_design_stage2(&rhif)
}
