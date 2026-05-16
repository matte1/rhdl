//! Smoke test: build a trivial identity kernel programmatically (no `#[kernel]`
//! macro) and verify the full RHDL pipeline (RHIF VM, RTL VM, iverilog) accepts
//! it and produces correct output.

use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::KernelFnKind;
use rhdl_core::Meta;
use rhdl_core::MetaDB;
use rhdl_core::SpanLoc;
use rhdl_core::ast::NodeId;
use rhdl_core::ast::builder::*;
use rhdl_core::compiler::driver::compile_kernel_stage1;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;

/// The compiler insists that `KernelFn::text` point at a real readable file
/// (used to render diagnostic source snippets). For runtime-built kernels we
/// don't have a real source, so write a one-line placeholder once and reuse
/// its leaked path on every call.
fn placeholder_source_path() -> &'static str {
    use std::sync::OnceLock;
    static PATH: OnceLock<&'static str> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join("rhdl-fuzz");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("placeholder.rs");
        std::fs::write(&path, "// rhdl-fuzz runtime kernel\n").expect("write placeholder");
        Box::leak(
            path.into_os_string()
                .into_string()
                .expect("utf-8 path")
                .into_boxed_str(),
        )
    })
}

/// Build `fn id(a: Signal<Bits<8>, Red>) -> Signal<Bits<8>, Red> { a }` as a
/// runtime KernelFnKind.
///
/// The function's NodeId is 0 so that the compiler's "every node below
/// func.id has a span" check loops zero times. Every NodeId we use *also*
/// gets a placeholder MetaDB entry pointing at the start of the placeholder
/// file, so diagnostic span lookups succeed.
fn build_identity_kernel() -> KernelFnKind {
    let mut meta_db = MetaDB::new();
    let placeholder_meta = || Meta {
        span: SpanLoc {
            start_line: 1,
            start_col: 0,
            end_line: 1,
            end_col: 1,
        },
        attributes: vec![],
    };

    let func_id = NodeId::from(0u32);
    meta_db.insert(0, placeholder_meta());

    let mut next: u32 = 1;
    let mut id = || {
        let n = next;
        next += 1;
        meta_db.insert(n, placeholder_meta());
        NodeId::from(n)
    };

    let arg_kind = <Signal<Bits<8>, Red> as Digital>::static_kind();
    let ret_kind = arg_kind;

    let arg_pat = type_pat(id(), ident_pat(id(), "a", false), arg_kind);
    let a_expr = path_expr(id(), path(vec![path_segment("a", path_arguments_none())]));
    let body_block = block(id(), vec![expr_stmt(id(), a_expr)]);

    kernel_fn(
        func_id,
        "id",
        vec![arg_pat],
        ret_kind,
        body_block,
        std::any::TypeId::of::<u8>(),
        Some(placeholder_source_path()),
        meta_db,
        vec![],
    )
}

#[test]
fn smoke_identity_kernel_compiles() {
    let kernel = build_identity_kernel();
    // Just compile to RHIF; if this works the AST is well-formed.
    compile_kernel_stage1(kernel, CompilationMode::Asynchronous)
        .expect("identity kernel should compile to RHIF");
}

#[test]
fn smoke_identity_kernel_round_trips() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = build_identity_kernel();
    // Native reference: literally returns the input unchanged.
    let native = |a: Signal<Bits<8>, Red>| -> Signal<Bits<8>, Red> { a };
    // Run a few hand-picked inputs through the full pipeline.
    let inputs = [0u128, 1, 42, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}
