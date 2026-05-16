//! Multi-kernel programs: a small DAG of kernels where some call others.
//!
//! `MultiKernelProgram` carries an indexed list of `KernelDef`s and a
//! `top` pointer. The DSL is shared with single-kernel programs; the
//! only multi-kernel-specific Expr variant is `Expr::Call { callee,
//! args }` which references a callee by index.
//!
//! ## fn_id pool
//!
//! Each rendered kernel needs a *distinct* `FunctionId` so the
//! compiler's `externals` map keeps them separate. `kernel_fn` builds
//! the FunctionId by hashing a `TypeId`, but at runtime we don't have
//! distinct Rust types to hash. We use a small fixed pool of marker
//! ZSTs (`Slot0` .. `Slot7`) and look them up by index. Caps the
//! number of kernels per program at `KERNEL_POOL_SIZE`.

use crate::dsl::{Expr, NominalRegistry, Stmt, Type};
use crate::render::{CalleeInfo, render_kernel, signal_kind_of};
use rhdl_core::types::digital_fn::DigitalSignature;
use rhdl_core::{Color, KernelFnKind};
use std::any::TypeId;

/// Maximum number of kernels in a single `MultiKernelProgram`.
pub const KERNEL_POOL_SIZE: usize = 8;

/// One kernel inside a multi-kernel program.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Display name (used for the Verilog function name + the
    /// call_expr path). Must come from `KERNEL_NAMES`.
    pub name: &'static str,
    pub arg_types: Vec<Type>,
    pub ret_type: Type,
    pub body: Expr,
}

/// A program comprising one or more kernels. The top kernel is the
/// program's entry point; calls flow from `top` to lower-indexed
/// kernels (see `is_dag_valid`).
#[derive(Debug, Clone)]
pub struct MultiKernelProgram {
    pub kernels: Vec<KernelDef>,
    pub top: usize,
    /// Nominal types shared across all kernels.
    pub nominal: NominalRegistry,
}

impl MultiKernelProgram {
    /// The top (entry-point) kernel.
    pub fn top_kernel(&self) -> &KernelDef {
        &self.kernels[self.top]
    }

    /// Validate the DAG invariant: every `Expr::Call { callee }` must
    /// reference a kernel with a strictly smaller index than the
    /// caller. Guarantees the call graph is acyclic and terminates.
    pub fn is_dag_valid(&self) -> bool {
        fn walk(expr: &Expr, caller_idx: usize) -> bool {
            match expr {
                Expr::Call { callee, args } => {
                    if *callee >= caller_idx {
                        return false;
                    }
                    args.iter().all(|a| walk(a, caller_idx))
                }
                Expr::BinOp(_, l, r) => walk(l, caller_idx) && walk(r, caller_idx),
                Expr::UnOp(_, x) => walk(x, caller_idx),
                Expr::Let { init, body } => walk(init, caller_idx) && walk(body, caller_idx),
                Expr::If {
                    cond: _,
                    then_branch,
                    else_branch,
                } => walk(then_branch, caller_idx) && walk(else_branch, caller_idx),
                Expr::Resize { inner, .. } => walk(inner, caller_idx),
                Expr::Cast { inner, .. } => walk(inner, caller_idx),
                Expr::Shift { value, amount, .. } => {
                    walk(value, caller_idx) && walk(amount, caller_idx)
                }
                Expr::TupleCtor(es) | Expr::ArrayCtor(es) => es.iter().all(|e| walk(e, caller_idx)),
                Expr::ArrayRepeat { value, .. } => walk(value, caller_idx),
                Expr::TupleProj { inner, .. } => walk(inner, caller_idx),
                Expr::ArrayIndex { inner, index } => {
                    walk(inner, caller_idx) && walk(index, caller_idx)
                }
                Expr::StructCtor { fields, .. } => fields.iter().all(|e| walk(e, caller_idx)),
                Expr::FieldAccess { inner, .. } => walk(inner, caller_idx),
                Expr::StructUpdate {
                    base, overrides, ..
                } => walk(base, caller_idx) && overrides.iter().all(|(_, e)| walk(e, caller_idx)),
                Expr::EnumCtor { payload, .. } => payload.iter().all(|e| walk(e, caller_idx)),
                Expr::Match { scrutinee, arms } => {
                    walk(scrutinee, caller_idx) && arms.iter().all(|a| walk(&a.body, caller_idx))
                }
                Expr::Block { stmts, tail } => {
                    stmts.iter().all(|s| match s {
                        Stmt::Let(e) | Stmt::LetMut(e) | Stmt::Assign { value: e, .. } => {
                            walk(e, caller_idx)
                        }
                    }) && walk(tail, caller_idx)
                }
                Expr::For { body, .. } => walk(body, caller_idx),
                Expr::Lit { .. } | Expr::Var(_) => true,
            }
        }
        self.kernels
            .iter()
            .enumerate()
            .all(|(i, def)| walk(&def.body, i))
    }
}

/// Static names for the kernel pool. Indexing by `KernelId` gives a
/// `&'static str` suitable for `kernel_fn`.
pub const KERNEL_NAMES: [&str; KERNEL_POOL_SIZE] = [
    "gen_kernel_0",
    "gen_kernel_1",
    "gen_kernel_2",
    "gen_kernel_3",
    "gen_kernel_4",
    "gen_kernel_5",
    "gen_kernel_6",
    "gen_kernel_7",
];

mod marker {
    pub struct Slot0;
    pub struct Slot1;
    pub struct Slot2;
    pub struct Slot3;
    pub struct Slot4;
    pub struct Slot5;
    pub struct Slot6;
    pub struct Slot7;
}

/// Pool of distinct `TypeId`s, one per kernel slot. Used as the
/// `fn_id` argument to `kernel_fn` so the compiler treats each
/// rendered kernel as a separately-identified function.
pub fn type_id_for(idx: usize) -> TypeId {
    use marker::*;
    match idx {
        0 => TypeId::of::<Slot0>(),
        1 => TypeId::of::<Slot1>(),
        2 => TypeId::of::<Slot2>(),
        3 => TypeId::of::<Slot3>(),
        4 => TypeId::of::<Slot4>(),
        5 => TypeId::of::<Slot5>(),
        6 => TypeId::of::<Slot6>(),
        7 => TypeId::of::<Slot7>(),
        n => panic!("kernel index {n} exceeds pool size {KERNEL_POOL_SIZE}"),
    }
}

/// Build the `DigitalSignature` for a call site referencing this
/// kernel. Both arguments and return are `Signal<T, Red>` because
/// callees, like the top kernel, have Signal-wrapped boundaries
/// (Asynchronous mode requirement).
fn signature_for(def: &KernelDef, nominal: &NominalRegistry) -> DigitalSignature {
    // Multi-kernel callees uniformly use the Red domain at their
    // boundaries; multi-color callees would require pinning each
    // callee's boundary to a chosen color (deferrable).
    DigitalSignature {
        arguments: def
            .arg_types
            .iter()
            .map(|t| signal_kind_of(t, nominal, Color::Red))
            .collect(),
        ret: signal_kind_of(&def.ret_type, nominal, Color::Red),
    }
}

/// Render a complete `MultiKernelProgram` into a single
/// `KernelFnKind` for the top kernel. Each callee is rendered first
/// (with the callees it itself depends on already in scope, by virtue
/// of the strict-DAG invariant) and threaded into the caller's render
/// context via `CalleeInfo`. The compiler then automatically stashes
/// the call graph into `rhif::Object::externals` during stage 1.
pub fn render_multi(program: &MultiKernelProgram) -> KernelFnKind {
    assert!(
        program.is_dag_valid(),
        "multi-kernel program violates the strict-DAG invariant"
    );
    assert!(
        program.kernels.len() <= KERNEL_POOL_SIZE,
        "multi-kernel program has {} kernels but the pool size is {}",
        program.kernels.len(),
        KERNEL_POOL_SIZE
    );

    // Render kernels in index order. Because every Expr::Call references
    // a strictly smaller index, by the time we render kernel `i` all
    // its callees already exist in `rendered`.
    let mut rendered: Vec<CalleeInfo> = Vec::with_capacity(program.kernels.len());
    for (idx, def) in program.kernels.iter().enumerate() {
        let kfk = render_kernel(
            &def.arg_types,
            &def.ret_type,
            &def.body,
            def.name,
            type_id_for(idx),
            rendered.clone(),
            program.nominal.clone(),
            Color::Red,
        );
        rendered.push(CalleeInfo {
            kfk,
            signature: signature_for(def, &program.nominal),
            name: def.name,
        });
    }

    rendered
        .into_iter()
        .nth(program.top)
        .expect("top index out of range")
        .kfk
}
