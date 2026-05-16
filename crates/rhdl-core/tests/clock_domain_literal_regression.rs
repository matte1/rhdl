//! Regression test for the `unify_projected_clocks` literal bailout
//! bug. See the commit "Include Const::Uncolored when collecting clock
//! domains" for the full diagnosis.
//!
//! ## Why a programmatic AST instead of a `#[kernel]` macro test
//!
//! The bug only fires when a *literal slot* (whose clock-domain type is
//! `Const::Uncolored`) flows directly into a Unary opcode that routes
//! through `unify_projected_clocks` -- i.e., `AluUnary::Signed` or
//! `AluUnary::Unsigned` (the lowering of `.as_signed()` /
//! `.as_unsigned()`). Hand-written Rust like `bits::<8>(0).as_signed()`
//! routes the literal through a `bits()` call first, which produces a
//! fresh *register* slot (clock-domain type `Var(_)`), not a literal
//! slot. The bug requires the literal to be the immediate input to the
//! method call -- which only the programmatic AST builder
//! (`expr_typed_bits`) can express.
//!
//! The test builds a kernel equivalent to:
//!
//! ```ignore
//! fn k(a: Signal<Bits<8>, Red>) -> Signal<Bits<8>, Red> {
//!     let a = a.val();
//!     // Literal-typed Signed<8> piped directly into .as_unsigned():
//!     let t = (LIT_SIGNED_0.as_unsigned(), a);
//!     signal(t.1)
//! }
//! ```
//!
//! Without the fix this compile fails with
//! `RHDLClockDomainViolation { cause: UnresolvedClock }`; with the fix
//! it compiles cleanly.

use rhdl_core::CompilationMode;
use rhdl_core::ast::NodeId;
use rhdl_core::ast::builder::{
    block, expr_signal, expr_stmt, expr_typed_bits, field_expr, ident_pat, kernel_fn, local_stmt,
    method_expr, path, path_arguments_none, path_expr, path_segment, tuple_expr, type_pat,
};
use rhdl_core::compiler::driver::compile_kernel_stage1;
use rhdl_core::rhif::spec::Member;
use rhdl_core::types::kind::Kind;
use rhdl_core::{BitX, Color, Meta, MetaDB, SpanLoc, TypedBits};

/// The compiler reads the source file referenced by `kernel_fn`'s
/// `text` argument to render diagnostics. Write a placeholder file
/// once and leak the path so the kernel can name a real readable file.
fn placeholder_source_path() -> &'static str {
    use std::sync::OnceLock;
    static PATH: OnceLock<&'static str> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join("rhdl-core-tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("clock_domain_literal_regression.rs");
        std::fs::write(&path, "// rhdl-core regression test kernel\n").expect("write placeholder");
        Box::leak(
            path.into_os_string()
                .into_string()
                .expect("utf-8 path")
                .into_boxed_str(),
        )
    })
}

/// Fresh-id helper that registers a placeholder span for each NodeId.
/// The compiler walks every NodeId and pulls its span from MetaDB, so
/// missing entries are a hard error -- not a clock-domain bug.
struct Ids {
    next: u32,
    meta_db: MetaDB,
}

impl Ids {
    fn new() -> Self {
        Self {
            next: 0,
            meta_db: MetaDB::new(),
        }
    }
    fn fresh(&mut self) -> NodeId {
        let n = self.next;
        self.next += 1;
        self.meta_db.insert(
            n,
            Meta {
                span: SpanLoc {
                    start_line: 1,
                    start_col: 0,
                    end_line: 1,
                    end_col: 1,
                },
                attributes: vec![],
            },
        );
        NodeId::from(n)
    }
}

/// Build a `TypedBits` of `Signed<8>` value 0 -- the literal that
/// triggers the bug when fed directly to `.as_unsigned()`.
fn signed_8_zero() -> TypedBits {
    TypedBits::new(vec![BitX::Zero; 8], Kind::make_signed(8))
}

#[test]
fn literal_through_method_cast_in_discarded_tuple_element_compiles() {
    let mut ids = Ids::new();
    let red_signal_b8 = Kind::make_signal(Kind::make_bits(8), Color::Red);

    // Input pattern: `a: Signal<Bits<8>, Red>`.
    let arg_name_id = ids.fresh();
    let arg_typed_id = ids.fresh();
    let arg_pat = type_pat(
        arg_typed_id,
        ident_pat(arg_name_id, "a", false),
        red_signal_b8,
    );

    // Statement: `let a = a.val();` -- shadows the Signal-typed binding
    // with its unwrapped Bits<8> value so the body sees a plain `a`.
    let recv_id = ids.fresh();
    let recv = path_expr(
        recv_id,
        path(vec![path_segment("a", path_arguments_none())]),
    );
    let val_id = ids.fresh();
    let val_call = method_expr(val_id, recv, vec![], "val", None);
    let unwrap_pat_id = ids.fresh();
    let unwrap_pat = ident_pat(unwrap_pat_id, "a", false);
    let unwrap_let_id = ids.fresh();
    let unwrap_let = local_stmt(unwrap_let_id, unwrap_pat, Some(val_call));

    // First tuple element: `LIT_SIGNED_0.as_unsigned()`. The literal is
    // a `Signed<8>` whose clock-domain type after import is Uncolored;
    // `.as_unsigned()` lowers to `op_unary(AluUnary::Unsigned)` whose
    // clock-domain check goes through `unify_projected_clocks` -- the
    // buggy path. The result slot is dropped via `t.1`.
    let lit_id = ids.fresh();
    let lit = expr_typed_bits(lit_id, path(vec![]), signed_8_zero(), "");
    let cast_id = ids.fresh();
    let cast = method_expr(cast_id, lit, vec![], "as_unsigned", None);

    // Second tuple element: `a` (the variable).
    let a_ref_id = ids.fresh();
    let a_ref = path_expr(
        a_ref_id,
        path(vec![path_segment("a", path_arguments_none())]),
    );

    // `(cast, a)` then `.1`.
    let tuple_id = ids.fresh();
    let tuple = tuple_expr(tuple_id, vec![cast, a_ref]);
    let proj_id = ids.fresh();
    let proj = field_expr(proj_id, tuple, Member::Unnamed(1));

    // `signal::<_, Red>(proj)` at the kernel boundary.
    let signal_id = ids.fresh();
    let signal_call = expr_signal(signal_id, proj, Some(Color::Red));
    let tail_id = ids.fresh();
    let tail = expr_stmt(tail_id, signal_call);

    let block_id = ids.fresh();
    let body = block(block_id, vec![unwrap_let, tail]);

    let func_id = ids.fresh();
    let kfk = kernel_fn(
        func_id,
        "regression_kernel",
        vec![arg_pat],
        red_signal_b8,
        body,
        std::any::TypeId::of::<()>(),
        Some(placeholder_source_path()),
        ids.meta_db,
        vec![],
    );

    // Stage 1 is enough to exercise the clock-domain check; we don't
    // need to go all the way to RTL for this regression.
    compile_kernel_stage1(kfk, CompilationMode::Asynchronous)
        .expect("kernel should compile cleanly after the Uncolored unify fix");
}
