//! Render a [`Program`] into a runtime-built RHDL kernel.
//!
//! Wraps the bookkeeping the `#[kernel]` proc macro normally handles:
//! NodeId allocation, MetaDB population (every node needs at least a
//! placeholder span entry, otherwise downstream span lookups panic), and
//! the source file path that the diagnostic infrastructure expects.
//!
//! Generated kernels follow the standard RHDL idiom for Asynchronous mode:
//! arguments and return type are `Signal<T, Red>` for whatever `T` the
//! program declares. The renderer emits a `let a = a.val();` shadowing
//! for each input at the top of the body so the inner expression operates
//! on the unwrapped value, then wraps the final value with `signal(...)`.

use crate::dsl::{
    BinOpKind, BoolExpr, CmpKind, Expr, FIELD_NAMES, LET_NAMES, MatchArm, NominalRegistry, Program,
    ReductionKind, ShiftKind, Stmt, Type, UnOpKind, VAR_NAMES,
};
use rhdl::prelude::*;
use rhdl_core::ast::builder::*;
use rhdl_core::ast::{Expr as AstExpr, NodeId};
use rhdl_core::rhif::spec::Member;
use rhdl_core::types::digital_fn::DigitalSignature;
use rhdl_core::types::kind::{
    DiscriminantAlignment, DiscriminantLayout, DiscriminantType, Field, Variant,
};
use rhdl_core::{Color, KernelFnKind, Meta, MetaDB, SpanLoc, TypedBits};
use std::any::TypeId;

/// Information needed to render a call site for a callee kernel: the
/// callee's already-rendered AST (which the compiler will stash into
/// the parent's `externals`), its function signature (for type
/// inference at the call site), and its name (used for the call's
/// path expression).
#[derive(Clone)]
pub struct CalleeInfo {
    pub kfk: KernelFnKind,
    pub signature: DigitalSignature,
    pub name: &'static str,
}

/// Name used for every generated kernel. The proptest seed is what
/// uniquely identifies a particular failing program.
pub const KERNEL_NAME: &str = "gen_kernel";

/// Convert a [`Type`] into its RHDL [`Kind`] representation.
/// Nominal types (`Struct`, and later `Enum`) require the registry
/// for their definitions.
pub fn kind_of(ty: &Type, nominal: &NominalRegistry) -> Kind {
    match ty {
        Type::Bits(n) => Kind::make_bits(*n as usize),
        Type::Signed(n) => Kind::make_signed(*n as usize),
        Type::Tuple(elems) => {
            let kinds: Vec<Kind> = elems.iter().map(|t| kind_of(t, nominal)).collect();
            Kind::make_tuple(kinds.into_boxed_slice())
        }
        Type::Array(elem, len) => Kind::make_array(kind_of(elem, nominal), *len),
        Type::Struct(idx) => {
            let s = &nominal.structs[*idx];
            let fields: Vec<Field> = s
                .fields
                .iter()
                .map(|(name, ty)| Kind::make_field(name, kind_of(ty, nominal)))
                .collect();
            Kind::make_struct(s.name, fields.into_boxed_slice())
        }
        Type::Enum(idx) => {
            let e = &nominal.enums[*idx];
            let variants: Vec<Variant> = e
                .variants
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // Empty payload -> Kind::Empty (unit variant);
                    // otherwise wrap in a tuple kind so the compiler sees
                    // a positional tuple-struct-like variant.
                    let kind = if v.payload.is_empty() {
                        Kind::Empty
                    } else {
                        let elems: Vec<Kind> =
                            v.payload.iter().map(|t| kind_of(t, nominal)).collect();
                        Kind::make_tuple(elems.into_boxed_slice())
                    };
                    Kind::make_variant(v.name, kind, i as i64)
                })
                .collect();
            let layout = DiscriminantLayout {
                width: e.discriminant_width as usize,
                alignment: DiscriminantAlignment::Msb,
                ty: DiscriminantType::Unsigned,
            };
            Kind::make_enum(e.name, variants, layout)
        }
    }
}

/// `Kind` for the Signal-wrapped boundary form of `ty` in `color`'s
/// clock domain. Used both at the kernel boundary (where `color` is
/// the kernel's boundary domain) and for multi-kernel callee
/// signatures.
pub fn signal_kind_of(ty: &Type, nominal: &NominalRegistry, color: Color) -> Kind {
    Kind::make_signal(kind_of(ty, nominal), color)
}

/// Construct a `TypedBits` value of `ty` from a raw `u128` (or `i128` for
/// signed types). Only valid for scalar types -- compound types are
/// constructed via `TupleCtor` / `ArrayCtor` / `StructCtor`.
pub fn typed_bits_for(value: u128, ty: &Type) -> TypedBits {
    match ty {
        Type::Bits(n) => {
            let n = *n as usize;
            let mut bits = Vec::with_capacity(n);
            let mut v = value;
            for _ in 0..n {
                bits.push((v & 1 != 0).into());
                v >>= 1;
            }
            TypedBits::new(bits, Kind::make_bits(n))
        }
        Type::Signed(n) => {
            let n = *n as usize;
            // Sign-extend to i128 first so `>> 1` walks the bits correctly.
            let mut v = sign_extend_i128(value as i128, n);
            let mut bits = Vec::with_capacity(n);
            for _ in 0..n {
                bits.push((v & 1 != 0).into());
                v >>= 1; // arithmetic shift on i128
            }
            TypedBits::new(bits, Kind::make_signed(n))
        }
        Type::Tuple(_) | Type::Array(_, _) | Type::Struct(_) | Type::Enum(_) => {
            panic!(
                "typed_bits_for: compound types are constructed via *Ctor expressions, not as literals"
            )
        }
    }
}

fn sign_extend_i128(raw: i128, width: usize) -> i128 {
    if width >= 128 {
        raw
    } else {
        let shift = 128 - width as u32;
        (raw << shift) >> shift
    }
}

/// Build the `TypedBits` discriminant value an `arm_kind_enum` arm
/// needs. RHDL stores match-arm discriminants as unsigned bits of the
/// enum's declared discriminant width; we use the variant index as
/// the discriminant value directly.
fn i64_to_discriminant_typed_bits(value: i64, width: u8) -> TypedBits {
    // Going through the existing `typed_bits_for` for `Type::Bits(width)`
    // keeps the bit packing identical to what the enum's `Kind` expects.
    typed_bits_for(value as u128, &Type::Bits(width))
}

/// Convert a [`Program`] into a `KernelFnKind` suitable for
/// [`rhdl_core::compiler::driver::compile_kernel`] /
/// [`rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast`].
pub fn render(program: &Program) -> KernelFnKind {
    render_kernel(
        &program.arg_types,
        &program.ret_type,
        &program.body,
        KERNEL_NAME,
        TypeId::of::<Program>(),
        Vec::new(),
        program.nominal.clone(),
        Color::Red,
    )
}

/// Same as `render`, but lets the caller pick the boundary clock
/// domain. Used by the multi-color domain proptest.
pub fn render_with_color(program: &Program, color: Color) -> KernelFnKind {
    render_kernel(
        &program.arg_types,
        &program.ret_type,
        &program.body,
        KERNEL_NAME,
        TypeId::of::<Program>(),
        Vec::new(),
        program.nominal.clone(),
        color,
    )
}

/// Render a single-arg `Program` as a synchronous-mode kernel.
///
/// The emitted kernel has the shape
/// `fn k(_cr: ClockReset, a: I, _q: ()) -> (O, ())`, where:
///   * `_cr` is the synchronous-mode clock/reset bundle (ignored by
///     the body -- our generator never references it).
///   * `a` is the single data input, named `VAR_NAMES[0]` so the
///     existing body generators can reference it as `Var(0)` without
///     any modification.
///   * `_q` is the read-state input, always unit `()` -- this first
///     cut of sync support exercises the synchronous boundary
///     without introducing actual state elements.
///   * The return type is a 2-tuple `(O, ())` per RHDL synchronous
///     kernel convention: first element is the combinational output,
///     second element is the next-state value (unit here).
///
/// The body inside the tuple is the program's body unchanged.
///
/// Sync kernels do NOT Signal-wrap their inputs (the `Signal` wrap is
/// an asynchronous-mode convention); there is therefore no
/// `let a = a.val();` shadow at the boundary.
pub fn render_sync(program: &Program) -> KernelFnKind {
    assert_eq!(
        program.arg_types.len(),
        1,
        "render_sync v1 supports arity 1 only; got {} args",
        program.arg_types.len()
    );
    let arg_type = &program.arg_types[0];
    let ret_type = &program.ret_type;

    let mut ctx = RenderCtx::new(1, Vec::new(), program.nominal.clone());

    // Build the three input patterns: _cr: ClockReset, a: <arg_type>, _q: ()
    let cr_kind = Kind::make_struct(
        "ClockReset",
        vec![
            Kind::make_field("clock", Kind::Clock),
            Kind::make_field("reset", Kind::Reset),
        ]
        .into_boxed_slice(),
    );
    let cr_ident_id = ctx.fresh();
    let cr_typed_id = ctx.fresh();
    let cr_pat = type_pat(cr_typed_id, ident_pat(cr_ident_id, "_cr", false), cr_kind);

    let arg_ident_id = ctx.fresh();
    let arg_typed_id = ctx.fresh();
    let arg_pat = type_pat(
        arg_typed_id,
        ident_pat(arg_ident_id, VAR_NAMES[0], false),
        kind_of(arg_type, &ctx.nominal),
    );

    let q_ident_id = ctx.fresh();
    let q_typed_id = ctx.fresh();
    let q_pat = type_pat(q_typed_id, ident_pat(q_ident_id, "_q", false), Kind::Empty);

    // Build the body: `(body_inner, ())` as the block's tail expression.
    let body_inner = ctx.expr(&program.body);
    let unit_id = ctx.fresh();
    let unit_expr = expr_typed_bits(
        unit_id,
        path(vec![]),
        TypedBits::new(vec![], Kind::Empty),
        "",
    );
    let tuple_id = ctx.fresh();
    let ret_tuple = tuple_expr(tuple_id, vec![body_inner, unit_expr]);
    let tail_stmt_id = ctx.fresh();
    let tail_stmt = expr_stmt(tail_stmt_id, ret_tuple);

    let block_id = ctx.fresh();
    let body_block = block(block_id, vec![tail_stmt]);

    // Return type: tuple of (ret, unit).
    let ret_kind =
        Kind::make_tuple(vec![kind_of(ret_type, &ctx.nominal), Kind::Empty].into_boxed_slice());

    let func_id = ctx.fresh();
    kernel_fn(
        func_id,
        KERNEL_NAME,
        vec![cr_pat, arg_pat, q_pat],
        ret_kind,
        body_block,
        TypeId::of::<Program>(),
        Some(placeholder_source_path()),
        ctx.meta_db,
        vec![],
    )
}

/// Low-level kernel renderer. Used by `render` (single-kernel) and by
/// `multi::render_multi` (multi-kernel). Each call to `render_kernel`
/// produces a `KernelFnKind` whose `FunctionId` is derived from
/// `fn_id`; multi-kernel callers must pass distinct `TypeId`s per
/// kernel (see `multi::type_id_for`) so the compiler doesn't merge
/// them in its `externals` map.
pub fn render_kernel(
    arg_types: &[Type],
    ret_type: &Type,
    body: &Expr,
    name: &'static str,
    fn_id: TypeId,
    callees: Vec<CalleeInfo>,
    nominal: NominalRegistry,
    boundary_color: Color,
) -> KernelFnKind {
    let arity = arg_types.len() as u8;
    assert!(
        arity >= 1 && arity as usize <= VAR_NAMES.len(),
        "program arity {} out of supported range 1..={}",
        arity,
        VAR_NAMES.len()
    );
    let mut ctx = RenderCtx::new(arity, callees, nominal);
    let ret_signal_kind = signal_kind_of(ret_type, &ctx.nominal, boundary_color);

    // `arg: Signal<T, boundary_color>` for each input.
    let arg_pats: Vec<_> = arg_types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let arg_signal_kind = signal_kind_of(ty, &ctx.nominal, boundary_color);
            let name_id = ctx.fresh();
            let typed_id = ctx.fresh();
            type_pat(
                typed_id,
                ident_pat(name_id, VAR_NAMES[i], false),
                arg_signal_kind,
            )
        })
        .collect();

    // `let a = a.val(); let b = b.val(); ...`
    let mut stmts = Vec::with_capacity(arity as usize + 1);
    for i in 0..arity {
        stmts.push(ctx.unwrap_signal_let(VAR_NAMES[i as usize]));
    }

    // `signal::<_, boundary_color>(<body>)` as the final tail expression.
    let body_inner = ctx.expr(body);
    let signal_call_id = ctx.fresh();
    let signal_call = expr_signal(signal_call_id, body_inner, Some(boundary_color));
    let final_stmt_id = ctx.fresh();
    stmts.push(expr_stmt(final_stmt_id, signal_call));

    let block_id = ctx.fresh();
    let body_block = block(block_id, stmts);

    // Allocate the function's NodeId LAST so the compiler's "every NodeId
    // below func.id has a span" check loops exactly over the nodes we
    // populated above.
    let func_id = ctx.fresh();

    kernel_fn(
        func_id,
        name,
        arg_pats,
        ret_signal_kind,
        body_block,
        fn_id,
        Some(placeholder_source_path()),
        ctx.meta_db,
        vec![],
    )
}

struct RenderCtx {
    next: u32,
    meta_db: MetaDB,
    /// Number of input arguments. Used to distinguish input bindings
    /// (`Var(i)` for `i < arity`) from `let` temporaries.
    arity: u8,
    /// How many `Let` scopes are currently open. Determines the name we
    /// pick for the next `let` we introduce.
    let_depth: u8,
    /// Callees indexable by `Expr::Call::callee`. Empty for
    /// single-kernel rendering -- `Expr::Call` panics in that mode.
    callees: Vec<CalleeInfo>,
    /// Nominal type registry for resolving `Type::Struct` (and later
    /// `Type::Enum`).
    nominal: NominalRegistry,
}

impl RenderCtx {
    fn new(arity: u8, callees: Vec<CalleeInfo>, nominal: NominalRegistry) -> Self {
        Self {
            next: 0,
            meta_db: MetaDB::new(),
            arity,
            let_depth: 0,
            callees,
            nominal,
        }
    }

    /// Allocate a fresh `NodeId` and register a placeholder span for it.
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

    /// Build `let <name> = <name>.val();` -- shadows the Signal-typed
    /// binding with its unwrapped value.
    fn unwrap_signal_let(&mut self, name: &'static str) -> rhdl_core::ast::Stmt {
        let receiver_id = self.fresh();
        let receiver = path_expr(
            receiver_id,
            path(vec![path_segment(name, path_arguments_none())]),
        );
        let method_id = self.fresh();
        let val_call = method_expr(method_id, receiver, vec![], "val", None);
        let pat_ident_id = self.fresh();
        let pat = ident_pat(pat_ident_id, name, false);
        let local_id = self.fresh();
        local_stmt(local_id, pat, Some(val_call))
    }

    fn expr(&mut self, e: &Expr) -> AstExpr {
        match e {
            Expr::Lit { value, ty } => self.lit(*value, ty),
            Expr::Var(i) => self.var(*i),
            Expr::BinOp(op, lhs, rhs) => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                let id = self.fresh();
                binary_expr(id, bin_op_to_ast(*op), l, r)
            }
            Expr::UnOp(op, inner) => {
                let v = self.expr(inner);
                let id = self.fresh();
                unary_expr(id, un_op_to_ast(*op), v)
            }
            Expr::Let { init, body } => self.let_block(init, body),
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => self.if_block(cond, then_branch, else_branch),
            Expr::Resize { inner, to_width } => {
                let receiver = self.expr(inner);
                let id = self.fresh();
                method_expr(id, receiver, vec![], "resize", Some(*to_width as usize))
            }
            Expr::Cast { inner, to } => {
                let receiver = self.expr(inner);
                let method = match to {
                    Type::Signed(_) => "as_signed",
                    Type::Bits(_) => "as_unsigned",
                    other => panic!("Cast to compound type not supported: {:?}", other),
                };
                let id = self.fresh();
                method_expr(id, receiver, vec![], method, None)
            }
            Expr::Shift {
                kind,
                value,
                amount,
            } => {
                let l = self.expr(value);
                let r = self.expr(amount);
                let op = match kind {
                    ShiftKind::Shl => BinOp::Shl,
                    ShiftKind::Shr => BinOp::Shr,
                };
                let id = self.fresh();
                binary_expr(id, op, l, r)
            }
            Expr::TupleCtor(elems) => {
                let rendered: Vec<AstExpr> = elems.iter().map(|e| self.expr(e)).collect();
                let id = self.fresh();
                tuple_expr(id, rendered)
            }
            Expr::TupleProj { inner, index } => {
                let inner_expr = self.expr(inner);
                let id = self.fresh();
                field_expr(id, inner_expr, Member::Unnamed(*index as u32))
            }
            Expr::ArrayCtor(elems) => {
                let rendered: Vec<AstExpr> = elems.iter().map(|e| self.expr(e)).collect();
                let id = self.fresh();
                array_expr(id, rendered)
            }
            Expr::ArrayRepeat { value, len } => {
                let rendered = self.expr(value);
                let id = self.fresh();
                repeat_expr(id, rendered, *len as i64)
            }
            Expr::ArrayIndex { inner, index } => {
                let inner_expr = self.expr(inner);
                // RHDL wants array indices as `ExprLit::Int` for static
                // indexing rather than the typed-bits literal that
                // `Expr::Lit` normally renders to. Special-case literal
                // indices; non-literal indices are passed through.
                let index_expr_val = match index.as_ref() {
                    Expr::Lit { value, .. } => {
                        let id = self.fresh();
                        lit_expr(id, expr_lit_int(&value.to_string()))
                    }
                    other => self.expr(other),
                };
                let id = self.fresh();
                index_expr(id, inner_expr, index_expr_val)
            }
            Expr::Call { callee, args } => self.call(*callee, args),
            Expr::StructCtor { struct_id, fields } => self.struct_ctor(*struct_id, fields),
            Expr::StructUpdate {
                struct_id,
                base,
                overrides,
            } => self.struct_update(*struct_id, base, overrides),
            Expr::EnumCtor {
                enum_id,
                variant_idx,
                payload,
            } => self.enum_ctor(*enum_id, *variant_idx, payload),
            Expr::Match { scrutinee, arms } => self.match_render(scrutinee, arms),
            Expr::Block { stmts, tail } => self.block_render(stmts, tail),
            Expr::For {
                loop_var_width,
                range_hi,
                body,
            } => self.for_render(*loop_var_width, *range_hi, body),
            Expr::FieldAccess { inner, field_idx } => {
                let inner_expr = self.expr(inner);
                // Look up the field name from the registry. We need to
                // know which struct the inner expression belongs to,
                // but we don't track inner expression types in the
                // renderer; we trust the generator to have produced
                // valid programs and emit the field name based on the
                // FIELD_NAMES pool indexed by field_idx (matches the
                // generator's naming convention).
                let id = self.fresh();
                field_expr(id, inner_expr, member_named(FIELD_NAMES[*field_idx]))
            }
        }
    }

    /// Render `StructCtor { struct_id, fields }` as a struct expression
    /// `S { f0: v0, f1: v1, ... }`.
    fn struct_ctor(&mut self, struct_id: usize, field_exprs: &[Expr]) -> AstExpr {
        let struct_def = self.nominal.structs[struct_id].clone();
        let kind = kind_of(&Type::Struct(struct_id), &self.nominal);
        let template = kind.place_holder();

        let path = path(vec![path_segment(struct_def.name, path_arguments_none())]);

        let fvs: Vec<_> = field_exprs
            .iter()
            .enumerate()
            .map(|(i, expr)| {
                let rendered = self.expr(expr);
                let field_name = struct_def.fields[i].0;
                field_value(member_named(field_name), rendered)
            })
            .collect();

        let id = self.fresh();
        struct_expr(id, path, fvs, None, template)
    }

    /// Render `StructUpdate { struct_id, base, overrides }` as
    /// `S { f_overridden: v, ..base }`. Lowers via the same builder
    /// as `struct_ctor` but with the `rest` argument set.
    fn struct_update(
        &mut self,
        struct_id: usize,
        base: &Expr,
        overrides: &[(usize, Expr)],
    ) -> AstExpr {
        let struct_def = self.nominal.structs[struct_id].clone();
        let kind = kind_of(&Type::Struct(struct_id), &self.nominal);
        let template = kind.place_holder();
        let path = path(vec![path_segment(struct_def.name, path_arguments_none())]);

        let base_rendered = self.expr(base);
        let fvs: Vec<_> = overrides
            .iter()
            .map(|(field_idx, expr)| {
                let rendered = self.expr(expr);
                let field_name = struct_def.fields[*field_idx].0;
                field_value(member_named(field_name), rendered)
            })
            .collect();

        let id = self.fresh();
        struct_expr(id, path, fvs, Some(base_rendered), template)
    }

    /// Render `EnumCtor { enum_id, variant_idx, payload }` as a call
    /// expression `EnumName::VariantName(p0, p1, ...)`. The compiler
    /// dispatches on the `EnumTupleStructConstructor` `KernelFnKind`
    /// code field to lower this into an `op_enum` RHIF op.
    fn enum_ctor(&mut self, enum_id: usize, variant_idx: usize, payload_exprs: &[Expr]) -> AstExpr {
        let enum_def = self.nominal.enums[enum_id].clone();
        let variant = &enum_def.variants[variant_idx];
        let enum_kind = kind_of(&Type::Enum(enum_id), &self.nominal);
        let template = enum_kind
            .enum_template(variant.name)
            .expect("enum_template lookup for declared variant");

        let payload_kinds: Vec<Kind> = variant
            .payload
            .iter()
            .map(|t| kind_of(t, &self.nominal))
            .collect();
        let signature = DigitalSignature {
            arguments: payload_kinds,
            ret: enum_kind,
        };

        let rendered_args: Vec<AstExpr> = payload_exprs.iter().map(|e| self.expr(e)).collect();
        let id = self.fresh();
        call_expr(
            id,
            path(vec![
                path_segment(enum_def.name, path_arguments_none()),
                path_segment(variant.name, path_arguments_none()),
            ]),
            rendered_args,
            signature,
            Some(KernelFnKind::EnumTupleStructConstructor(template)),
        )
    }

    /// Render `Match { scrutinee, arms }` as a match expression with
    /// one arm per variant. Payload-binding arms push positional
    /// bindings onto the let-name stack (same naming pool as `Let`),
    /// so the body's `Var(i)` references resolve to them.
    fn match_render(&mut self, scrutinee: &Expr, arms: &[MatchArm]) -> AstExpr {
        let scrut_rendered = self.expr(scrutinee);

        let rendered_arms: Vec<_> = arms
            .iter()
            .map(|m_arm| {
                let enum_def = self.nominal.enums[m_arm.enum_id].clone();
                let variant = &enum_def.variants[m_arm.variant_idx];
                let n_payload = variant.payload.len();

                // Bind payload positions as `tN` idents (or a wildcard
                // for unit variants). Allocate names BEFORE rendering the
                // body so the body's Var(i) references find them.
                let pat = if n_payload == 0 {
                    let pat_id = self.fresh();
                    wild_pat(pat_id)
                } else {
                    let elems: Vec<_> = (0..n_payload)
                        .map(|i| {
                            let id = self.fresh();
                            let name = LET_NAMES[self.let_depth as usize + i];
                            ident_pat(id, name, false)
                        })
                        .collect();
                    let pat_id = self.fresh();
                    tuple_struct_pat(
                        pat_id,
                        path(vec![
                            path_segment(enum_def.name, path_arguments_none()),
                            path_segment(variant.name, path_arguments_none()),
                        ]),
                        elems,
                    )
                };

                // Push payload bindings into scope for the body.
                self.let_depth += n_payload as u8;
                let body = self.expr(&m_arm.body);
                self.let_depth -= n_payload as u8;

                let discriminant = i64_to_discriminant_typed_bits(
                    m_arm.variant_idx as i64,
                    enum_def.discriminant_width,
                );

                let arm_id = self.fresh();
                arm(arm_id, arm_kind_enum(pat, discriminant), body)
            })
            .collect();

        let id = self.fresh();
        match_expr(id, scrut_rendered, rendered_arms)
    }

    /// Render `Block { stmts, tail }` as a Rust block expression
    /// `{ stmt0; stmt1; ...; tail }`. Each `Let`/`LetMut` extends the
    /// renderer's binding stack for the duration of the block, mirroring
    /// the interpreter's binding-stack discipline.
    fn block_render(&mut self, stmts: &[Stmt], tail: &Expr) -> AstExpr {
        let start_depth = self.let_depth;
        let mut rendered_stmts: Vec<rhdl_core::ast::Stmt> = Vec::with_capacity(stmts.len() + 1);
        for stmt in stmts {
            match stmt {
                Stmt::Let(init) | Stmt::LetMut(init) => {
                    let init_expr = self.expr(init);
                    let name = LET_NAMES[self.let_depth as usize];
                    self.let_depth += 1;
                    let pat_id = self.fresh();
                    let mutable = matches!(stmt, Stmt::LetMut(_));
                    let pat = ident_pat(pat_id, name, mutable);
                    let local_id = self.fresh();
                    rendered_stmts.push(local_stmt(local_id, pat, Some(init_expr)));
                }
                Stmt::Assign { var_idx, value } => {
                    let rhs = self.expr(value);
                    let lhs_id = self.fresh();
                    let name = self.var_name(*var_idx);
                    let lhs = path_expr(
                        lhs_id,
                        path(vec![path_segment(name, path_arguments_none())]),
                    );
                    let assign_id = self.fresh();
                    let stmt_id = self.fresh();
                    rendered_stmts.push(semi_stmt(stmt_id, assign_expr(assign_id, lhs, rhs)));
                }
            }
        }
        let tail_expr = self.expr(tail);
        let tail_stmt_id = self.fresh();
        rendered_stmts.push(expr_stmt(tail_stmt_id, tail_expr));

        // Pop any bindings introduced by the block so subsequent
        // siblings see the original depth.
        self.let_depth = start_depth;

        let block_id = self.fresh();
        let blk = block(block_id, rendered_stmts);
        let block_expr_id = self.fresh();
        block_expr(block_expr_id, blk)
    }

    /// Render `For { loop_var_width, range_hi, body }` as
    /// `for tN in (0..range_hi) { let tN = signed::<32>(tN); body }`.
    ///
    /// The loop variable is given the next `LET_NAMES` slot. RHDL
    /// requires the for-loop pattern to be a bare `Ident`, not a typed
    /// pat, so the outer `tN` is whatever type RHDL infers from the
    /// range expression (i.e., an unsized literal int -- context-
    /// dependent and ambiguous in expressions that mix multiple
    /// constraints, e.g., `t << bits::<32>(_)` triggers a `(s128, b1)
    /// vs (b32, b1)` `MismatchInDataTypes` ICE).
    ///
    /// We sidestep that ambiguity by **shadowing the loop variable
    /// with an explicit `signed::<32>(tN)` cast at the top of the
    /// body**. Every subsequent reference to `tN` in the body
    /// resolves to the shadow, which RHDL types unambiguously as
    /// `Signed<32>`. This matches our DSL's `LOOP_VAR_TYPE` and our
    /// interpreter's `Value::Signed(_, 32)` push, so the pipeline and
    /// reference agree.
    ///
    /// `loop_var_width` is currently always `LOOP_VAR_WIDTH` (32);
    /// kept as a DSL field for future flexibility but not consumed
    /// here -- the shadow hardcodes the width to 32.
    fn for_render(&mut self, loop_var_width: u8, range_hi: u32, body: &Expr) -> AstExpr {
        let _ = loop_var_width;
        let name = LET_NAMES[self.let_depth as usize];
        let ident_id = self.fresh();
        let pat = ident_pat(ident_id, name, false);

        // Range `0..range_hi`.
        let lo_id = self.fresh();
        let lo = lit_expr(lo_id, expr_lit_int("0"));
        let hi_id = self.fresh();
        let hi = lit_expr(hi_id, expr_lit_int(&range_hi.to_string()));
        let range_id = self.fresh();
        let range = range_expr(range_id, Some(lo), range_limits_half_open(), Some(hi));

        // Shadow: `let tN = signed::<32>(tN);`
        let shadow_recv_id = self.fresh();
        let shadow_recv = path_expr(
            shadow_recv_id,
            path(vec![path_segment(name, path_arguments_none())]),
        );
        let shadow_rhs_id = self.fresh();
        let shadow_rhs = expr_signed_with_length(shadow_rhs_id, shadow_recv, 32);
        let shadow_pat_id = self.fresh();
        let shadow_pat = ident_pat(shadow_pat_id, name, false);
        let shadow_stmt_id = self.fresh();
        let shadow_stmt = local_stmt(shadow_stmt_id, shadow_pat, Some(shadow_rhs));

        // Body: render with the loop variable in scope.
        self.let_depth += 1;
        let body_inner = self.expr(body);
        self.let_depth -= 1;
        let body_stmt_id = self.fresh();
        let body_block_id = self.fresh();
        let body_block = block(
            body_block_id,
            vec![shadow_stmt, expr_stmt(body_stmt_id, body_inner)],
        );

        let for_id = self.fresh();
        for_expr(for_id, pat, range, body_block)
    }

    /// Render a kernel-to-kernel call. The callee's interface is
    /// `Signal<T, Red>`-typed (Asynchronous mode requirement), but the
    /// surrounding body of OUR kernel operates on unwrapped values, so
    /// we wrap each argument with `signal(...)` and unwrap the result
    /// with `.val()`.
    fn call(&mut self, callee_idx: usize, args: &[Expr]) -> AstExpr {
        let callee = self
            .callees
            .get(callee_idx)
            .unwrap_or_else(|| {
                panic!(
                    "Expr::Call references callee {} but only {} callees are in scope \
                     (single-kernel mode does not support Expr::Call)",
                    callee_idx,
                    self.callees.len()
                )
            })
            .clone();

        // Wrap each unwrapped argument back into a Signal<T, Red>.
        let wrapped_args: Vec<AstExpr> = args
            .iter()
            .map(|arg| {
                let inner = self.expr(arg);
                let id = self.fresh();
                expr_signal(id, inner, Some(<Red as Domain>::color()))
            })
            .collect();

        // Build the call: `callee(wrapped_args...)` returns Signal<U, Red>.
        let call_id = self.fresh();
        let call = call_expr(
            call_id,
            path(vec![path_segment(callee.name, path_arguments_none())]),
            wrapped_args,
            callee.signature,
            Some(callee.kfk),
        );

        // Unwrap the Signal-wrapped result with `.val()`.
        let val_id = self.fresh();
        method_expr(val_id, call, vec![], "val", None)
    }

    /// Render `Let { init, body }` as the block expression
    /// `{ let tN = init; body }`.
    fn let_block(&mut self, init: &Expr, body: &Expr) -> AstExpr {
        let init_expr = self.expr(init);
        let name = LET_NAMES[self.let_depth as usize];
        self.let_depth += 1;
        let body_expr = self.expr(body);
        self.let_depth -= 1;

        let pat_id = self.fresh();
        let pat = ident_pat(pat_id, name, false);
        let local_id = self.fresh();
        let let_stmt = local_stmt(local_id, pat, Some(init_expr));
        let tail_id = self.fresh();
        let tail_stmt = expr_stmt(tail_id, body_expr);

        let block_id = self.fresh();
        let inner = block(block_id, vec![let_stmt, tail_stmt]);
        let block_expr_id = self.fresh();
        block_expr(block_expr_id, inner)
    }

    /// Render `if cond { then } else { else }` as an `if_expr`.
    fn if_block(&mut self, cond: &BoolExpr, then_branch: &Expr, else_branch: &Expr) -> AstExpr {
        let cond_expr = self.bool_expr(cond);
        let then_expr = self.expr(then_branch);
        let else_expr = self.expr(else_branch);

        let then_stmt_id = self.fresh();
        let then_block_id = self.fresh();
        let then_block = block(then_block_id, vec![expr_stmt(then_stmt_id, then_expr)]);

        let id = self.fresh();
        if_expr(id, cond_expr, then_block, Some(else_expr))
    }

    /// Render a `BoolExpr` into the corresponding RHDL AST node.
    fn bool_expr(&mut self, b: &BoolExpr) -> AstExpr {
        match b {
            BoolExpr::Lit(v) => {
                let id = self.fresh();
                lit_expr(id, expr_lit_bool(*v))
            }
            BoolExpr::Not(x) => {
                let inner = self.bool_expr(x);
                let id = self.fresh();
                unary_expr(id, UnOp::Not, inner)
            }
            BoolExpr::And(l, r) => {
                let l = self.bool_expr(l);
                let r = self.bool_expr(r);
                let id = self.fresh();
                binary_expr(id, BinOp::And, l, r)
            }
            BoolExpr::Or(l, r) => {
                let l = self.bool_expr(l);
                let r = self.bool_expr(r);
                let id = self.fresh();
                binary_expr(id, BinOp::Or, l, r)
            }
            BoolExpr::Cmp(op, lhs, rhs) => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                let id = self.fresh();
                binary_expr(id, cmp_to_ast(*op), l, r)
            }
            BoolExpr::Reduction(kind, inner) => {
                let receiver = self.expr(inner);
                let method = match kind {
                    ReductionKind::Any => "any",
                    ReductionKind::All => "all",
                    ReductionKind::Xor => "xor",
                };
                let id = self.fresh();
                method_expr(id, receiver, vec![], method, None)
            }
        }
    }

    fn lit(&mut self, value: u128, ty: &Type) -> AstExpr {
        let tb = typed_bits_for(value, ty);
        let id = self.fresh();
        expr_typed_bits(id, path(vec![]), tb, "")
    }

    fn var(&mut self, i: u8) -> AstExpr {
        let name = self.var_name(i);
        let id = self.fresh();
        path_expr(id, path(vec![path_segment(name, path_arguments_none())]))
    }

    /// Look up the binding name for `Var(i)` without emitting an AST
    /// expression. Used by `block_render` to build the LHS of `Assign`.
    fn var_name(&self, i: u8) -> &'static str {
        if (i as usize) < VAR_NAMES.len() && i < self.arity {
            VAR_NAMES[i as usize]
        } else {
            LET_NAMES[(i - self.arity) as usize]
        }
    }
}

fn bin_op_to_ast(op: BinOpKind) -> BinOp {
    match op {
        BinOpKind::Add => BinOp::Add,
        BinOpKind::Sub => BinOp::Sub,
        BinOpKind::Mul => BinOp::Mul,
        BinOpKind::BitAnd => BinOp::BitAnd,
        BinOpKind::BitOr => BinOp::BitOr,
        BinOpKind::BitXor => BinOp::BitXor,
    }
}

fn un_op_to_ast(op: UnOpKind) -> UnOp {
    match op {
        UnOpKind::Not => UnOp::Not,
        UnOpKind::Neg => UnOp::Neg,
    }
}

fn cmp_to_ast(op: CmpKind) -> BinOp {
    match op {
        CmpKind::Eq => BinOp::Eq,
        CmpKind::Ne => BinOp::Ne,
        CmpKind::Lt => BinOp::Lt,
        CmpKind::Le => BinOp::Le,
        CmpKind::Gt => BinOp::Gt,
        CmpKind::Ge => BinOp::Ge,
    }
}

/// Path of a placeholder source file on disk. The compiler reads from
/// this path to render diagnostic snippets; for runtime kernels we never
/// produce diagnostics that would surface real source, but the read still
/// has to succeed. Created lazily on first use.
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
