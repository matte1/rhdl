//! Proptest [`Strategy`]s for generating random DSL programs.
//!
//! After M0 the generator is *type-directed*: dispatch on a `target: Type`
//! and produce expressions that evaluate to that type. The `TypingCtx`
//! tracks the types of in-scope bindings so `Var` references stay in
//! range and so the body of a `Let` can reference one more binding than
//! its `init`.
//!
//! Today only `Type::Bits(8)` is reachable; subsequent milestones add
//! variants to `Type` and corresponding arms in `gen_expr_of_type`.

use crate::coverage;
use crate::dsl::{
    BinOpKind, BoolExpr, CmpKind, ENUM_NAMES, EnumDef, Expr, FIELD_NAMES, MatchArm,
    NominalRegistry, Program, ReductionKind, STRUCT_NAMES, ShiftKind, Stmt, StructDef, Type,
    UnOpKind, VARIANT_NAMES, VariantDef,
};
use crate::multi::{KERNEL_NAMES, KERNEL_POOL_SIZE, KernelDef, MultiKernelProgram};
use proptest::prelude::*;
use std::sync::atomic::Ordering;

/// Maximum recursion depth for a generated `Expr` tree.
pub const DEFAULT_MAX_DEPTH: u32 = 4;

/// Generate literal `u128` values biased toward boundary patterns. The
/// returned values fit in `width` bits (always `& mask`).
///
/// Heavily weighted toward 0, 1, all-ones, and the sign-bit boundary
/// (which corresponds to the smallest negative for signed types). Under
/// uniform random over `u128 & mask` these patterns are vanishingly
/// rare for wide types -- the bias gives ~60% boundary values vs ~5%
/// uniform, which roughly 10x's the rate at which boundary-related
/// codegen bugs are surfaced.
pub fn lit_value_strategy(width: u8) -> BoxedStrategy<u128> {
    debug_assert!(width >= 1 && width <= 128);
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let sign_bit = if width >= 128 {
        1u128 << 127
    } else {
        1u128 << (width - 1)
    };
    let max_signed = sign_bit.saturating_sub(1);
    let max_minus_one = mask.saturating_sub(1);
    prop_oneof![
        3 => Just(0u128),
        3 => Just(1u128 & mask),
        3 => Just(mask),               // all-ones; -1 if interpreted signed
        2 => Just(max_minus_one),      // mask - 1
        2 => Just(sign_bit),           // MIN_SIGNED if interpreted signed
        2 => Just(max_signed),         // MAX_SIGNED if interpreted signed
        // Single-bit-set patterns (powers of two within range).
        1 => (0u8..width).prop_map(move |b| 1u128 << b),
        // Uniform random tail for the long tail of values.
        4 => any::<u128>().prop_map(move |v| v & mask),
    ]
    .boxed()
}

/// Widths the generator may use for `Type::Bits(_)`. Powers of two plus
/// 1 cover the high-leverage cases (single bit, byte, common
/// machine-word multiples). Bounded so the iverilog cost per case stays
/// manageable.
pub const WIDTH_POOL: &[u8] = &[1, 2, 4, 8, 16, 32, 64, 128];

/// Width and signedness assigned to the for-loop induction variable,
/// both in the generated DSL (`Expr::For::loop_var_width`) and the
/// body's `TypingCtx` binding.
///
/// RHDL compile-time-unrolls `for i in 0..N` and inlines `i` as a
/// literal int coerced to `i32`
/// (`crates/rhdl-core/src/compiler/mir/compiler.rs:1001-1002`, via
/// `coerce_literal_to_i32`). Inference then unifies that literal
/// against its use context; in the absence of an explicit cast, the
/// inferred type is **signed** -- so e.g. `(!i).resize::<W>()`
/// sign-extends rather than zero-extends.
///
/// Typing the loop var as `Signed<32>` matches that behavior: our
/// generator only emits `Var(loop_var)` in `Signed<32>`-typed
/// contexts, and any use in a `Bits<W>` context goes through an
/// explicit `Cast(Resize(Var, W), Bits(W))` chain that renders to
/// `t.resize::<W>().as_unsigned()` -- which RHDL accepts and types
/// consistently.
///
/// Previously typed as `Bits<32>` (commit a4bbcdd8); a 10k stress
/// run exposed the signedness drift (commit d196715e log).
pub const LOOP_VAR_WIDTH: u8 = 32;
pub const LOOP_VAR_TYPE: Type = Type::Signed(LOOP_VAR_WIDTH);

/// Signature of a callee kernel as seen from a call site.
#[derive(Debug, Clone)]
pub struct CalleeSig {
    pub arg_types: Vec<Type>,
    pub ret_type: Type,
}

/// In-scope bindings + callees that can be invoked from this scope.
#[derive(Debug, Clone)]
pub struct TypingCtx {
    pub bindings: Vec<Type>,
    /// Callees indexable by `Expr::Call::callee`. Empty for
    /// single-kernel programs.
    pub callees: Vec<CalleeSig>,
    /// Nominal types in scope. Lets the body generator emit
    /// StructCtor / FieldAccess / EnumCtor / Match arms that reference
    /// the surrounding program's structs and enums.
    pub nominal: NominalRegistry,
}

impl TypingCtx {
    pub fn new() -> Self {
        TypingCtx {
            bindings: vec![],
            callees: vec![],
            nominal: NominalRegistry::new(),
        }
    }

    pub fn from_args(arg_types: &[Type]) -> Self {
        TypingCtx {
            bindings: arg_types.to_vec(),
            callees: vec![],
            nominal: NominalRegistry::new(),
        }
    }

    pub fn from_args_with_callees(arg_types: &[Type], callees: Vec<CalleeSig>) -> Self {
        TypingCtx {
            bindings: arg_types.to_vec(),
            callees,
            nominal: NominalRegistry::new(),
        }
    }

    pub fn from_args_with_nominal(arg_types: &[Type], nominal: NominalRegistry) -> Self {
        TypingCtx {
            bindings: arg_types.to_vec(),
            callees: vec![],
            nominal,
        }
    }

    /// Indices of bindings whose type matches `ty`.
    pub fn bindings_of(&self, ty: &Type) -> Vec<u8> {
        self.bindings
            .iter()
            .enumerate()
            .filter(|(_, t)| *t == ty)
            .map(|(i, _)| i as u8)
            .collect()
    }

    /// Indices of callees whose return type matches `ty`.
    pub fn callees_returning(&self, ty: &Type) -> Vec<usize> {
        self.callees
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ret_type == *ty)
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for TypingCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate any of the supported binary operators with equal weight.
pub fn bin_op_kind() -> impl Strategy<Value = BinOpKind> {
    prop_oneof![
        Just(BinOpKind::Add),
        Just(BinOpKind::Sub),
        Just(BinOpKind::Mul),
        Just(BinOpKind::BitAnd),
        Just(BinOpKind::BitOr),
        Just(BinOpKind::BitXor),
    ]
}

/// Generate any of the supported unary operators valid for `target`.
/// `Neg` is only emitted for signed targets.
pub fn un_op_kind_for(target: &Type) -> BoxedStrategy<UnOpKind> {
    if target.is_signed() {
        prop_oneof![Just(UnOpKind::Not), Just(UnOpKind::Neg)].boxed()
    } else {
        Just(UnOpKind::Not).boxed()
    }
}

/// Generate either shift direction with equal weight.
pub fn shift_kind() -> impl Strategy<Value = ShiftKind> {
    prop_oneof![Just(ShiftKind::Shl), Just(ShiftKind::Shr)]
}

/// Generate any of the supported comparison operators with equal weight.
pub fn cmp_kind() -> impl Strategy<Value = CmpKind> {
    prop_oneof![
        Just(CmpKind::Eq),
        Just(CmpKind::Ne),
        Just(CmpKind::Lt),
        Just(CmpKind::Le),
        Just(CmpKind::Gt),
        Just(CmpKind::Ge),
    ]
}

/// Generate an expression that evaluates to `target` in a scope where
/// `ctx.bindings` are visible. `depth` bounds recursion.
pub fn gen_expr_of_type(target: Type, ctx: TypingCtx, depth: u32) -> BoxedStrategy<Expr> {
    match target {
        Type::Bits(_) | Type::Signed(_) => gen_scalar_expr(target, ctx, depth),
        Type::Tuple(_) => gen_tuple_expr(target, ctx, depth),
        Type::Array(_, _) => gen_array_expr(target, ctx, depth),
        Type::Struct(_) | Type::Enum(_) => {
            // Nominal-type return values are not generated at the
            // program-return-type level in v1; they appear only inside
            // hand-rolled sanity tests via ctors and projections /
            // matches that produce scalar results.
            unreachable!(
                "gen_expr_of_type called with nominal target {:?} — \
                 not produced by current program-level strategy",
                target
            )
        }
    }
}

/// Generate an expression of a scalar (`Bits` or `Signed`) type. The
/// strategy is the same for both modulo (a) `UnOpKind::Neg` is only
/// valid on signed and (b) `Cast` flips signedness.
fn gen_scalar_expr(target: Type, ctx: TypingCtx, depth: u32) -> BoxedStrategy<Expr> {
    let leaves = scalar_leaves(target.clone(), ctx.clone()).boxed();
    if depth == 0 {
        return leaves;
    }
    let next = depth - 1;

    let bin = (
        bin_op_kind(),
        gen_expr_of_type(target.clone(), ctx.clone(), next),
        gen_expr_of_type(target.clone(), ctx.clone(), next),
    )
        .prop_map(|(op, l, r)| {
            coverage::BINOP.fetch_add(1, Ordering::Relaxed);
            Expr::BinOp(op, Box::new(l), Box::new(r))
        })
        .boxed();

    let un = (
        un_op_kind_for(&target),
        gen_expr_of_type(target.clone(), ctx.clone(), next),
    )
        .prop_map(|(op, x)| {
            match op {
                UnOpKind::Not => coverage::UNOP_NOT.fetch_add(1, Ordering::Relaxed),
                UnOpKind::Neg => coverage::UNOP_NEG.fetch_add(1, Ordering::Relaxed),
            };
            Expr::UnOp(op, Box::new(x))
        })
        .boxed();

    let lett = {
        let ctx_inner = {
            let mut c = ctx.clone();
            c.bindings.push(target.clone());
            c
        };
        (
            gen_expr_of_type(target.clone(), ctx.clone(), next),
            gen_expr_of_type(target.clone(), ctx_inner, next),
        )
            .prop_map(|(init, body)| {
                coverage::LET.fetch_add(1, Ordering::Relaxed);
                Expr::Let {
                    init: Box::new(init),
                    body: Box::new(body),
                }
            })
            .boxed()
    };

    let if_ = (
        bool_expr_strategy(ctx.clone(), next),
        gen_expr_of_type(target.clone(), ctx.clone(), next),
        gen_expr_of_type(target.clone(), ctx.clone(), next),
    )
        .prop_map(|(c, t, e)| {
            coverage::IF.fetch_add(1, Ordering::Relaxed);
            Expr::If {
                cond: Box::new(c),
                then_branch: Box::new(t),
                else_branch: Box::new(e),
            }
        })
        .boxed();

    // Resize bridges any width to the target width while preserving
    // signedness.
    let resize = {
        let target_w = target.width();
        let target_signed = target.is_signed();
        let ctx_resize = ctx.clone();
        proptest::sample::select(WIDTH_POOL)
            .prop_flat_map(move |inner_w| {
                let inner_ty = if target_signed {
                    Type::Signed(inner_w)
                } else {
                    Type::Bits(inner_w)
                };
                gen_expr_of_type(inner_ty, ctx_resize.clone(), next)
            })
            .prop_map(move |inner| {
                coverage::RESIZE.fetch_add(1, Ordering::Relaxed);
                Expr::Resize {
                    inner: Box::new(inner),
                    to_width: target_w,
                }
            })
            .boxed()
    };

    // Cast: flip signedness at the target's width.
    let cast = {
        let target_for_cast = target.clone();
        let inner_ty = if target.is_signed() {
            Type::Bits(target.width())
        } else {
            Type::Signed(target.width())
        };
        gen_expr_of_type(inner_ty, ctx.clone(), next)
            .prop_map(move |inner| {
                coverage::CAST.fetch_add(1, Ordering::Relaxed);
                Expr::Cast {
                    inner: Box::new(inner),
                    to: target_for_cast.clone(),
                }
            })
            .boxed()
    };

    // Shift: value at target type, amount at unsigned Bits<W>. RHDL
    // runtime-rejects shift amounts >= operand width, so constrain the
    // amount to a literal in [0, width). Dynamic shift amounts would
    // require value-range analysis that the DSL can't currently express.
    let shift = {
        let target_w = target.width();
        let amount_ty = Type::Bits(target.width());
        let amount_strat = (0u128..target_w as u128).prop_map(move |amt| Expr::Lit {
            value: amt,
            ty: amount_ty.clone(),
        });
        (
            shift_kind(),
            gen_expr_of_type(target.clone(), ctx.clone(), next),
            amount_strat,
        )
            .prop_map(|(kind, value, amount)| {
                coverage::SHIFT.fetch_add(1, Ordering::Relaxed);
                Expr::Shift {
                    kind,
                    value: Box::new(value),
                    amount: Box::new(amount),
                }
            })
            .boxed()
    };

    // Project a scalar out of an inline 2-element tuple. The other
    // tuple element gets a random scalar type from the pool.
    let tuple_proj = {
        let target_for_proj = target.clone();
        let ctx_for_proj = ctx.clone();
        (any::<bool>(), proptest::sample::select(WIDTH_POOL))
            .prop_flat_map(move |(target_first, other_w)| {
                // Other element gets the same kind (Bits/Signed) as target
                // for symmetry; a different kind would also be valid.
                let other_ty = if target_for_proj.is_signed() {
                    Type::Signed(other_w)
                } else {
                    Type::Bits(other_w)
                };
                let (t0, t1, target_idx) = if target_first {
                    (target_for_proj.clone(), other_ty, 0usize)
                } else {
                    (other_ty, target_for_proj.clone(), 1usize)
                };
                (
                    Just(target_idx),
                    gen_expr_of_type(t0, ctx_for_proj.clone(), next),
                    gen_expr_of_type(t1, ctx_for_proj.clone(), next),
                )
            })
            .prop_map(|(idx, e0, e1)| {
                coverage::TUPLE_CTOR.fetch_add(1, Ordering::Relaxed);
                coverage::TUPLE_PROJ.fetch_add(1, Ordering::Relaxed);
                Expr::TupleProj {
                    inner: Box::new(Expr::TupleCtor(vec![e0, e1])),
                    index: idx,
                }
            })
            .boxed()
    };

    // Index into an inline 2-element array of the target type with a
    // literal index in [0, 2).
    let array_index = {
        let target_for_idx = target.clone();
        let ctx_for_idx = ctx.clone();
        (
            (0u128..2u128),
            gen_expr_of_type(target.clone(), ctx.clone(), next),
            gen_expr_of_type(target.clone(), ctx.clone(), next),
        )
            .prop_map(move |(idx, e0, e1)| {
                let _ = (&target_for_idx, &ctx_for_idx);
                coverage::ARRAY_CTOR.fetch_add(1, Ordering::Relaxed);
                coverage::ARRAY_INDEX.fetch_add(1, Ordering::Relaxed);
                Expr::ArrayIndex {
                    inner: Box::new(Expr::ArrayCtor(vec![e0, e1])),
                    index: Box::new(Expr::Lit {
                        value: idx,
                        ty: Type::Bits(1),
                    }),
                }
            })
            .boxed()
    };

    // Index into a `[value; N]`-style repeat array of the target type.
    // The index is always 0 (any index yields the same element), but
    // emits the RHIF `Repeat` opcode that ArrayCtor does not.
    let array_repeat_index = {
        let ctx_for_rep = ctx.clone();
        gen_expr_of_type(target.clone(), ctx_for_rep, next)
            .prop_map(|value| {
                coverage::ARRAY_REPEAT.fetch_add(1, Ordering::Relaxed);
                coverage::ARRAY_INDEX.fetch_add(1, Ordering::Relaxed);
                Expr::ArrayIndex {
                    inner: Box::new(Expr::ArrayRepeat {
                        value: Box::new(value),
                        len: 2,
                    }),
                    index: Box::new(Expr::Lit {
                        value: 0,
                        ty: Type::Bits(1),
                    }),
                }
            })
            .boxed()
    };

    // Optional Call arm: only included if at least one callee in scope
    // returns a value of the target type.
    let call_arm: Option<BoxedStrategy<Expr>> = build_call_arm(&target, &ctx, next);
    // Optional nominal-type round-trip arms: only included if there is
    // a struct in scope with a field of the target type / an enum in
    // scope (whose arms can produce the target). Arms terminate one
    // level of recursion via `next`.
    let struct_arm: Option<BoxedStrategy<Expr>> = build_struct_roundtrip_arm(&target, &ctx, next);
    let struct_update_arm: Option<BoxedStrategy<Expr>> =
        build_struct_update_arm(&target, &ctx, next);
    let match_arm: Option<BoxedStrategy<Expr>> = build_match_roundtrip_arm(&target, &ctx, next);
    let mutation_arm: BoxedStrategy<Expr> = build_mutation_arm(&target, &ctx, next);
    let for_arm: Option<BoxedStrategy<Expr>> = build_for_accumulator_arm(&target, &ctx, next);

    let mut arms: Vec<(u32, BoxedStrategy<Expr>)> = vec![
        (4, leaves),
        (2, bin),
        (1, un),
        (2, lett),
        (2, if_),
        (1, resize),
        (1, cast),
        (1, shift),
        (1, tuple_proj),
        (1, array_index),
        (1, array_repeat_index),
    ];
    if let Some(call) = call_arm {
        arms.push((3, call));
    }
    if let Some(s) = struct_arm {
        arms.push((1, s));
    }
    if let Some(su) = struct_update_arm {
        arms.push((1, su));
    }
    if let Some(m) = match_arm {
        arms.push((1, m));
    }
    arms.push((1, mutation_arm));
    if let Some(f) = for_arm {
        arms.push((1, f));
    }
    weighted_oneof(arms)
}

/// `prop_oneof!` only accepts a fixed-size tuple, so for the
/// dynamically-sized arm list (where `Call` is conditionally present)
/// we fold a sequence of weighted strategies into a single boxed
/// `Union`-like strategy via the `Strategy::prop_union` chain. proptest
/// exposes this as `proptest::strategy::Union::new_weighted`.
fn weighted_oneof(arms: Vec<(u32, BoxedStrategy<Expr>)>) -> BoxedStrategy<Expr> {
    use proptest::strategy::Union;
    let weighted: Vec<(u32, BoxedStrategy<Expr>)> = arms;
    Union::new_weighted(weighted).boxed()
}

/// Build the optional Call arm of `gen_scalar_expr`. Returns `None`
/// if no callee in scope has the target return type.
fn build_call_arm(target: &Type, ctx: &TypingCtx, depth: u32) -> Option<BoxedStrategy<Expr>> {
    let matching: Vec<usize> = ctx.callees_returning(target);
    if matching.is_empty() {
        return None;
    }
    let ctx_for_call = ctx.clone();
    Some(
        proptest::sample::select(matching)
            .prop_flat_map(move |idx| {
                let sig = ctx_for_call.callees[idx].clone();
                let args_strat = build_args_strategy(&sig.arg_types, &ctx_for_call, depth);
                args_strat.prop_map(move |args| Expr::Call { callee: idx, args })
            })
            .boxed(),
    )
}

/// Build the optional struct round-trip arm: pick a `(struct_id,
/// field_idx)` pair where the field's type is `target`, generate one
/// scalar expression per declared field, wrap in `StructCtor` and
/// project the matching field via `FieldAccess`. Returns `None` if no
/// struct in scope has a field of type `target`.
fn build_struct_roundtrip_arm(
    target: &Type,
    ctx: &TypingCtx,
    depth: u32,
) -> Option<BoxedStrategy<Expr>> {
    let candidates: Vec<(usize, usize)> = ctx
        .nominal
        .structs
        .iter()
        .enumerate()
        .flat_map(|(si, s)| {
            s.fields
                .iter()
                .enumerate()
                .filter(|(_, (_, ty))| ty == target)
                .map(move |(fi, _)| (si, fi))
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let ctx_outer = ctx.clone();
    Some(
        proptest::sample::select(candidates)
            .prop_flat_map(move |(si, fi)| {
                let s = ctx_outer.nominal.structs[si].clone();
                let ctx_inner = ctx_outer.clone();
                let field_strats: Vec<BoxedStrategy<Expr>> = s
                    .fields
                    .iter()
                    .map(|(_, ty)| gen_expr_of_type(ty.clone(), ctx_inner.clone(), depth))
                    .collect();
                field_strats.prop_map(move |fields| {
                    coverage::STRUCT_CTOR.fetch_add(1, Ordering::Relaxed);
                    coverage::FIELD_ACCESS.fetch_add(1, Ordering::Relaxed);
                    Expr::FieldAccess {
                        inner: Box::new(Expr::StructCtor {
                            struct_id: si,
                            fields,
                        }),
                        field_idx: fi,
                    }
                })
            })
            .boxed(),
    )
}

/// Build the mutation round-trip arm: `{ let mut t = init; t = new; t }`.
/// Always available -- exercises the `Block` + `Stmt::LetMut` +
/// `Stmt::Assign` codepath, which lowers to `OpCode::Assign` in RHIF.
/// Both `init` and `new` have type `target`; the block's value is the
/// final read of `t`, which equals `new`.
fn build_mutation_arm(target: &Type, ctx: &TypingCtx, depth: u32) -> BoxedStrategy<Expr> {
    // The mutable binding lives at position `arity + ctx.bindings.len()
    // - (input arity)` -- since ctx.bindings starts as the args and is
    // extended only by parent generators' lets/match-payloads. The
    // new binding's index is exactly `ctx.bindings.len() as u8`.
    let mut_idx = ctx.bindings.len() as u8;
    let mut ctx_after_let = ctx.clone();
    ctx_after_let.bindings.push(target.clone());
    let target_for_init = target.clone();
    let target_for_new = target.clone();
    let ctx_for_init = ctx.clone();
    (
        gen_expr_of_type(target_for_init, ctx_for_init, depth),
        gen_expr_of_type(target_for_new, ctx_after_let.clone(), depth),
    )
        .prop_map(move |(init, new)| {
            coverage::BLOCK.fetch_add(1, Ordering::Relaxed);
            coverage::STMT_LET_MUT.fetch_add(1, Ordering::Relaxed);
            coverage::STMT_ASSIGN.fetch_add(1, Ordering::Relaxed);
            Expr::Block {
                stmts: vec![
                    Stmt::LetMut(init),
                    Stmt::Assign {
                        var_idx: mut_idx,
                        value: new,
                    },
                ],
                tail: Box::new(Expr::Var(mut_idx)),
            }
        })
        .boxed()
}

/// Build the for-accumulator round-trip arm:
///   `{ let mut acc = init; for i in 0..4 { acc = acc + delta; } acc }`
/// Exercises `Expr::For` + the accumulator pattern. The loop variable
/// `i` is never referenced (it just controls iteration count); the
/// inner Assign computes `acc + delta` where `delta` is a scalar
/// expression sampled at generation time (constant across iterations
/// because the loop body sees the same outer bindings each iteration).
fn build_for_accumulator_arm(
    target: &Type,
    ctx: &TypingCtx,
    depth: u32,
) -> Option<BoxedStrategy<Expr>> {
    // The accumulator binop only makes sense on scalar types; for
    // compound targets, skip.
    if !target.is_scalar() {
        return None;
    }
    let acc_idx = ctx.bindings.len() as u8;
    let mut ctx_after_let = ctx.clone();
    ctx_after_let.bindings.push(target.clone());
    // Inside the for body, the loop variable is also in scope at
    // `acc_idx + 1` (the renderer/interpreter pushes it onto the
    // bindings stack). The delta expression is generated INSIDE the
    // for body, so its ctx must include the loop variable; otherwise
    // any nested generator (e.g., a recursive for-accumulator) will
    // compute Var positions that are off by one.
    //
    // The loop variable's type is `LOOP_VAR_TYPE` -- wide enough that
    // `i op i` for any binop and our small `range_hi` cannot wrap
    // (matching RHDL, which compile-time unrolls `for i in 0..N` and
    // inlines `i` as an unsized literal integer per
    // crates/rhdl-core/src/compiler/mir/compiler.rs:1001-1002).
    let mut ctx_for_delta = ctx_after_let.clone();
    ctx_for_delta.bindings.push(LOOP_VAR_TYPE);
    let target_for_init = target.clone();
    let target_for_delta = target.clone();
    let ctx_for_init = ctx.clone();
    (
        gen_expr_of_type(target_for_init, ctx_for_init, depth),
        gen_expr_of_type(target_for_delta, ctx_for_delta, depth),
    )
        .prop_map(move |(init, delta)| {
            coverage::BLOCK.fetch_add(1, Ordering::Relaxed);
            coverage::STMT_LET_MUT.fetch_add(1, Ordering::Relaxed);
            coverage::STMT_ASSIGN.fetch_add(1, Ordering::Relaxed);
            coverage::FOR.fetch_add(1, Ordering::Relaxed);
            let acc_plus_delta = Expr::BinOp(
                BinOpKind::Add,
                Box::new(Expr::Var(acc_idx)),
                Box::new(delta),
            );
            let inner_block = Expr::Block {
                stmts: vec![Stmt::Assign {
                    var_idx: acc_idx,
                    value: acc_plus_delta,
                }],
                tail: Box::new(Expr::Lit {
                    value: 0,
                    ty: Type::Bits(1),
                }),
            };
            let for_expr = Expr::For {
                loop_var_width: LOOP_VAR_WIDTH,
                range_hi: 4,
                body: Box::new(inner_block),
            };
            Expr::Block {
                stmts: vec![Stmt::LetMut(init), Stmt::Let(for_expr)],
                tail: Box::new(Expr::Var(acc_idx)),
            }
        })
        .boxed()
        .into()
}

/// Build the optional struct-update round-trip arm: pick a
/// `(struct_id, field_idx)` where the field's type is `target`,
/// generate a base struct expression (via `StructCtor`), generate an
/// override for the targeted field, then project the targeted field
/// from the resulting `StructUpdate`. The projected field always
/// comes from the override (since we override exactly that field),
/// but the path through `op_struct(rest=Some)` is distinct from the
/// no-rest form and worth exercising on its own.
fn build_struct_update_arm(
    target: &Type,
    ctx: &TypingCtx,
    depth: u32,
) -> Option<BoxedStrategy<Expr>> {
    let candidates: Vec<(usize, usize)> = ctx
        .nominal
        .structs
        .iter()
        .enumerate()
        .flat_map(|(si, s)| {
            s.fields
                .iter()
                .enumerate()
                .filter(|(_, (_, ty))| ty == target)
                .map(move |(fi, _)| (si, fi))
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let ctx_outer = ctx.clone();
    Some(
        proptest::sample::select(candidates)
            .prop_flat_map(move |(si, fi)| {
                let s = ctx_outer.nominal.structs[si].clone();
                let ctx_inner = ctx_outer.clone();
                // Base: a fresh StructCtor of the right type.
                let base_field_strats: Vec<BoxedStrategy<Expr>> = s
                    .fields
                    .iter()
                    .map(|(_, ty)| gen_expr_of_type(ty.clone(), ctx_inner.clone(), depth))
                    .collect();
                let base_strat = base_field_strats.prop_map(move |fields| Expr::StructCtor {
                    struct_id: si,
                    fields,
                });
                // Override expression for the targeted field.
                let override_ty = s.fields[fi].1.clone();
                let override_strat = gen_expr_of_type(override_ty, ctx_inner.clone(), depth);
                (base_strat, override_strat).prop_map(move |(base, ov)| {
                    coverage::STRUCT_UPDATE.fetch_add(1, Ordering::Relaxed);
                    coverage::FIELD_ACCESS.fetch_add(1, Ordering::Relaxed);
                    Expr::FieldAccess {
                        inner: Box::new(Expr::StructUpdate {
                            struct_id: si,
                            base: Box::new(base),
                            overrides: vec![(fi, ov)],
                        }),
                        field_idx: fi,
                    }
                })
            })
            .boxed(),
    )
}

/// Build the optional match round-trip arm: pick an enum, generate a
/// scrutinee via `EnumCtor` (picking a random variant + payload
/// expressions), and emit a `Match` whose arms each produce `target`.
/// Arms that bind a payload push the payload's type onto the binding
/// stack for body generation. Returns `None` if no enum is in scope.
fn build_match_roundtrip_arm(
    target: &Type,
    ctx: &TypingCtx,
    depth: u32,
) -> Option<BoxedStrategy<Expr>> {
    if ctx.nominal.enums.is_empty() {
        return None;
    }
    let enum_ids: Vec<usize> = (0..ctx.nominal.enums.len()).collect();
    let ctx_outer = ctx.clone();
    let target_outer = target.clone();
    Some(
        proptest::sample::select(enum_ids)
            .prop_flat_map(move |enum_id| {
                let edef = ctx_outer.nominal.enums[enum_id].clone();
                let n_variants = edef.variants.len();
                let ctx_inner = ctx_outer.clone();
                let target_inner = target_outer.clone();

                // Strategy for the scrutinee: pick a variant index,
                // then generate one expression per payload field.
                let variants_for_scrut: Vec<usize> = (0..n_variants).collect();
                let edef_for_scrut = edef.clone();
                let ctx_for_scrut = ctx_inner.clone();
                let scrut_strat = proptest::sample::select(variants_for_scrut)
                    .prop_flat_map(move |variant_idx| {
                        let v = edef_for_scrut.variants[variant_idx].clone();
                        let ctx_pl = ctx_for_scrut.clone();
                        let payload_strats: Vec<BoxedStrategy<Expr>> = v
                            .payload
                            .iter()
                            .map(|ty| gen_expr_of_type(ty.clone(), ctx_pl.clone(), depth))
                            .collect();
                        payload_strats.prop_map(move |payload| Expr::EnumCtor {
                            enum_id,
                            variant_idx,
                            payload,
                        })
                    })
                    .boxed();

                // Strategy for each arm body: bindings extended with
                // that variant's payload types.
                let edef_for_arms = edef.clone();
                let ctx_for_arms = ctx_inner.clone();
                let target_for_arms = target_inner.clone();
                let arm_body_strats: Vec<BoxedStrategy<Expr>> = (0..n_variants)
                    .map(|vidx| {
                        let mut ctx_arm = ctx_for_arms.clone();
                        ctx_arm
                            .bindings
                            .extend(edef_for_arms.variants[vidx].payload.iter().cloned());
                        gen_expr_of_type(target_for_arms.clone(), ctx_arm, depth)
                    })
                    .collect();

                (scrut_strat, arm_body_strats).prop_map(move |(scrutinee, bodies)| {
                    coverage::ENUM_CTOR.fetch_add(1, Ordering::Relaxed);
                    coverage::MATCH.fetch_add(1, Ordering::Relaxed);
                    let arms = bodies
                        .into_iter()
                        .enumerate()
                        .map(|(vidx, body)| MatchArm {
                            enum_id,
                            variant_idx: vidx,
                            body,
                        })
                        .collect();
                    Expr::Match {
                        scrutinee: Box::new(scrutinee),
                        arms,
                    }
                })
            })
            .boxed(),
    )
}

/// Build a `Strategy<Vec<Expr>>` that produces one expression per
/// element of `arg_types`, each typed accordingly. Supports arity 1..=4
/// (covers everything the fuzzer needs in M4 v1).
fn build_args_strategy(
    arg_types: &[Type],
    ctx: &TypingCtx,
    depth: u32,
) -> BoxedStrategy<Vec<Expr>> {
    match arg_types.len() {
        1 => gen_expr_of_type(arg_types[0].clone(), ctx.clone(), depth)
            .prop_map(|e| vec![e])
            .boxed(),
        2 => (
            gen_expr_of_type(arg_types[0].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[1].clone(), ctx.clone(), depth),
        )
            .prop_map(|(a, b)| vec![a, b])
            .boxed(),
        3 => (
            gen_expr_of_type(arg_types[0].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[1].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[2].clone(), ctx.clone(), depth),
        )
            .prop_map(|(a, b, c)| vec![a, b, c])
            .boxed(),
        4 => (
            gen_expr_of_type(arg_types[0].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[1].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[2].clone(), ctx.clone(), depth),
            gen_expr_of_type(arg_types[3].clone(), ctx.clone(), depth),
        )
            .prop_map(|(a, b, c, d)| vec![a, b, c, d])
            .boxed(),
        n => panic!("build_args_strategy: arity {n} not supported (max 4)"),
    }
}

/// Generate an expression evaluating to a `Type::Tuple(...)`. For v1
/// only arity-2 tuples are produced.
fn gen_tuple_expr(target: Type, ctx: TypingCtx, depth: u32) -> BoxedStrategy<Expr> {
    let elem_types = match &target {
        Type::Tuple(t) => t.clone(),
        _ => unreachable!("gen_tuple_expr called with non-tuple target"),
    };
    let next = depth.saturating_sub(1);
    match elem_types.len() {
        2 => (
            gen_expr_of_type(elem_types[0].clone(), ctx.clone(), next),
            gen_expr_of_type(elem_types[1].clone(), ctx.clone(), next),
        )
            .prop_map(|(a, b)| {
                coverage::TUPLE_CTOR.fetch_add(1, Ordering::Relaxed);
                Expr::TupleCtor(vec![a, b])
            })
            .boxed(),
        n => panic!("gen_tuple_expr: tuple arity {n} not supported (v1: 2 only)"),
    }
}

/// Generate an expression evaluating to a `Type::Array(...)`. For v1
/// only length-2 arrays are produced.
fn gen_array_expr(target: Type, ctx: TypingCtx, depth: u32) -> BoxedStrategy<Expr> {
    let (elem_ty, len) = match &target {
        Type::Array(et, n) => ((**et).clone(), *n),
        _ => unreachable!("gen_array_expr called with non-array target"),
    };
    let next = depth.saturating_sub(1);
    match len {
        2 => (
            gen_expr_of_type(elem_ty.clone(), ctx.clone(), next),
            gen_expr_of_type(elem_ty, ctx, next),
        )
            .prop_map(|(a, b)| {
                coverage::ARRAY_CTOR.fetch_add(1, Ordering::Relaxed);
                Expr::ArrayCtor(vec![a, b])
            })
            .boxed(),
        n => panic!("gen_array_expr: array length {n} not supported (v1: 2 only)"),
    }
}

fn scalar_leaves(target: Type, ctx: TypingCtx) -> BoxedStrategy<Expr> {
    let lit_target = target.clone();
    let width = target.width();
    let lit_signed = target.is_signed();
    let lit = lit_value_strategy(width)
        .prop_map(move |v| {
            if lit_signed {
                coverage::LIT_SIGNED.fetch_add(1, Ordering::Relaxed);
            } else {
                coverage::LIT_BITS.fetch_add(1, Ordering::Relaxed);
            }
            Expr::Lit {
                value: v,
                ty: lit_target.clone(),
            }
        })
        .boxed();
    let matching: Vec<u8> = ctx.bindings_of(&target);
    if matching.is_empty() {
        return lit;
    }
    let var = proptest::sample::select(matching)
        .prop_map(|i| {
            coverage::VAR.fetch_add(1, Ordering::Relaxed);
            Expr::Var(i)
        })
        .boxed();
    prop_oneof![1 => lit, 1 => var].boxed()
}

/// Generate a [`BoolExpr`] valid in `ctx`. `depth` bounds recursion.
pub fn bool_expr_strategy(ctx: TypingCtx, depth: u32) -> BoxedStrategy<BoolExpr> {
    let leaf = any::<bool>()
        .prop_map(|v| {
            coverage::BOOL_LIT.fetch_add(1, Ordering::Relaxed);
            BoolExpr::Lit(v)
        })
        .boxed();
    if depth == 0 {
        return leaf;
    }
    let next = depth - 1;
    // Comparisons take two operands of the same scalar type. Pick a
    // width AND signedness, then generate both operands at that type.
    let cmp = {
        let ctx_cmp = ctx.clone();
        (proptest::sample::select(WIDTH_POOL), any::<bool>())
            .prop_flat_map(move |(w, signed)| {
                let cmp_target = if signed {
                    Type::Signed(w)
                } else {
                    Type::Bits(w)
                };
                (
                    cmp_kind(),
                    gen_expr_of_type(cmp_target.clone(), ctx_cmp.clone(), next),
                    gen_expr_of_type(cmp_target, ctx_cmp.clone(), next),
                )
            })
            .prop_map(|(op, l, r)| {
                coverage::BOOL_CMP.fetch_add(1, Ordering::Relaxed);
                BoolExpr::Cmp(op, Box::new(l), Box::new(r))
            })
            .boxed()
    };
    let not_ = bool_expr_strategy(ctx.clone(), next)
        .prop_map(|x| {
            coverage::BOOL_NOT.fetch_add(1, Ordering::Relaxed);
            BoolExpr::Not(Box::new(x))
        })
        .boxed();
    let and_ = (
        bool_expr_strategy(ctx.clone(), next),
        bool_expr_strategy(ctx.clone(), next),
    )
        .prop_map(|(l, r)| {
            coverage::BOOL_AND.fetch_add(1, Ordering::Relaxed);
            BoolExpr::And(Box::new(l), Box::new(r))
        })
        .boxed();
    let or_ = (
        bool_expr_strategy(ctx.clone(), next),
        bool_expr_strategy(ctx.clone(), next),
    )
        .prop_map(|(l, r)| {
            coverage::BOOL_OR.fetch_add(1, Ordering::Relaxed);
            BoolExpr::Or(Box::new(l), Box::new(r))
        })
        .boxed();
    // Reductions (`.any()`, `.all()`, `.xor()`) take a scalar of any
    // width and return bool. Pick width + signedness, generate a scalar
    // expression of that type, wrap in the chosen reduction.
    let reduction = {
        let ctx_red = ctx;
        (
            proptest::sample::select(WIDTH_POOL),
            any::<bool>(),
            prop_oneof![
                Just(ReductionKind::Any),
                Just(ReductionKind::All),
                Just(ReductionKind::Xor),
            ],
        )
            .prop_flat_map(move |(w, signed, kind)| {
                let inner_ty = if signed {
                    Type::Signed(w)
                } else {
                    Type::Bits(w)
                };
                gen_expr_of_type(inner_ty, ctx_red.clone(), next).prop_map(move |inner| {
                    coverage::BOOL_REDUCTION.fetch_add(1, Ordering::Relaxed);
                    BoolExpr::Reduction(kind, Box::new(inner))
                })
            })
            .boxed()
    };
    prop_oneof![
        2 => leaf,
        4 => cmp,
        1 => not_,
        1 => and_,
        1 => or_,
        2 => reduction,
    ]
    .boxed()
}

/// A small palette of scalar types used as struct fields and enum
/// payloads. Kept narrow so the body generator has a reasonable chance
/// of producing values that fit those fields/payloads.
fn scalar_palette_strategy() -> impl Strategy<Value = Type> {
    prop_oneof![
        Just(Type::Bits(8)),
        Just(Type::Bits(16)),
        Just(Type::Signed(8)),
    ]
}

/// Strategy producing a small `NominalRegistry` for a single program:
/// always one 2-field struct and one 2-variant enum (one variant has a
/// scalar payload, the other is unit). The field/payload scalar types
/// are sampled from `scalar_palette_strategy` so different programs
/// see different shapes.
pub fn nominal_registry_strategy() -> impl Strategy<Value = NominalRegistry> {
    (
        scalar_palette_strategy(),
        scalar_palette_strategy(),
        scalar_palette_strategy(),
    )
        .prop_map(|(f0_ty, f1_ty, v0_payload_ty)| NominalRegistry {
            structs: vec![StructDef {
                name: STRUCT_NAMES[0],
                fields: vec![(FIELD_NAMES[0], f0_ty), (FIELD_NAMES[1], f1_ty)],
            }],
            enums: vec![EnumDef {
                name: ENUM_NAMES[0],
                variants: vec![
                    VariantDef {
                        name: VARIANT_NAMES[0],
                        payload: vec![v0_payload_ty],
                    },
                    VariantDef {
                        name: VARIANT_NAMES[1],
                        payload: vec![],
                    },
                ],
                discriminant_width: 1,
            }],
        })
}

/// Generate a `Program` whose argument types are `arg_types` and whose
/// body has type `ret_type`. A non-empty nominal registry is always
/// generated; the body may or may not reference its structs/enums
/// depending on whether the round-trip arms fire.
pub fn program_strategy_typed(
    arg_types: Vec<Type>,
    ret_type: Type,
) -> impl Strategy<Value = Program> {
    nominal_registry_strategy().prop_flat_map(move |nominal| {
        let ctx = TypingCtx::from_args_with_nominal(&arg_types, nominal.clone());
        let arg_types_inner = arg_types.clone();
        let ret_type_inner = ret_type.clone();
        let nominal_inner = nominal.clone();
        gen_expr_of_type(ret_type.clone(), ctx, DEFAULT_MAX_DEPTH).prop_map(move |body| Program {
            arg_types: arg_types_inner.clone(),
            ret_type: ret_type_inner.clone(),
            body,
            nominal: nominal_inner.clone(),
        })
    })
}

/// Mono-typed convenience: produce a `Bits<8>` -> `Bits<8>` program with
/// the given arity.
pub fn program_strategy(arity: u8) -> impl Strategy<Value = Program> {
    let arg_types = vec![Type::Bits(8); arity as usize];
    program_strategy_typed(arg_types, Type::Bits(8))
}

// ---------- M4: multi-kernel program strategy ----------

/// Generate a `MultiKernelProgram` with `top_arg_types` / `top_ret_type`
/// at the entry kernel and `num_callees` lower-indexed kernels each
/// returning the same scalar type as the top.
///
/// Callees have arity 1 over the same scalar type as the top. The top
/// kernel sees all callees in scope. This is M4 v1 -- richer signature
/// variety can come later.
pub fn multi_program_strategy_v1(
    top_arg_types: Vec<Type>,
    top_ret_type: Type,
    num_callees: usize,
) -> BoxedStrategy<MultiKernelProgram> {
    assert!(top_ret_type.is_scalar(), "multi v1 requires scalar return");
    assert!(
        num_callees + 1 <= KERNEL_POOL_SIZE,
        "num_callees {} + 1 exceeds pool size {}",
        num_callees,
        KERNEL_POOL_SIZE
    );

    // Callee shape: arity-1, same type in/out as the top's ret_type.
    // Each callee can call earlier callees (true DAG).
    let callee_arg_types = vec![top_ret_type.clone()];
    let callee_ret_type = top_ret_type.clone();

    let top_args = top_arg_types;
    let top_ret = top_ret_type;

    // Build callees one at a time. Each callee's body strategy is
    // parameterized on the callees that came before it (lower indices).
    fn build_callee_strategies(
        n: usize,
        arg_types: Vec<Type>,
        ret_type: Type,
    ) -> BoxedStrategy<Vec<KernelDef>> {
        if n == 0 {
            return Just(Vec::<KernelDef>::new()).boxed();
        }
        let arg_types_for_recurse = arg_types.clone();
        let ret_type_for_recurse = ret_type.clone();
        build_callee_strategies(n - 1, arg_types.clone(), ret_type.clone())
            .prop_flat_map(move |earlier| {
                // Build CalleeSig list for already-rendered callees.
                let callees: Vec<CalleeSig> = earlier
                    .iter()
                    .map(|d| CalleeSig {
                        arg_types: d.arg_types.clone(),
                        ret_type: d.ret_type.clone(),
                    })
                    .collect();
                let ctx = TypingCtx::from_args_with_callees(&arg_types_for_recurse, callees);
                let arg_types_inner = arg_types_for_recurse.clone();
                let ret_type_inner = ret_type_for_recurse.clone();
                let new_idx = earlier.len();
                gen_expr_of_type(ret_type_inner.clone(), ctx, DEFAULT_MAX_DEPTH).prop_map(
                    move |body| {
                        let mut combined = earlier.clone();
                        combined.push(KernelDef {
                            name: KERNEL_NAMES[new_idx],
                            arg_types: arg_types_inner.clone(),
                            ret_type: ret_type_inner.clone(),
                            body,
                        });
                        combined
                    },
                )
            })
            .boxed()
    }

    build_callee_strategies(num_callees, callee_arg_types, callee_ret_type)
        .prop_flat_map(move |callees| {
            // Build TypingCtx for the top kernel: bindings = top args,
            // callees = all the previously-built ones.
            let callee_sigs: Vec<CalleeSig> = callees
                .iter()
                .map(|d| CalleeSig {
                    arg_types: d.arg_types.clone(),
                    ret_type: d.ret_type.clone(),
                })
                .collect();
            let ctx = TypingCtx::from_args_with_callees(&top_args, callee_sigs);
            let top_args_inner = top_args.clone();
            let top_ret_inner = top_ret.clone();
            let top_idx = callees.len();
            gen_expr_of_type(top_ret_inner.clone(), ctx, DEFAULT_MAX_DEPTH).prop_map(move |body| {
                let mut all = callees.clone();
                all.push(KernelDef {
                    name: KERNEL_NAMES[top_idx],
                    arg_types: top_args_inner.clone(),
                    ret_type: top_ret_inner.clone(),
                    body,
                });
                MultiKernelProgram {
                    kernels: all,
                    top: top_idx,
                    nominal: NominalRegistry::new(),
                }
            })
        })
        .boxed()
}
