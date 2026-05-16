//! Reference interpreter for the DSL: executes a program directly in Rust
//! over a tagged `Value` ADT, producing the value the RHDL pipeline is
//! expected to compute.

use crate::dsl::{
    BinOpKind, BoolExpr, CmpKind, Expr, Program, ReductionKind, ShiftKind, Stmt, Type, UnOpKind,
};
use crate::multi::{KernelDef, MultiKernelProgram};

/// A runtime value flowing through the interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Unsigned N-bit integer. The `u128` payload is masked to `width`.
    Bits(u128, u8),
    /// Signed N-bit integer. The `i128` payload is sign-extended from
    /// `width` bits to 128 bits.
    Signed(i128, u8),
    /// Heterogeneous tuple of values.
    Tuple(Vec<Value>),
    /// Homogeneous fixed-length array of values.
    Array(Vec<Value>),
    /// Nominal struct instance. The `usize` is the `struct_id`
    /// matching `Type::Struct`; the `Vec<Value>` holds the field
    /// values in declaration order.
    Struct(usize, Vec<Value>),
    /// Nominal enum instance. `enum_id` matches `Type::Enum`;
    /// `variant_idx` selects the variant; `payload` carries the
    /// positional payload values for that variant (empty for unit).
    Enum {
        enum_id: usize,
        variant_idx: usize,
        payload: Vec<Value>,
    },
}

impl Value {
    /// Width in bits of a *scalar* value. Panics on compound values.
    pub fn width(&self) -> u8 {
        match self {
            Value::Bits(_, w) | Value::Signed(_, w) => *w,
            _ => panic!("width() called on compound Value {:?}", self),
        }
    }

    /// True if this value is signed.
    pub fn is_signed(&self) -> bool {
        matches!(self, Value::Signed(_, _))
    }

    /// Mask `raw` to `width` bits and produce a `Value::Bits`.
    pub fn bits(raw: u128, width: u8) -> Value {
        Value::Bits(mask_to_width(raw, width), width)
    }

    /// Sign-extend `raw` from `width` bits to 128 and produce a
    /// `Value::Signed`.
    pub fn signed(raw: i128, width: u8) -> Value {
        Value::Signed(sign_extend(raw, width), width)
    }

    /// Extract the raw `u128` payload of a `Value::Bits`. Panics on
    /// any other variant.
    pub fn expect_bits_raw(&self) -> u128 {
        match self {
            Value::Bits(v, _) => *v,
            other => panic!("expect_bits_raw called on {:?}", other),
        }
    }

    /// Extract the raw `i128` payload of a `Value::Signed`. Panics on
    /// any other variant.
    pub fn expect_signed_raw(&self) -> i128 {
        match self {
            Value::Signed(v, _) => *v,
            other => panic!("expect_signed_raw called on {:?}", other),
        }
    }

    /// Recursively check whether this value's structure matches `ty`.
    /// For nominal types (`Struct`/`Enum`), only the id is checked
    /// and the inner field/payload structure is trusted (the renderer
    /// would have rejected a mismatch at compile time).
    pub fn matches_type(&self, ty: &Type) -> bool {
        match (self, ty) {
            (Value::Bits(_, w), Type::Bits(tw)) => w == tw,
            (Value::Signed(_, w), Type::Signed(tw)) => w == tw,
            (Value::Tuple(elems), Type::Tuple(tys)) => {
                elems.len() == tys.len() && elems.iter().zip(tys).all(|(v, t)| v.matches_type(t))
            }
            (Value::Array(elems), Type::Array(et, n)) => {
                elems.len() == *n && elems.iter().all(|v| v.matches_type(et))
            }
            (Value::Struct(sid, _), Type::Struct(tid)) => sid == tid,
            (Value::Enum { enum_id, .. }, Type::Enum(tid)) => enum_id == tid,
            _ => false,
        }
    }
}

fn mask_to_width(raw: u128, width: u8) -> u128 {
    if width >= 128 {
        raw
    } else {
        raw & ((1u128 << width) - 1)
    }
}

/// Sign-extend `raw` from `width` bits up to 128. For values already
/// representing the correct numeric value at `width`, this is a no-op.
fn sign_extend(raw: i128, width: u8) -> i128 {
    if width >= 128 {
        raw
    } else {
        let shift = 128 - width as u32;
        (raw << shift) >> shift
    }
}

/// Execute a program with the given input bindings.
pub fn interpret(program: &Program, inputs: &[Value]) -> Value {
    assert_eq!(
        inputs.len(),
        program.arity() as usize,
        "interpret: input count {} does not match program arity {}",
        inputs.len(),
        program.arity()
    );
    for (i, (val, ty)) in inputs.iter().zip(program.arg_types.iter()).enumerate() {
        assert!(
            val.matches_type(ty),
            "interpret: input {} {:?} does not structurally match arg type {:?}",
            i,
            val,
            ty
        );
    }
    let mut bindings: Vec<Value> = inputs.to_vec();
    eval(&program.body, &mut bindings, &[])
}

/// Execute a `MultiKernelProgram`. The top kernel's body may include
/// `Expr::Call` referencing lower-indexed kernels in the `kernels`
/// table.
pub fn interpret_multi(program: &MultiKernelProgram, inputs: &[Value]) -> Value {
    let top = program.top_kernel();
    assert_eq!(
        inputs.len(),
        top.arg_types.len(),
        "interpret_multi: input count {} does not match top kernel arity {}",
        inputs.len(),
        top.arg_types.len()
    );
    for (i, (val, ty)) in inputs.iter().zip(top.arg_types.iter()).enumerate() {
        assert!(
            val.matches_type(ty),
            "interpret_multi: top input {} {:?} does not match arg type {:?}",
            i,
            val,
            ty
        );
    }
    eval_kernel(top, inputs, &program.kernels)
}

/// Execute a single `KernelDef` body with the given inputs and access
/// to the full kernel table for `Expr::Call` resolution.
fn eval_kernel(def: &KernelDef, inputs: &[Value], all_kernels: &[KernelDef]) -> Value {
    let mut bindings: Vec<Value> = inputs.to_vec();
    eval(&def.body, &mut bindings, all_kernels)
}

fn eval(expr: &Expr, bindings: &mut Vec<Value>, callees: &[KernelDef]) -> Value {
    match expr {
        Expr::Lit { value, ty } => match ty {
            Type::Bits(w) => Value::bits(*value, *w),
            Type::Signed(w) => Value::signed(*value as i128, *w),
            other => panic!("Lit cannot have compound type {:?}", other),
        },
        Expr::Var(i) => bindings[*i as usize].clone(),
        Expr::BinOp(op, lhs, rhs) => {
            let l = eval(lhs, bindings, callees);
            let r = eval(rhs, bindings, callees);
            eval_binop(*op, l, r)
        }
        Expr::UnOp(op, inner) => {
            let v = eval(inner, bindings, callees);
            eval_unop(*op, v)
        }
        Expr::Let { init, body } => {
            let v = eval(init, bindings, callees);
            bindings.push(v);
            let result = eval(body, bindings, callees);
            bindings.pop();
            result
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            if eval_bool(cond, bindings, callees) {
                eval(then_branch, bindings, callees)
            } else {
                eval(else_branch, bindings, callees)
            }
        }
        Expr::Resize { inner, to_width } => {
            let v = eval(inner, bindings, callees);
            match v {
                Value::Bits(raw, _) => Value::bits(raw, *to_width),
                Value::Signed(raw, _) => Value::signed(raw, *to_width),
                other => panic!("Resize on non-scalar value {:?}", other),
            }
        }
        Expr::Cast { inner, to } => {
            let v = eval(inner, bindings, callees);
            match (v, to) {
                (Value::Bits(raw, w), Type::Signed(tw)) => {
                    debug_assert_eq!(w, *tw);
                    Value::signed(raw as i128, w)
                }
                (Value::Signed(raw, w), Type::Bits(tw)) => {
                    debug_assert_eq!(w, *tw);
                    let as_unsigned =
                        (raw as u128) & if w >= 128 { !0u128 } else { (1u128 << w) - 1 };
                    Value::bits(as_unsigned, w)
                }
                (v, _) => v,
            }
        }
        Expr::Shift {
            kind,
            value,
            amount,
        } => {
            let v = eval(value, bindings, callees);
            let amt = eval(amount, bindings, callees);
            let amt_raw = amt.expect_bits_raw() as u32;
            // Clamp shift amount to width to avoid overshift UB on the
            // u128/i128 host. RHDL's actual semantics for over-shift are
            // implementation-defined; this matches typical hardware.
            let safe_amt = amt_raw.min(v.width() as u32);
            match (v, kind) {
                (Value::Bits(raw, w), ShiftKind::Shl) => {
                    Value::bits(if safe_amt >= 128 { 0 } else { raw << safe_amt }, w)
                }
                (Value::Bits(raw, w), ShiftKind::Shr) => {
                    Value::bits(if safe_amt >= 128 { 0 } else { raw >> safe_amt }, w)
                }
                (Value::Signed(raw, w), ShiftKind::Shl) => {
                    // Logical-equivalent on the underlying bits, then re-mask.
                    let shifted = if safe_amt >= 128 {
                        0
                    } else {
                        (raw as u128).wrapping_shl(safe_amt) as i128
                    };
                    Value::signed(shifted, w)
                }
                (Value::Signed(raw, w), ShiftKind::Shr) => {
                    // Arithmetic shift on i128 sign-extends.
                    let shifted = if safe_amt >= 128 {
                        if raw < 0 { -1 } else { 0 }
                    } else {
                        raw >> safe_amt
                    };
                    Value::signed(shifted, w)
                }
                (other, _) => panic!("Shift on non-scalar value {:?}", other),
            }
        }
        Expr::TupleCtor(elems) => {
            Value::Tuple(elems.iter().map(|e| eval(e, bindings, callees)).collect())
        }
        Expr::TupleProj { inner, index } => match eval(inner, bindings, callees) {
            Value::Tuple(elems) => elems[*index].clone(),
            other => panic!("TupleProj on non-tuple value {:?}", other),
        },
        Expr::ArrayCtor(elems) => {
            Value::Array(elems.iter().map(|e| eval(e, bindings, callees)).collect())
        }
        Expr::ArrayRepeat { value, len } => {
            // Evaluate the element once and replicate. Matches RHIF
            // semantics: `[x; N]` does not re-evaluate `x` per slot.
            let v = eval(value, bindings, callees);
            Value::Array(vec![v; *len])
        }
        Expr::ArrayIndex { inner, index } => {
            let arr = eval(inner, bindings, callees);
            let idx = eval(index, bindings, callees).expect_bits_raw() as usize;
            match arr {
                Value::Array(elems) => elems[idx].clone(),
                other => panic!("ArrayIndex on non-array value {:?}", other),
            }
        }
        Expr::Call { callee, args } => {
            let callee_def = callees.get(*callee).unwrap_or_else(|| {
                panic!(
                    "Expr::Call references callee {} but only {} callees are available \
                     (single-kernel `interpret` does not support Expr::Call)",
                    callee,
                    callees.len()
                )
            });
            let arg_values: Vec<Value> = args.iter().map(|a| eval(a, bindings, callees)).collect();
            eval_kernel(callee_def, &arg_values, callees)
        }
        Expr::StructCtor { struct_id, fields } => {
            let field_values: Vec<Value> =
                fields.iter().map(|e| eval(e, bindings, callees)).collect();
            Value::Struct(*struct_id, field_values)
        }
        Expr::FieldAccess { inner, field_idx } => match eval(inner, bindings, callees) {
            Value::Struct(_, fields) => fields[*field_idx].clone(),
            other => panic!("FieldAccess on non-struct value {:?}", other),
        },
        Expr::StructUpdate {
            struct_id,
            base,
            overrides,
        } => {
            let base_val = eval(base, bindings, callees);
            let mut fields = match base_val {
                Value::Struct(sid, fs) => {
                    debug_assert_eq!(sid, *struct_id, "StructUpdate base struct_id mismatch");
                    fs
                }
                other => panic!("StructUpdate base evaluated to non-struct {:?}", other),
            };
            for (field_idx, expr) in overrides {
                fields[*field_idx] = eval(expr, bindings, callees);
            }
            Value::Struct(*struct_id, fields)
        }
        Expr::EnumCtor {
            enum_id,
            variant_idx,
            payload,
        } => {
            let payload_values: Vec<Value> =
                payload.iter().map(|e| eval(e, bindings, callees)).collect();
            Value::Enum {
                enum_id: *enum_id,
                variant_idx: *variant_idx,
                payload: payload_values,
            }
        }
        Expr::Match { scrutinee, arms } => {
            let scrut = eval(scrutinee, bindings, callees);
            let (sid, vidx, payload) = match scrut {
                Value::Enum {
                    enum_id,
                    variant_idx,
                    payload,
                } => (enum_id, variant_idx, payload),
                other => panic!("Match scrutinee evaluated to non-enum {:?}", other),
            };
            let arm = arms
                .iter()
                .find(|a| a.enum_id == sid && a.variant_idx == vidx)
                .unwrap_or_else(|| {
                    panic!(
                        "no Match arm for enum {} variant {} (non-exhaustive)",
                        sid, vidx
                    )
                });
            // Push payload values onto the binding stack in declaration
            // order, evaluate the body, then pop them back off.
            let n_pushed = payload.len();
            bindings.extend(payload);
            let result = eval(&arm.body, bindings, callees);
            bindings.truncate(bindings.len() - n_pushed);
            result
        }
        Expr::Block { stmts, tail } => {
            // Track how many bindings the block introduced so we can
            // pop them all at the end. `Assign` does not introduce a
            // binding, so it doesn't change the count.
            let start_len = bindings.len();
            for stmt in stmts {
                match stmt {
                    Stmt::Let(init) | Stmt::LetMut(init) => {
                        let v = eval(init, bindings, callees);
                        bindings.push(v);
                    }
                    Stmt::Assign { var_idx, value } => {
                        let v = eval(value, bindings, callees);
                        bindings[*var_idx as usize] = v;
                    }
                }
            }
            let result = eval(tail, bindings, callees);
            bindings.truncate(start_len);
            result
        }
        Expr::For {
            loop_var_width,
            range_hi,
            body,
        } => {
            // The loop variable occupies one binding slot for the
            // duration of each iteration. Push a fresh value, evaluate
            // the body for its side effects (assigns to outer mutable
            // bindings), then pop. The for expression's value is unit;
            // we return an `()`-like marker by using a zero-width Bits
            // value -- in practice the caller will be inside a `Block`
            // and discard the result.
            //
            // RHDL unrolls `for i in 0..N` at compile time and inlines
            // `i` as a literal coerced to `i32`; inference in the
            // absence of an explicit cast types it as **signed**. The
            // value bound here must therefore be `Value::Signed` so
            // downstream binops / Resize on the loop var preserve sign
            // semantics. See `LOOP_VAR_TYPE` in the generator crate.
            let start_len = bindings.len();
            for i in 0..*range_hi {
                bindings.push(Value::signed(i as i128, *loop_var_width));
                let _ = eval(body, bindings, callees);
                bindings.truncate(start_len);
            }
            // RHDL for-expressions are unit-typed; we return a 1-bit
            // sentinel value that any wrapping `Stmt::ExprStmt`/`Block`
            // will discard.
            Value::bits(0, 1)
        }
    }
}

fn eval_binop(op: BinOpKind, l: Value, r: Value) -> Value {
    assert_eq!(
        l.width(),
        r.width(),
        "binop {:?} on mismatched widths {} vs {}",
        op,
        l.width(),
        r.width()
    );
    assert_eq!(
        l.is_signed(),
        r.is_signed(),
        "binop {:?} on mismatched signedness",
        op
    );
    let width = l.width();
    match (l, r) {
        (Value::Bits(lv, _), Value::Bits(rv, _)) => {
            let raw = match op {
                BinOpKind::Add => lv.wrapping_add(rv),
                BinOpKind::Sub => lv.wrapping_sub(rv),
                BinOpKind::Mul => lv.wrapping_mul(rv),
                BinOpKind::BitAnd => lv & rv,
                BinOpKind::BitOr => lv | rv,
                BinOpKind::BitXor => lv ^ rv,
            };
            Value::bits(raw, width)
        }
        (Value::Signed(lv, _), Value::Signed(rv, _)) => {
            let raw = match op {
                BinOpKind::Add => lv.wrapping_add(rv),
                BinOpKind::Sub => lv.wrapping_sub(rv),
                BinOpKind::Mul => lv.wrapping_mul(rv),
                BinOpKind::BitAnd => lv & rv,
                BinOpKind::BitOr => lv | rv,
                BinOpKind::BitXor => lv ^ rv,
            };
            Value::signed(raw, width)
        }
        _ => unreachable!("signedness mismatch already checked"),
    }
}

fn eval_unop(op: UnOpKind, v: Value) -> Value {
    match (op, v) {
        (UnOpKind::Not, Value::Bits(raw, w)) => Value::bits(!raw, w),
        (UnOpKind::Not, Value::Signed(raw, w)) => Value::signed(!raw, w),
        (UnOpKind::Neg, Value::Signed(raw, w)) => Value::signed(raw.wrapping_neg(), w),
        (UnOpKind::Neg, Value::Bits(_, _)) => panic!("UnOp::Neg is invalid on Value::Bits"),
        (op, other) => panic!("UnOp {:?} on non-scalar value {:?}", op, other),
    }
}

fn eval_bool(b: &BoolExpr, bindings: &mut Vec<Value>, callees: &[KernelDef]) -> bool {
    match b {
        BoolExpr::Lit(v) => *v,
        BoolExpr::Not(x) => !eval_bool(x, bindings, callees),
        BoolExpr::And(l, r) => eval_bool(l, bindings, callees) && eval_bool(r, bindings, callees),
        BoolExpr::Or(l, r) => eval_bool(l, bindings, callees) || eval_bool(r, bindings, callees),
        BoolExpr::Cmp(op, lhs, rhs) => {
            let l = eval(lhs, bindings, callees);
            let r = eval(rhs, bindings, callees);
            assert_eq!(
                l.width(),
                r.width(),
                "cmp on mismatched widths {} vs {}",
                l.width(),
                r.width()
            );
            assert_eq!(l.is_signed(), r.is_signed(), "cmp on mismatched signedness");
            match (l, r) {
                (Value::Bits(lv, _), Value::Bits(rv, _)) => match op {
                    CmpKind::Eq => lv == rv,
                    CmpKind::Ne => lv != rv,
                    CmpKind::Lt => lv < rv,
                    CmpKind::Le => lv <= rv,
                    CmpKind::Gt => lv > rv,
                    CmpKind::Ge => lv >= rv,
                },
                (Value::Signed(lv, _), Value::Signed(rv, _)) => match op {
                    CmpKind::Eq => lv == rv,
                    CmpKind::Ne => lv != rv,
                    CmpKind::Lt => lv < rv,
                    CmpKind::Le => lv <= rv,
                    CmpKind::Gt => lv > rv,
                    CmpKind::Ge => lv >= rv,
                },
                _ => unreachable!("signedness mismatch already checked"),
            }
        }
        BoolExpr::Reduction(kind, inner) => {
            let v = eval(inner, bindings, callees);
            let (raw, width) = match v {
                Value::Bits(r, w) => (r, w),
                Value::Signed(r, w) => (r as u128 & mask_to_width(u128::MAX, w), w),
                other => panic!("Reduction on non-scalar value {:?}", other),
            };
            let masked = mask_to_width(raw, width);
            let bit_count = masked.count_ones();
            match kind {
                // .any() == OR of all bits.
                ReductionKind::Any => bit_count > 0,
                // .all() == AND of all bits == every bit set.
                ReductionKind::All => bit_count as u8 == width,
                // .xor() == XOR of all bits == parity (1 iff odd count).
                ReductionKind::Xor => bit_count % 2 == 1,
            }
        }
    }
}
