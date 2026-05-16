//! Hand-rolled regression tests for the DSL pipeline.
//!
//! **Charter (read before adding more tests here):**
//!
//! These tests exist to give a *deterministic, fast* signal that
//! `dsl + interpret + render` round-trips end-to-end for specific shapes
//! that are hard to generate randomly or that have caused real bugs.
//! They are NOT the place to test "does feature X work" -- the proptest
//! suite covers that ground much more thoroughly with random input.
//!
//! Add a sanity test only if it falls into one of these categories:
//!
//!   1. **Smallest case for a new feature.** When you add (e.g.) tuples
//!      to the DSL, one sanity test exercising a tuple round-trip is
//!      worth keeping as a "did the renderer/interpret addition land?"
//!      smoke test that runs in milliseconds without proptest setup.
//!
//!   2. **A specific shrunk failing program from a bug.** If proptest
//!      finds a real bug, lift the shrunk program here so the test
//!      survives across regressions.
//!
//!   3. **A pattern that the generator can't produce yet** but should
//!      still type-check and run -- e.g., a deeply nested let-of-let
//!      that exceeds proptest's depth budget.
//!
//! Everything else (operator interactions, edge values, signedness
//! interactions) belongs in the proptest suite, not here. The reason
//! the M2 sanity tests didn't catch the localparam-signedness bug was
//! exactly that they tested features in isolation; the bug needs three
//! features interacting, and the proptest's random program generation
//! found it within ~3 minutes.

use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_fuzz::dsl::{
    BinOpKind, BoolExpr, CmpKind, Expr, NominalRegistry, Program, Type, UnOpKind,
};
use rhdl_fuzz::interpret::{Value, interpret};
use rhdl_fuzz::render::render;

type Sig = Signal<Bits<8>, Red>;

fn lit(n: u8) -> Box<Expr> {
    Box::new(Expr::Lit {
        value: u128::from(n),
        ty: Type::Bits(8),
    })
}
fn var(i: u8) -> Box<Expr> {
    Box::new(Expr::Var(i))
}
fn binop(op: BinOpKind, l: Box<Expr>, r: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::BinOp(op, l, r))
}
fn unop(op: UnOpKind, x: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::UnOp(op, x))
}

/// Convenience: build a mono-typed `Bits(8)` program with `arity` inputs.
fn mono_program(arity: u8, body: Expr) -> Program {
    Program {
        arg_types: vec![Type::Bits(8); arity as usize],
        ret_type: Type::Bits(8),
        body,
        nominal: NominalRegistry::new(),
    }
}

/// Lift a `Sig` to a `Value::Bits` for the interpreter.
fn from_sig(s: Sig) -> Value {
    Value::bits(s.val().raw(), 8)
}

/// Drop a `Value::Bits(_, 8)` back to a `Sig`.
fn to_sig(v: Value) -> Sig {
    signal(bits(v.expect_bits_raw()))
}

#[test]
fn sanity_identity_arity_1() -> Result<(), Box<dyn std::error::Error>> {
    let program = mono_program(1, *var(0));
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 42, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_struct_update_overrides_field() -> Result<(), Box<dyn std::error::Error>> {
    // Build S0 { f0: 0, f1: 0 } as base, override f0 to `a`, project f0:
    //   body: S0 { f0: a, ..S0 { f0: 0, f1: 0 } }.f0  ==  a
    use rhdl_fuzz::dsl::{FIELD_NAMES, STRUCT_NAMES, StructDef};
    let nominal = NominalRegistry {
        structs: vec![StructDef {
            name: STRUCT_NAMES[0],
            fields: vec![
                (FIELD_NAMES[0], Type::Bits(8)),
                (FIELD_NAMES[1], Type::Bits(8)),
            ],
        }],
        enums: vec![],
    };
    let base = Expr::StructCtor {
        struct_id: 0,
        fields: vec![
            Expr::Lit {
                value: 0,
                ty: Type::Bits(8),
            },
            Expr::Lit {
                value: 0,
                ty: Type::Bits(8),
            },
        ],
    };
    let program = Program {
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::FieldAccess {
            inner: Box::new(Expr::StructUpdate {
                struct_id: 0,
                base: Box::new(base),
                overrides: vec![(0, *var(0))],
            }),
            field_idx: 0,
        },
        nominal,
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 42, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_array_repeat_index_picks_value() -> Result<(), Box<dyn std::error::Error>> {
    // Smallest case for `[x; N]`: body is `[a; 4][2]`, should equal `a`.
    let program = mono_program(
        1,
        Expr::ArrayIndex {
            inner: Box::new(Expr::ArrayRepeat {
                value: var(0),
                len: 4,
            }),
            index: Box::new(Expr::Lit {
                value: 2,
                ty: Type::Bits(2),
            }),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 42, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_reduction_any_in_if() -> Result<(), Box<dyn std::error::Error>> {
    // Smallest case for `.any()`: `if a.any() { 1 } else { 0 }`
    use rhdl_fuzz::dsl::ReductionKind;
    let program = mono_program(
        1,
        Expr::If {
            cond: Box::new(BoolExpr::Reduction(ReductionKind::Any, var(0))),
            then_branch: Box::new(Expr::Lit {
                value: 1,
                ty: Type::Bits(8),
            }),
            else_branch: Box::new(Expr::Lit {
                value: 0,
                ty: Type::Bits(8),
            }),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 0x80, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_sync_identity_arity_1() -> Result<(), Box<dyn std::error::Error>> {
    // Synchronous mode equivalent of the above: kernel shape is
    // `fn k(_cr: ClockReset, a: Bits<8>, _q: ()) -> (Bits<8>, ())`.
    use rhdl_core::types::clock_reset::ClockReset;
    use rhdl_fuzz::render::render_sync;
    let program = mono_program(1, *var(0));
    let kernel = render_sync(&program);
    let p = program.clone();
    let native = move |_cr: ClockReset, a: Bits<8>, _q: ()| -> (Bits<8>, ()) {
        let v = interpret(&p, &[Value::bits(a.raw(), 8)]);
        (bits(v.expect_bits_raw()), ())
    };
    let inputs = [0u128, 1, 42, 0xFF].into_iter().map(|n| {
        (
            rhdl_core::types::clock_reset::clock_reset(
                rhdl::prelude::clock(true),
                rhdl::prelude::reset(false),
            ),
            bits::<8>(n),
            (),
        )
    });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Synchronous)?;
    Ok(())
}

#[test]
fn sanity_a_plus_b() -> Result<(), Box<dyn std::error::Error>> {
    let program = mono_program(2, *binop(BinOpKind::Add, var(0), var(1)));
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(0u128, 0u128), (1, 2), (255, 1), (100, 200), (42, 42)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_let_binding() -> Result<(), Box<dyn std::error::Error>> {
    // body: { let t0 = a + b; t0 ^ t0 }
    let program = mono_program(
        2,
        Expr::Let {
            init: binop(BinOpKind::Add, var(0), var(1)),
            body: binop(BinOpKind::BitXor, var(2), var(2)),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (10, 20), (0xFF, 0x01)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_nested_let() -> Result<(), Box<dyn std::error::Error>> {
    // body: { let t0 = a; { let t1 = t0 + a; t1 * 2 } }
    let program = mono_program(
        1,
        Expr::Let {
            init: var(0),
            body: Box::new(Expr::Let {
                init: binop(BinOpKind::Add, var(1), var(0)),
                body: binop(BinOpKind::Mul, var(2), lit(2)),
            }),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 3, 50, 0x7F]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_if_else() -> Result<(), Box<dyn std::error::Error>> {
    // body: if a < b { a + b } else { a - b }
    let program = mono_program(
        2,
        Expr::If {
            cond: Box::new(BoolExpr::Cmp(CmpKind::Lt, var(0), var(1))),
            then_branch: binop(BinOpKind::Add, var(0), var(1)),
            else_branch: binop(BinOpKind::Sub, var(0), var(1)),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (5, 3), (10, 10), (0xFF, 0)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_nested_if_with_bool_combinators() -> Result<(), Box<dyn std::error::Error>> {
    // body:
    //   if !(a == 0) && (b > 10) {
    //       a * b
    //   } else {
    //       if a == b { 0xFF } else { a ^ b }
    //   }
    let program = mono_program(
        2,
        Expr::If {
            cond: Box::new(BoolExpr::And(
                Box::new(BoolExpr::Not(Box::new(BoolExpr::Cmp(
                    CmpKind::Eq,
                    var(0),
                    lit(0),
                )))),
                Box::new(BoolExpr::Cmp(CmpKind::Gt, var(1), lit(10))),
            )),
            then_branch: binop(BinOpKind::Mul, var(0), var(1)),
            else_branch: Box::new(Expr::If {
                cond: Box::new(BoolExpr::Cmp(CmpKind::Eq, var(0), var(1))),
                then_branch: lit(0xFF),
                else_branch: binop(BinOpKind::BitXor, var(0), var(1)),
            }),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(0u128, 0u128), (0, 20), (5, 5), (3, 11), (0xFF, 0x80)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M1: multiple bit widths + Resize ----------

#[test]
fn sanity_width_16_addition() -> Result<(), Box<dyn std::error::Error>> {
    // body: a + 0x1234 at width 16
    type Sig16 = Signal<Bits<16>, Red>;
    let program = Program {
        arg_types: vec![Type::Bits(16)],
        ret_type: Type::Bits(16),
        body: Expr::BinOp(
            BinOpKind::Add,
            var(0),
            Box::new(Expr::Lit {
                value: 0x1234,
                ty: Type::Bits(16),
            }),
        ),
        nominal: NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig16| -> Sig16 {
        let v = interpret(&p, &[Value::bits(a.val().raw(), 16)]);
        signal(bits(v.expect_bits_raw()))
    };
    let inputs = [0u128, 1, 0xFFFF, 0xCAFE]
        .into_iter()
        .map(|n| (signal::<Bits<16>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_width_1_xor() -> Result<(), Box<dyn std::error::Error>> {
    // body: a ^ b at width 1 (single bit XOR == NEQ)
    type Sig1 = Signal<Bits<1>, Red>;
    let program = Program {
        arg_types: vec![Type::Bits(1), Type::Bits(1)],
        ret_type: Type::Bits(1),
        body: Expr::BinOp(BinOpKind::BitXor, var(0), var(1)),
        nominal: NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig1, b: Sig1| -> Sig1 {
        let v = interpret(
            &p,
            &[Value::bits(a.val().raw(), 1), Value::bits(b.val().raw(), 1)],
        );
        signal(bits(v.expect_bits_raw()))
    };
    let inputs = [(0u128, 0u128), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<1>, Red>(bits(a)),
                signal::<Bits<1>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_resize_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    // body: ((a as u16) + 0x100) as u8
    // i.e. Resize(BinOp(Add, Resize(Var(0), 16), Lit{value:0x100, ty:Bits<16>}), 8)
    let program = mono_program(
        1,
        Expr::Resize {
            inner: Box::new(Expr::BinOp(
                BinOpKind::Add,
                Box::new(Expr::Resize {
                    inner: var(0),
                    to_width: 16,
                }),
                Box::new(Expr::Lit {
                    value: 0x100,
                    ty: Type::Bits(16),
                }),
            )),
            to_width: 8,
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    // For each a in {0, 1, 0xFF}, expected = ((a + 0x100) & 0xFF) = a (since 0x100 & 0xFF == 0)
    let inputs = [0u128, 1, 42, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M2: signed types, cast, shift ----------

#[test]
fn sanity_signed_8_addition() -> Result<(), Box<dyn std::error::Error>> {
    // body: a + (-1 as Signed<8>)
    type SSig8 = Signal<SignedBits<8>, Red>;
    let program = Program {
        arg_types: vec![Type::Signed(8)],
        ret_type: Type::Signed(8),
        body: Expr::BinOp(
            BinOpKind::Add,
            var(0),
            Box::new(Expr::Lit {
                value: (-1i8) as u128, // bit pattern of -1 in low 8 bits
                ty: Type::Signed(8),
            }),
        ),
        nominal: NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: SSig8| -> SSig8 {
        let v = interpret(&p, &[Value::signed(a.val().raw() as i128, 8)]);
        let raw = v.expect_signed_raw();
        // Build SignedBits<8> from the i128 value (low 8 bits).
        signal(signed::<8>(raw))
    };
    let inputs = [-128i64, -1, 0, 1, 42, 127]
        .into_iter()
        .map(|n| (signal::<SignedBits<8>, Red>(signed(n as i128)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_signed_arithmetic_shift_right() -> Result<(), Box<dyn std::error::Error>> {
    // body: a >> 2 where a: Signed<8>; checks arithmetic-shift semantics
    type SSig8 = Signal<SignedBits<8>, Red>;
    let program = Program {
        arg_types: vec![Type::Signed(8)],
        ret_type: Type::Signed(8),
        body: Expr::Shift {
            kind: rhdl_fuzz::dsl::ShiftKind::Shr,
            value: var(0),
            // Shift amount must be unsigned Bits<8>.
            amount: Box::new(Expr::Lit {
                value: 2,
                ty: Type::Bits(8),
            }),
        },
        nominal: NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: SSig8| -> SSig8 {
        let v = interpret(&p, &[Value::signed(a.val().raw() as i128, 8)]);
        signal(signed::<8>(v.expect_signed_raw()))
    };
    let inputs = [-128i64, -64, -1, 0, 1, 64, 127]
        .into_iter()
        .map(|n| (signal::<SignedBits<8>, Red>(signed(n as i128)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_signed_unsigned_cast_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    // body: ((a as Signed<8>) + 1) as Bits<8>
    let program = mono_program(
        1,
        Expr::Cast {
            inner: Box::new(Expr::BinOp(
                BinOpKind::Add,
                Box::new(Expr::Cast {
                    inner: var(0),
                    to: Type::Signed(8),
                }),
                Box::new(Expr::Lit {
                    value: 1,
                    ty: Type::Signed(8),
                }),
            )),
            to: Type::Bits(8),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 0x7F, 0x80, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M3: tuples + arrays ----------

#[test]
fn sanity_tuple_proj_returns_first() -> Result<(), Box<dyn std::error::Error>> {
    // body: (a, b).0  ==  a
    let program = mono_program(
        2,
        Expr::TupleProj {
            inner: Box::new(Expr::TupleCtor(vec![*var(0), *var(1)])),
            index: 0,
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (0xFF, 0x00), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_array_index_picks_element() -> Result<(), Box<dyn std::error::Error>> {
    // body: [a, b][1]  ==  b
    let program = mono_program(
        2,
        Expr::ArrayIndex {
            inner: Box::new(Expr::ArrayCtor(vec![*var(0), *var(1)])),
            index: Box::new(Expr::Lit {
                value: 1,
                ty: Type::Bits(1),
            }),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (0xFF, 0x00), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M5a: structs ----------

#[test]
fn sanity_struct_ctor_then_field_access() -> Result<(), Box<dyn std::error::Error>> {
    // Smallest case for struct support:
    //   struct S0 { f0: b8, f1: b8 }
    //   body: S0 { f0: a, f1: b }.f0  ==  a
    use rhdl_fuzz::dsl::{FIELD_NAMES, STRUCT_NAMES, StructDef};
    let nominal = NominalRegistry {
        structs: vec![StructDef {
            name: STRUCT_NAMES[0],
            fields: vec![
                (FIELD_NAMES[0], Type::Bits(8)),
                (FIELD_NAMES[1], Type::Bits(8)),
            ],
        }],
        enums: vec![],
    };
    let program = Program {
        arg_types: vec![Type::Bits(8), Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::FieldAccess {
            inner: Box::new(Expr::StructCtor {
                struct_id: 0,
                fields: vec![*var(0), *var(1)],
            }),
            field_idx: 0,
        },
        nominal,
    };
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (0xFF, 0x00), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M5b: enums + match ----------

#[test]
fn sanity_enum_ctor_then_match_payload() -> Result<(), Box<dyn std::error::Error>> {
    // Smallest case for enum + match:
    //   enum E0 { V0(b8), V1 }
    //   body: match E0::V0(a) { E0::V0(x) => x, E0::V1 => b }
    // Should always return `a` since the constructor selects V0.
    use rhdl_fuzz::dsl::{ENUM_NAMES, EnumDef, MatchArm, VARIANT_NAMES, VariantDef};
    let nominal = NominalRegistry {
        structs: vec![],
        enums: vec![EnumDef {
            name: ENUM_NAMES[0],
            variants: vec![
                VariantDef {
                    name: VARIANT_NAMES[0],
                    payload: vec![Type::Bits(8)],
                },
                VariantDef {
                    name: VARIANT_NAMES[1],
                    payload: vec![],
                },
            ],
            discriminant_width: 1,
        }],
    };
    // The match arm for V0 binds the payload as `t0` (let-depth 0).
    // Inside the body, `Var(2)` therefore resolves to that payload,
    // since arity=2 (a, b) and there's exactly one let-depth-0 binding.
    let body = Expr::Match {
        scrutinee: Box::new(Expr::EnumCtor {
            enum_id: 0,
            variant_idx: 0,
            payload: vec![*var(0)],
        }),
        arms: vec![
            MatchArm {
                enum_id: 0,
                variant_idx: 0,
                body: *var(2), // the bound payload `t0` (== a)
            },
            MatchArm {
                enum_id: 0,
                variant_idx: 1,
                body: *var(1), // dead branch; would return b
            },
        ],
    };
    let program = Program {
        arg_types: vec![Type::Bits(8), Type::Bits(8)],
        ret_type: Type::Bits(8),
        body,
        nominal,
    };
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (0xFF, 0x00), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_enum_unit_variant_arm() -> Result<(), Box<dyn std::error::Error>> {
    // Exercises the unit-variant arm (wild_pat) code path:
    //   enum E0 { V0, V1 }
    //   body: match E0::V1 { E0::V0 => a, E0::V1 => b }
    // Should always return `b`.
    use rhdl_fuzz::dsl::{ENUM_NAMES, EnumDef, MatchArm, VARIANT_NAMES, VariantDef};
    let nominal = NominalRegistry {
        structs: vec![],
        enums: vec![EnumDef {
            name: ENUM_NAMES[0],
            variants: vec![
                VariantDef {
                    name: VARIANT_NAMES[0],
                    payload: vec![],
                },
                VariantDef {
                    name: VARIANT_NAMES[1],
                    payload: vec![],
                },
            ],
            discriminant_width: 1,
        }],
    };
    let body = Expr::Match {
        scrutinee: Box::new(Expr::EnumCtor {
            enum_id: 0,
            variant_idx: 1,
            payload: vec![],
        }),
        arms: vec![
            MatchArm {
                enum_id: 0,
                variant_idx: 0,
                body: *var(0),
            },
            MatchArm {
                enum_id: 0,
                variant_idx: 1,
                body: *var(1),
            },
        ],
    };
    let program = Program {
        arg_types: vec![Type::Bits(8), Type::Bits(8)],
        ret_type: Type::Bits(8),
        body,
        nominal,
    };
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (0xFF, 0x00), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M6: statement-level features ----------

#[test]
fn sanity_let_mut_assign_block() -> Result<(), Box<dyn std::error::Error>> {
    // Smallest case for let mut + Assign:
    //   { let mut t0 = a; t0 = t0 + b; t0 }
    // Should equal a + b.
    use rhdl_fuzz::dsl::Stmt;
    let program = mono_program(
        2,
        Expr::Block {
            stmts: vec![
                Stmt::LetMut(*var(0)),
                Stmt::Assign {
                    var_idx: 2,
                    value: Expr::BinOp(BinOpKind::Add, var(2), var(1)),
                },
            ],
            tail: var(2),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native =
        move |a: Sig, b: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a), from_sig(b)])) };
    let inputs = [(1u128, 2u128), (10, 20), (0xFF, 0x01), (42, 100)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_for_loop_var_typed_signed() -> Result<(), Box<dyn std::error::Error>> {
    // Regression test for a fuzzer / RHDL signedness mismatch: RHDL
    // unrolls `for i in 0..N` at compile time and inlines `i` as a
    // literal coerced to `i32` -- inference in the absence of an
    // explicit cast treats `i` as **signed**. Earlier the fuzzer
    // modelled the loop var as `Bits<32>`, so e.g. `(!i).resize::<W>()`
    // sign-extended in RHDL but zero-extended in the interpreter,
    // producing pipeline mismatches.
    //
    // Body equivalent to:
    //   {
    //     let mut t0 = 0;
    //     for i in 0..4 {
    //       // (i + i) is Signed<32>; resize to Signed<8> (sign-truncate);
    //       // cast to Bits<8> for the unsigned accumulator.
    //       t0 += (i + i).resize::<8>().as_unsigned();
    //     }
    //     t0
    //   }
    // Expected: 0 + 2 + 4 + 6 = 12 (= 0x0C). With Bits<32> loop var
    // we'd compute the same, but the signedness path is the focus
    // here -- any wider extension that involves `!i` would diverge.
    use rhdl_fuzz::dsl::Stmt;
    use rhdl_fuzz::generator::{LOOP_VAR_TYPE, LOOP_VAR_WIDTH};
    let _ = LOOP_VAR_TYPE; // marker for the constant we depend on
    let program = mono_program(
        1,
        Expr::Block {
            stmts: vec![
                Stmt::LetMut(Expr::Lit {
                    value: 0,
                    ty: Type::Bits(8),
                }),
                Stmt::Let(Expr::For {
                    loop_var_width: LOOP_VAR_WIDTH,
                    range_hi: 4,
                    body: Box::new(Expr::Block {
                        stmts: vec![Stmt::Assign {
                            var_idx: 1, // t0
                            value: Expr::BinOp(
                                BinOpKind::Add,
                                var(1),
                                // (i + i).resize::<8>().as_unsigned() where i = Var(2)
                                Box::new(Expr::Cast {
                                    inner: Box::new(Expr::Resize {
                                        inner: Box::new(Expr::BinOp(
                                            BinOpKind::Add,
                                            var(2),
                                            var(2),
                                        )),
                                        to_width: 8,
                                    }),
                                    to: Type::Bits(8),
                                }),
                            ),
                        }],
                        tail: Box::new(Expr::Lit {
                            value: 0,
                            ty: Type::Bits(1),
                        }),
                    }),
                }),
            ],
            tail: var(1),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    // Input doesn't affect the result (body references only the loop
    // var and the accumulator); use a single deterministic case.
    let inputs = std::iter::once((signal::<Bits<8>, Red>(bits(0)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_for_loop_var_not_sign_extends() -> Result<(), Box<dyn std::error::Error>> {
    // Directly pins the signedness behavior: `(!i).resize::<64>()`
    // sign-extends because `i` is `Signed<32>`. Expected sum across
    // i in 0..4 is `-1 + -2 + -3 + -4 = -10`, then cast to Bits<8>
    // (truncating to low 8 bits of -10) = 0xF6.
    use rhdl_fuzz::dsl::{Stmt, UnOpKind};
    use rhdl_fuzz::generator::LOOP_VAR_WIDTH;
    let program = Program {
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::Block {
            stmts: vec![
                Stmt::LetMut(Expr::Lit {
                    value: 0,
                    ty: Type::Signed(8),
                }),
                Stmt::Let(Expr::For {
                    loop_var_width: LOOP_VAR_WIDTH,
                    range_hi: 4,
                    body: Box::new(Expr::Block {
                        stmts: vec![Stmt::Assign {
                            var_idx: 1, // s_acc: Signed<8>
                            value: Expr::BinOp(
                                BinOpKind::Add,
                                var(1),
                                // (!i).resize::<8>() preserves the
                                // signed sign-extension behavior.
                                Box::new(Expr::Resize {
                                    inner: Box::new(Expr::UnOp(UnOpKind::Not, var(2))),
                                    to_width: 8,
                                }),
                            ),
                        }],
                        tail: Box::new(Expr::Lit {
                            value: 0,
                            ty: Type::Bits(1),
                        }),
                    }),
                }),
            ],
            // Cast Signed<8> accumulator to Bits<8> as the kernel return.
            tail: Box::new(Expr::Cast {
                inner: var(1),
                to: Type::Bits(8),
            }),
        },
        nominal: rhdl_fuzz::dsl::NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = std::iter::once((signal::<Bits<8>, Red>(bits(0)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_for_loop_accumulator() -> Result<(), Box<dyn std::error::Error>> {
    // Accumulator pattern:
    //   { let mut t0 = 0; for i in 0..4 { t0 = t0 + a; } t0 }
    // Should equal 4 * a (mod 256).
    use rhdl_fuzz::dsl::Stmt;
    let program = mono_program(
        1,
        Expr::Block {
            stmts: vec![
                Stmt::LetMut(Expr::Lit {
                    value: 0,
                    ty: Type::Bits(8),
                }),
                // Index i (loop var) is at Var(2): arity 1 + LetMut at depth 0 (Var(1)) + i at depth 1 (Var(2)).
                Stmt::Let(Expr::For {
                    loop_var_width: 2, // covers 0..4
                    range_hi: 4,
                    // Loop body: { t0 = t0 + a; () }  -- expressed as a Block.
                    body: Box::new(Expr::Block {
                        stmts: vec![Stmt::Assign {
                            var_idx: 1,
                            value: Expr::BinOp(BinOpKind::Add, var(1), var(0)),
                        }],
                        tail: Box::new(Expr::Lit {
                            value: 0,
                            ty: Type::Bits(1),
                        }),
                    }),
                }),
                // After the for-loop, t0 holds 4 * a (mod 256).
            ],
            tail: var(1),
        },
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig { to_sig(interpret(&p, &[from_sig(a)])) };
    let inputs = [0u128, 1, 0x10, 0x40, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- M4: multi-kernel programs ----------

#[test]
fn sanity_two_kernel_call() -> Result<(), Box<dyn std::error::Error>> {
    // kernel_0: (x: b8) -> b8 { x + x }                      // doubler
    // kernel_1 (top): (a: b8, b: b8) -> b8 { kernel_0(a) + b } // calls kernel_0
    use rhdl_fuzz::interpret::interpret_multi;
    use rhdl_fuzz::multi::{KERNEL_NAMES, KernelDef, MultiKernelProgram, render_multi};

    let kernel_0 = KernelDef {
        name: KERNEL_NAMES[0],
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: *binop(BinOpKind::Add, var(0), var(0)),
    };
    let kernel_1 = KernelDef {
        name: KERNEL_NAMES[1],
        arg_types: vec![Type::Bits(8), Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::BinOp(
            BinOpKind::Add,
            Box::new(Expr::Call {
                callee: 0,
                args: vec![*var(0)],
            }),
            var(1),
        ),
    };
    let program = MultiKernelProgram {
        kernels: vec![kernel_0, kernel_1],
        top: 1,
        nominal: NominalRegistry::new(),
    };

    let kernel = render_multi(&program);
    let p = program.clone();
    let native = move |a: Sig, b: Sig| -> Sig {
        let v = interpret_multi(&p, &[from_sig(a), from_sig(b)]);
        to_sig(v)
    };
    let inputs = [(1u128, 2u128), (10, 20), (100, 100), (0xFF, 0x01)]
        .into_iter()
        .map(|(a, b)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

#[test]
fn sanity_three_kernel_diamond() -> Result<(), Box<dyn std::error::Error>> {
    // kernel_0: (x: b8) -> b8 { !x }              // bitwise complement
    // kernel_1: (x: b8) -> b8 { kernel_0(x) + 1 } // calls 0
    // kernel_2 (top): (a: b8) -> b8 { kernel_0(a) ^ kernel_1(a) }  // calls 0 and 1
    use rhdl_fuzz::interpret::interpret_multi;
    use rhdl_fuzz::multi::{KERNEL_NAMES, KernelDef, MultiKernelProgram, render_multi};

    let kernel_0 = KernelDef {
        name: KERNEL_NAMES[0],
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: *unop(UnOpKind::Not, var(0)),
    };
    let kernel_1 = KernelDef {
        name: KERNEL_NAMES[1],
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::BinOp(
            BinOpKind::Add,
            Box::new(Expr::Call {
                callee: 0,
                args: vec![*var(0)],
            }),
            lit(1),
        ),
    };
    let kernel_2 = KernelDef {
        name: KERNEL_NAMES[2],
        arg_types: vec![Type::Bits(8)],
        ret_type: Type::Bits(8),
        body: Expr::BinOp(
            BinOpKind::BitXor,
            Box::new(Expr::Call {
                callee: 0,
                args: vec![*var(0)],
            }),
            Box::new(Expr::Call {
                callee: 1,
                args: vec![*var(0)],
            }),
        ),
    };
    let program = MultiKernelProgram {
        kernels: vec![kernel_0, kernel_1, kernel_2],
        top: 2,
        nominal: NominalRegistry::new(),
    };

    let kernel = render_multi(&program);
    let p = program.clone();
    let native = move |a: Sig| -> Sig {
        let v = interpret_multi(&p, &[from_sig(a)]);
        to_sig(v)
    };
    let inputs = [0u128, 1, 0x7F, 0x80, 0xFF]
        .into_iter()
        .map(|n| (signal::<Bits<8>, Red>(bits(n)),));
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- existing tests ----------

#[test]
fn sanity_complex_expression() -> Result<(), Box<dyn std::error::Error>> {
    // body: !((a + b) ^ (c & 0xF0))
    let program = mono_program(
        3,
        *unop(
            UnOpKind::Not,
            binop(
                BinOpKind::BitXor,
                binop(BinOpKind::Add, var(0), var(1)),
                binop(BinOpKind::BitAnd, var(2), lit(0xF0)),
            ),
        ),
    );
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: Sig, b: Sig, c: Sig| -> Sig {
        to_sig(interpret(&p, &[from_sig(a), from_sig(b), from_sig(c)]))
    };
    let inputs = [(0u128, 0u128, 0u128), (1, 2, 3), (0xFF, 0x01, 0xAA)]
        .into_iter()
        .map(|(a, b, c)| {
            (
                signal::<Bits<8>, Red>(bits(a)),
                signal::<Bits<8>, Red>(bits(b)),
                signal::<Bits<8>, Red>(bits(c)),
            )
        });
    test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous)?;
    Ok(())
}
