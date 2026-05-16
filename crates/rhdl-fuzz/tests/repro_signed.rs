//! Reproducer: dump the generated Verilog for a minimal signed-cast
//! kernel that the fuzzer flagged. Used to confirm whether the bug is
//! the localparam-signedness issue at builder.rs:26-37.

use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::compiler::driver::compile_kernel;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_fuzz::dsl::{BinOpKind, BoolExpr, CmpKind, Expr, NominalRegistry, Program, Type};
use rhdl_fuzz::interpret::{Value, interpret};
use rhdl_fuzz::render::render;

#[test]
fn dump_failing_signed_cast_kernel() {
    // Minimal extract from the shrunken fuzzer failure:
    //
    //   fn k(_a: Signed<8>) -> Signed<8> {
    //       (b8(127) | b8(255)) as_signed::<8>()
    //   }
    //
    // Pure-Rust expected: Signed<8>(-1) = 0xFF.
    // Suspected Verilog bug: localparams emitted as `8'b...` (unsigned).
    let program = Program {
        arg_types: vec![Type::Signed(8)],
        ret_type: Type::Signed(8),
        body: Expr::Cast {
            inner: Box::new(Expr::BinOp(
                BinOpKind::BitOr,
                Box::new(Expr::Lit {
                    value: 127,
                    ty: Type::Bits(8),
                }),
                Box::new(Expr::Lit {
                    value: 255,
                    ty: Type::Bits(8),
                }),
            )),
            to: Type::Signed(8),
        },
        nominal: NominalRegistry::new(),
    };

    println!("=== Interpreter result ===");
    let interp_result = interpret(&program, &[Value::signed(0, 8)]);
    println!("{:?}", interp_result);

    let kernel = render(&program);
    let rtl = compile_kernel(kernel, CompilationMode::Asynchronous).expect("compile");

    println!("\n=== Generated Verilog ===");
    let func = rtl.as_vlog().expect("as_vlog");
    println!("{}", func.pretty());
}

/// Run the simplified body through the differential pipeline. This one
/// PASSES -- so the bug is somewhere in the wrapping structure, not in
/// the cast itself.
#[test]
fn run_simplified_signed_kernel_through_pipeline() {
    type SSig8 = Signal<SignedBits<8>, Red>;
    let program = Program {
        arg_types: vec![Type::Signed(8)],
        ret_type: Type::Signed(8),
        body: Expr::Cast {
            inner: Box::new(Expr::BinOp(
                BinOpKind::BitOr,
                Box::new(Expr::Lit {
                    value: 127,
                    ty: Type::Bits(8),
                }),
                Box::new(Expr::Lit {
                    value: 255,
                    ty: Type::Bits(8),
                }),
            )),
            to: Type::Signed(8),
        },
        nominal: NominalRegistry::new(),
    };
    let kernel = render(&program);
    let p = program.clone();
    let native = move |a: SSig8| -> SSig8 {
        let v = interpret(&p, &[Value::signed(a.val().raw() as i128, 8)]);
        signal(signed::<8>(v.expect_signed_raw()))
    };
    // One input case is enough -- the kernel ignores its argument.
    let inputs = std::iter::once((signal::<SignedBits<8>, Red>(signed(0)),));
    let result =
        test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous);
    println!("Simplified pipeline result: {result:?}");
    result.unwrap();
}

/// Reproduce the EXACT shrunken program proptest found from the latest
/// seed replay. The cond uses `Resize { Lit{value:3, ty:Signed(2)}, 1 }`
/// where the value 3 in `Signed(2)` is the bit pattern `11` = -1.
#[test]
fn run_full_shrunken_program() {
    type SSig8 = Signal<SignedBits<8>, Red>;

    fn lit_bits(n: u128, w: u8) -> Box<Expr> {
        Box::new(Expr::Lit {
            value: n,
            ty: Type::Bits(w),
        })
    }
    fn lit_signed(n: u128, w: u8) -> Box<Expr> {
        Box::new(Expr::Lit {
            value: n,
            ty: Type::Signed(w),
        })
    }
    fn var(i: u8) -> Box<Expr> {
        Box::new(Expr::Var(i))
    }

    let body = Expr::If {
        cond: Box::new(BoolExpr::Not(Box::new(BoolExpr::Cmp(
            CmpKind::Lt,
            Box::new(Expr::Resize {
                inner: lit_signed(3, 2), // 3 in Signed(2) = bit pattern 11 = -1
                to_width: 1,
            }),
            Box::new(Expr::Resize {
                inner: Box::new(Expr::Var(1)),
                to_width: 1,
            }),
        )))),
        then_branch: Box::new(Expr::Let {
            init: Box::new(Expr::If {
                cond: Box::new(BoolExpr::Or(
                    Box::new(BoolExpr::Lit(false)),
                    Box::new(BoolExpr::Lit(true)),
                )),
                then_branch: var(0),
                else_branch: var(0),
            }),
            body: Box::new(Expr::Cast {
                inner: Box::new(Expr::BinOp(
                    BinOpKind::BitOr,
                    lit_bits(127, 8),
                    lit_bits(255, 8),
                )),
                to: Type::Signed(8),
            }),
        }),
        else_branch: Box::new(Expr::BinOp(
            BinOpKind::Mul,
            var(0),
            Box::new(Expr::Let {
                init: Box::new(Expr::Let {
                    init: lit_signed(111, 8),
                    body: var(0),
                }),
                body: var(0),
            }),
        )),
    };
    let program = Program {
        arg_types: vec![Type::Signed(8), Type::Signed(8)],
        ret_type: Type::Signed(8),
        body,
        nominal: NominalRegistry::new(),
    };

    println!("=== Interpreter result for input (-128, -128) ===");
    let interp_result = interpret(&program, &[Value::signed(-128, 8), Value::signed(-128, 8)]);
    println!("{:?}", interp_result);

    let kernel = render(&program);
    let rtl = compile_kernel(kernel.clone(), CompilationMode::Asynchronous).expect("compile");
    println!("\n=== Generated Verilog ===");
    let func = rtl.as_vlog().expect("as_vlog");
    println!("{}", func.pretty());

    println!("\n=== Pipeline run ===");
    let p = program.clone();
    let native = move |a: SSig8, b: SSig8| -> SSig8 {
        let v = interpret(
            &p,
            &[
                Value::signed(a.val().raw() as i128, 8),
                Value::signed(b.val().raw() as i128, 8),
            ],
        );
        signal(signed::<8>(v.expect_signed_raw()))
    };
    let inputs = std::iter::once((
        signal::<SignedBits<8>, Red>(signed(-128)),
        signal::<SignedBits<8>, Red>(signed(-128)),
    ));
    let result =
        test_kernel_vm_and_verilog_ast(kernel, native, inputs, CompilationMode::Asynchronous);
    println!("Pipeline result: {result:?}");
}
