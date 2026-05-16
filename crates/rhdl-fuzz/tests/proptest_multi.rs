//! Property test for multi-kernel programs.
//!
//! Generates small DAGs of kernels (top + 1-2 callees) and runs each
//! through the differential pipeline. Each generated case costs N
//! iverilog invocations (one per kernel rendered + netlist), so this
//! suite is gated behind `RHDL_FUZZ_MULTI=1` to keep routine `cargo
//! test` runs fast. Run thoroughly with:
//!
//! ```sh
//! RHDL_FUZZ_MULTI=1 PROPTEST_CASES=64 cargo test -p rhdl-fuzz \
//!     --test proptest_multi
//! ```

use proptest::prelude::*;
use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_fuzz::dsl::Type;
use rhdl_fuzz::generator::multi_program_strategy_v1;
use rhdl_fuzz::interpret::{Value, interpret_multi};
use rhdl_fuzz::multi::{MultiKernelProgram, render_multi};

type Sig = Signal<Bits<8>, Red>;

fn val_bits<const W: usize>(s: Signal<Bits<W>, Red>) -> Value
where
    rhdl::bits::W<W>: BitWidth,
{
    Value::bits(s.val().raw(), W as u8)
}

fn sig_bits<const W: usize>(v: Value) -> Signal<Bits<W>, Red>
where
    rhdl::bits::W<W>: BitWidth,
{
    signal(bits(v.expect_bits_raw()))
}

fn check_top_arity_2(program: &MultiKernelProgram) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = render_multi(program);
    let p = program.clone();
    let native = move |a: Sig, b: Sig| -> Sig {
        sig_bits::<8>(interpret_multi(&p, &[val_bits::<8>(a), val_bits::<8>(b)]))
    };
    let cases = [
        (0u128, 0u128),
        (1, 2),
        (0xFF, 0x01),
        (0x80, 0x80),
        (42, 100),
    ]
    .map(|(a, b)| {
        (
            signal::<Bits<8>, Red>(bits(a)),
            signal::<Bits<8>, Red>(bits(b)),
        )
    });
    test_kernel_vm_and_verilog_ast(
        kernel,
        native,
        cases.into_iter(),
        CompilationMode::Asynchronous,
    )?;
    Ok(())
}

fn enabled() -> bool {
    std::env::var("RHDL_FUZZ_MULTI").is_ok()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8),
        // See proptest_arithmetic.rs for rationale.
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    /// 2-kernel programs (top + 1 callee). Top has arity 2 over Bits<8>.
    #[test]
    fn multi_2_kernels(
        program in multi_program_strategy_v1(
            vec![Type::Bits(8), Type::Bits(8)],
            Type::Bits(8),
            1, // 1 callee
        )
    ) {
        if !enabled() { return Ok(()); }
        check_top_arity_2(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    /// 3-kernel programs (top + 2 callees). Top has arity 2 over Bits<8>.
    #[test]
    fn multi_3_kernels(
        program in multi_program_strategy_v1(
            vec![Type::Bits(8), Type::Bits(8)],
            Type::Bits(8),
            2, // 2 callees
        )
    ) {
        if !enabled() { return Ok(()); }
        check_top_arity_2(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }
}
