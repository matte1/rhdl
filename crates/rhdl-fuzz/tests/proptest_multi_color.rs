//! Property test for non-Red boundary clock domains.
//!
//! RHDL defines seven `Color` domains (Red..Violet). The existing
//! `proptest_arithmetic` suite pins the kernel boundary to `Red`, so
//! six of the seven are untouched by the fuzzer. This file runs the
//! same `Bits<8>` body generator across several non-Red colors via
//! `render_with_color`, giving the clock-domain check, signal
//! coercion, and Verilog domain emission paths exercise across the
//! whole color set.
//!
//! Coverage choice: one program shape (single-arg `Bits<8>`) varied
//! across several colors is cheaper than every (shape × color) pair
//! and still distinguishes color-specific lowering bugs from
//! shape-specific ones. We pick Orange / Green / Violet -- one
//! near-Red color, one middle, one at the far end of the enum.

use proptest::prelude::*;
use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_fuzz::dsl::{Program, Type};
use rhdl_fuzz::generator::program_strategy_typed;
use rhdl_fuzz::interpret::{Value, interpret};
use rhdl_fuzz::render::render_with_color;

const INPUT_SAMPLES: [u128; 6] = [0, 1, 0x7F, 0x80, 0xFE, 0xFF];

/// Run `program` through the differential pipeline with the kernel
/// boundary pinned to `C`'s clock domain.
fn check_for_color<C: Domain>(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |a: Signal<Bits<8>, C>| -> Signal<Bits<8>, C> {
        let v = interpret(&p, &[Value::bits(a.val().raw(), 8)]);
        signal(bits(v.expect_bits_raw()))
    };
    let cases = INPUT_SAMPLES.map(|n| (signal::<Bits<8>, C>(bits(n)),));
    let kernel = render_with_color(program, <C as Domain>::color());
    test_kernel_vm_and_verilog_ast(
        kernel,
        native,
        cases.into_iter(),
        CompilationMode::Asynchronous,
    )?;
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64),
        // See proptest_arithmetic.rs for rationale.
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn multi_color_orange(
        program in program_strategy_typed(vec![Type::Bits(8)], Type::Bits(8))
    ) {
        check_for_color::<Orange>(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn multi_color_green(
        program in program_strategy_typed(vec![Type::Bits(8)], Type::Bits(8))
    ) {
        check_for_color::<Green>(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn multi_color_violet(
        program in program_strategy_typed(vec![Type::Bits(8)], Type::Bits(8))
    ) {
        check_for_color::<Violet>(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }
}
