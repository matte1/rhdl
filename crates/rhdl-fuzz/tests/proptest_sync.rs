//! Property test for `CompilationMode::Synchronous`.
//!
//! The single-kernel async proptest [`proptest_arithmetic`] exercises
//! the asynchronous lowering path end-to-end. This file does the same
//! for the synchronous path: every generated program is rendered as a
//! sync kernel via `render_sync` and pushed through the differential
//! pipeline with `CompilationMode::Synchronous`.
//!
//! Synchronous kernels have a different boundary shape:
//!   `fn k(_cr: ClockReset, a: I, _q: ()) -> (O, ())`
//! The body language is identical to async; what changes is the
//! lowering pass (clock-domain check is skipped for sync; sequential
//! state-element wiring is added). Reusing the same body generators
//! gives broad sync-mode coverage at little extra implementation cost.
//!
//! v1 covers arity-1 programs of width 1, 8, and 64 (mirroring the
//! async suite's width pool). Multi-arg sync kernels require either
//! tuple-bundling the input or a Vec<Pat> boundary -- deferrable.

use proptest::prelude::*;
use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_core::types::clock_reset::{ClockReset, clock_reset};
use rhdl_fuzz::dsl::{Program, Type};
use rhdl_fuzz::generator::program_strategy_typed;
use rhdl_fuzz::interpret::{Value, interpret};
use rhdl_fuzz::render::render_sync;

/// Render a sync kernel and run it through the differential pipeline
/// in `CompilationMode::Synchronous`.
fn run_sync_pipeline<F, Args, T0>(
    program: &Program,
    native: F,
    cases: impl Iterator<Item = Args> + Clone,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: rhdl_core::sim::testbench::kernel::Testable<Args, T0>,
    T0: Digital,
    Args: rhdl_core::sim::testbench::kernel::TestArg,
{
    let kernel = render_sync(program);
    test_kernel_vm_and_verilog_ast(kernel, native, cases, CompilationMode::Synchronous)?;
    Ok(())
}

/// One ClockReset value reused across every test case (rising clock,
/// reset deasserted). Sync kernels here have no state so the value of
/// the clock/reset doesn't affect the output.
fn cr() -> ClockReset {
    clock_reset(clock(true), reset(false))
}

// ---------- Width-parameterized check helpers ----------

fn check_sync_arity_1_width_8(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |_cr: ClockReset, a: Bits<8>, _q: ()| -> (Bits<8>, ()) {
        let v = interpret(&p, &[Value::bits(a.raw(), 8)]);
        (bits(v.expect_bits_raw()), ())
    };
    let cases = [0u128, 1, 0x7F, 0x80, 0xFE, 0xFF]
        .into_iter()
        .map(|n| (cr(), bits::<8>(n), ()));
    run_sync_pipeline(program, native, cases)
}

fn check_sync_arity_1_width_1(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |_cr: ClockReset, a: Bits<1>, _q: ()| -> (Bits<1>, ()) {
        let v = interpret(&p, &[Value::bits(a.raw(), 1)]);
        (bits(v.expect_bits_raw()), ())
    };
    let cases = [0u128, 1].into_iter().map(|n| (cr(), bits::<1>(n), ()));
    run_sync_pipeline(program, native, cases)
}

fn check_sync_arity_1_width_64(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |_cr: ClockReset, a: Bits<64>, _q: ()| -> (Bits<64>, ()) {
        let v = interpret(&p, &[Value::bits(a.raw(), 64)]);
        (bits(v.expect_bits_raw()), ())
    };
    let cases = [0u128, 1, u64::MAX as u128 / 2, u64::MAX as u128]
        .into_iter()
        .map(|n| (cr(), bits::<64>(n), ()));
    run_sync_pipeline(program, native, cases)
}

fn check_sync_arity_1_signed_8(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |_cr: ClockReset, a: SignedBits<8>, _q: ()| -> (SignedBits<8>, ()) {
        let v = interpret(&p, &[Value::signed(a.raw() as i128, 8)]);
        (signed::<8>(v.expect_signed_raw()), ())
    };
    let cases = [-128i128, -1, 0, 1, 42, 127]
        .into_iter()
        .map(|n| (cr(), signed::<8>(n), ()));
    run_sync_pipeline(program, native, cases)
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Default 64 cases; PROPTEST_CASES env var overrides.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64),
        // See proptest_arithmetic.rs for rationale.
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn sync_arity_1_width_8(
        program in program_strategy_typed(vec![Type::Bits(8)], Type::Bits(8))
    ) {
        check_sync_arity_1_width_8(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn sync_arity_1_width_1(
        program in program_strategy_typed(vec![Type::Bits(1)], Type::Bits(1))
    ) {
        check_sync_arity_1_width_1(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn sync_arity_1_width_64(
        program in program_strategy_typed(vec![Type::Bits(64)], Type::Bits(64))
    ) {
        check_sync_arity_1_width_64(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn sync_arity_1_signed_8(
        program in program_strategy_typed(vec![Type::Signed(8)], Type::Signed(8))
    ) {
        check_sync_arity_1_signed_8(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }
}
