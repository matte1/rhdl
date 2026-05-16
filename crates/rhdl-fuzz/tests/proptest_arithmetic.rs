//! Property test: generate random `Type`-directed programs using
//! proptest, render each through the RHDL pipeline, and assert that
//! the differential pipeline (RHIF VM, RTL VM, iverilog, netlist
//! iverilog) agrees with the reference interpreter on a fixed set of
//! inputs.
//!
//! Each generated program is exercised against a small deterministic
//! input set (so a single proptest case is cheap-ish). Bug-catchability
//! comes from the program-shape variety (proptest cases) plus the
//! boundary-biased literal generator -- not from input variety.
//!
//! Tunables: set `PROPTEST_CASES` env var to override the default case
//! count per suite (default 64). Thorough runs at 1000 take ~2 min.
//! Generated programs trigger many iverilog invocations.

use proptest::prelude::*;
use rhdl::prelude::*;
use rhdl_core::CompilationMode;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog_ast;
use rhdl_fuzz::dsl::{Program, Type};
use rhdl_fuzz::generator::{program_strategy, program_strategy_typed};
use rhdl_fuzz::interpret::{Value, interpret};
use rhdl_fuzz::render::render;

// ---------- Conversion helpers ----------
//
// These collapse the per-(arity, width, signedness) boilerplate so each
// `check_*` function is a closure body + an input case list. Generic
// over `W` so the same helpers serve every Bits<W> / SignedBits<W>
// width without repeating the conversion code.

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

fn val_signed<const W: usize>(s: Signal<SignedBits<W>, Red>) -> Value
where
    rhdl::bits::W<W>: BitWidth,
{
    Value::signed(s.val().raw() as i128, W as u8)
}

fn sig_signed<const W: usize>(v: Value) -> Signal<SignedBits<W>, Red>
where
    rhdl::bits::W<W>: BitWidth,
{
    signal(signed::<W>(v.expect_signed_raw()))
}

/// Render the program and run it through the differential pipeline.
fn run_pipeline<F, Args, T0>(
    program: &Program,
    native: F,
    cases: impl Iterator<Item = Args> + Clone,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: rhdl_core::sim::testbench::kernel::Testable<Args, T0>,
    T0: Digital,
    Args: rhdl_core::sim::testbench::kernel::TestArg,
{
    let kernel = render(program);
    test_kernel_vm_and_verilog_ast(kernel, native, cases, CompilationMode::Asynchronous)?;
    Ok(())
}

// ---------- Mono-typed Bits<8> suites ----------

type Sig = Signal<Bits<8>, Red>;

/// Small deterministic input pool covering edge cases (zeros, ones,
/// boundaries) and a handful of "interesting" middle values.
const INPUT_SAMPLES: [u128; 6] = [0, 1, 0x7F, 0x80, 0xFE, 0xFF];

fn sig(n: u128) -> Sig {
    signal::<Bits<8>, Red>(bits(n))
}

fn check_arity_1(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |a: Sig| -> Sig { sig_bits::<8>(interpret(&p, &[val_bits::<8>(a)])) };
    let cases = INPUT_SAMPLES.map(|a| (sig(a),));
    run_pipeline(program, native, cases.into_iter())
}

fn check_arity_2(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |a: Sig, b: Sig| -> Sig {
        sig_bits::<8>(interpret(&p, &[val_bits::<8>(a), val_bits::<8>(b)]))
    };
    let cases = [
        (0u128, 0u128),
        (0, 1),
        (1, 1),
        (0xFF, 0x01),
        (0x80, 0x80),
        (0xFE, 0xFE),
        (0x7F, 0x80),
        (42, 100),
    ]
    .map(|(a, b)| (sig(a), sig(b)));
    run_pipeline(program, native, cases.into_iter())
}

fn check_arity_3(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |a: Sig, b: Sig, c: Sig| -> Sig {
        sig_bits::<8>(interpret(
            &p,
            &[val_bits::<8>(a), val_bits::<8>(b), val_bits::<8>(c)],
        ))
    };
    let cases = [
        (0u128, 0u128, 0u128),
        (1, 2, 3),
        (0xFF, 0x01, 0x80),
        (0x7F, 0x80, 0xFE),
        (42, 100, 200),
        (0, 0xFF, 0x55),
    ]
    .map(|(a, b, c)| (sig(a), sig(b), sig(c)));
    run_pipeline(program, native, cases.into_iter())
}

fn check_arity_4(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    let p = program.clone();
    let native = move |a: Sig, b: Sig, c: Sig, d: Sig| -> Sig {
        sig_bits::<8>(interpret(
            &p,
            &[
                val_bits::<8>(a),
                val_bits::<8>(b),
                val_bits::<8>(c),
                val_bits::<8>(d),
            ],
        ))
    };
    let cases = [
        (0u128, 0u128, 0u128, 0u128),
        (1, 2, 3, 4),
        (0xFF, 0x01, 0x80, 0x7F),
        (42, 100, 200, 0xAA),
    ]
    .map(|(a, b, c, d)| (sig(a), sig(b), sig(c), sig(d)));
    run_pipeline(program, native, cases.into_iter())
}

// ---------- M1: width-varying suites ----------

fn check_arity_2_width_1(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    type Sig1 = Signal<Bits<1>, Red>;
    let p = program.clone();
    let native = move |a: Sig1, b: Sig1| -> Sig1 {
        sig_bits::<1>(interpret(&p, &[val_bits::<1>(a), val_bits::<1>(b)]))
    };
    // Single-bit space is small; cover all four pairs.
    let cases = [(0u128, 0u128), (0, 1), (1, 0), (1, 1)]
        .map(|(a, b)| (signal(bits::<1>(a)), signal(bits::<1>(b))));
    run_pipeline(program, native, cases.into_iter())
}

fn check_arity_2_width_16(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    type Sig16 = Signal<Bits<16>, Red>;
    let p = program.clone();
    let native = move |a: Sig16, b: Sig16| -> Sig16 {
        sig_bits::<16>(interpret(&p, &[val_bits::<16>(a), val_bits::<16>(b)]))
    };
    let cases = [
        (0u128, 0u128),
        (1, 0xFFFF),
        (0x1234, 0xCAFE),
        (0x8000, 0x8000),
        (0x7FFF, 0x0001),
    ]
    .map(|(a, b)| (signal(bits::<16>(a)), signal(bits::<16>(b))));
    run_pipeline(program, native, cases.into_iter())
}

fn check_arity_1_width_64(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    type Sig64 = Signal<Bits<64>, Red>;
    let p = program.clone();
    let native = move |a: Sig64| -> Sig64 { sig_bits::<64>(interpret(&p, &[val_bits::<64>(a)])) };
    let cases = [
        0u128,
        1,
        0xDEAD_BEEF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
    ]
    .map(|a| (signal(bits::<64>(a)),));
    run_pipeline(program, native, cases.into_iter())
}

// ---------- M2: signed suites ----------

fn check_arity_2_signed_8(program: &Program) -> Result<(), Box<dyn std::error::Error>> {
    type SSig8 = Signal<SignedBits<8>, Red>;
    let p = program.clone();
    let native = move |a: SSig8, b: SSig8| -> SSig8 {
        sig_signed::<8>(interpret(&p, &[val_signed::<8>(a), val_signed::<8>(b)]))
    };
    let cases = [
        (-128i64, -128),
        (-1, 1),
        (0, 0),
        (42, -100),
        (127, 1),
        (-64, -64),
    ]
    .map(|(a, b)| {
        (
            signal(signed::<8>(a as i128)),
            signal(signed::<8>(b as i128)),
        )
    });
    run_pipeline(program, native, cases.into_iter())
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Default 64 cases per suite. Set PROPTEST_CASES=1000 (or
        // higher) for a thorough run; ~2 minutes for all suites at 1000.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        // Lowered from the proptest default (4096). Our generated
        // programs run through the full pipeline (interpret + RHIF VM
        // + RTL VM + iverilog) which takes ~5-10s per shrink candidate;
        // at the default cap a single failure can take 11+ hours to
        // bottom out. 256 still produces fairly minimal shrunken
        // programs in our experience while bounding shrink time to
        // ~30-45 minutes per failure.
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn arity_1_round_trip(program in program_strategy(1)) {
        check_arity_1(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_2_round_trip(program in program_strategy(2)) {
        check_arity_2(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_3_round_trip(program in program_strategy(3)) {
        check_arity_3(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_4_round_trip(program in program_strategy(4)) {
        check_arity_4(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_2_width_1_round_trip(
        program in program_strategy_typed(vec![Type::Bits(1), Type::Bits(1)], Type::Bits(1))
    ) {
        check_arity_2_width_1(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_2_width_16_round_trip(
        program in program_strategy_typed(vec![Type::Bits(16), Type::Bits(16)], Type::Bits(16))
    ) {
        check_arity_2_width_16(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_1_width_64_round_trip(
        program in program_strategy_typed(vec![Type::Bits(64)], Type::Bits(64))
    ) {
        check_arity_1_width_64(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }

    #[test]
    fn arity_2_signed_8_round_trip(
        program in program_strategy_typed(vec![Type::Signed(8), Type::Signed(8)], Type::Signed(8))
    ) {
        check_arity_2_signed_8(&program)
            .map_err(|e| TestCaseError::fail(format!("{e:?}\nprogram: {:?}", program)))?;
    }
}

/// Dump the generator's per-AST-kind coverage histogram. Always passes;
/// run with `--nocapture --test-threads=1` to see the distribution.
/// The "zzz_" prefix sorts this test alphabetically after every
/// `arity_*` suite so the cumulative counts reflect all of them.
#[test]
fn zzz_coverage_summary() {
    rhdl_fuzz::coverage::print_summary();
}
