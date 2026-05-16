//! Regression test for a bug in `#[derive(Digital)]` interacting with the
//! kernel compiler.
//!
//! # Symptom
//!
//! When a struct uses `#[derive(Digital)]` with a `SignedBits<N>` field,
//! the kernel compiler treats the value obtained via field access as
//! **unsigned** bits, even though Rust's type system correctly reports
//! it as `SignedBits<N>`. Subsequent operations that depend on
//! signedness (`xmul`, `xshr`, `as_signed_bits`) then produce wrong
//! values inside `#[kernel]` bodies.
//!
//! The unsigned analog (struct wraps a `Bits<N>` field) works correctly.
//! That difference is what makes this a derive/lowering bug rather than
//! a fundamental kernel-compiler limitation.
//!
//! # Minimal repro
//!
//! 1. `UnsignedWrap { raw: Bits<6> }` and `SignedWrap { raw: SignedBits<6> }`
//!    both `#[derive(Digital)]`.
//! 2. For each, a kernel that takes two wrappers, accesses `.raw`, and
//!    multiplies the two values with `xmul`.
//! 3. Each result is compared host-vs-kernel via the differential
//!    pipeline (RHIF VM, RTL VM, iverilog, netlist iverilog).
//!
//! Expected behavior: both wrappers pass the differential test.
//!
//! Observed behavior: the unsigned wrapper passes. The signed wrapper
//! fails because the kernel-side `xmul` is dispatched as an unsigned
//! multiplication. Concretely, for inputs whose underlying bit patterns
//! have the MSB set (e.g. `-2` = `0b111110`), the kernel multiplies the
//! raw bit pattern as if it were `62` instead of `-2`.
//!
//! # Why it matters
//!
//! Any user-defined wrapper type that embeds a `SignedBits<N>` field
//! and is used as a kernel input/output silently produces wrong results
//! for any negative-input case. This blocks any library that wants to
//! provide typed wrappers over `SignedBits<N>` (e.g. fixed-point
//! libraries, signed-magnitude types).
//!
//! # Root cause
//!
//! Traced via the `dump_failing_kernel_rhif` test below. Three layers,
//! each behaving locally correctly, combine to produce the bug:
//!
//! 1. **Digital derive (correct):** Produces
//!    `Kind::Struct { fields: [("raw", Kind::Signed(6))] }`. Verified
//!    by `print_signed_wrap_kind`.
//!
//! 2. **MIR type inference (correct):** The dump shows the RHIF
//!    register holding `a.raw` is correctly typed `s6`, and `xmul`
//!    produces `s12`. Inference threads signedness through field
//!    access perfectly.
//!
//! 3. **RHIF → RTL (wrong):** The RTL has its own register table.
//!    Input arguments of struct type are stored with their full
//!    `Kind::Struct` kind. Then [`LowerIndexAllToCopy`] folds the
//!    field access `r2 <- r0.raw` into an `Assign r2 <- r0` (since
//!    the field spans all bits). Then [`RemoveExtraRegistersPass`]
//!    runs a union-find over assignment chains and replaces all uses
//!    of one register with the other — and crucially, **the picked
//!    survivor may be the struct-kinded register, not the
//!    properly-signed inner register.** The downstream `Cast` op then
//!    reads the survivor's signedness via [`Kind::is_signed()`] which
//!    is hard-coded to return `false` for `Kind::Struct(_)`, treating
//!    the value as unsigned. The cast then zero-extends instead of
//!    sign-extending, and the multiplication operates on positive
//!    `62 * 62 = 3844` instead of `-2 * -2 = 4`.
//!
//! Concretely visible in the dump from `dump_failing_kernel_rhif`:
//!
//! ```text
//! --- RHIF (correct) ---
//! Reg r0 : SignedWrap { raw: s6 }
//! Reg r2 : s6                       // r2 <- r0.raw, properly signed
//! Reg r4 : s12                      // r4 <- r2 xmul r3, properly signed
//!
//! --- RTL (bug) ---
//! reg r0 : b6 // SignedWrap { raw: s6 }     <-- r2 was folded away;
//! reg r3 : s12                              //   r3 reads from r0 directly
//! r3 <- r0 as x12                           <-- Cast zero-extends (b6 -> s12)
//! r2 <- r3 * r4                             //   instead of sign-extending
//! ```
//!
//! # Fix candidates
//!
//! - **Narrow fix in `LowerIndexAllToCopy`:** when the field access
//!   has a different signedness from the parent, emit an `AsSigned`
//!   / `AsBits` cast instead of an `Assign`. This preserves the
//!   semantics that the field access reinterprets the bits.
//! - **Narrow fix in `RemoveExtraRegistersPass`:** when unioning two
//!   registers with different kinds, prefer the one whose
//!   `is_signed()` matches the use site, or refuse to union when
//!   kinds differ.
//! - **Broader fix:** at RHIF→RTL lowering, flatten single-field
//!   aggregate kinds (struct/tuple of one field) to their inner kind
//!   so that subsequent passes see a consistent signedness.
//! - **Broadest fix:** change `Kind::is_signed()` to recurse into
//!   single-field structs/tuples. (Not recommended — pollutes the
//!   semantics for the common multi-field case.)

use rhdl::core::sim::testbench::kernel::test_kernel_vm_and_verilog_synchronous;
use rhdl::prelude::*;

#[derive(Digital, Debug, Clone, Copy, PartialEq, Default)]
struct UnsignedWrap {
    raw: Bits<6>,
}

#[derive(Digital, Debug, Clone, Copy, PartialEq, Default)]
struct SignedWrap {
    raw: SignedBits<6>,
}

/// Unsigned baseline: multiply two values pulled out of a wrapper struct.
/// Should pass the differential pipeline cleanly.
#[kernel]
fn mul_unsigned_wrap(a: UnsignedWrap, b: UnsignedWrap) -> Bits<12> {
    let x = a.raw;
    let y = b.raw;
    x.xmul(y).as_bits()
}

/// Signed analog: same shape, but with `SignedBits<6>` instead of
/// `Bits<6>`. This is the failing case.
#[kernel]
fn mul_signed_wrap(a: SignedWrap, b: SignedWrap) -> SignedBits<12> {
    let x = a.raw;
    let y = b.raw;
    x.xmul(y).as_signed_bits()
}

/// Control: same math as `mul_signed_wrap` but with `SignedBits<6>`
/// passed *directly* as kernel arguments — no wrapper struct. This case
/// works, which is the smoking gun: the bug appears only when the
/// signed value reaches the kernel as a struct field.
#[kernel]
fn mul_signed_no_wrap(a: SignedBits<6>, b: SignedBits<6>) -> SignedBits<12> {
    a.xmul(b).as_signed_bits()
}

#[test]
fn unsigned_wrap_works() -> miette::Result<()> {
    // Sweep the full 6-bit input range.
    let cases: Vec<_> = (0u128..64)
        .flat_map(|a| {
            (0u128..64).map(move |b| (UnsignedWrap { raw: b6(a) }, UnsignedWrap { raw: b6(b) }))
        })
        .collect();
    test_kernel_vm_and_verilog_synchronous::<mul_unsigned_wrap, _, _, _>(
        mul_unsigned_wrap,
        cases.into_iter(),
    )?;
    Ok(())
}

#[test]
fn signed_no_wrap_works() -> miette::Result<()> {
    // Demonstrates that SignedBits<6> arguments do flow through the
    // kernel correctly when passed directly, so the bug is not in
    // SignedBits or xmul or as_signed_bits — it's specifically in the
    // struct-field access path.
    let cases: Vec<_> = (-32i128..32)
        .flat_map(|a| {
            (-32i128..32).map(move |b| (rhdl::bits::signed::<6>(a), rhdl::bits::signed::<6>(b)))
        })
        .collect();
    test_kernel_vm_and_verilog_synchronous::<mul_signed_no_wrap, _, _, _>(
        mul_signed_no_wrap,
        cases.into_iter(),
    )?;
    Ok(())
}

/// This test is **expected to fail** until the underlying bug is fixed.
///
/// When it fails, the verification error shows the kernel producing
/// `(raw_a as u6) * (raw_b as u6)` rather than `(raw_a as i6) *
/// (raw_b as i6)`. For example:
///
/// Inputs: `SignedWrap { raw: 1 }, SignedWrap { raw: -2 }`
///   Host (correct):     `1 * -2 = -2` = `0b111111111110` (12-bit signed)
///   Kernel (incorrect): `1 * 62 = 62` = `0b000000111110`
///
/// The two interpretations agree on the low bits and diverge on the
/// high bits, because the kernel-side multiplication zero-extends what
/// should have been a sign-extension.
#[test]
fn signed_wrap_demonstrates_bug() -> miette::Result<()> {
    // Sweep cases including negatives so the bug actually triggers.
    let cases: Vec<_> = (-4i128..4)
        .flat_map(|a| {
            (-4i128..4).map(move |b| {
                (
                    SignedWrap {
                        raw: rhdl::bits::signed::<6>(a),
                    },
                    SignedWrap {
                        raw: rhdl::bits::signed::<6>(b),
                    },
                )
            })
        })
        .collect();
    test_kernel_vm_and_verilog_synchronous::<mul_signed_wrap, _, _, _>(
        mul_signed_wrap,
        cases.into_iter(),
    )?;
    Ok(())
}

/// Compile the failing kernel and dump the RHIF symbol table. Each
/// register's `Kind` tells us whether the compiler considers it signed
/// or unsigned at every intermediate step. If we see `b6` instead of
/// `s6` for the slot that holds `a.raw`, the bug is in MIR inference
/// or the field-access lowering. If we see `s6` correctly here but
/// the RTL VM still produces unsigned bits, the bug is in the
/// RHIF→RTL bridge.
#[test]
fn dump_failing_kernel_rhif() {
    use rhdl::core::CompilationMode;
    use rhdl::core::compiler::driver::compile_design_stage1;

    let obj =
        compile_design_stage1::<mul_signed_wrap>(CompilationMode::Synchronous).expect("compile");
    println!("--- RHIF for mul_signed_wrap ---");
    println!("{obj:?}");

    use rhdl::core::compiler::driver::compile_design_stage2;
    let rtl = compile_design_stage2(&obj).expect("rtl");
    println!("--- RTL for mul_signed_wrap (post-passes, BUG) ---");
    println!("{rtl:?}");

    // Compare with the unsigned wrapper, which works.
    let obj_u = compile_design_stage1::<mul_unsigned_wrap>(CompilationMode::Synchronous)
        .expect("compile u");
    println!("--- RHIF for mul_unsigned_wrap (control) ---");
    println!("{obj_u:?}");
    let rtl_u = compile_design_stage2(&obj_u).expect("rtl u");
    println!("--- RTL for mul_unsigned_wrap (control) ---");
    println!("{rtl_u:?}");
}

/// Print the runtime [`Kind`] of `SignedWrap` to verify that
/// `#[derive(Digital)]` correctly tags the `raw` field as signed.
///
/// If the field shows up as `Bits(6)` instead of `Signed(6)`, the bug
/// is in the derive. Otherwise it's downstream (likely in
/// codegen — the type inference treats `.raw` as signed but the
/// generated hardware doesn't honor it).
#[test]
fn print_signed_wrap_kind() {
    let kind = <SignedWrap as Digital>::static_kind();
    println!("SignedWrap kind: {kind:?}");

    let unsigned_kind = <UnsignedWrap as Digital>::static_kind();
    println!("UnsignedWrap kind: {unsigned_kind:?}");

    // Also dump the raw field type's Kind directly.
    let raw_kind = <SignedBits<6> as Digital>::static_kind();
    println!("SignedBits<6> kind: {raw_kind:?}");

    let raw_unsigned_kind = <Bits<6> as Digital>::static_kind();
    println!("Bits<6> kind: {raw_unsigned_kind:?}");
}

/// Sanity: the host-side `mul_signed_wrap` function (without the kernel
/// pipeline) computes correctly. This isolates the bug to the kernel
/// compiler path — not the Rust impl, not the wrapper struct itself.
#[test]
fn signed_wrap_host_side_is_correct() {
    let a = SignedWrap {
        raw: rhdl::bits::signed::<6>(1),
    };
    let b = SignedWrap {
        raw: rhdl::bits::signed::<6>(-2),
    };
    let got = mul_signed_wrap(a, b);
    assert_eq!(got.raw(), -2, "host: 1 * -2 = -2");

    let a = SignedWrap {
        raw: rhdl::bits::signed::<6>(-3),
    };
    let b = SignedWrap {
        raw: rhdl::bits::signed::<6>(-4),
    };
    let got = mul_signed_wrap(a, b);
    assert_eq!(got.raw(), 12, "host: -3 * -4 = 12");
}
