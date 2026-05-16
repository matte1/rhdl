//! Regression test for an RHIF -> RTL lowering bug where the
//! signedness of an enum variant's signed payload is lost during
//! bit-slice extraction.
//!
//! ## Symptom
//!
//! A kernel that matches an enum with a `SignedBits<W>` payload, binds
//! the payload, and uses it in any binary operation alongside another
//! `SignedBits<W>` value fails at the RTL VM with:
//!
//! ```text
//! BinaryOperationRequiresCompatibleType { lhs: sN, rhs: bN }
//! ```
//!
//! ## Root cause
//!
//! The RHIF `op_index(payload, enum_value, [EnumPayloadByValue(disc)])`
//! correctly tracks the variant's payload kind (`(s8)` -- a 1-tuple of
//! `Signed<8>`) at the RHIF level. When lowered to RTL the payload
//! becomes a flat bit slice `r3 <- r1[0..8]`, and the RTL register
//! type drops to `b8` (unsigned BitString) -- the variant's payload
//! signedness doesn't propagate through the slice.
//!
//! Subsequent unary ops (`!`, `~`) propagate the wrong signedness,
//! and a binop that mixes the extracted payload with a signed literal
//! fails at the runtime type check in the RTL VM.
//!
//! ## Why it wasn't caught before
//!
//! Stage 1 + stage 2 compilation both succeed -- the bug only fires
//! when the kernel is run through the **RTL VM** (the differential
//! testbench path). Existing `#[kernel]` tests that just call
//! `compile_design` don't surface it.
//!
//! ## Reproducer scope
//!
//! Two variants below: the payload-bearing variant at discriminant 0
//! (`E_DiscZero`) and at discriminant 1 (`E_DiscOne`). Both fail with
//! the same error -- the bug is independent of the discriminant
//! value.

use rhdl::prelude::*;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog;

#[derive(PartialEq, Debug, Default, Clone, Copy, Digital)]
enum EDiscZero {
    V0(SignedBits<8>),
    #[default]
    V1_,
}

#[derive(PartialEq, Debug, Default, Clone, Copy, Digital)]
enum EDiscOne {
    #[default]
    V0_,
    V1(SignedBits<8>),
}

#[test]
fn signed_enum_payload_at_disc_zero_through_vm() -> miette::Result<()> {
    #[kernel]
    fn k(a: Signal<SignedBits<8>, Red>) -> Signal<SignedBits<8>, Red> {
        let a = a.val();
        let e = EDiscZero::V0(a);
        let r = match e {
            EDiscZero::V0(t0) => !!t0,
            EDiscZero::V1_ => signed(0),
        };
        // Combining the match result with a signed literal triggers
        // the RTL VM's runtime type check; the match result's stored
        // BitString is unsigned even though the static kind is signed.
        let s = signed(0) + r;
        signal(s)
    }
    let native = |a: Signal<SignedBits<8>, Red>| -> Signal<SignedBits<8>, Red> {
        let a = a.val();
        signal::<SignedBits<8>, Red>(signed::<8>(0) + (!!a))
    };
    let inputs = [-128i128, -1, 0, 1, 42, 127]
        .into_iter()
        .map(|n| (signal::<SignedBits<8>, Red>(signed::<8>(n)),));
    test_kernel_vm_and_verilog::<k, _, _, _>(native, inputs)?;
    Ok(())
}

#[test]
fn signed_enum_payload_at_disc_one_through_vm() -> miette::Result<()> {
    #[kernel]
    fn k(a: Signal<SignedBits<8>, Red>) -> Signal<SignedBits<8>, Red> {
        let a = a.val();
        let e = EDiscOne::V1(a);
        let r = match e {
            EDiscOne::V0_ => signed(0),
            EDiscOne::V1(t0) => !!t0,
        };
        let s = signed(0) + r;
        signal(s)
    }
    let native = |a: Signal<SignedBits<8>, Red>| -> Signal<SignedBits<8>, Red> {
        let a = a.val();
        signal::<SignedBits<8>, Red>(signed::<8>(0) + (!!a))
    };
    let inputs = [-128i128, -1, 0, 1, 42, 127]
        .into_iter()
        .map(|n| (signal::<SignedBits<8>, Red>(signed::<8>(n)),));
    test_kernel_vm_and_verilog::<k, _, _, _>(native, inputs)?;
    Ok(())
}
