//! Regression test for an RHIF -> RTL lowering bug where a path
//! projection that unwraps a single-element tuple of a signed scalar
//! produces an RTL register with the tuple kind (s8) instead of the
//! scalar kind s8. The subsequent `.as_unsigned()` call on that
//! register fails the RTL VM's signedness check.
//!
//! ## Root cause sketch
//!
//! 1. The RHIF for `match e { V0(t0) => t0.as_unsigned(), ... }` emits:
//!    ```
//!    r9  <- r5#0     ; extract V0 payload  (kind (s8))
//!    r10 <- r9.0     ; project tuple field (kind s8)
//!    r13 <- unsigned r10
//!    ```
//! 2. Lowering to RTL emits one Index op per RHIF Index op. Each RTL
//!    operand carries its RHIF declared kind: r9_rtl = (s8), r10_rtl
//!    = s8.
//! 3. `lower_index_all_to_copy` turns the full-range `r10 <- r9[0..8]`
//!    Index into an Assign `r10 <- r9`.
//! 4. `remove_extra_registers` unifies r9_rtl and r10_rtl via
//!    union-find. The root keeps the kind of whichever was inserted
//!    first; in this case r9_rtl with kind `(s8)`.
//! 5. The downstream `unsigned r10` becomes `unsigned r9` (root).
//!    State.write coerces the BitString tag using `kind.is_signed()`
//!    on `(s8)`, which is false (only scalar Signed counts).
//!    Subsequent `as_unsigned` on the unsigned-tagged value fails.

use rhdl::prelude::*;
use rhdl_core::sim::testbench::kernel::test_kernel_vm_and_verilog;

#[derive(PartialEq, Debug, Default, Clone, Copy, Digital)]
enum E {
    V0(SignedBits<8>),
    #[default]
    V1_,
}

#[test]
fn enum_signed_payload_as_unsigned_through_vm() -> miette::Result<()> {
    #[kernel]
    fn k(a: Signal<SignedBits<8>, Red>) -> Signal<b8, Red> {
        let a = a.val();
        let e = E::V0(a);
        // The payload extraction has RHIF kind chain (s8) -> s8.
        // `.as_unsigned()` expects the consumer slot to be Signed.
        // After lower_index_all_to_copy + remove_extra_registers
        // collapse the tuple-unwrap, the unified RTL register keeps
        // the tuple kind, losing the signedness signal.
        let r = match e {
            E::V0(t0) => t0.as_unsigned(),
            E::V1_ => bits(0),
        };
        signal(r)
    }
    let native = |a: Signal<SignedBits<8>, Red>| -> Signal<b8, Red> {
        let a = a.val();
        signal::<b8, Red>(a.as_unsigned())
    };
    let inputs = [-128i128, -1, 0, 1, 42, 127]
        .into_iter()
        .map(|n| (signal::<SignedBits<8>, Red>(signed::<8>(n)),));
    test_kernel_vm_and_verilog::<k, _, _, _>(native, inputs)?;
    Ok(())
}
