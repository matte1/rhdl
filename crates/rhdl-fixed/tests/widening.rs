//! Exhaustive tests for width-extending ops.
//!
//! Verifies width arithmetic, frac arithmetic, and bit-exact result
//! values against `i128`/`u128` reference computations.

use rhdl_bits::{bits, signed};
use rhdl_fixed::{DynFixed, Fixed, SignedDynFixed, SignedFixed};

// -----------------------------------------------------------------------------
// xadd
// -----------------------------------------------------------------------------

#[test]
fn fixed_xadd_widths_and_values() {
    for a in 0u128..16 {
        for b in 0u128..32 {
            let x = Fixed::<4, 2>::from_raw(bits(a));
            let y = Fixed::<5, 2>::from_raw(bits(b));
            let s: DynFixed = x.xadd(y);
            assert_eq!(s.width(), 6, "width = max(4,5)+1 = 6");
            assert_eq!(s.frac, 2);
            assert_eq!(s.raw.raw(), a + b, "{a} + {b}");
        }
    }
}

#[test]
fn signed_fixed_xadd_widths_and_values() {
    for a in -8i128..8 {
        for b in -16i128..16 {
            let x = SignedFixed::<4, 1>::from_raw(signed(a));
            let y = SignedFixed::<5, 1>::from_raw(signed(b));
            let s: SignedDynFixed = x.xadd(y);
            assert_eq!(s.width(), 6);
            assert_eq!(s.frac, 1);
            assert_eq!(s.raw.raw(), a + b, "{a} + {b}");
        }
    }
}

// -----------------------------------------------------------------------------
// xsub
// -----------------------------------------------------------------------------

#[test]
fn fixed_xsub_always_returns_signed() {
    for a in 0u128..16 {
        for b in 0u128..16 {
            let x = Fixed::<4, 2>::from_raw(bits(a));
            let y = Fixed::<4, 2>::from_raw(bits(b));
            let d: SignedDynFixed = x.xsub(y);
            assert_eq!(d.width(), 5);
            assert_eq!(d.frac, 2);
            let want = (a as i128) - (b as i128);
            assert_eq!(d.raw.raw(), want, "{a} - {b}");
        }
    }
}

#[test]
fn signed_fixed_xsub() {
    for a in -8i128..8 {
        for b in -8i128..8 {
            let x = SignedFixed::<4, 0>::from_raw(signed(a));
            let y = SignedFixed::<4, 0>::from_raw(signed(b));
            let d: SignedDynFixed = x.xsub(y);
            assert_eq!(d.width(), 5);
            assert_eq!(d.frac, 0);
            assert_eq!(d.raw.raw(), a - b, "{a} - {b}");
        }
    }
}

// -----------------------------------------------------------------------------
// xmul
// -----------------------------------------------------------------------------

#[test]
fn fixed_xmul_widths_and_frac_arithmetic() {
    for a in 0u128..16 {
        for b in 0u128..8 {
            let x = Fixed::<4, 2>::from_raw(bits(a));
            let y = Fixed::<3, 1>::from_raw(bits(b));
            let p: DynFixed = x.xmul(y);
            assert_eq!(p.width(), 7, "width = N1 + N2 = 4 + 3");
            assert_eq!(p.frac, 3, "frac = F1 + F2 = 2 + 1");
            assert_eq!(p.raw.raw(), a * b, "{a} * {b}");
        }
    }
}

#[test]
fn signed_fixed_xmul_signed_results() {
    for a in -8i128..8 {
        for b in -4i128..4 {
            let x = SignedFixed::<4, 1>::from_raw(signed(a));
            let y = SignedFixed::<3, 0>::from_raw(signed(b));
            let p: SignedDynFixed = x.xmul(y);
            assert_eq!(p.width(), 7);
            assert_eq!(p.frac, 1);
            assert_eq!(p.raw.raw(), a * b, "{a} * {b}");
        }
    }
}

// -----------------------------------------------------------------------------
// xshl / xshr
// -----------------------------------------------------------------------------

#[test]
fn fixed_xshl_grows_width_and_scales_value() {
    for a in 0u128..16 {
        let x = Fixed::<4, 0>::from_raw(bits(a));
        let s: DynFixed = x.xshl::<3>();
        assert_eq!(s.width(), 7);
        assert_eq!(s.frac, 0);
        assert_eq!(s.raw.raw(), a << 3);
    }
}

#[test]
fn fixed_xshr_shrinks_width_and_drops_low_bits() {
    for a in 0u128..256 {
        let x = Fixed::<8, 0>::from_raw(bits(a));
        let s: DynFixed = x.xshr::<3>();
        assert_eq!(s.width(), 5);
        assert_eq!(s.frac, 0);
        assert_eq!(s.raw.raw(), a >> 3);
    }
}

#[test]
fn signed_fixed_xshr_is_arithmetic() {
    // Sign bits must propagate.
    for a in -8i128..8 {
        let x = SignedFixed::<4, 0>::from_raw(signed(a));
        let s: SignedDynFixed = x.xshr::<1>();
        assert_eq!(s.width(), 3);
        assert_eq!(s.raw.raw(), a >> 1, "{a} >> 1");
    }
}

// -----------------------------------------------------------------------------
// xsgn / xneg
// -----------------------------------------------------------------------------

#[test]
fn fixed_xsgn_promotes_to_signed_with_extra_bit() {
    for a in 0u128..16 {
        let x = Fixed::<4, 2>::from_raw(bits(a));
        let s: SignedDynFixed = x.xsgn();
        assert_eq!(s.width(), 5);
        assert_eq!(s.frac, 2);
        assert_eq!(s.raw.raw(), a as i128);
    }
}

#[test]
fn signed_fixed_xneg_handles_min() {
    for a in -8i128..8 {
        let x = SignedFixed::<4, 0>::from_raw(signed(a));
        let n: SignedDynFixed = x.xneg();
        assert_eq!(n.width(), 5);
        assert_eq!(n.raw.raw(), -a, "-{a}");
    }
}

// -----------------------------------------------------------------------------
// DynFixed chaining
// -----------------------------------------------------------------------------

#[test]
fn dyn_fixed_chain_add_and_mul() {
    // (a + b) * c at appropriate widths.
    let a = Fixed::<4, 0>::from_raw(bits(3)).to_dyn();
    let b = Fixed::<4, 0>::from_raw(bits(5)).to_dyn();
    let c = Fixed::<3, 0>::from_raw(bits(2)).to_dyn();
    let sum = a.xadd(b); // width 5, value 8
    let prod = sum.xmul(c); // width 8, value 16, frac 0+0
    assert_eq!(prod.width(), 8);
    assert_eq!(prod.raw.raw(), 16);
}

#[test]
#[should_panic(expected = "matching frac")]
fn dyn_fixed_xadd_panics_on_frac_mismatch() {
    let a = Fixed::<4, 2>::from_raw(bits(3)).to_dyn();
    let b = Fixed::<4, 0>::from_raw(bits(3)).to_dyn();
    let _ = a.xadd(b);
}

#[test]
fn dyn_fixed_xmul_adds_fracs() {
    let a = Fixed::<4, 2>::from_raw(bits(0b1010)).to_dyn(); // 2.5
    let b = Fixed::<4, 1>::from_raw(bits(0b0110)).to_dyn(); // 3.0
    let p = a.xmul(b);
    assert_eq!(p.frac, 3);
    assert_eq!(p.width(), 8);
    // raw 10 * raw 6 = 60; binary point at F=3 -> 60 / 8 = 7.5 = 2.5 * 3.0 ✓
    assert_eq!(p.raw.raw(), 60);
}

// -----------------------------------------------------------------------------
// 128-bit width budget
// -----------------------------------------------------------------------------

#[test]
fn fixed_xmul_at_budget_limit() {
    // 64 * 64 = 128 bits, right at the edge.
    let a = Fixed::<64, 0>::from_raw(bits(0xFFFF_FFFF_FFFF_FFFF));
    let b = Fixed::<64, 0>::from_raw(bits(2));
    let p = a.xmul(b);
    assert_eq!(p.width(), 128);
    assert_eq!(p.raw.raw(), 0x1_FFFF_FFFF_FFFF_FFFEu128);
}
