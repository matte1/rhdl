//! Tests for the `Mac` and `SignedMac` accumulators.

use rhdl_bits::{bits, signed};
use rhdl_fixed::mac::{Mac, SignedMac};
use rhdl_fixed::{Fixed, SignedFixed};

#[test]
fn mac_zero_init_sum_of_products() {
    // Q4.4 inputs, Q8.8 accumulator (F1 + F2 = 4 + 4 = 8 ✓)
    let m = Mac::<16, 8>::zero();
    let a1 = Fixed::<4, 4>::from_raw(bits(0b0010)); // 0.125
    let b1 = Fixed::<4, 4>::from_raw(bits(0b0100)); // 0.25
    let a2 = Fixed::<4, 4>::from_raw(bits(0b1000)); // 0.5
    let b2 = Fixed::<4, 4>::from_raw(bits(0b0010)); // 0.125
    let m = m.mac(a1, b1).mac(a2, b2);
    // sum = 2*4 + 8*2 = 8 + 16 = 24 at raw scale (frac 8)
    // = 24 / 256 = 0.09375 = 0.125*0.25 + 0.5*0.125 = 0.03125 + 0.0625 = 0.09375 ✓
    assert_eq!(m.result().raw.raw(), 24);
}

#[test]
fn mac_new_seeds_accumulator() {
    let init = Fixed::<16, 8>::from_raw(bits(100));
    let m = Mac::<16, 8>::new(init);
    let a = Fixed::<4, 4>::from_raw(bits(3));
    let b = Fixed::<4, 4>::from_raw(bits(7));
    let m = m.mac(a, b);
    assert_eq!(m.result().raw.raw(), 100 + 21);
}

#[test]
#[should_panic(expected = "product frac")]
fn mac_panics_on_frac_mismatch() {
    let m = Mac::<16, 8>::zero();
    // F1 + F2 = 2 + 2 = 4, accumulator F = 8 → mismatch
    let a = Fixed::<4, 2>::from_raw(bits(1));
    let b = Fixed::<4, 2>::from_raw(bits(1));
    let _ = m.mac(a, b);
}

#[test]
fn mac_wraps_on_overflow() {
    // Accumulate enough to wrap a 4-bit accumulator.
    let m = Mac::<4, 0>::new(Fixed::<4, 0>::from_raw(bits(15)));
    let one = Fixed::<2, 0>::from_raw(bits(1));
    let one_x = Fixed::<2, 0>::from_raw(bits(1));
    let m = m.mac(one, one_x); // 15 + 1 = 16 wraps to 0
    assert_eq!(m.result().raw.raw(), 0);
}

#[test]
fn signed_mac_handles_negatives() {
    let m = SignedMac::<16, 4>::zero();
    let a = SignedFixed::<6, 2>::from_raw(signed(-3));
    let b = SignedFixed::<6, 2>::from_raw(signed(5));
    let c = SignedFixed::<6, 2>::from_raw(signed(2));
    let d = SignedFixed::<6, 2>::from_raw(signed(-7));
    let m = m.mac(a, b).mac(c, d);
    // (-3)*5 + 2*(-7) = -15 + -14 = -29 (at raw scale)
    assert_eq!(m.result().raw.raw(), -29);
}

#[test]
fn signed_mac_exhaustive_4_term_sum_of_products() {
    // For small inputs, exhaustively verify Mac equals i128 reference.
    for a in -4i128..4 {
        for b in -4i128..4 {
            for c in -4i128..4 {
                for d in -4i128..4 {
                    let m = SignedMac::<16, 0>::zero()
                        .mac(
                            SignedFixed::<4, 0>::from_raw(signed(a)),
                            SignedFixed::<4, 0>::from_raw(signed(b)),
                        )
                        .mac(
                            SignedFixed::<4, 0>::from_raw(signed(c)),
                            SignedFixed::<4, 0>::from_raw(signed(d)),
                        );
                    let want = a * b + c * d;
                    assert_eq!(m.result().raw.raw(), want, "({a})*({b}) + ({c})*({d})");
                }
            }
        }
    }
}
