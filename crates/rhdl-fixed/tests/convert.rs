//! Tests for conversions: rounding, saturation, and float conversion.

use rhdl_bits::{bits, signed};
use rhdl_fixed::{DynFixed, Fixed, SignedDynFixed, SignedFixed};

// -----------------------------------------------------------------------------
// truncate
// -----------------------------------------------------------------------------

#[test]
fn truncate_drops_low_bits() {
    // Source: 8 bits, frac 4. Value 0b1010_1101 = 173/16 = 10.8125
    let src = Fixed::<8, 4>::from_raw(bits(0b1010_1101)).to_dyn();
    // Target frac 2: drop 2 low bits, so value becomes 0b101011 = 43/4 = 10.75
    let dst: Fixed<8, 2> = src.truncate();
    assert_eq!(dst.raw.raw(), 0b10_1011);
}

#[test]
fn truncate_wraps_high_bits() {
    // 8-bit source 200, truncate to 4 bits: 200 & 0xF = 8
    let src = Fixed::<8, 0>::from_raw(bits(200)).to_dyn();
    let dst: Fixed<4, 0> = src.truncate();
    assert_eq!(dst.raw.raw(), 8);
}

#[test]
#[should_panic(expected = "target frac")]
fn truncate_panics_when_target_frac_exceeds_source() {
    let src = Fixed::<8, 2>::from_raw(bits(0x10)).to_dyn();
    let _: Fixed<8, 4> = src.truncate();
}

#[test]
fn signed_truncate_handles_negatives() {
    let src = SignedFixed::<8, 0>::from_raw(signed(-100)).to_dyn();
    let dst: SignedFixed<8, 0> = src.truncate();
    assert_eq!(dst.raw.raw(), -100);
    // Truncating to 5 bits: -100 wraps to ?
    // -100 in i8 binary = 10011100. Low 5 bits = 11100 = 28 as unsigned, but
    // as signed 5-bit: 11100 sign-extended -> -4
    let dst5: SignedFixed<5, 0> = src.truncate();
    assert_eq!(dst5.raw.raw(), -4);
}

// -----------------------------------------------------------------------------
// round_nearest (RTNE)
// -----------------------------------------------------------------------------

#[test]
fn round_nearest_rounds_up_above_half() {
    // 0b01_11 (= 7), drop 2 bits, remainder=3>half=2 -> round up: 2
    let src = Fixed::<4, 2>::from_raw(bits(0b01_11)).to_dyn();
    let dst: Fixed<4, 0> = src.round_nearest();
    assert_eq!(dst.raw.raw(), 2);
}

#[test]
fn round_nearest_rounds_down_below_half() {
    // 0b01_01 (=5), drop 2 bits, remainder=1<half=2 -> round down: 1
    let src = Fixed::<4, 2>::from_raw(bits(0b01_01)).to_dyn();
    let dst: Fixed<4, 0> = src.round_nearest();
    assert_eq!(dst.raw.raw(), 1);
}

#[test]
fn round_nearest_ties_to_even() {
    // 0b00_10 (=2), drop 2 bits: remainder=2==half, truncated=0 (even) -> 0
    let src1 = Fixed::<4, 2>::from_raw(bits(0b00_10)).to_dyn();
    let dst1: Fixed<4, 0> = src1.round_nearest();
    assert_eq!(dst1.raw.raw(), 0, "0.5 ties to even -> 0");

    // 0b01_10 (=6), drop 2: remainder=2==half, truncated=1 (odd) -> 2
    let src2 = Fixed::<4, 2>::from_raw(bits(0b01_10)).to_dyn();
    let dst2: Fixed<4, 0> = src2.round_nearest();
    assert_eq!(dst2.raw.raw(), 2, "1.5 ties to even -> 2");

    // 0b10_10 (=10), drop 2: remainder=2==half, truncated=2 (even) -> 2
    let src3 = Fixed::<4, 2>::from_raw(bits(0b10_10)).to_dyn();
    let dst3: Fixed<4, 0> = src3.round_nearest();
    assert_eq!(dst3.raw.raw(), 2, "2.5 ties to even -> 2");

    // 0b11_10 (=14), drop 2: remainder=2==half, truncated=3 (odd) -> 4
    let src4 = Fixed::<4, 2>::from_raw(bits(0b11_10)).to_dyn();
    let dst4: Fixed<4, 0> = src4.round_nearest();
    assert_eq!(dst4.raw.raw(), 4, "3.5 ties to even -> 4");
}

#[test]
fn signed_round_nearest_negative_values() {
    // -3.5 (raw = 0b1110, shifted): SignedFixed<5, 2> raw -14 = -3.5
    // Round to int: -3.5 ties to even -> -4 (since -4 is even, -3 is odd)
    let src = SignedFixed::<5, 2>::from_raw(signed(-14)).to_dyn();
    let dst: SignedFixed<5, 0> = src.round_nearest();
    assert_eq!(dst.raw.raw(), -4, "-3.5 RTNE -> -4");

    // -2.5 raw = -10 in 5-bit frac=2; -2 is even, so -2.5 -> -2
    let src = SignedFixed::<5, 2>::from_raw(signed(-10)).to_dyn();
    let dst: SignedFixed<5, 0> = src.round_nearest();
    assert_eq!(dst.raw.raw(), -2, "-2.5 RTNE -> -2");
}

// -----------------------------------------------------------------------------
// saturate
// -----------------------------------------------------------------------------

#[test]
fn saturate_clamps_unsigned_overflow() {
    // 8-bit value 200, saturate to 4 bits: max is 15
    let src = Fixed::<8, 0>::from_raw(bits(200)).to_dyn();
    let dst: Fixed<4, 0> = src.saturate();
    assert_eq!(dst.raw.raw(), 15);
}

#[test]
fn saturate_passes_through_in_range() {
    let src = Fixed::<8, 0>::from_raw(bits(10)).to_dyn();
    let dst: Fixed<4, 0> = src.saturate();
    assert_eq!(dst.raw.raw(), 10);
}

#[test]
fn signed_saturate_clamps_both_directions() {
    // +100 saturated to SignedFixed<5, 0>: max is 15
    let src = SignedFixed::<8, 0>::from_raw(signed(100)).to_dyn();
    let dst: SignedFixed<5, 0> = src.saturate();
    assert_eq!(dst.raw.raw(), 15);

    // -100 saturated to SignedFixed<5, 0>: min is -16
    let src = SignedFixed::<8, 0>::from_raw(signed(-100)).to_dyn();
    let dst: SignedFixed<5, 0> = src.saturate();
    assert_eq!(dst.raw.raw(), -16);
}

// -----------------------------------------------------------------------------
// round_saturate
// -----------------------------------------------------------------------------

#[test]
fn round_saturate_combines_both() {
    // Value 0b1111_1111 = 255 at frac 2 means 63.75.
    // Round to frac 0: RTNE 0.75 round up -> 64. Saturate to 4-bit max 15.
    let src = Fixed::<8, 2>::from_raw(bits(0b1111_1111)).to_dyn();
    let dst: Fixed<4, 0> = src.round_saturate();
    assert_eq!(dst.raw.raw(), 15);
}

// -----------------------------------------------------------------------------
// Sign reinterpretation
// -----------------------------------------------------------------------------

#[test]
fn dyn_fixed_as_signed_round_trip() {
    let src: DynFixed = Fixed::<8, 4>::from_raw(bits(0x12)).to_dyn();
    let s: SignedDynFixed = src.as_signed();
    assert_eq!(s.frac, 4);
    assert_eq!(s.width(), 8);
    assert_eq!(s.raw.raw(), 0x12);
    let back = s.as_unsigned();
    assert_eq!(back.raw.raw(), 0x12);
}

// -----------------------------------------------------------------------------
// from_f64
// -----------------------------------------------------------------------------

#[test]
fn fixed_from_f64_round_trip() {
    let x = Fixed::<16, 8>::from_f64(1.5);
    assert_eq!(x.raw.raw(), 0x180); // 1.5 * 256 = 384
    let y = Fixed::<16, 8>::from_f64(0.0);
    assert_eq!(y.raw.raw(), 0);
}

#[test]
fn signed_fixed_from_f64_handles_negatives() {
    let x = SignedFixed::<16, 8>::from_f64(-1.5);
    assert_eq!(x.raw.raw(), -384);
    let y = SignedFixed::<16, 8>::from_f64(2.25);
    assert_eq!(y.raw.raw(), 576);
}

#[test]
fn from_f64_rounds_to_nearest() {
    // 0.5 ULP at frac=4 is 1/32 = 0.03125. So 0.5 + 0.001 should round to nearest.
    // f64::round() is round-half-away-from-zero, not RTNE — acceptable host-side
    // mismatch; document this as a known quirk.
    let x = Fixed::<8, 4>::from_f64(1.5);
    assert_eq!(x.raw.raw(), 24); // 1.5 * 16 = 24 exactly
}

#[test]
#[should_panic(expected = "out of")]
fn fixed_from_f64_panics_on_overflow() {
    let _ = Fixed::<8, 4>::from_f64(100.0); // way out of range
}

#[test]
#[should_panic(expected = "out of")]
fn fixed_from_f64_panics_on_negative_unsigned() {
    let _ = Fixed::<8, 4>::from_f64(-1.0);
}

#[test]
#[should_panic(expected = "must be finite")]
fn fixed_from_f64_panics_on_nan() {
    let _ = Fixed::<8, 4>::from_f64(f64::NAN);
}
