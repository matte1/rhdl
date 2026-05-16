use rhdl_bits::{bits, signed};
use rhdl_core::Digital;
use rhdl_fixed::{DynFixed, Fixed, SignedDynFixed, SignedFixed};

#[test]
fn fixed_round_trip_via_from_raw() {
    let x = Fixed::<8, 4>::from_raw(bits(0xA5));
    assert_eq!(x.raw.raw(), 0xA5);
    assert_eq!(Fixed::<8, 4>::WIDTH, 8);
    assert_eq!(Fixed::<8, 4>::FRAC, 4);
}

#[test]
fn fixed_round_trip_via_u128() {
    let x = Fixed::<8, 4>::from_raw(bits(0x12));
    assert_eq!(x.raw.raw(), 0x12);
}

#[test]
fn signed_fixed_round_trip() {
    let x = SignedFixed::<8, 4>::from_raw(signed(-5));
    assert_eq!(x.raw.raw(), -5);
    assert_eq!(SignedFixed::<8, 4>::WIDTH, 8);
    assert_eq!(SignedFixed::<8, 4>::FRAC, 4);
}

#[test]
fn fixed_to_dyn_preserves_width_and_frac() {
    let x = Fixed::<12, 5>::from_raw(bits(0x123));
    let d: DynFixed = x.to_dyn();
    assert_eq!(d.width(), 12);
    assert_eq!(d.frac, 5);
    assert_eq!(d.int_bits(), 7);
    assert_eq!(d.raw.raw(), 0x123);
}

#[test]
fn signed_fixed_to_dyn_preserves_width_and_frac() {
    let x = SignedFixed::<8, 4>::from_raw(signed(-3));
    let d: SignedDynFixed = x.to_dyn();
    assert_eq!(d.width(), 8);
    assert_eq!(d.frac, 4);
    assert_eq!(d.int_bits(), 3); // 8 - 4 - 1 (sign)
    assert_eq!(d.raw.raw(), -3);
}

#[test]
fn fixed_is_digital() {
    assert_eq!(<Fixed<8, 4> as Digital>::BITS, 8);
    let x = Fixed::<8, 4>::from_raw(bits(0xA5));
    assert_eq!(x.bin().len(), 8);
}

#[test]
fn signed_fixed_is_digital() {
    assert_eq!(<SignedFixed<8, 4> as Digital>::BITS, 8);
    let x = SignedFixed::<8, 4>::from_raw(signed(-1));
    assert_eq!(x.bin().len(), 8);
}

#[test]
fn dont_care_defaults_to_zero() {
    let x = <Fixed<8, 4> as Digital>::dont_care();
    assert_eq!(x.raw.raw(), 0);
}

// Operator-overload smoke tests. These delegate to rhdl-bits ops
// directly; exhaustive verification lives in rhdl-bits' own test suite.
// For saturating variants, see `crates/rhdl-fixed/src/saturating.rs`.

#[test]
fn fixed_add_sub_wrap() {
    let a = Fixed::<4, 0>::from_raw(bits(10));
    let b = Fixed::<4, 0>::from_raw(bits(7));
    assert_eq!((a + b).raw.raw(), 1, "10 + 7 wraps at 4 bits");
    assert_eq!((a - b).raw.raw(), 3, "10 - 7 = 3");
}

#[test]
fn signed_fixed_add_sub_neg_wrap() {
    let a = SignedFixed::<5, 0>::from_raw(signed(-3));
    let b = SignedFixed::<5, 0>::from_raw(signed(5));
    assert_eq!((a + b).raw.raw(), 2);
    assert_eq!((a - b).raw.raw(), -8);
    assert_eq!((-a).raw.raw(), 3);
    // MIN.neg() == MIN
    let min = SignedFixed::<5, 0>::from_raw(signed(-16));
    assert_eq!((-min).raw.raw(), -16);
}
