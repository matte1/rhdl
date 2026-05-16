//! Conversions: [`DynFixed`] / [`SignedDynFixed`] back into typed
//! [`Fixed`] / [`SignedFixed`] with optional rounding and saturation,
//! plus the host-only `f64`-to-Fixed helpers.
//!
//! The target [`Fixed<N, F>`] determines two things:
//!
//! 1. How many low (fractional) bits are dropped: `source.frac - F`.
//! 2. How the result is fit into `N` bits: truncate (wrap) or saturate
//!    (clip) the high bits.
//!
//! All four combinations of rounding × overflow handling are exposed:
//!
//! | Method            | Low bits     | High bits |
//! |-------------------|--------------|-----------|
//! | `truncate`        | discard      | wrap      |
//! | `round_nearest`   | round (RTNE) | wrap      |
//! | `saturate`        | discard      | clip      |
//! | `round_saturate`  | round (RTNE) | clip      |
//!
//! `target.frac > source.frac` panics — it would require inventing
//! precision out of nothing.

use rhdl_bits::bitwidth::{BitWidth, W};
use rhdl_bits::{bits, signed};

use crate::dyn_fixed::{DynFixed, SignedDynFixed};
use crate::fixed::{Fixed, SignedFixed};

// -----------------------------------------------------------------------------
// Sign-aware conversions between Dyn variants
// -----------------------------------------------------------------------------

impl DynFixed {
    /// Reinterpret as a signed value (no bit change). Useful before
    /// signed-only conversions. Width and frac unchanged.
    pub fn as_signed(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.as_signed(),
            frac: self.frac,
        }
    }
}

impl SignedDynFixed {
    /// Reinterpret as an unsigned value (no bit change). Width and frac
    /// unchanged. Use with care — semantically valid only when the value
    /// is known non-negative.
    pub fn as_unsigned(self) -> DynFixed {
        DynFixed {
            raw: self.raw.as_unsigned(),
            frac: self.frac,
        }
    }
}

// -----------------------------------------------------------------------------
// Unsigned conversions
// -----------------------------------------------------------------------------

impl DynFixed {
    /// Truncate to [`Fixed<N, F>`]. Low bits beyond `F` are discarded
    /// (truncation), high bits beyond `N` are dropped (wrap on overflow).
    pub fn truncate<const N: usize, const F: usize>(self) -> Fixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "truncate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let dropped = self.raw.raw() >> shift;
        Fixed::<N, F>::from_raw(bits(dropped & u_mask::<N>()))
    }

    /// Round-to-nearest-even on the discarded low bits, then truncate
    /// high bits (wrap on overflow).
    pub fn round_nearest<const N: usize, const F: usize>(self) -> Fixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "round_nearest target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let rounded = round_nearest_u128(self.raw.raw(), shift);
        Fixed::<N, F>::from_raw(bits(rounded & u_mask::<N>()))
    }

    /// Truncate low bits, then saturate to `[0, 2^N - 1]`.
    pub fn saturate<const N: usize, const F: usize>(self) -> Fixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "saturate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let dropped = self.raw.raw() >> shift;
        let max = u_mask::<N>();
        let clamped = if dropped > max { max } else { dropped };
        Fixed::<N, F>::from_raw(bits(clamped))
    }

    /// Round low, saturate high.
    pub fn round_saturate<const N: usize, const F: usize>(self) -> Fixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "round_saturate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let rounded = round_nearest_u128(self.raw.raw(), shift);
        let max = u_mask::<N>();
        let clamped = if rounded > max { max } else { rounded };
        Fixed::<N, F>::from_raw(bits(clamped))
    }
}

// -----------------------------------------------------------------------------
// Signed conversions
// -----------------------------------------------------------------------------

impl SignedDynFixed {
    /// Truncate to [`SignedFixed<N, F>`]. Wraps signed overflow.
    pub fn truncate<const N: usize, const F: usize>(self) -> SignedFixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "truncate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let dropped = self.raw.raw() >> shift; // arithmetic right shift on i128
        SignedFixed::<N, F>::from_raw(signed(s_wrap::<N>(dropped)))
    }

    /// RTNE on discarded low bits, wrap on overflow.
    pub fn round_nearest<const N: usize, const F: usize>(self) -> SignedFixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "round_nearest target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let rounded = round_nearest_i128(self.raw.raw(), shift);
        SignedFixed::<N, F>::from_raw(signed(s_wrap::<N>(rounded)))
    }

    /// Truncate low, saturate to `[MIN, MAX]`.
    pub fn saturate<const N: usize, const F: usize>(self) -> SignedFixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "saturate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let dropped = self.raw.raw() >> shift;
        let (lo, hi) = s_bounds::<N>();
        SignedFixed::<N, F>::from_raw(signed(dropped.clamp(lo, hi)))
    }

    /// Round low, saturate high.
    pub fn round_saturate<const N: usize, const F: usize>(self) -> SignedFixed<N, F>
    where
        W<N>: BitWidth,
    {
        assert!(
            F <= self.frac,
            "round_saturate target frac {F} exceeds source frac {}",
            self.frac
        );
        let shift = self.frac - F;
        let rounded = round_nearest_i128(self.raw.raw(), shift);
        let (lo, hi) = s_bounds::<N>();
        SignedFixed::<N, F>::from_raw(signed(rounded.clamp(lo, hi)))
    }
}

// -----------------------------------------------------------------------------
// Float-to-Fixed (host-side only)
// -----------------------------------------------------------------------------

impl<const N: usize, const F: usize> Fixed<N, F>
where
    W<N>: BitWidth,
{
    /// Convert an `f64` to [`Fixed<N, F>`] using round-to-nearest-even.
    /// Panics if the value is out of range or not finite.
    ///
    /// This is host-only — `f64` cannot appear inside `#[kernel]`
    /// bodies. Use it at startup or as the right-hand side of `const`
    /// initializers that fit a `const fn` context only if no float
    /// arithmetic is required (in this implementation, `f64::round` is
    /// not `const`, so calls must run at host runtime).
    pub fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "from_f64: value must be finite");
        let scaled = value * (1u128 << F) as f64;
        let rounded = scaled.round();
        let max = u_mask::<N>() as f64;
        assert!(
            (0.0..=max).contains(&rounded),
            "from_f64: {value} out of Fixed<{N}, {F}> range [0, {max}/2^{F}]"
        );
        Self::from_raw(bits(rounded as u128))
    }
}

impl<const N: usize, const F: usize> SignedFixed<N, F>
where
    W<N>: BitWidth,
{
    /// Convert an `f64` to [`SignedFixed<N, F>`] using
    /// round-to-nearest-even. Panics if out of range or not finite.
    pub fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "from_f64: value must be finite");
        let scaled = value * (1u128 << F) as f64;
        let rounded = scaled.round();
        let (lo, hi) = s_bounds::<N>();
        assert!(
            (lo as f64..=hi as f64).contains(&rounded),
            "from_f64: {value} out of SignedFixed<{N}, {F}> range",
        );
        Self::from_raw(signed(rounded as i128))
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn u_mask<const N: usize>() -> u128 {
    if N == 128 {
        u128::MAX
    } else {
        (1u128 << N) - 1
    }
}

fn s_bounds<const N: usize>() -> (i128, i128) {
    if N == 128 {
        (i128::MIN, i128::MAX)
    } else {
        let half = 1i128 << (N - 1);
        (-half, half - 1)
    }
}

fn s_wrap<const N: usize>(v: i128) -> i128 {
    if N == 128 {
        v
    } else {
        let mask = (1i128 << N) - 1;
        let v = v & mask;
        if v & (1i128 << (N - 1)) != 0 {
            v - (1i128 << N)
        } else {
            v
        }
    }
}

/// Round-to-nearest-even, dropping `shift` low bits from an unsigned value.
fn round_nearest_u128(value: u128, shift: usize) -> u128 {
    if shift == 0 {
        return value;
    }
    let half = 1u128 << (shift - 1);
    let mask = (1u128 << shift) - 1;
    let remainder = value & mask;
    let truncated = value >> shift;
    if remainder > half {
        truncated + 1
    } else if remainder == half {
        truncated + (truncated & 1) // round to even
    } else {
        truncated
    }
}

/// Round-to-nearest-even, dropping `shift` low bits from a signed value.
/// Handles negative values correctly by working in two's-complement.
fn round_nearest_i128(value: i128, shift: usize) -> i128 {
    if shift == 0 {
        return value;
    }
    // Treat as unsigned for the rounding decision, then sign-extend.
    let u = value as u128;
    let rounded_u = round_nearest_u128(u, shift);
    // Sign-extend from (128 - shift) bits to 128 bits.
    let used_bits = 128 - shift;
    let sign_bit_pos = used_bits - 1;
    if (rounded_u >> sign_bit_pos) & 1 == 1 {
        // Negative: fill high bits with 1.
        let mask = !((1u128 << used_bits) - 1);
        (rounded_u | mask) as i128
    } else {
        rounded_u as i128
    }
}
