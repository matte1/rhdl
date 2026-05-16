//! Multiply-accumulate (MAC) **test helper**.
//!
//! [`Mac`] and [`SignedMac`] are host-side accumulators for computing
//! reference values in DSP tests — e.g. the expected output of a FIR
//! filter, or a coefficient-sum sanity check. They are not intended for
//! use inside `#[kernel]` bodies and are not [`Digital`](rhdl_core::Digital).
//!
//! For combinatorial sum-of-products in a kernel, inline the
//! `xmul`/`resize`/`xadd` chain directly — see
//! [`rhdl-fpga::dsp::fir`](../../rhdl_fpga/dsp/fir/index.html) for the
//! canonical pattern.
//!
//! ## Why not kernel-compatible?
//!
//! A kernel-callable `Mac::mac(self, a, b)` would need
//! `xshr::<{F1+F2-F}>()` for the truncation step, which requires a
//! const-generic expression in turbofish — not available on stable Rust.
//! The constraint also has to be checked at runtime (`F1+F2 == F`),
//! which adds a panic path the kernel compiler can't synthesize. The
//! kernel-side pattern is short enough to inline without losing clarity.
//!
//! ## Constraint
//!
//! The product's frac count (`F1 + F2`) must equal the accumulator's
//! frac (`F`). Panics on mismatch.

use crate::fixed::{Fixed, SignedFixed};
use rhdl_bits::bitwidth::{BitWidth, W};

/// Host-side unsigned multiply-accumulate test helper.
///
/// ```
/// use rhdl_bits::bits;
/// use rhdl_fixed::{Fixed, mac::Mac};
///
/// // Q8.8 accumulator, Q4.4 multiplicands. F1 + F2 = 4 + 4 = 8 = acc F.
/// let s = [Fixed::<4, 4>::from_raw(bits(0b0010)), // 0.125
///          Fixed::<4, 4>::from_raw(bits(0b1000))];// 0.5
/// let c = [Fixed::<4, 4>::from_raw(bits(0b0100)), // 0.25
///          Fixed::<4, 4>::from_raw(bits(0b0010))];// 0.125
///
/// let out = Mac::<16, 8>::zero()
///     .mac(s[0], c[0])
///     .mac(s[1], c[1])
///     .result();
///
/// // 0.125*0.25 + 0.5*0.125 = 0.03125 + 0.0625 = 0.09375
/// // At Q8.8: 0.09375 * 256 = 24
/// assert_eq!(out.raw.raw(), 24);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Mac<const N: usize, const F: usize>
where
    W<N>: BitWidth,
{
    acc: Fixed<N, F>,
}

impl<const N: usize, const F: usize> Mac<N, F>
where
    W<N>: BitWidth,
{
    /// Accumulator initialized to zero.
    pub fn zero() -> Self {
        Self {
            acc: Fixed::default(),
        }
    }

    /// Accumulator seeded with `init`.
    pub fn new(init: Fixed<N, F>) -> Self {
        Self { acc: init }
    }

    /// Add `a * b` to the accumulator (wrapping). Panics if
    /// `F1 + F2 != F`.
    pub fn mac<const N1: usize, const F1: usize, const N2: usize, const F2: usize>(
        self,
        a: Fixed<N1, F1>,
        b: Fixed<N2, F2>,
    ) -> Self
    where
        W<N1>: BitWidth,
        W<N2>: BitWidth,
    {
        assert_eq!(
            F1 + F2,
            F,
            "Mac: product frac {} != accumulator frac {F}",
            F1 + F2
        );
        let product = a.xmul(b);
        let product_typed: Fixed<N, F> = product.truncate();
        Self {
            acc: self.acc + product_typed,
        }
    }

    /// Current accumulator value.
    pub fn result(self) -> Fixed<N, F> {
        self.acc
    }
}

/// Host-side signed multiply-accumulate test helper.
#[derive(Debug, Clone, Copy)]
pub struct SignedMac<const N: usize, const F: usize>
where
    W<N>: BitWidth,
{
    acc: SignedFixed<N, F>,
}

impl<const N: usize, const F: usize> SignedMac<N, F>
where
    W<N>: BitWidth,
{
    /// Accumulator initialized to zero.
    pub fn zero() -> Self {
        Self {
            acc: SignedFixed::default(),
        }
    }

    /// Accumulator seeded with `init`.
    pub fn new(init: SignedFixed<N, F>) -> Self {
        Self { acc: init }
    }

    /// Add `a * b` to the accumulator (wrapping). Panics if
    /// `F1 + F2 != F`.
    pub fn mac<const N1: usize, const F1: usize, const N2: usize, const F2: usize>(
        self,
        a: SignedFixed<N1, F1>,
        b: SignedFixed<N2, F2>,
    ) -> Self
    where
        W<N1>: BitWidth,
        W<N2>: BitWidth,
    {
        assert_eq!(
            F1 + F2,
            F,
            "SignedMac: product frac {} != accumulator frac {F}",
            F1 + F2
        );
        let product = a.xmul(b);
        let product_typed: SignedFixed<N, F> = product.truncate();
        Self {
            acc: self.acc + product_typed,
        }
    }

    /// Current accumulator value.
    pub fn result(self) -> SignedFixed<N, F> {
        self.acc
    }
}
