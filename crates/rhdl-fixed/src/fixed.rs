//! The typed fixed-point types [`Fixed`] and [`SignedFixed`].
//!
//! Both are zero-cost wrappers over the corresponding `rhdl-bits` integer
//! type, plus a compile-time fractional-bits parameter `F`. They both
//! implement [`Digital`](rhdl_core::Digital), so they flow through
//! `#[kernel]` function signatures as ordinary input/output types.
//!
//! ## Construction
//!
//! Outside a kernel body:
//!
//! - `Fixed::<N, F>::from_raw(bits(value))` — wrap any value at width `N`
//! - `Fixed::<N, F>::from_f64(v)` — convert a float with round-to-nearest
//!   (see [`crate::convert`])
//!
//! Inside a `#[kernel]` body, the constructors are not recognized by the
//! kernel compiler; use struct syntax against the public `raw` field:
//!
//! ```ignore
//! Fixed::<8, 4> { raw: some_bits_value }
//! ```
//!
//! ## Same-format ops
//!
//! `+`, `-`, unary `-` are wrapping (matching `rhdl-bits`). For saturating
//! semantics, use the kernel-callable functions in [`crate::saturating`].
//!
//! ## Width-extending ops
//!
//! `xadd`, `xsub`, `xmul`, `xshl`, `xshr`, `xsgn`/`xneg` return a
//! [`DynFixed`] / [`SignedDynFixed`] with the new width and frac
//! tracked at runtime. These method names are recognized by the kernel
//! compiler, so they work inside `#[kernel]` bodies. Convert back to a
//! typed `Fixed` via [`crate::convert`].

use rhdl::prelude::{BitWidth, Bits, Digital, SignedBits};
use rhdl_bits::bitwidth::W;
use rhdl_bits::{XAdd, XMul, XNeg, XSgn, XSub};
use std::ops::{Add, Neg, Sub};

use crate::dyn_fixed::{DynFixed, SignedDynFixed};

/// Unsigned fixed-point value with `N` total bits and `F` fractional bits.
///
/// The represented rational value is `raw / 2^F`, where `raw` is the
/// underlying [`Bits<N>`]. Integer bit count is `N - F`.
///
/// ```
/// use rhdl_bits::bits;
/// use rhdl_fixed::Fixed;
///
/// // Q8.4 — 8 total bits, 4 fractional. Represents 1.5.
/// let x = Fixed::<8, 4>::from_raw(bits(0x18));
/// assert_eq!(x.raw.raw(), 0x18);
/// assert_eq!(Fixed::<8, 4>::WIDTH, 8);
/// assert_eq!(Fixed::<8, 4>::FRAC, 4);
/// ```
#[derive(Digital, Debug, Clone, Copy, PartialEq, Default)]
pub struct Fixed<const N: usize, const F: usize>
where
    W<N>: BitWidth,
{
    /// The raw underlying bits. Public so `#[kernel]` bodies can build
    /// values via `Fixed { raw: ... }` struct syntax.
    pub raw: Bits<N>,
}

/// Signed fixed-point value with `N` total bits (sign bit included) and
/// `F` fractional bits.
///
/// The represented rational value is `raw / 2^F`, where `raw` is the
/// underlying [`SignedBits<N>`]. Integer bit count excluding sign is
/// `N - F - 1`.
///
/// ```
/// use rhdl_bits::signed;
/// use rhdl_fixed::SignedFixed;
///
/// // 8 bits total, 4 fractional, representing -1.5.
/// let x = SignedFixed::<8, 4>::from_raw(signed(-24));
/// assert_eq!(x.raw.raw(), -24);
/// ```
#[derive(Digital, Debug, Clone, Copy, PartialEq, Default)]
pub struct SignedFixed<const N: usize, const F: usize>
where
    W<N>: BitWidth,
{
    /// The raw underlying signed bits. Public so `#[kernel]` bodies can
    /// build values via `SignedFixed { raw: ... }` struct syntax.
    pub raw: SignedBits<N>,
}

// -----------------------------------------------------------------------------
// Fixed: constructors, conversions, same-format and widening ops
// -----------------------------------------------------------------------------

impl<const N: usize, const F: usize> Fixed<N, F>
where
    W<N>: BitWidth,
{
    /// Total width in bits.
    pub const WIDTH: usize = N;
    /// Number of fractional bits.
    pub const FRAC: usize = F;

    /// Wrap a raw [`Bits<N>`] as a [`Fixed<N, F>`].
    pub const fn from_raw(raw: Bits<N>) -> Self {
        Self { raw }
    }

    /// Convert to a runtime-tracked [`DynFixed`].
    pub fn to_dyn(self) -> DynFixed {
        DynFixed {
            raw: self.raw.dyn_bits(),
            frac: F,
        }
    }

    /// Width-extending add. Result width is `max(N, N2) + 1`, frac `F`.
    /// Requires both operands to share the same fractional-bit count `F`.
    pub fn xadd<const N2: usize>(self, rhs: Fixed<N2, F>) -> DynFixed
    where
        W<N2>: BitWidth,
    {
        DynFixed {
            raw: self.raw.xadd(rhs.raw),
            frac: F,
        }
    }

    /// Width-extending sub. Result is signed (matching `rhdl-bits::xsub`).
    /// Width is `max(N, N2) + 1`, frac `F`.
    pub fn xsub<const N2: usize>(self, rhs: Fixed<N2, F>) -> SignedDynFixed
    where
        W<N2>: BitWidth,
    {
        SignedDynFixed {
            raw: self.raw.xsub(rhs.raw),
            frac: F,
        }
    }

    /// Width-extending mul. Result width is `N + N2`, frac `F + F2`.
    ///
    /// ```
    /// use rhdl_bits::bits;
    /// use rhdl_fixed::{DynFixed, Fixed};
    ///
    /// // Q4.2 (raw 10 = 2.5) * Q4.1 (raw 6 = 3.0) = 7.5 at Q8.3
    /// let a = Fixed::<4, 2>::from_raw(bits(0b1010));
    /// let b = Fixed::<4, 1>::from_raw(bits(0b0110));
    /// let p: DynFixed = a.xmul(b);
    /// assert_eq!(p.width(), 8);
    /// assert_eq!(p.frac, 3);
    /// assert_eq!(p.raw.raw(), 60); // 60 / 2^3 = 7.5
    /// ```
    pub fn xmul<const N2: usize, const F2: usize>(self, rhs: Fixed<N2, F2>) -> DynFixed
    where
        W<N2>: BitWidth,
    {
        DynFixed {
            raw: self.raw.xmul(rhs.raw),
            frac: F + F2,
        }
    }

    /// Left-shift by `M` bits. Width grows by `M`, frac unchanged.
    /// Represented value scales by `2^M`.
    pub fn xshl<const M: usize>(self) -> DynFixed {
        DynFixed {
            raw: self.raw.xshl::<M>(),
            frac: F,
        }
    }

    /// Right-shift by `M` bits. Width shrinks by `M`, frac unchanged.
    /// Represented value scales by `2^-M` (low bits truncated).
    pub fn xshr<const M: usize>(self) -> DynFixed {
        DynFixed {
            raw: self.raw.xshr::<M>(),
            frac: F,
        }
    }

    /// Promote to signed by prepending a zero sign bit. Width grows by 1.
    pub fn xsgn(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xsgn(),
            frac: F,
        }
    }
}

// -----------------------------------------------------------------------------
// SignedFixed: constructors, conversions, same-format and widening ops
// -----------------------------------------------------------------------------

impl<const N: usize, const F: usize> SignedFixed<N, F>
where
    W<N>: BitWidth,
{
    /// Total width in bits, including the sign bit.
    pub const WIDTH: usize = N;
    /// Number of fractional bits.
    pub const FRAC: usize = F;

    /// Wrap a raw [`SignedBits<N>`] as a [`SignedFixed<N, F>`].
    pub const fn from_raw(raw: SignedBits<N>) -> Self {
        Self { raw }
    }

    /// Convert to a runtime-tracked [`SignedDynFixed`].
    pub fn to_dyn(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.dyn_bits(),
            frac: F,
        }
    }

    /// Width-extending add. Result width is `max(N, N2) + 1`, frac `F`.
    pub fn xadd<const N2: usize>(self, rhs: SignedFixed<N2, F>) -> SignedDynFixed
    where
        W<N2>: BitWidth,
    {
        SignedDynFixed {
            raw: self.raw.xadd(rhs.raw),
            frac: F,
        }
    }

    /// Width-extending sub. Result width is `max(N, N2) + 1`, frac `F`.
    pub fn xsub<const N2: usize>(self, rhs: SignedFixed<N2, F>) -> SignedDynFixed
    where
        W<N2>: BitWidth,
    {
        SignedDynFixed {
            raw: self.raw.xsub(rhs.raw),
            frac: F,
        }
    }

    /// Width-extending mul. Result width is `N + N2`, frac `F + F2`.
    pub fn xmul<const N2: usize, const F2: usize>(self, rhs: SignedFixed<N2, F2>) -> SignedDynFixed
    where
        W<N2>: BitWidth,
    {
        SignedDynFixed {
            raw: self.raw.xmul(rhs.raw),
            frac: F + F2,
        }
    }

    /// Left-shift by `M` bits. Width grows by `M`, frac unchanged.
    pub fn xshl<const M: usize>(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xshl::<M>(),
            frac: F,
        }
    }

    /// Right-shift by `M` bits (arithmetic shift). Width shrinks by `M`,
    /// frac unchanged.
    pub fn xshr<const M: usize>(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xshr::<M>(),
            frac: F,
        }
    }

    /// Width-extending negation. Width grows by 1 so `MIN` becomes
    /// representable.
    pub fn xneg(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xneg(),
            frac: F,
        }
    }
}

// -----------------------------------------------------------------------------
// Same-format wrapping ops (delegate to rhdl-bits)
// -----------------------------------------------------------------------------

impl<const N: usize, const F: usize> Add for Fixed<N, F>
where
    W<N>: BitWidth,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl<const N: usize, const F: usize> Sub for Fixed<N, F>
where
    W<N>: BitWidth,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl<const N: usize, const F: usize> Add for SignedFixed<N, F>
where
    W<N>: BitWidth,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl<const N: usize, const F: usize> Sub for SignedFixed<N, F>
where
    W<N>: BitWidth,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl<const N: usize, const F: usize> Neg for SignedFixed<N, F>
where
    W<N>: BitWidth,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self { raw: -self.raw }
    }
}
