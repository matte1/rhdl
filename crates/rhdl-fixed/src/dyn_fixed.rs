//! Runtime-tracked fixed-point types [`DynFixed`] and [`SignedDynFixed`].
//!
//! These mirror [`rhdl_bits::dyn_bits::DynBits`] /
//! [`rhdl_bits::signed_dyn_bits::SignedDynBits`] and carry a
//! fractional-bits count alongside the runtime bit width. They are used
//! for intermediate values in width-extending expressions and are
//! converted back to typed [`Fixed`](crate::Fixed) /
//! [`SignedFixed`](crate::SignedFixed) before flowing out of a kernel
//! function.
//!
//! Unlike [`Fixed`](crate::Fixed) / [`SignedFixed`](crate::SignedFixed),
//! these types are **not** [`Digital`](rhdl_core::Digital): they cannot
//! appear as kernel inputs, outputs, or stateful values. They exist only
//! to thread width and binary-point information through expressions on
//! the host side and the AST emitted from it.

use rhdl_bits::dyn_bits::DynBits;
use rhdl_bits::signed_dyn_bits::SignedDynBits;
use rhdl_bits::{XAdd, XMul, XNeg, XSgn, XSub};

/// Runtime-tracked unsigned fixed-point value.
///
/// `raw.bits()` gives the bit width and `frac` gives the number of
/// fractional bits. The represented rational value is
/// `raw.raw() / 2^frac`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynFixed {
    /// The underlying runtime-width unsigned bits.
    pub raw: DynBits,
    /// Number of fractional bits.
    pub frac: usize,
}

/// Runtime-tracked signed fixed-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedDynFixed {
    /// The underlying runtime-width signed bits.
    pub raw: SignedDynBits,
    /// Number of fractional bits.
    pub frac: usize,
}

// -----------------------------------------------------------------------------
// DynFixed
// -----------------------------------------------------------------------------

impl DynFixed {
    /// Total bit width.
    pub fn width(&self) -> usize {
        self.raw.bits()
    }
    /// Number of integer bits (`width - frac`).
    pub fn int_bits(&self) -> usize {
        self.width() - self.frac
    }

    /// Width-extending add. Panics if the frac counts differ.
    pub fn xadd(self, rhs: DynFixed) -> DynFixed {
        assert_eq!(self.frac, rhs.frac, "DynFixed::xadd requires matching frac");
        DynFixed {
            raw: self.raw.xadd(rhs.raw),
            frac: self.frac,
        }
    }

    /// Width-extending sub. Result is signed (matching `rhdl-bits::xsub`).
    pub fn xsub(self, rhs: DynFixed) -> SignedDynFixed {
        assert_eq!(self.frac, rhs.frac, "DynFixed::xsub requires matching frac");
        SignedDynFixed {
            raw: self.raw.xsub(rhs.raw),
            frac: self.frac,
        }
    }

    /// Width-extending mul. Result frac is the sum of input fracs.
    pub fn xmul(self, rhs: DynFixed) -> DynFixed {
        DynFixed {
            raw: self.raw.xmul(rhs.raw),
            frac: self.frac + rhs.frac,
        }
    }

    /// Left-shift; width grows by `M`, frac unchanged.
    pub fn xshl<const M: usize>(self) -> DynFixed {
        DynFixed {
            raw: self.raw.xshl::<M>(),
            frac: self.frac,
        }
    }

    /// Right-shift; width shrinks by `M`, frac unchanged.
    pub fn xshr<const M: usize>(self) -> DynFixed {
        DynFixed {
            raw: self.raw.xshr::<M>(),
            frac: self.frac,
        }
    }

    /// Promote to signed by prepending a zero sign bit. Width grows by 1.
    pub fn xsgn(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xsgn(),
            frac: self.frac,
        }
    }
}

// -----------------------------------------------------------------------------
// SignedDynFixed
// -----------------------------------------------------------------------------

impl SignedDynFixed {
    /// Total bit width including the sign bit.
    pub fn width(&self) -> usize {
        self.raw.bits()
    }
    /// Number of integer bits excluding the sign bit (`width - frac - 1`).
    pub fn int_bits(&self) -> usize {
        self.width() - self.frac - 1
    }

    /// Width-extending add. Panics if the frac counts differ.
    pub fn xadd(self, rhs: SignedDynFixed) -> SignedDynFixed {
        assert_eq!(
            self.frac, rhs.frac,
            "SignedDynFixed::xadd requires matching frac"
        );
        SignedDynFixed {
            raw: self.raw.xadd(rhs.raw),
            frac: self.frac,
        }
    }

    /// Width-extending sub.
    pub fn xsub(self, rhs: SignedDynFixed) -> SignedDynFixed {
        assert_eq!(
            self.frac, rhs.frac,
            "SignedDynFixed::xsub requires matching frac"
        );
        SignedDynFixed {
            raw: self.raw.xsub(rhs.raw),
            frac: self.frac,
        }
    }

    /// Width-extending mul. Result frac is the sum.
    pub fn xmul(self, rhs: SignedDynFixed) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xmul(rhs.raw),
            frac: self.frac + rhs.frac,
        }
    }

    /// Left-shift; width grows by `M`, frac unchanged.
    pub fn xshl<const M: usize>(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xshl::<M>(),
            frac: self.frac,
        }
    }

    /// Right-shift (arithmetic); width shrinks by `M`, frac unchanged.
    pub fn xshr<const M: usize>(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xshr::<M>(),
            frac: self.frac,
        }
    }

    /// Width-extending negation. Width grows by 1.
    pub fn xneg(self) -> SignedDynFixed {
        SignedDynFixed {
            raw: self.raw.xneg(),
            frac: self.frac,
        }
    }
}
