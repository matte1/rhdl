//! Saturating arithmetic on [`Fixed`] / [`SignedFixed`] as
//! `#[kernel]` functions.
//!
//! These functions compile to hardware (multiplexed comparison + select
//! around the wrapping op) and also run on the host as ordinary Rust
//! when invoked outside a kernel. They use only operations recognized
//! by the kernel compiler: `xadd`, `xsub`, `xneg`, shifts, comparisons,
//! and select.

use rhdl::prelude::*;

use crate::fixed::{Fixed, SignedFixed};

/// Saturating add: clamps to `[0, 2^N - 1]` on overflow.
#[kernel]
pub fn saturating_add<const N: usize, const F: usize>(a: Fixed<N, F>, b: Fixed<N, F>) -> Fixed<N, F>
where
    rhdl::bits::W<N>: BitWidth,
{
    // sum has N + 1 bits; overflow iff the top bit is set.
    let sum = a.raw.xadd(b.raw);
    let high_bit_set = sum.xshr::<N>().any();
    let truncated: Bits<N> = sum.resize::<N>().as_bits();
    let max: Bits<N> = !Bits::<N>::default();
    let result = if high_bit_set { max } else { truncated };
    Fixed::<N, F> { raw: result }
}

/// Saturating sub: clamps to 0 on underflow.
#[kernel]
pub fn saturating_sub<const N: usize, const F: usize>(a: Fixed<N, F>, b: Fixed<N, F>) -> Fixed<N, F>
where
    rhdl::bits::W<N>: BitWidth,
{
    let underflow = a.raw < b.raw;
    let diff: Bits<N> = a.raw - b.raw;
    let result = if underflow {
        Bits::<N>::default()
    } else {
        diff
    };
    Fixed::<N, F> { raw: result }
}

/// Saturating signed add: clamps to `[MIN, MAX]` of the N-bit signed range.
///
/// Detects overflow by re-sign-extending the truncated result and comparing
/// to the N+1-bit sum.
#[kernel]
pub fn saturating_add_signed<const N: usize, const F: usize>(
    a: SignedFixed<N, F>,
    b: SignedFixed<N, F>,
) -> SignedFixed<N, F>
where
    rhdl::bits::W<N>: BitWidth,
{
    let sum = a.raw.xadd(b.raw); // SignedDynBits, width N+1
    let truncated: SignedBits<N> = sum.resize::<N>().as_signed_bits();
    let re_extended = truncated.xext::<1>(); // SignedDynBits, width N+1 (sign-extended)
    let no_overflow = sum == re_extended;
    let result_negative = sum.xshr::<N>().any(); // top bit of N+1-bit sum
    // MAX = 0b0111...1, MIN = 0b1000...0 at width N.
    // !Bits::<N>::default() = all-ones; the Shr below is unsigned.
    let all_ones_u: Bits<N> = !Bits::<N>::default();
    let max_u: Bits<N> = all_ones_u >> bits::<N>(1);
    let max_s: SignedBits<N> = max_u.as_signed();
    let min_s: SignedBits<N> = (!max_u).as_signed();
    let result = if no_overflow {
        truncated
    } else if result_negative {
        min_s
    } else {
        max_s
    };
    SignedFixed::<N, F> { raw: result }
}

/// Saturating signed sub: clamps to `[MIN, MAX]`.
#[kernel]
pub fn saturating_sub_signed<const N: usize, const F: usize>(
    a: SignedFixed<N, F>,
    b: SignedFixed<N, F>,
) -> SignedFixed<N, F>
where
    rhdl::bits::W<N>: BitWidth,
{
    let diff = a.raw.xsub(b.raw); // SignedDynBits, width N+1
    let truncated: SignedBits<N> = diff.resize::<N>().as_signed_bits();
    let re_extended = truncated.xext::<1>();
    let no_overflow = diff == re_extended;
    let result_negative = diff.xshr::<N>().any();
    let all_ones_u: Bits<N> = !Bits::<N>::default();
    let max_u: Bits<N> = all_ones_u >> bits::<N>(1);
    let max_s: SignedBits<N> = max_u.as_signed();
    let min_s: SignedBits<N> = (!max_u).as_signed();
    let result = if no_overflow {
        truncated
    } else if result_negative {
        min_s
    } else {
        max_s
    };
    SignedFixed::<N, F> { raw: result }
}

/// Saturating signed negation: `MIN.saturating_neg() == MAX`.
#[kernel]
pub fn saturating_neg_signed<const N: usize, const F: usize>(
    a: SignedFixed<N, F>,
) -> SignedFixed<N, F>
where
    rhdl::bits::W<N>: BitWidth,
{
    // xneg widens to N+1 bits, so -MIN fits without wrap.
    let neg = a.raw.xneg();
    let truncated: SignedBits<N> = neg.resize::<N>().as_signed_bits();
    let re_extended = truncated.xext::<1>();
    let no_overflow = neg == re_extended;
    // The only overflow case is `-MIN`, which would be `MAX + 1`. Saturate to MAX.
    let all_ones_u: Bits<N> = !Bits::<N>::default();
    let max_u: Bits<N> = all_ones_u >> bits::<N>(1);
    let max_s: SignedBits<N> = max_u.as_signed();
    let result = if no_overflow { truncated } else { max_s };
    SignedFixed::<N, F> { raw: result }
}

#[cfg(test)]
mod tests {
    use rhdl::core::sim::testbench::kernel::test_kernel_vm_and_verilog_synchronous;

    use super::*;

    fn u_max<const N: usize>() -> u128 {
        if N == 128 {
            u128::MAX
        } else {
            (1u128 << N) - 1
        }
    }

    #[test]
    fn saturating_add_host_exhaustive_4bit() {
        let max = u_max::<4>();
        for a in 0u128..16 {
            for b in 0u128..16 {
                let x = Fixed::<4, 0> { raw: bits(a) };
                let y = Fixed::<4, 0> { raw: bits(b) };
                let want = (a + b).min(max);
                assert_eq!(saturating_add(x, y).raw.raw(), want, "{a}+{b}");
            }
        }
    }

    #[test]
    fn saturating_sub_host_exhaustive_4bit() {
        for a in 0u128..16 {
            for b in 0u128..16 {
                let x = Fixed::<4, 0> { raw: bits(a) };
                let y = Fixed::<4, 0> { raw: bits(b) };
                let want = a.saturating_sub(b);
                assert_eq!(saturating_sub(x, y).raw.raw(), want, "{a}-{b}");
            }
        }
    }

    #[test]
    fn saturating_add_kernel_pipeline() -> Result<(), rhdl::core::RHDLError> {
        let cases: Vec<_> = (0u128..8)
            .flat_map(|a| {
                (0u128..8).map(move |b| {
                    (
                        Fixed::<4, 0> { raw: bits(a) },
                        Fixed::<4, 0> { raw: bits(b) },
                    )
                })
            })
            .collect();
        test_kernel_vm_and_verilog_synchronous::<saturating_add<4, 0>, _, _, _>(
            saturating_add::<4, 0>,
            cases.into_iter(),
        )?;
        Ok(())
    }

    #[test]
    fn saturating_sub_kernel_pipeline() -> Result<(), rhdl::core::RHDLError> {
        let cases: Vec<_> = (0u128..8)
            .flat_map(|a| {
                (0u128..8).map(move |b| {
                    (
                        Fixed::<4, 0> { raw: bits(a) },
                        Fixed::<4, 0> { raw: bits(b) },
                    )
                })
            })
            .collect();
        test_kernel_vm_and_verilog_synchronous::<saturating_sub<4, 0>, _, _, _>(
            saturating_sub::<4, 0>,
            cases.into_iter(),
        )?;
        Ok(())
    }

    fn s_bounds<const N: usize>() -> (i128, i128) {
        let half = 1i128 << (N - 1);
        (-half, half - 1)
    }

    #[test]
    fn saturating_add_signed_host_exhaustive_5bit() {
        let (lo, hi) = s_bounds::<5>();
        for a in -16i128..16 {
            for b in -16i128..16 {
                let x = SignedFixed::<5, 0> { raw: signed(a) };
                let y = SignedFixed::<5, 0> { raw: signed(b) };
                let want = (a + b).clamp(lo, hi);
                assert_eq!(
                    saturating_add_signed(x, y).raw.raw(),
                    want,
                    "sat_add {a} + {b}"
                );
            }
        }
    }

    #[test]
    fn saturating_sub_signed_host_exhaustive_5bit() {
        let (lo, hi) = s_bounds::<5>();
        for a in -16i128..16 {
            for b in -16i128..16 {
                let x = SignedFixed::<5, 0> { raw: signed(a) };
                let y = SignedFixed::<5, 0> { raw: signed(b) };
                let want = (a - b).clamp(lo, hi);
                assert_eq!(
                    saturating_sub_signed(x, y).raw.raw(),
                    want,
                    "sat_sub {a} - {b}"
                );
            }
        }
    }

    #[test]
    fn saturating_neg_signed_host_handles_min() {
        let (lo, hi) = s_bounds::<5>();
        for a in -16i128..16 {
            let x = SignedFixed::<5, 0> { raw: signed(a) };
            let want = (-a).clamp(lo, hi);
            assert_eq!(saturating_neg_signed(x).raw.raw(), want, "sat_neg {a}");
        }
    }

    #[test]
    fn saturating_add_signed_kernel_pipeline() -> Result<(), rhdl::core::RHDLError> {
        let cases: Vec<_> = (-4i128..4)
            .flat_map(|a| {
                (-4i128..4).map(move |b| {
                    (
                        SignedFixed::<4, 0> { raw: signed(a) },
                        SignedFixed::<4, 0> { raw: signed(b) },
                    )
                })
            })
            .collect();
        test_kernel_vm_and_verilog_synchronous::<saturating_add_signed<4, 0>, _, _, _>(
            saturating_add_signed::<4, 0>,
            cases.into_iter(),
        )?;
        Ok(())
    }

    #[test]
    fn saturating_neg_signed_kernel_pipeline() -> Result<(), rhdl::core::RHDLError> {
        let cases: Vec<_> = (-4i128..4)
            .map(|a| (SignedFixed::<4, 0> { raw: signed(a) },))
            .collect();
        test_kernel_vm_and_verilog_synchronous::<saturating_neg_signed<4, 0>, _, _, _>(
            saturating_neg_signed::<4, 0>,
            cases.into_iter(),
        )?;
        Ok(())
    }
}
