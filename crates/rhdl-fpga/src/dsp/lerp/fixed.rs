//!# Linear Interpolation Functions
//!
//! In DSP work, you often need to linearly interpolate
//! between two values.  These functions provide that
//! computation, with a variable number of bits in the
//! two input arguments and a variable number of bits in
//! the interpolation factor.
//!
//!# Internal Details
//! There are various subtleties at play here.
//! Given A: Bits<N>, B: Bits<N> and a factor: Bits<M>,
//!
//! we want to compute
//!
//!> A * (1 - delta) + B * delta = Y
//!
//! where
//!
//!> 0 <= delta = factor / 2^M < 1
//!
//! The `< 1` part is important, since with `M` bits, it is
//! not possible to represent `2^M`.  The largest value that
//! `delta` can take is `2^(M-1)/2^M`.  Normally, when
//! linear interpolation is used, this limitation is not a problem.
//! However, you should be aware, in case you need the function
//! to be able to handle the case of `delta = 1`.
//!
//!
//! Substituting delta, we get
//!
//!> A * ( 1 - factor / 2^M) + B * factor / 2^M = Y
//!
//! Multiplying out by 2^M, we get
//!
//!> A * 2^M - A * factor + B * factor = Y * 2^M
//!
//! To get this into a single multiplication, we need
//!
//!> A * 2^M + (B - A) * factor = Y * 2^M
//!
//! Even if `B` and `A` are unsigned, the `B - A` term is
//! signed, so we need to promote the factor to be signed as well
//!
//!> A * 2^M + Diff * signed_factor = Y * 2^M
//!
//! Here `signed_factor`` will be `M+1`` bits wide, and `Diff` will be `N+1`
//! bits wide
//!
//! The product will thus be `M+N+2` bits wide.  The factor `A * 2^M`
//! will be `N+M` bits wide,
//! and is unsigned.  So we need to convert it to a signed value (which adds 1 bit) and
//! then extend it (signed) by a bit.
//!
//! We can (after adding it), right shift by `M` bits to retrieve `Y`, and then
//! truncate the value to `N` bits, and safely cast as unsigned.  Because the
//! output is guaranteed to be unsigned and the number of bits cannot increase
//! (ignoring fractional bits, of course), this operation can be safely
//! carried out with a truncation operation.
//!
//!# Example
//!
//! The following example demos the unsigned `lerp` function, by wrapping
//! it into a core using the [Func] wrapper.
//!
//!```
#![doc = include_str!("../../../examples/lerp.rs")]
//!```
//!
//! The resulting trace shows the linear interpolation
//!
#![doc = include_str!("../../../doc/lerp.md")]
//!
use rhdl::prelude::*;
use rhdl_fixed::{Fixed, SignedFixed};

#[kernel]
/// Linearly interpolate between unsigned values.
///
/// Interpolates between `lower_value` and `upper_value` with a factor
/// of `factor/2^M` where `M` is the number of bits in `factor`. Note
/// that `factor/2^M < 1`, so the output cannot equal `upper_value`.
/// This core is just a function since it has no state. It _does_
/// require a multiplier.
pub fn lerp_unsigned<const N: usize, const M: usize>(
    lower_value: Fixed<N, 0>,
    upper_value: Fixed<N, 0>,
    factor: Fixed<M, M>,
) -> Fixed<N, 0>
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<M>: BitWidth,
{
    let lower_value = lower_value.raw.dyn_bits();
    let upper_value = upper_value.raw.dyn_bits();
    let factor = factor.raw.dyn_bits();
    let signed_factor = factor.xsgn();
    let diff = upper_value.xsub(lower_value);
    let correction = signed_factor.xmul(diff);
    let lower_value = lower_value.xshl::<M>();
    let lower_value = lower_value.xsgn();
    let y = lower_value.xadd(correction);
    let y = y.xshr::<M>();
    let y = y.as_unsigned().resize::<N>();
    Fixed::<N, 0> { raw: y.as_bits() }
}

#[kernel]
/// Linearly interpolate between signed values.
///
/// Interpolates between `lower_value` and `upper_value` with a factor
/// of `factor/2^M` where `M` is the number of bits in `factor`. The
/// factor is an unsigned fraction in `[0, 1)`.
pub fn lerp_signed<const N: usize, const M: usize>(
    lower_value: SignedFixed<N, 0>,
    upper_value: SignedFixed<N, 0>,
    factor: Fixed<M, M>,
) -> SignedFixed<N, 0>
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<M>: BitWidth,
{
    let lower_value = lower_value.raw.dyn_bits();
    let upper_value = upper_value.raw.dyn_bits();
    let factor = factor.raw.dyn_bits();
    let signed_factor = factor.xsgn();
    let diff = upper_value.xsub(lower_value);
    let correction = signed_factor.xmul(diff);
    let lower_value = lower_value.xshl::<M>();
    let y = lower_value.xadd(correction);
    let y = y.xshr::<M>();
    let y = y.resize::<N>();
    SignedFixed::<N, 0> {
        raw: y.as_signed_bits(),
    }
}

#[cfg(test)]
mod tests {
    use rhdl::core::sim::testbench::kernel::test_kernel_vm_and_verilog_synchronous;

    use super::*;

    /// i32-grounded reference implementation. Independent of rhdl-fixed
    /// so a bug there doesn't mask itself in the expected outputs.
    fn lerp_i32(a: i32, b: i32, f: i32, shift: u8) -> i32 {
        ((a << shift) + (b - a) * f) >> shift
    }

    #[test]
    fn lerp_unsigned_exhaustive() {
        for a in 0u128..16 {
            for b in 0u128..16 {
                for factor in 0u128..32 {
                    let x = Fixed::<4, 0> { raw: b4(a) };
                    let y = Fixed::<4, 0> { raw: b4(b) };
                    let f = Fixed::<5, 5> { raw: b5(factor) };
                    let expected = lerp_i32(a as i32, b as i32, factor as i32, 5) as u128;
                    assert_eq!(
                        lerp_unsigned::<4, 5>(x, y, f).raw.raw(),
                        expected,
                        "{a} {b} {factor}"
                    );
                }
            }
        }
    }

    #[test]
    fn lerp_signed_exhaustive() {
        for a in -8i128..7 {
            for b in -8i128..7 {
                for factor in 0u128..32 {
                    let x = SignedFixed::<4, 0> { raw: s4(a) };
                    let y = SignedFixed::<4, 0> { raw: s4(b) };
                    let f = Fixed::<5, 5> { raw: b5(factor) };
                    let expected = lerp_i32(a as i32, b as i32, factor as i32, 5) as i128;
                    assert_eq!(
                        lerp_signed::<4, 5>(x, y, f).raw.raw(),
                        expected,
                        "{a} {b} {factor}"
                    );
                }
            }
        }
    }

    #[test]
    fn lerp_unsigned_kernel_pipeline() -> miette::Result<()> {
        let vals = (0..16)
            .map(|a| Fixed::<4, 0> { raw: b4(a) })
            .flat_map(|x| (0..16).map(move |b| (x, Fixed::<4, 0> { raw: b4(b) })))
            .flat_map(|(x, y)| (0..32).map(move |f| (x, y, Fixed::<5, 5> { raw: b5(f) })))
            .collect::<Vec<_>>();
        test_kernel_vm_and_verilog_synchronous::<lerp_unsigned<4, 5>, _, _, _>(
            lerp_unsigned,
            vals.into_iter(),
        )?;
        Ok(())
    }

    #[test]
    fn lerp_signed_kernel_pipeline() -> miette::Result<()> {
        let vals = (-8..7)
            .map(|a| SignedFixed::<4, 0> { raw: s4(a) })
            .flat_map(|x| (-8..7).map(move |b| (x, SignedFixed::<4, 0> { raw: s4(b) })))
            .flat_map(|(x, y)| (0..32).map(move |f| (x, y, Fixed::<5, 5> { raw: b5(f) })))
            .collect::<Vec<_>>();
        test_kernel_vm_and_verilog_synchronous::<lerp_signed<4, 5>, _, _, _>(
            lerp_signed,
            vals.into_iter(),
        )?;
        Ok(())
    }
}
