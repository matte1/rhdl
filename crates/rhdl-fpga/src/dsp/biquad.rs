//! # Biquad IIR filter (Direct Form II Transposed)
//!
//! A second-order IIR filter implemented as a single-step combinatorial
//! kernel. The user owns the state registers (`s1`, `s2`) and wraps the
//! step kernel in a `Synchronous` circuit for actual filtering.
//!
//! ## Difference equations (Direct Form II Transposed)
//!
//! ```text
//! y[n]      = b0 * x[n] + s1[n-1]
//! s1_next   = b1 * x[n] - a1 * y[n] + s2[n-1]
//! s2_next   = b2 * x[n] - a2 * y[n]
//! ```
//!
//! Coefficients `{b0, b1, b2, a1, a2}` are passed as a
//! `[SignedFixed<N, F>; 5]` array. `a0` is assumed to be 1.
//!
//! Each multiply produces `2N` bits; the result is shifted right by `F`
//! to bring it back to the chosen Q-format, then resized to the
//! accumulator width before being added with wrap.

use rhdl::prelude::*;
pub use rhdl_fixed::SignedFixed;

/// Coefficient index for `b0` in the coefficient array.
pub const B0: usize = 0;
/// Coefficient index for `b1`.
pub const B1: usize = 1;
/// Coefficient index for `b2`.
pub const B2: usize = 2;
/// Coefficient index for `a1`.
pub const A1: usize = 3;
/// Coefficient index for `a2`.
pub const A2: usize = 4;

/// One step of a Direct Form II Transposed biquad.
///
/// `N` is the coefficient/sample bit width; `F` is the fractional bits
/// (shared by samples and coefficients); `ACC_N` is the state/accumulator
/// width. Returns `(y, s1_next, s2_next)`.
#[kernel]
pub fn biquad_step<const N: usize, const F: usize, const ACC_N: usize>(
    x: SignedFixed<N, F>,
    s1_prev: SignedFixed<ACC_N, F>,
    s2_prev: SignedFixed<ACC_N, F>,
    coeffs: [SignedFixed<N, F>; 5],
) -> (
    SignedFixed<N, F>,
    SignedFixed<ACC_N, F>,
    SignedFixed<ACC_N, F>,
)
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<ACC_N>: BitWidth,
{
    let xr = x.raw;
    // y = b0 * x + s1_prev (truncated to N)
    let b0x = coeffs[B0].raw.xmul(xr).xshr::<F>();
    let b0x_acc: SignedBits<ACC_N> = b0x.resize::<ACC_N>().as_signed_bits();
    let y_acc = s1_prev.raw + b0x_acc;
    let yr: SignedBits<N> = y_acc.dyn_bits().resize::<N>().as_signed_bits();

    // s1_next = b1 * x - a1 * y + s2_prev
    let b1x = coeffs[B1].raw.xmul(xr).xshr::<F>();
    let a1y = coeffs[A1].raw.xmul(yr).xshr::<F>();
    let b1x_acc: SignedBits<ACC_N> = b1x.resize::<ACC_N>().as_signed_bits();
    let a1y_acc: SignedBits<ACC_N> = a1y.resize::<ACC_N>().as_signed_bits();
    let s1_next = s2_prev.raw + b1x_acc - a1y_acc;

    // s2_next = b2 * x - a2 * y
    let b2x = coeffs[B2].raw.xmul(xr).xshr::<F>();
    let a2y = coeffs[A2].raw.xmul(yr).xshr::<F>();
    let b2x_acc: SignedBits<ACC_N> = b2x.resize::<ACC_N>().as_signed_bits();
    let a2y_acc: SignedBits<ACC_N> = a2y.resize::<ACC_N>().as_signed_bits();
    let s2_next = b2x_acc - a2y_acc;

    (
        SignedFixed::<N, F> { raw: yr },
        SignedFixed::<ACC_N, F> { raw: s1_next },
        SignedFixed::<ACC_N, F> { raw: s2_next },
    )
}

#[cfg(test)]
mod tests {
    use rhdl::core::sim::testbench::kernel::test_kernel_vm_and_verilog_synchronous;

    use super::*;

    fn sf<const N: usize, const F: usize>(v: i128) -> SignedFixed<N, F>
    where
        rhdl::bits::W<N>: BitWidth,
    {
        SignedFixed::<N, F>::from_raw(signed(v))
    }

    /// i128-grounded reference for one biquad step (DF-II-T).
    fn biquad_step_ref(
        x: i128,
        s1_prev: i128,
        s2_prev: i128,
        coeffs: &[i128; 5],
        f: u32,
        acc_mask: i128,
        n_mask: i128,
    ) -> (i128, i128, i128) {
        let s = |v: i128| -> i128 {
            let v = v & acc_mask;
            if v & (1i128 << (acc_mask.count_ones() - 1)) != 0 {
                v - (acc_mask + 1)
            } else {
                v
            }
        };
        let s_n = |v: i128| -> i128 {
            let v = v & n_mask;
            if v & (1i128 << (n_mask.count_ones() - 1)) != 0 {
                v - (n_mask + 1)
            } else {
                v
            }
        };
        let y_acc = s(s1_prev.wrapping_add((coeffs[B0] * x) >> f));
        let y = s_n(y_acc);
        let s1_next = s(s2_prev
            .wrapping_add((coeffs[B1] * x) >> f)
            .wrapping_sub((coeffs[A1] * y) >> f));
        let s2_next = s((coeffs[B2] * x >> f).wrapping_sub((coeffs[A2] * y) >> f));
        (y, s1_next, s2_next)
    }

    #[test]
    fn biquad_step_passthrough_b0_one() {
        // Coefficients: b0=1, others=0. Should pass input through.
        // At F=4, value 1.0 is raw 16.
        let coeffs = [
            sf::<8, 4>(16),
            sf::<8, 4>(0),
            sf::<8, 4>(0),
            sf::<8, 4>(0),
            sf::<8, 4>(0),
        ];
        let (y, s1, s2) =
            biquad_step::<8, 4, 12>(sf::<8, 4>(5), sf::<12, 4>(0), sf::<12, 4>(0), coeffs);
        assert_eq!(y.raw.raw(), 5);
        assert_eq!(s1.raw.raw(), 0);
        assert_eq!(s2.raw.raw(), 0);
    }

    #[test]
    fn biquad_step_pure_delay_b0_zero_b1_one() {
        // b0=0, b1=1 means output = previous input (one-step delay,
        // ignoring the feedback path).
        let coeffs = [
            sf::<8, 4>(0),
            sf::<8, 4>(16),
            sf::<8, 4>(0),
            sf::<8, 4>(0),
            sf::<8, 4>(0),
        ];
        let (y0, s1_0, s2_0) =
            biquad_step::<8, 4, 12>(sf::<8, 4>(7), sf::<12, 4>(0), sf::<12, 4>(0), coeffs);
        assert_eq!(
            y0.raw.raw(),
            0,
            "first sample: y = 0 since b0=0 and state=0"
        );
        assert_eq!(s1_0.raw.raw(), 7, "s1 holds b1*x = 7 for the next step");

        let (y1, _, _) = biquad_step::<8, 4, 12>(sf::<8, 4>(0), s1_0, s2_0, coeffs);
        assert_eq!(y1.raw.raw(), 7, "delayed value emerges as y");
    }

    #[test]
    fn biquad_step_matches_reference() {
        let coeffs_i: [i128; 5] = [10, -5, 3, -7, 2];
        let coeffs = [
            sf::<8, 4>(coeffs_i[0]),
            sf::<8, 4>(coeffs_i[1]),
            sf::<8, 4>(coeffs_i[2]),
            sf::<8, 4>(coeffs_i[3]),
            sf::<8, 4>(coeffs_i[4]),
        ];
        let acc_mask = (1i128 << 12) - 1;
        let n_mask = (1i128 << 8) - 1;
        let mut s1: i128 = 0;
        let mut s2: i128 = 0;
        for &x_val in &[3i128, -7, 0, 11, -2] {
            let (want_y, want_s1, want_s2) =
                biquad_step_ref(x_val, s1, s2, &coeffs_i, 4, acc_mask, n_mask);
            let (got_y, got_s1, got_s2) = biquad_step::<8, 4, 12>(
                sf::<8, 4>(x_val),
                sf::<12, 4>(s1),
                sf::<12, 4>(s2),
                coeffs,
            );
            assert_eq!(got_y.raw.raw(), want_y, "x={x_val}");
            assert_eq!(got_s1.raw.raw(), want_s1, "x={x_val} s1");
            assert_eq!(got_s2.raw.raw(), want_s2, "x={x_val} s2");
            s1 = want_s1;
            s2 = want_s2;
        }
    }

    #[test]
    fn biquad_step_kernel_pipeline() -> miette::Result<()> {
        let cases: Vec<_> = (-3i128..3)
            .flat_map(|x| {
                (-3i128..3).map(move |s1| {
                    (
                        sf::<6, 2>(x),
                        sf::<10, 2>(s1),
                        sf::<10, 2>(0),
                        [
                            sf::<6, 2>(2),
                            sf::<6, 2>(-1),
                            sf::<6, 2>(1),
                            sf::<6, 2>(-1),
                            sf::<6, 2>(0),
                        ],
                    )
                })
            })
            .collect();
        test_kernel_vm_and_verilog_synchronous::<biquad_step<6, 2, 10>, _, _, _>(
            biquad_step::<6, 2, 10>,
            cases.into_iter(),
        )?;
        Ok(())
    }
}
