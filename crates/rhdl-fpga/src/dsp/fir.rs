//! # Finite Impulse Response (FIR) filter core
//!
//! A combinatorial N-tap FIR filter taking signed fixed-point samples
//! and signed fixed-point coefficients, producing a wider signed
//! fixed-point output.
//!
//! ## Math
//!
//! For `TAPS` taps with samples `s[0..TAPS]` and coefficients
//! `c[0..TAPS]`, the output is
//!
//! ```text
//! y = sum_{i=0..TAPS} s[i] * c[i]
//! ```
//!
//! Each product `s[i] * c[i]` widens from [`SignedFixed<N, F>`] to
//! `2N` bits. Shifting right by `F` and resizing to the accumulator width
//! brings each product back to scale `F` for accumulation with wrap.
//!
//! ## Choosing the accumulator width
//!
//! Worst-case bound: `|y| <= TAPS * (2^(N-1) - 1)^2 / 2^F`. The
//! accumulator must hold this without overflow. For typical FIR usage
//! (coefficient magnitudes <= 1, sample magnitudes <= 1, both Q1.{N-1}),
//! `ACC_N = N + ceil(log2(TAPS)) + 1` is safe.

use rhdl::prelude::*;
pub use rhdl_fixed::SignedFixed;

/// Combinatorial signed FIR filter.
///
/// `TAPS` is the number of taps; `N` is the per-sample bit width; `F` is
/// the fractional bit count of both samples and coefficients; `ACC_N` is
/// the accumulator/output bit width.
#[kernel]
pub fn fir_signed<const TAPS: usize, const N: usize, const F: usize, const ACC_N: usize>(
    samples: [SignedFixed<N, F>; TAPS],
    coeffs: [SignedFixed<N, F>; TAPS],
) -> SignedFixed<ACC_N, F>
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<ACC_N>: BitWidth,
{
    let mut acc = SignedBits::<ACC_N>::default();
    for i in 0..TAPS {
        let product = samples[i].raw.xmul(coeffs[i].raw);
        let scaled = product.xshr::<F>();
        let typed: SignedBits<ACC_N> = scaled.resize::<ACC_N>().as_signed_bits();
        acc = acc + typed;
    }
    SignedFixed::<ACC_N, F> { raw: acc }
}

#[cfg(test)]
mod tests {
    use rhdl::core::sim::testbench::kernel::test_kernel_vm_and_verilog_synchronous;

    use super::*;

    /// Reference impl using i128 arithmetic so we don't depend on
    /// rhdl-fixed for ground truth.
    fn fir_ref(samples: &[i128], coeffs: &[i128], f: u32) -> i128 {
        let mut acc: i128 = 0;
        for i in 0..samples.len() {
            let product = samples[i] * coeffs[i];
            let scaled = product >> f;
            acc = acc.wrapping_add(scaled);
        }
        acc
    }

    fn sf<const N: usize, const F: usize>(v: i128) -> SignedFixed<N, F>
    where
        rhdl::bits::W<N>: BitWidth,
    {
        SignedFixed::<N, F>::from_raw(signed(v))
    }

    #[test]
    fn fir_signed_impulse_response_equals_coeffs() {
        // Impulse: 1 at tap 0, 0 elsewhere. Output equals coeff[0].
        let coeffs = [sf::<8, 4>(8), sf::<8, 4>(-4), sf::<8, 4>(2), sf::<8, 4>(-1)];
        let samples_first = [sf::<8, 4>(16), sf::<8, 4>(0), sf::<8, 4>(0), sf::<8, 4>(0)];
        let y = fir_signed::<4, 8, 4, 12>(samples_first, coeffs);
        // 16 * 8 = 128, shifted right by 4 = 8. Same as coeffs[0].
        assert_eq!(y.raw.raw(), 8);
    }

    #[test]
    fn fir_signed_step_response_equals_sum_of_coeffs() {
        let coeffs = [sf::<8, 4>(8), sf::<8, 4>(-4), sf::<8, 4>(2), sf::<8, 4>(-1)];
        // value 1.0 at Q4.4 is raw 16
        let samples = [sf::<8, 4>(16); 4];
        let y = fir_signed::<4, 8, 4, 12>(samples, coeffs);
        // Each tap: 16 * c >> 4 = c. Sum = 8 + (-4) + 2 + (-1) = 5.
        assert_eq!(y.raw.raw(), 5);
    }

    #[test]
    fn fir_signed_random_matches_reference() {
        let samples_i: [i128; 4] = [3, -7, 11, -2];
        let coeffs_i: [i128; 4] = [4, -5, 1, 6];
        let samples = [
            sf::<8, 4>(samples_i[0]),
            sf::<8, 4>(samples_i[1]),
            sf::<8, 4>(samples_i[2]),
            sf::<8, 4>(samples_i[3]),
        ];
        let coeffs = [
            sf::<8, 4>(coeffs_i[0]),
            sf::<8, 4>(coeffs_i[1]),
            sf::<8, 4>(coeffs_i[2]),
            sf::<8, 4>(coeffs_i[3]),
        ];
        let want = fir_ref(&samples_i, &coeffs_i, 4);
        let y = fir_signed::<4, 8, 4, 12>(samples, coeffs);
        assert_eq!(y.raw.raw(), want);
    }

    #[test]
    fn fir_signed_kernel_pipeline() -> miette::Result<()> {
        // 2 taps to keep the iverilog runs short.
        let cases: Vec<_> = (-4i128..4)
            .flat_map(|a| {
                (-4i128..4).map(move |b| {
                    let samples = [sf::<6, 2>(a), sf::<6, 2>(b)];
                    let coeffs = [sf::<6, 2>(3), sf::<6, 2>(-2)];
                    (samples, coeffs)
                })
            })
            .collect();
        test_kernel_vm_and_verilog_synchronous::<fir_signed<2, 6, 2, 10>, _, _, _>(
            fir_signed::<2, 6, 2, 10>,
            cases.into_iter(),
        )?;
        Ok(())
    }
}
