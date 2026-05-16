//! FIR filter example: a 4-tap moving-average filter applied to a
//! triangle-wave input, producing an SVG waveform trace.
//!
//! Wraps `fir_signed` in a [`Func`] so it can be simulated and traced
//! like any other combinatorial core.

use rhdl::prelude::*;
use rhdl_fixed::SignedFixed;
use rhdl_fpga::doc::write_svg_as_markdown;
use rhdl_fpga::dsp::fir::fir_signed;

const N: usize = 8;
const F: usize = 4;
const TAPS: usize = 4;
const ACC_N: usize = 12;

#[derive(PartialEq, Clone, Copy, Digital)]
pub struct FirIn {
    pub samples: [SignedFixed<N, F>; TAPS],
    pub coeffs: [SignedFixed<N, F>; TAPS],
}

#[kernel]
pub fn wrap_fir(_cr: ClockReset, i: FirIn) -> SignedFixed<ACC_N, F> {
    fir_signed::<TAPS, N, F, ACC_N>(i.samples, i.coeffs)
}

fn main() -> Result<(), RHDLError> {
    let uut: Func<FirIn, SignedFixed<ACC_N, F>> = Func::try_new::<wrap_fir>()?;

    // Moving-average filter: all four coefficients = 0.25.
    let coeffs = [
        SignedFixed::<N, F>::from_f64(0.25),
        SignedFixed::<N, F>::from_f64(0.25),
        SignedFixed::<N, F>::from_f64(0.25),
        SignedFixed::<N, F>::from_f64(0.25),
    ];

    // Sliding window of a triangle wave (peak 1.0, trough -1.0).
    let triangle: Vec<i128> = (-8..=8).chain((-8..=7).rev()).cycle().take(32).collect();
    let stream = (0..triangle.len() - TAPS)
        .map(move |start| {
            let mut samples = [SignedFixed::<N, F>::default(); TAPS];
            for j in 0..TAPS {
                samples[j] = SignedFixed::<N, F>::from_raw(signed(triangle[start + j]));
            }
            FirIn { samples, coeffs }
        })
        .without_reset()
        .clock_pos_edge(100);

    let trace = uut.run(stream).collect::<SvgFile>();
    write_svg_as_markdown(trace, "fir.md", SvgOptions::default())?;
    Ok(())
}
