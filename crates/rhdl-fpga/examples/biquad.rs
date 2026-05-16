//! Biquad IIR filter example: run the single-step `biquad_step` kernel
//! in a host-side loop, threading state from cycle N to cycle N+1.
//!
//! Demonstrates the impulse and step responses of a lowpass biquad. The
//! same `biquad_step` kernel that compiles to Verilog is invoked here
//! as a native Rust function — the same code path RHDL's differential
//! testing uses for the "expected" side.

use rhdl_fixed::SignedFixed;
use rhdl_fpga::dsp::biquad::biquad_step;

const N: usize = 12;
const F: usize = 8;
const ACC_N: usize = 18;

/// Approximate one-pole lowpass at fc/fs ≈ 0.1. Coefficients quantized
/// to Q4.8.
fn lowpass_coeffs() -> [SignedFixed<N, F>; 5] {
    [
        SignedFixed::from_f64(0.067),  // b0
        SignedFixed::from_f64(0.135),  // b1
        SignedFixed::from_f64(0.067),  // b2
        SignedFixed::from_f64(-1.143), // a1
        SignedFixed::from_f64(0.413),  // a2
    ]
}

fn main() {
    let coeffs = lowpass_coeffs();

    let one = SignedFixed::<N, F>::from_f64(1.0);
    let zero = SignedFixed::<N, F>::default();
    let zero_acc = SignedFixed::<ACC_N, F>::default();

    // Impulse response: input is 1.0 at t=0, 0 elsewhere.
    println!("# Impulse response (Q4.8 raw values, divide by 256 for float)");
    println!("{:>3} {:>8} {:>10} {:>10}", "t", "x", "y", "s1");
    let (mut s1, mut s2) = (zero_acc, zero_acc);
    for t in 0..24 {
        let x = if t == 0 { one } else { zero };
        let (y, s1n, s2n) = biquad_step::<N, F, ACC_N>(x, s1, s2, coeffs);
        println!(
            "{t:>3} {:>8} {:>10} {:>10}   ({:.4})",
            x.raw.raw(),
            y.raw.raw(),
            s1.raw.raw(),
            y.raw.raw() as f64 / (1 << F) as f64
        );
        s1 = s1n;
        s2 = s2n;
    }

    // Step response: input is 1.0 forever after t=0.
    println!();
    println!("# Step response");
    println!("{:>3} {:>8} {:>10}", "t", "x", "y");
    let (mut s1, mut s2) = (zero_acc, zero_acc);
    for t in 0..32 {
        let x = if t < 1 { zero } else { one };
        let (y, s1n, s2n) = biquad_step::<N, F, ACC_N>(x, s1, s2, coeffs);
        println!(
            "{t:>3} {:>8} {:>10}   ({:.4})",
            x.raw.raw(),
            y.raw.raw(),
            y.raw.raw() as f64 / (1 << F) as f64
        );
        s1 = s1n;
        s2 = s2n;
    }
}
