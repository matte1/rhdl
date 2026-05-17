//! # CORDIC sine/cosine (iterative)
//!
//! Iterative implementation of the COordinate Rotation DIgital Computer
//! algorithm for computing `sin(phase)` and `cos(phase)` using only
//! shifts, additions, and a small ROM of `atan(2^-i)` constants — no
//! multipliers.
//!
//! ## Topology
//!
//! One iteration per clock, 14 iterations total. State holds the
//! `(x, y, z, iter, busy)` registers. The user pulses `start` with a
//! `phase`; 14 cycles later, `valid` asserts for one cycle with the
//! result on `(cos, sin)`.
//!
//! This is the iterative form because the combinatorial alternative
//! chains 14 stages of (compare → 2 adders → 2 shifts → mux), giving a
//! ~35 ns critical path that limits the clock to ~28 MHz. The iterative
//! form trades a 14-cycle latency for ~1/14th the area and meets fast
//! clocks. For maximum throughput (1 sample per clock), a pipelined
//! version would add a register between each stage of the combinatorial
//! version.
//!
//! ## Algorithm
//!
//! Each iteration rotates `(x, y)` by ±atan(2^-i):
//!
//! ```text
//! if z >= 0:
//!     x' = x - (y >> i)
//!     y' = y + (x >> i)
//!     z' = z - atan(2^-i)
//! else:
//!     x' = x + (y >> i)
//!     y' = y - (x >> i)
//!     z' = z + atan(2^-i)
//! ```
//!
//! Starting from `(x_0, y_0, z_0) = (K_INV, 0, phase)`, after 14
//! iterations `(x, y) ≈ (cos(phase), sin(phase))`. The `K_INV` constant
//! cancels the cumulative rotation gain.
//!
//! ## Convergence range
//!
//! The algorithm converges only for `phase ∈ [-π/2, π/2]`. For inputs
//! outside that range, do a quadrant reduction in the caller. The
//! `Q1.14` format used here can't represent π anyway (range `[-2, 2)`).
//!
//! ## Precision
//!
//! Each iteration adds ~1 bit of precision. With 14 iterations and 14
//! fractional bits, the worst-case error is around `2^-13 ≈ 1.2e-4`.

use rhdl::prelude::*;
use rhdl_fixed::SignedFixed;

use crate::core::dff;

/// Number of CORDIC iterations.
pub const N_ITER: usize = 14;

/// `K_inv = 1 / prod_i cos(atan(2^-i))` for `i ∈ 0..N_ITER`.
/// Pre-loading `x = K_INV` cancels the cumulative scaling of the
/// rotations, so the final `(x, y)` lands on the unit circle.
///
/// Quantized to `Q1.14`: `round(0.6072529 * 16384) = 9949`.
pub const K_INV: SignedFixed<16, 14> = SignedFixed {
    raw: signed::<16>(9949),
};

/// Per-iteration `atan(2^-i)` constants quantized to `Q1.14`.
pub const ATAN_TABLE: [SignedFixed<16, 14>; N_ITER] = [
    SignedFixed {
        raw: signed::<16>(12868),
    }, // atan(1)      ≈ 0.7854
    SignedFixed {
        raw: signed::<16>(7596),
    }, // atan(1/2)    ≈ 0.4636
    SignedFixed {
        raw: signed::<16>(4014),
    }, // atan(1/4)    ≈ 0.2450
    SignedFixed {
        raw: signed::<16>(2037),
    }, // atan(1/8)    ≈ 0.1244
    SignedFixed {
        raw: signed::<16>(1023),
    }, // atan(1/16)   ≈ 0.0624
    SignedFixed {
        raw: signed::<16>(512),
    }, // atan(1/32)   ≈ 0.0312
    SignedFixed {
        raw: signed::<16>(256),
    }, // atan(1/64)   ≈ 0.0156
    SignedFixed {
        raw: signed::<16>(128),
    }, // atan(1/128)  ≈ 0.0078
    SignedFixed {
        raw: signed::<16>(64),
    }, // atan(1/256)  ≈ 0.0039
    SignedFixed {
        raw: signed::<16>(32),
    }, // atan(1/512)  ≈ 0.00195
    SignedFixed {
        raw: signed::<16>(16),
    }, // atan(1/1024) ≈ 0.000977
    SignedFixed {
        raw: signed::<16>(8),
    }, // atan(1/2048) ≈ 0.000488
    SignedFixed {
        raw: signed::<16>(4),
    }, // atan(1/4096) ≈ 0.000244
    SignedFixed {
        raw: signed::<16>(2),
    }, // atan(1/8192) ≈ 0.000122
];

/// Input to the CORDIC core. Pulse `start` high for one cycle with a
/// `phase` value to begin a new computation.
#[derive(PartialEq, Clone, Copy, Digital, Default)]
pub struct CordicIn {
    /// Phase in radians, `Q1.14`. Must be in `[-π/2, π/2]`.
    pub phase: SignedFixed<16, 14>,
    /// One-cycle pulse to start a new computation. Ignored if already busy.
    pub start: bool,
}

/// Output from the CORDIC core. `cos` and `sin` are valid only on the
/// cycle when `valid` is asserted.
#[derive(PartialEq, Clone, Copy, Digital, Default)]
pub struct CordicOut {
    /// Cosine of the input phase, `Q1.14`.
    pub cos: SignedFixed<16, 14>,
    /// Sine of the input phase, `Q1.14`.
    pub sin: SignedFixed<16, 14>,
    /// Asserted for one cycle when `cos` / `sin` are valid.
    pub valid: bool,
}

/// Iterative CORDIC core.
#[derive(Clone, Debug, Synchronous, SynchronousDQ)]
#[rhdl(dq_no_prefix)]
pub struct CordicCore {
    x: dff::DFF<SignedFixed<16, 14>>,
    y: dff::DFF<SignedFixed<16, 14>>,
    z: dff::DFF<SignedFixed<16, 14>>,
    iter: dff::DFF<Bits<4>>,
    busy: dff::DFF<bool>,
}

impl Default for CordicCore {
    fn default() -> Self {
        Self {
            x: dff::DFF::new(SignedFixed::<16, 14>::default()),
            y: dff::DFF::new(SignedFixed::<16, 14>::default()),
            z: dff::DFF::new(SignedFixed::<16, 14>::default()),
            iter: dff::DFF::new(Bits::<4>::default()),
            busy: dff::DFF::new(false),
        }
    }
}

impl SynchronousIO for CordicCore {
    type I = CordicIn;
    type O = CordicOut;
    type Kernel = cordic_kernel;
}

/// CORDIC state-machine kernel.
///
/// Manually unrolled: each iteration index has its own `if` arm with a
/// const-generic `xshr` and a literal `ATAN_TABLE` index. This is what
/// the unroll looks like when const-generic turbofish can't take a
/// loop variable.
#[kernel]
pub fn cordic_kernel(cr: ClockReset, i: CordicIn, q: Q) -> (CordicOut, D) {
    // Default: hold state.
    let mut d = D {
        x: q.x,
        y: q.y,
        z: q.z,
        iter: q.iter,
        busy: q.busy,
    };

    if !q.busy && i.start {
        // Load a new computation.
        d.x = K_INV;
        d.y = SignedFixed::<16, 14>::default();
        d.z = i.phase;
        d.iter = bits::<4>(0);
        d.busy = true;
    } else if q.busy {
        // Work in raw SignedBits<16> inside the iteration body. The
        // operator overloads on SignedFixed itself aren't recognized by
        // the kernel compiler (Digital-struct types don't get Add/Sub
        // lowered). Same pattern as the FIR and biquad cores.
        let xr = q.x.raw;
        let yr = q.y.raw;
        let zr = q.z.raw;
        let z_neg = zr < signed::<16>(0);
        let (nx_raw, ny_raw, nz_raw): (SignedBits<16>, SignedBits<16>, SignedBits<16>) =
            if q.iter == bits::<4>(0) {
                let ys: SignedBits<16> = yr.xshr::<0>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<0>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[0].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(1) {
                let ys: SignedBits<16> = yr.xshr::<1>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<1>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[1].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(2) {
                let ys: SignedBits<16> = yr.xshr::<2>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<2>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[2].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(3) {
                let ys: SignedBits<16> = yr.xshr::<3>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<3>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[3].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(4) {
                let ys: SignedBits<16> = yr.xshr::<4>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<4>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[4].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(5) {
                let ys: SignedBits<16> = yr.xshr::<5>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<5>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[5].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(6) {
                let ys: SignedBits<16> = yr.xshr::<6>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<6>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[6].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(7) {
                let ys: SignedBits<16> = yr.xshr::<7>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<7>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[7].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(8) {
                let ys: SignedBits<16> = yr.xshr::<8>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<8>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[8].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(9) {
                let ys: SignedBits<16> = yr.xshr::<9>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<9>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[9].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(10) {
                let ys: SignedBits<16> = yr.xshr::<10>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<10>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[10].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(11) {
                let ys: SignedBits<16> = yr.xshr::<11>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<11>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[11].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else if q.iter == bits::<4>(12) {
                let ys: SignedBits<16> = yr.xshr::<12>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<12>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[12].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            } else {
                let ys: SignedBits<16> = yr.xshr::<13>().resize::<16>().as_signed_bits();
                let xs: SignedBits<16> = xr.xshr::<13>().resize::<16>().as_signed_bits();
                let a = ATAN_TABLE[13].raw;
                if z_neg {
                    (xr + ys, yr - xs, zr + a)
                } else {
                    (xr - ys, yr + xs, zr - a)
                }
            };
        d.x = SignedFixed::<16, 14> { raw: nx_raw };
        d.y = SignedFixed::<16, 14> { raw: ny_raw };
        d.z = SignedFixed::<16, 14> { raw: nz_raw };
        let last = q.iter == bits::<4>(13);
        d.iter = if last { bits::<4>(0) } else { q.iter + 1 };
        d.busy = !last;
    }

    // Reset takes precedence.
    if cr.reset.any() {
        d.x = SignedFixed::<16, 14>::default();
        d.y = SignedFixed::<16, 14>::default();
        d.z = SignedFixed::<16, 14>::default();
        d.iter = bits::<4>(0);
        d.busy = false;
    }

    let o = CordicOut {
        cos: q.x,
        sin: q.y,
        valid: q.busy && q.iter == bits::<4>(13),
    };

    (o, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert raw Q1.14 to f64.
    fn q_to_f64(v: SignedFixed<16, 14>) -> f64 {
        v.raw.raw() as f64 / 16384.0
    }
    /// Convert f64 to Q1.14 with rounding.
    fn f64_to_q(v: f64) -> SignedFixed<16, 14> {
        SignedFixed::<16, 14>::from_f64(v)
    }

    /// Worst-case CORDIC error at Q1.14 with 14 iterations.
    /// Empirically ~2.4e-4 near the convergence boundary at ±π/2.
    const TOL: f64 = 5e-4;

    /// Drive a stream of inputs through the circuit and return all
    /// outputs with `valid` asserted, paired with the input that
    /// produced them.
    fn run_circuit(
        inputs: impl IntoIterator<Item = CordicIn>,
    ) -> Vec<(SignedFixed<16, 14>, CordicOut)> {
        let stream = inputs.into_iter().with_reset(1).clock_pos_edge(100);
        let uut = CordicCore::default();
        let mut last_phase = SignedFixed::<16, 14>::default();
        let mut results = vec![];
        for t in uut.run(stream).synchronous_sample() {
            // t.input is (ClockReset, CordicIn); t.output is CordicOut.
            if t.input.1.start {
                last_phase = t.input.1.phase;
            }
            if t.output.valid {
                results.push((last_phase, t.output));
            }
        }
        results
    }

    /// Build an input stream: for each phase, one cycle of `start` plus
    /// 15 cycles of idle (enough for the 14-iter computation to
    /// complete and produce a `valid` output).
    fn drive(phases: &[f64]) -> Vec<CordicIn> {
        let mut stream = vec![];
        for &p in phases {
            let phase = f64_to_q(p);
            stream.push(CordicIn { phase, start: true });
            for _ in 0..15 {
                stream.push(CordicIn {
                    phase,
                    start: false,
                });
            }
        }
        stream
    }

    #[test]
    fn cordic_known_phases() {
        let phases = [
            0.0,
            std::f64::consts::FRAC_PI_4,
            -std::f64::consts::FRAC_PI_4,
        ];
        let expected = [
            (1.0_f64, 0.0_f64),
            (
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ),
            (
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
            ),
        ];
        let results = run_circuit(drive(&phases));
        assert_eq!(results.len(), phases.len(), "one valid per phase");
        for (i, (phase, out)) in results.iter().enumerate() {
            let cos_err = (q_to_f64(out.cos) - expected[i].0).abs();
            let sin_err = (q_to_f64(out.sin) - expected[i].1).abs();
            assert!(
                cos_err < TOL,
                "phase={} cos err {cos_err}",
                q_to_f64(*phase)
            );
            assert!(
                sin_err < TOL,
                "phase={} sin err {sin_err}",
                q_to_f64(*phase)
            );
        }
    }

    #[test]
    fn cordic_sweep_matches_f64() {
        let mut phases = vec![];
        let mut p = -std::f64::consts::FRAC_PI_2;
        while p <= std::f64::consts::FRAC_PI_2 {
            phases.push(p);
            p += 0.05;
        }
        let results = run_circuit(drive(&phases));
        assert_eq!(results.len(), phases.len());
        for ((phase_q, out), &want_p) in results.iter().zip(phases.iter()) {
            let cos_err = (q_to_f64(out.cos) - want_p.cos()).abs();
            let sin_err = (q_to_f64(out.sin) - want_p.sin()).abs();
            assert!(
                cos_err < TOL,
                "phase={want_p}: cos {} err {cos_err}",
                q_to_f64(out.cos)
            );
            assert!(
                sin_err < TOL,
                "phase={want_p}: sin {} err {sin_err}",
                q_to_f64(out.sin)
            );
            assert!((q_to_f64(*phase_q) - want_p).abs() < TOL);
        }
    }

    #[test]
    fn cordic_kernel_pipeline() -> miette::Result<()> {
        use rhdl::core::sim::testbench::synchronous::SynchronousTestBench;
        let phases = [0.0_f64, 0.5, -0.5, 1.0];
        let stream = drive(&phases).into_iter().with_reset(1).clock_pos_edge(100);
        let uut = CordicCore::default();
        let tb = uut.run(stream).collect::<SynchronousTestBench<_, _>>();
        let tm = tb.rtl(&uut, &Default::default())?;
        tm.run_iverilog()?;
        let tm = tb.ntl(&uut, &Default::default())?;
        tm.run_iverilog()?;
        Ok(())
    }
}
