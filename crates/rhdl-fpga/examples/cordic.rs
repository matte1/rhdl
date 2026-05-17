//! CORDIC sine/cosine example.
//!
//! Sweeps the input phase from -π/2 to π/2 in 16 steps, pulsing
//! `start` for each phase and giving the iterative core 15 cycles to
//! complete each computation. Output trace shows `phase` stepping
//! through values, `busy` rising during each 14-cycle computation,
//! `valid` pulsing when the result is ready, and the `(cos, sin)`
//! outputs settling into their final waveforms.

use rhdl::prelude::*;
use rhdl_fixed::SignedFixed;
use rhdl_fpga::doc::write_svg_as_markdown;
use rhdl_fpga::dsp::cordic::{CordicCore, CordicIn};

fn main() -> Result<(), RHDLError> {
    let uut = CordicCore::default();

    // 6 phases across [-π/2, π/2]. Each phase takes ~16 cycles in the
    // trace; keeping the count small keeps the SVG readable.
    let n_phases = 6;
    let step = std::f64::consts::PI / (n_phases - 1) as f64;
    let phases: Vec<f64> = (0..n_phases)
        .map(|i| -std::f64::consts::FRAC_PI_2 + (i as f64) * step)
        .collect();

    // Drive: for each phase, one cycle of start, then 15 cycles of idle.
    let mut stream = vec![];
    for p in phases {
        let phase = SignedFixed::<16, 14>::from_f64(p);
        stream.push(CordicIn { phase, start: true });
        for _ in 0..15 {
            stream.push(CordicIn {
                phase,
                start: false,
            });
        }
    }

    let input = stream.into_iter().with_reset(1).clock_pos_edge(100);
    let trace = uut.run(input).collect::<SvgFile>();
    // Show only top-level inputs/outputs (hide the (x, y, z, iter, busy)
    // internal state) — keeps the SVG compact and focused on the
    // observable behavior. Drop `.with_io_filter()` to see the full
    // iteration.
    let options = SvgOptions::default().with_io_filter();
    write_svg_as_markdown(trace, "cordic.md", options)?;
    Ok(())
}
