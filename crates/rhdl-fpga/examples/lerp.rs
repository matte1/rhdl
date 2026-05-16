use rhdl::prelude::*;
use rhdl_fixed::Fixed;
use rhdl_fpga::doc::write_svg_as_markdown;
use rhdl_fpga::dsp::lerp::fixed::lerp_unsigned;

// Because the `lerp` function is just a function, if we want to make a
// core out of it, we need a wrapper. In this example, we will use the
// [Func] wrapper. We need to fit the type signature of that function,
// so the inputs of the `lerp` function need to be put into a single
// struct.
#[derive(PartialEq, Clone, Copy, Digital)]
pub struct LerpIn<const N: usize, const M: usize>
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<M>: BitWidth,
{
    pub lower_value: Fixed<N, 0>,
    pub upper_value: Fixed<N, 0>,
    pub factor: Fixed<M, M>,
}

// A wrapper function to call the `lerp_unsigned`.
#[kernel]
pub fn wrap_lerp<const N: usize, const M: usize>(_cr: ClockReset, i: LerpIn<N, M>) -> Fixed<N, 0>
where
    rhdl::bits::W<N>: BitWidth,
    rhdl::bits::W<M>: BitWidth,
{
    lerp_unsigned::<N, M>(i.lower_value, i.upper_value, i.factor)
}

fn main() -> Result<(), RHDLError> {
    // The [Func] wrapper gives us a core we can simulate.
    let uut: Func<LerpIn<8, 4>, Fixed<8, 0>> = Func::try_new::<wrap_lerp<8, 4>>()?;
    // Simulate a ramp through the interpolation factor.
    let ramp = (0..15)
        .map(|x| LerpIn {
            upper_value: Fixed::<8, 0> { raw: bits(255) },
            lower_value: Fixed::<8, 0> { raw: bits(0) },
            factor: Fixed::<4, 4> { raw: bits(x) },
        })
        .without_reset()
        .clock_pos_edge(100);
    let vcd = uut.run(ramp).collect::<SvgFile>();
    write_svg_as_markdown(vcd, "lerp.md", SvgOptions::default())?;
    Ok(())
}
