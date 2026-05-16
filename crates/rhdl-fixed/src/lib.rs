#![deny(missing_docs)]
//! # rhdl-fixed
//!
//! Fixed-point numeric types for RHDL hardware design, layered on top of
//! the integer primitives in `rhdl-bits`. Values are stored as `Bits<W>`
//! (or `SignedBits<W>`) and the position of the binary point is tracked
//! at the type level via a fractional-bits parameter `F`.
//!
//! ## Why a separate type
//!
//! `rhdl-bits` gives you integer arithmetic with defined wrap and
//! width-extension semantics. Fixed-point arithmetic layers a scaling
//! convention on top: a value with `W` total bits and `F` fractional bits
//! represents the rational number `raw / 2^F`. Without that layer, users
//! must hand-track the binary point in comments (see
//! `rhdl-fpga::dsp::lerp::fixed` for the canonical pre-`rhdl-fixed`
//! pattern).
//!
//! ## Type parameterization
//!
//! Stable Rust cannot evaluate `Bits<{I + F}>` in a type position, so the
//! types in this crate are parameterized by **total width** and
//! **fractional bits** rather than by integer and fractional bits:
//!
//! ```text
//! Fixed<W, F>          ->  unsigned, W total bits, F fractional bits
//! SignedFixed<W, F>    ->  signed,   W total bits, F fractional bits
//! ```
//!
//! For [`SignedFixed`], one of the `W` bits is the sign bit (the same
//! convention as [`SignedBits`](rhdl_bits::SignedBits)).
//!
//! ## Bit-exact host and hardware semantics
//!
//! Every operation in this crate is implemented using only the
//! deterministic integer primitives from `rhdl-bits`. No floating-point
//! values appear on the kernel path. This guarantees that host execution
//! of a `#[kernel]` function and the synthesized hardware produce
//! bit-identical results — the same property the underlying integer ops
//! already provide.
//!
//! ## What lives where
//!
//! - **Type definitions, same-format ops, width-extending ops**:
//!   [`crate::fixed`] (typed) and [`crate::dyn_fixed`] (runtime-tracked).
//!   `+`, `-`, unary `-` delegate to the wrapping `rhdl-bits` ops.
//!   `xadd` / `xsub` / `xmul` / `xshl` / `xshr` / `xsgn` / `xneg` return
//!   `DynFixed` / `SignedDynFixed`. All work inside `#[kernel]` bodies.
//! - **Saturating ops**: [`crate::saturating`] —
//!   `saturating_add` / `saturating_sub` and signed variants, as
//!   kernel-callable `#[kernel]` free functions.
//! - **Conversions back to typed Fixed**: [`crate::convert`] —
//!   `truncate`, `round_nearest`, `saturate`, `round_saturate`,
//!   `from_f64`. These are host-only; inside a kernel, compose
//!   `xshr` + `resize` + `as_bits` / `as_signed_bits` directly.
//! - **Host-only test helpers**: [`crate::mac::Mac`] for computing
//!   reference sum-of-products in tests.
//!
//! ## Typical flow
//!
//! A representative fixed-point computation — compute `(a + b) * c` with
//! `Q4.4` inputs, rounding the result back to `Q8.4`:
//!
//! ```
//! use rhdl_fixed::{Fixed, SignedFixed};
//!
//! // Construct via the f64 helper (host-side only; the f64 is
//! // resolved before the bits flow into a kernel).
//! let a = SignedFixed::<8, 4>::from_f64(1.5);   // raw = 0x18
//! let b = SignedFixed::<8, 4>::from_f64(-0.25); // raw = -4
//! let c = SignedFixed::<8, 4>::from_f64(2.0);   // raw = 0x20
//!
//! // Same-format wrapping add via `+`.
//! let sum = a + b;
//! assert_eq!(sum.raw.raw(), 0x14); // 1.25 * 16 = 20
//!
//! // Width-extending multiply: SignedDynFixed with width 16, frac 8.
//! let product = sum.xmul(c);
//! assert_eq!(product.width(), 16);
//! assert_eq!(product.frac, 8);
//! assert_eq!(product.raw.raw(), 0x14 * 0x20); // 0x280 -> 2.5 at Q8.8
//!
//! // Round back to SignedFixed<8, 4> with RTNE + saturation (host).
//! let result: SignedFixed<8, 4> = product.round_saturate();
//! assert_eq!(result.raw.raw(), 0x28); // 2.5 * 16
//! ```
//!
//! ## Using `Fixed` inside a `#[kernel]`
//!
//! The typed `Fixed<N, F>` types are [`Digital`](rhdl_core::Digital) and
//! flow through kernels as inputs/outputs. Inside a kernel body, access
//! the underlying [`Bits`](rhdl_bits::Bits) /
//! [`SignedBits`](rhdl_bits::SignedBits) via the `raw` field, do the
//! math with the recognized `rhdl-bits` operations, and reconstruct
//! with struct syntax:
//!
//! ```ignore
//! use rhdl::prelude::*;
//! use rhdl_fixed::Fixed;
//!
//! #[kernel]
//! fn add_q8_4(a: Fixed<8, 4>, b: Fixed<8, 4>) -> Fixed<8, 4> {
//!     Fixed::<8, 4> { raw: a.raw + b.raw }
//! }
//! ```
//!
//! See [`crate::fixed`] for details on the wrapper-construction
//! convention.

pub mod convert;
pub mod dyn_fixed;
pub mod fixed;
pub mod mac;
pub mod saturating;

pub use dyn_fixed::{DynFixed, SignedDynFixed};
pub use fixed::{Fixed, SignedFixed};
