//! Generative testing for RHDL.
//!
//! This crate defines a small typed mini-DSL ([`dsl`]), a reference
//! interpreter ([`interpret`]), and a renderer ([`render`]) that converts
//! DSL programs into runtime-built RHDL kernels. Property tests in
//! `tests/` use these to generate random programs and verify that RHDL's
//! compiled output matches the reference interpreter.

pub mod coverage;
pub mod dsl;
pub mod generator;
pub mod interpret;
pub mod multi;
pub mod render;
