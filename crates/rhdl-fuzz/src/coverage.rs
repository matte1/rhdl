//! AST-node coverage tracking for the generator.
//!
//! Each strategy arm in `generator.rs` increments one of these counters
//! when it produces a value, so we can dump a histogram of "what kinds
//! of nodes did this run actually generate?" The answer guides the
//! `prop_oneof!` weights -- if `Cast` is consistently <1% of generated
//! nodes, that's a hint to weight it higher (or accept that it's
//! intrinsically rare because of where in the tree it can appear).
//!
//! Counters are global atomics (proptest runs sequentially per test).
//! Shrinking re-runs `prop_map` closures, so shrunk programs inflate
//! counts; that's accepted noise for v1.

use std::sync::atomic::{AtomicUsize, Ordering};

macro_rules! counters {
    ($($vis:vis $name:ident),* $(,)?) => {
        $(
            $vis static $name: AtomicUsize = AtomicUsize::new(0);
        )*

        /// Snapshot of all counter values, in the order they were declared.
        pub fn snapshot() -> Vec<(&'static str, usize)> {
            vec![$(
                (stringify!($name), $name.load(Ordering::Relaxed)),
            )*]
        }

        /// Reset all counters to zero.
        pub fn reset() {
            $(
                $name.store(0, Ordering::Relaxed);
            )*
        }
    };
}

counters!(
    // Scalar Expr
    pub LIT_BITS,
    pub LIT_SIGNED,
    pub VAR,
    pub BINOP,
    pub UNOP_NOT,
    pub UNOP_NEG,
    pub LET,
    pub IF,
    pub RESIZE,
    pub CAST,
    pub SHIFT,
    // Compound Expr (M3)
    pub TUPLE_CTOR,
    pub TUPLE_PROJ,
    pub ARRAY_CTOR,
    pub ARRAY_INDEX,
    pub ARRAY_REPEAT,
    // Nominal-type Expr (M5a)
    pub STRUCT_CTOR,
    pub FIELD_ACCESS,
    pub STRUCT_UPDATE,
    // Nominal-type Expr (M5b)
    pub ENUM_CTOR,
    pub MATCH,
    // Statement-level Expr (M6)
    pub BLOCK,
    pub FOR,
    pub STMT_LET,
    pub STMT_LET_MUT,
    pub STMT_ASSIGN,
    // BoolExpr
    pub BOOL_LIT,
    pub BOOL_NOT,
    pub BOOL_AND,
    pub BOOL_OR,
    pub BOOL_CMP,
    pub BOOL_REDUCTION,
);

/// Print a histogram of generator coverage to stderr. Counts are
/// cumulative since the last `reset()`.
pub fn print_summary() {
    let snap = snapshot();
    let total: usize = snap.iter().map(|(_, n)| n).sum();
    if total == 0 {
        eprintln!("(no generator coverage data)");
        return;
    }
    eprintln!("--- Generator coverage (total {} nodes) ---", total);
    for (name, count) in snap {
        let pct = 100.0 * count as f64 / total as f64;
        eprintln!("  {name:20} {count:8}  {pct:5.1}%");
    }
}
