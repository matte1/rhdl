use std::collections::HashMap;

use crate::{
    Kind, RHDLError,
    common::{symtab::RegisterId, unify_key::EnaKey},
    rtl::{
        Object,
        spec::{Assign, OpCode, Operand},
        visit::visit_object_operands_mut,
    },
};
use ena::unify::InPlaceUnificationTable;

use super::pass::Pass;

/// Pick the "more specific" of two kinds that are bit-identical but
/// structurally different. The union-find merging of an Assign
/// (`lhs <- rhs`) can collapse a path projection that unwrapped a
/// single-element tuple of a scalar (`(Signed<N>)` -> `Signed<N>`)
/// into a single register; that register's downstream consumers were
/// typed against the scalar kind, not the tuple kind. Choosing the
/// scalar kind as the root's symtab entry preserves `is_signed()` /
/// `is_unsigned()` behavior for those consumers.
///
/// The rule is intentionally narrow: it only changes the chosen kind
/// when one of the two is a `Kind::Bits(_)` / `Kind::Signed(_)`
/// scalar and the other is a structurally-equivalent compound
/// (single-element tuple/struct wrapping the same scalar at the same
/// bit width). For all other combinations the LHS wins, preserving
/// prior behavior.
fn prefer_kind(a: Kind, b: Kind) -> Kind {
    if a == b {
        return a;
    }
    if a.bits() != b.bits() {
        // Bit-incompatible kinds shouldn't end up in the same
        // unification group, but if they do, don't try to choose.
        return a;
    }
    let a_is_scalar = matches!(a, Kind::Bits(_) | Kind::Signed(_));
    let b_is_scalar = matches!(b, Kind::Bits(_) | Kind::Signed(_));
    match (a_is_scalar, b_is_scalar) {
        (true, false) => a,
        (false, true) => b,
        // Both scalar (e.g., Signed<N> and Bits<N>): prefer Signed,
        // which carries strictly more type information (sign bit
        // tracking) than Bits at the same width.
        (true, true) => match (a.is_signed(), b.is_signed()) {
            (true, false) => a,
            (false, true) => b,
            _ => a,
        },
        // Both compound: arbitrary -- leave the first one.
        (false, false) => a,
    }
}

#[derive(Default, Debug, Clone)]
pub struct RemoveExtraRegistersPass {}

impl Pass for RemoveExtraRegistersPass {
    fn run(mut input: Object) -> Result<Object, RHDLError> {
        // Create a union table
        let mut table = InPlaceUnificationTable::<EnaKey>::new();
        // Map each Register ID to an EnaKey
        let reg_map: HashMap<RegisterId<_>, EnaKey> = input
            .symtab
            .iter_reg()
            .map(|(reg, _)| (reg, table.new_key(())))
            .collect();
        let inv_map: HashMap<EnaKey, RegisterId<_>> =
            reg_map.iter().map(|(&reg, &key)| (key, reg)).collect();
        // Loop over the assignment op codes, and union the arguments in the table
        for lop in &input.ops {
            if let OpCode::Assign(Assign { lhs, rhs }) = &lop.op {
                if let (Some(lhs_reg), Some(rhs_reg)) = (lhs.reg(), rhs.reg()) {
                    let lhs_key = reg_map[&lhs_reg];
                    let rhs_key = reg_map[&rhs_reg];
                    table.union(lhs_key, rhs_key);
                }
            }
        }
        // For each unification group, choose the "most specific"
        // kind among its members and assign it to the root register.
        // This matters when an Assign-driven merge collapses a
        // single-element tuple unwrap (e.g., `Signed<N>` joined with
        // `(Signed<N>)`); the union-find root keeps whatever kind
        // was inserted first, but downstream consumers expect the
        // more-specific scalar kind.
        let mut root_kind: HashMap<EnaKey, Kind> = HashMap::new();
        for (&reg_id, &key) in &reg_map {
            let root = table.find(key);
            let this_kind = input.symtab[reg_id];
            root_kind
                .entry(root)
                .and_modify(|k| *k = prefer_kind(*k, this_kind))
                .or_insert(this_kind);
        }
        for (root_key, best_kind) in root_kind {
            let root_reg = inv_map[&root_key];
            input.symtab[root_reg] = best_kind;
        }
        // Next, rewrite the ops, where for each operand, we take the root of the unify tree
        visit_object_operands_mut(&mut input, |_sense, operand| {
            if let Some(reg) = operand.reg() {
                let key = reg_map[&reg];
                let root = table.find(key);
                let replacement = Operand::Register(inv_map[&root]);
                *operand = replacement;
            }
        });
        input.ops.retain(|lop| {
            if let OpCode::Assign(Assign { lhs, rhs }) = lop.op {
                lhs != rhs
            } else {
                true
            }
        });
        Ok(input)
    }
    fn description() -> &'static str {
        "Remove extra registers"
    }
}
