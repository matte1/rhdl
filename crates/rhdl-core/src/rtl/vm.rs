use crate::{
    Kind, RHDLError, TypedBits,
    ast::ast_impl::SourceLocation,
    bitx::BitX,
    common::slot_vec::SlotKey,
    compiler::mir::error::{ICE, RHDLCompileError},
    error::rhdl_error,
    rhif::object::SourceDetails,
    rtl::spec::{
        AluBinary, AluUnary, Case, CaseArgument, Cast, CastKind, Concat, Index, Select, Splice,
        Unary,
    },
    types::bit_string::BitString,
};

use super::{
    Object,
    object::LocatedOpCode,
    runtime_ops::{binary, unary},
    spec::{Assign, Binary, OpCode, Operand},
};

type Result<T> = core::result::Result<T, RHDLError>;

struct VMState<'a> {
    reg_stack: &'a mut [Option<BitString>],
    literals: &'a [(TypedBits, SourceDetails)],
    obj: &'a Object,
}

impl VMState<'_> {
    fn raise_ice(&self, cause: ICE, loc: SourceLocation) -> RHDLError {
        let symbols = &self.obj.symbols;
        RHDLError::RHDLInternalCompilerError(Box::new(RHDLCompileError {
            cause,
            src: symbols.source(),
            err_span: symbols.span(loc).into(),
        }))
    }
    fn binary(
        &self,
        op: AluBinary,
        arg1: BitString,
        arg2: BitString,
        loc: SourceLocation,
    ) -> Result<BitString> {
        let arg1: TypedBits = arg1.into();
        let arg2: TypedBits = arg2.into();
        match binary(op, arg1, arg2) {
            Ok(result) => Ok(result.into()),
            Err(e) => Err(self.raise_ice(ICE::BinaryOperatorError(Box::new(e)), loc)),
        }
    }

    fn unary(&self, op: AluUnary, arg1: BitString, loc: SourceLocation) -> Result<BitString> {
        let arg1: TypedBits = arg1.into();
        match unary(op, arg1) {
            Ok(result) => Ok(result.into()),
            Err(e) => Err(self.raise_ice(ICE::UnaryOperatorError(Box::new(e)), loc)),
        }
    }
    fn read(&self, operand: Operand, loc: SourceLocation) -> Result<BitString> {
        match operand {
            Operand::Literal(l) => Ok((&self.literals[l.index()].0).into()),
            Operand::Register(r) => self.reg_stack[r.index()]
                .clone()
                .ok_or(self.raise_ice(ICE::UninitializedRTLRegister { r }, loc)),
        }
    }
    fn write(&mut self, operand: Operand, value: BitString, loc: SourceLocation) -> Result<()> {
        match operand {
            Operand::Literal(ndx) => Err(self.raise_ice(ICE::CannotWriteToRTLLiteral { ndx }, loc)),
            Operand::Register(r) => {
                // Coerce the BitString tag (signed/unsigned) to match
                // the register's declared kind. Some opcodes (Unary,
                // Binary) write the natural BitString tag from the
                // computed TypedBits, which can drift from the static
                // kind when the op's argument was extracted via a
                // bit-slice that flattened a compound kind (e.g., an
                // enum variant's `(Signed<N>)` payload yields a
                // BitString::Unsigned slice even though downstream
                // ops expect Signed<N>). Without this coercion, a
                // signed-typed register can hold an unsigned BitString
                // and trigger spurious runtime type errors.
                let coerced = match self.obj.kind(Operand::Register(r)).is_signed() {
                    true => BitString::Signed(value.bits().to_vec()),
                    false => BitString::Unsigned(value.bits().to_vec()),
                };
                self.reg_stack[r.index()] = Some(coerced);
                Ok(())
            }
        }
    }
}

fn execute_block(ops: &[LocatedOpCode], state: &mut VMState) -> Result<()> {
    for lop in ops {
        let loc = lop.loc;
        let op = &lop.op;
        match op {
            OpCode::Noop => {}
            OpCode::Assign(Assign { lhs, rhs }) => {
                let rhs = state.read(*rhs, loc)?;
                state.write(*lhs, rhs, loc)?;
            }
            OpCode::Binary(Binary {
                op,
                lhs,
                arg1,
                arg2,
            }) => {
                let arg1 = state.read(*arg1, loc)?;
                let arg2 = state.read(*arg2, loc)?;
                let result = state.binary(*op, arg1, arg2, loc)?;
                state.write(*lhs, result, loc)?;
            }
            OpCode::Case(Case {
                lhs,
                discriminant,
                table,
            }) => {
                let lhs_kind = state.obj.kind(*lhs);
                let lhs_dont_care = BitString::dont_care_from_kind(lhs_kind);
                let discriminant = state.read(*discriminant, loc)?;
                let arm = table
                    .iter()
                    .find(|(disc, _)| match disc {
                        CaseArgument::Literal(l) => {
                            discriminant == (&state.literals[l.index()].0).into()
                        }
                        CaseArgument::Wild => true,
                    })
                    .map(|x| x.1);
                let arm = if let Some(arm) = arm {
                    state.read(arm, loc)?
                } else {
                    lhs_dont_care
                };
                state.write(*lhs, arm, loc)?;
            }
            OpCode::Cast(Cast {
                lhs,
                arg,
                len,
                kind,
            }) => {
                let arg_value = state.read(*arg, loc)?;
                let result = match kind {
                    CastKind::Signed => arg_value.signed_cast(*len),
                    CastKind::Unsigned => arg_value.unsigned_cast(*len),
                    // Resize is the truncating/extending width change.
                    // Whether to sign-extend or zero-pad must be driven
                    // by the source register's *declared* Kind, not by
                    // the BitString variant. The BitString variant can
                    // be Unsigned even when the source is semantically
                    // signed -- this happens when the source register's
                    // Kind is a single-field struct/tuple wrapping a
                    // signed scalar, because `Kind::is_signed()` returns
                    // false for those wrapper kinds at the harness
                    // TypedBits -> BitString boundary. Mirror what
                    // `translate_resize` in the Verilog backend does.
                    CastKind::Resize => {
                        let source_kind_signed = state.obj.kind(*arg).is_signed();
                        let bits_vec = arg_value.bits().to_vec();
                        let tb_kind = if source_kind_signed {
                            Kind::make_signed(bits_vec.len())
                        } else {
                            Kind::make_bits(bits_vec.len())
                        };
                        TypedBits::new(bits_vec, tb_kind)
                            .resize(*len)
                            .map(BitString::from)
                    }
                }?;
                state.write(*lhs, result, loc)?;
            }
            OpCode::Concat(Concat { lhs, args }) => {
                let result = args
                    .iter()
                    .map(|a| state.read(*a, loc))
                    .collect::<Result<Vec<_>>>()?;
                let combined = result
                    .iter()
                    .flat_map(|x| x.bits())
                    .copied()
                    .collect::<Vec<BitX>>();
                if state.obj.kind(*lhs).is_signed() {
                    state.write(*lhs, BitString::Signed(combined), loc)?;
                } else {
                    state.write(*lhs, BitString::Unsigned(combined), loc)?;
                }
            }
            OpCode::Index(Index {
                lhs,
                arg,
                bit_range,
                path: _,
            }) => {
                let arg = state.read(*arg, loc)?;
                let slice = arg.bits()[bit_range.clone()].to_vec();
                if state.obj.kind(*lhs).is_signed() {
                    state.write(*lhs, BitString::Signed(slice), loc)?;
                } else {
                    state.write(*lhs, BitString::Unsigned(slice), loc)?;
                }
            }
            OpCode::Select(Select {
                lhs,
                cond,
                true_value,
                false_value,
            }) => {
                let cond = state.read(*cond, loc)?;
                let true_value = state.read(*true_value, loc)?;
                let false_value = state.read(*false_value, loc)?;
                match cond.bits()[0] {
                    BitX::Zero => state.write(*lhs, false_value, loc)?,
                    BitX::One => state.write(*lhs, true_value, loc)?,
                    BitX::X => state.write(*lhs, true_value.dont_care(), loc)?,
                }
            }
            OpCode::Splice(Splice {
                lhs,
                orig,
                value,
                bit_range,
                path: _,
            }) => {
                let orig = state.read(*orig, loc)?;
                let value = state.read(*value, loc)?;
                let mut orig = orig.bits().to_vec();
                let value = value.bits();
                orig.splice(bit_range.clone(), value.iter().copied());
                if state.obj.kind(*lhs).is_signed() {
                    state.write(*lhs, BitString::Signed(orig), loc)?;
                } else {
                    state.write(*lhs, BitString::Unsigned(orig), loc)?;
                }
            }
            OpCode::Unary(Unary { op, lhs, arg1 }) => {
                let arg1 = state.read(*arg1, loc)?;
                let result = state.unary(*op, arg1, loc)?;
                state.write(*lhs, result, loc)?;
            }
        }
    }
    Ok(())
}

pub fn execute(obj: &Object, arguments: Vec<BitString>) -> Result<BitString> {
    let symbols = &obj.symbols;
    let loc = symbols.fallback(obj.fn_id);
    // Load the object for this function
    if obj.arguments.len() != arguments.len() {
        return Err(rhdl_error(RHDLCompileError {
            cause: ICE::ArgumentCountMismatchOnCall,
            src: symbols.source(),
            err_span: symbols.span(loc).into(),
        }));
    }
    for (ndx, arg) in arguments.iter().enumerate() {
        if obj.arguments[ndx].is_none() ^ arg.is_empty() {
            return Err(rhdl_error(RHDLCompileError {
                cause: ICE::NonemptyToEmptyArgumentMismatch,
                src: symbols.source(),
                err_span: symbols.span(loc).into(),
            }));
        }
    }
    // Allocate registers for the function call.
    let max_reg = obj.symtab.reg_vec().len();
    let mut reg_stack = vec![None; max_reg];
    // Copy the arguments into the appropriate registers, coercing the
    // BitString tag (signed/unsigned) to match the register's declared
    // kind. The caller's TypedBits -> BitString conversion picks the
    // variant from `Kind::is_signed()`, which returns false for
    // wrapper kinds (single-field struct/tuple over a signed scalar).
    // Without this coercion, the argument register holds an Unsigned
    // BitString even though its declared kind is Signed; downstream
    // ops (Cast/Resize, Binary) then misinterpret it. This mirrors
    // the coercion `state.write` does on every internally-computed
    // value.
    for (ndx, arg) in arguments.into_iter().enumerate() {
        if let Some(r) = obj.arguments[ndx] {
            let coerced = if obj.symtab[r].is_signed() {
                BitString::Signed(arg.bits().to_vec())
            } else {
                BitString::Unsigned(arg.bits().to_vec())
            };
            reg_stack[r.index()] = Some(coerced);
        }
    }
    let mut state = VMState {
        reg_stack: &mut reg_stack,
        literals: obj.symtab.lit_vec(),
        obj,
    };
    execute_block(&obj.ops, &mut state)?;
    match obj.return_register {
        Operand::Register(r) => reg_stack
            .get(r.index())
            .cloned()
            .ok_or(RHDLError::RHDLInternalCompilerError(Box::new(
                RHDLCompileError {
                    cause: ICE::ReturnSlotNotFound {
                        name: format!("{r:?}"),
                    },
                    src: symbols.source(),
                    err_span: symbols.span(loc).into(),
                },
            )))?
            .ok_or(RHDLError::RHDLInternalCompilerError(Box::new(
                RHDLCompileError {
                    cause: ICE::ReturnSlotNotInitialized,
                    src: symbols.source(),
                    err_span: symbols.span(loc).into(),
                },
            ))),
        Operand::Literal(ndx) => Ok((&obj.symtab[ndx]).into()),
    }
}
