//! Typed mini-DSL representing the subset of Rust that the generator emits.
//!
//! After M0 the DSL is *type-aware*: every program declares the types of
//! its arguments and return value, and `Lit` carries its type alongside
//! its value. Today the only `Type` variant is `Bits(8)`, so the new
//! infrastructure is dormant -- it exists so M1+ can add variants
//! (multiple widths, signed, tuples, structs, enums) as pure additions.

/// Maximum number of input bindings a generated program may take.
/// Keeping this small bounds the variable-pool size and lets us use a
/// fixed `&'static str` table for binding names.
pub const MAX_ARITY: u8 = 4;

/// Static binding names for input arguments. Indexed by `Var(i)` where
/// `i` is in `0..MAX_ARITY`.
pub const VAR_NAMES: [&str; MAX_ARITY as usize] = ["a", "b", "c", "d"];

/// Static binding names for `let` temporaries. `Var(i)` where
/// `i >= arity` resolves to `LET_NAMES[i - arity]`. Sized to comfortably
/// exceed the max nesting depth proptest will produce at the configured
/// recursion limit.
pub const LET_NAMES: [&str; 16] = [
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14",
    "t15",
];

/// Type of a value flowing through a DSL expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Unsigned N-bit integer (`Bits<N>`). N is in 1..=128.
    Bits(u8),
    /// Signed N-bit integer (`SignedBits<N>`). N is in 1..=128.
    Signed(u8),
    /// Tuple of heterogeneous element types.
    Tuple(Vec<Type>),
    /// Fixed-length array of homogeneous element type.
    Array(Box<Type>, usize),
    /// Nominal struct type. The `usize` indexes into the surrounding
    /// program's `NominalRegistry::structs`.
    Struct(usize),
    /// Nominal enum type. The `usize` indexes into the surrounding
    /// program's `NominalRegistry::enums`.
    Enum(usize),
}

impl Type {
    /// Width in bits of a *scalar* type. Panics on compound types
    /// (caller must check with `is_scalar` first).
    pub fn width(&self) -> u8 {
        match self {
            Type::Bits(n) | Type::Signed(n) => *n,
            Type::Tuple(_) | Type::Array(_, _) | Type::Struct(_) | Type::Enum(_) => {
                panic!("width() called on compound Type {:?}", self)
            }
        }
    }

    /// True if this type is a signed scalar.
    pub fn is_signed(&self) -> bool {
        matches!(self, Type::Signed(_))
    }

    /// True if this is a scalar (`Bits` or `Signed`).
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::Bits(_) | Type::Signed(_))
    }
}

/// Nominal struct definition. Lives in `NominalRegistry::structs` and
/// is referenced by `Type::Struct(i)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructDef {
    /// Display name (must come from `STRUCT_NAMES`).
    pub name: &'static str,
    /// Ordered list of (field_name, field_type) pairs. Field name must
    /// come from `FIELD_NAMES`.
    pub fields: Vec<(&'static str, Type)>,
}

/// Nominal enum definition. Lives in `NominalRegistry::enums` and is
/// referenced by `Type::Enum(i)`. Variants are positional; each variant
/// may carry zero or more payload fields, each of any `Type`. The
/// discriminant for variant `idx` is just `idx` (kept small so the
/// computed width fits in `discriminant_width`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumDef {
    /// Display name (must come from `ENUM_NAMES`).
    pub name: &'static str,
    /// Ordered variants. Variant index doubles as discriminant value.
    pub variants: Vec<VariantDef>,
    /// Width in bits of the discriminant field in the rendered Verilog.
    /// Must satisfy `2^width >= variants.len()`.
    pub discriminant_width: u8,
}

/// One variant of an `EnumDef`. Payload is a positional list of types
/// (empty for unit variants).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantDef {
    /// Display name (must come from `VARIANT_NAMES`).
    pub name: &'static str,
    /// Positional payload types. Empty list means a unit variant.
    pub payload: Vec<Type>,
}

/// Per-program registry of nominal types. `Type::Struct(i)` indexes
/// into `structs`; `Type::Enum(i)` indexes into `enums`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct NominalRegistry {
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
}

impl NominalRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Static struct names for the registry pool. Indexable up to
/// `STRUCT_NAMES.len()` distinct structs per program.
pub const STRUCT_NAMES: [&str; 4] = ["S0", "S1", "S2", "S3"];

/// Static field names. A struct may have up to `FIELD_NAMES.len()`
/// fields, indexed positionally.
pub const FIELD_NAMES: [&str; 4] = ["f0", "f1", "f2", "f3"];

/// Static enum names for the registry pool.
pub const ENUM_NAMES: [&str; 4] = ["E0", "E1", "E2", "E3"];

/// Static variant names. An enum may have up to `VARIANT_NAMES.len()`
/// variants, indexed positionally.
pub const VARIANT_NAMES: [&str; 4] = ["V0", "V1", "V2", "V3"];

/// A complete generated kernel: typed argument list, return type, and a
/// body expression that evaluates to `ret_type`.
#[derive(Debug, Clone)]
pub struct Program {
    /// Types of the input bindings, in order. `Var(i)` for `i < arg_types.len()`
    /// references the `i`-th argument.
    pub arg_types: Vec<Type>,
    /// Type the body is expected to evaluate to.
    pub ret_type: Type,
    /// Body expression. Must produce a value of `ret_type`.
    pub body: Expr,
    /// Nominal types referenced by `Type::Struct` (and, after M5b,
    /// `Type::Enum`) anywhere in the program. Empty for programs that
    /// use only built-in types.
    pub nominal: NominalRegistry,
}

impl Program {
    /// Number of input arguments (a.k.a. arity).
    pub fn arity(&self) -> u8 {
        self.arg_types.len() as u8
    }
}

/// An expression in the DSL.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A typed literal value. The value is interpreted modulo the type's
    /// width when cast into the target representation.
    Lit { value: u128, ty: Type },
    /// A reference to a binding in scope. `Var(i)` for `i < arity` is an
    /// input argument; `Var(i)` for `i >= arity` is the `(i - arity)`-th
    /// `let`-bound temporary in the current scope.
    Var(u8),
    /// A binary operation between two sub-expressions. Both operands and
    /// the result must have the same scalar type.
    BinOp(BinOpKind, Box<Expr>, Box<Expr>),
    /// A unary operation on a sub-expression.
    UnOp(UnOpKind, Box<Expr>),
    /// A let binding: `{ let tN = init; body }`.
    Let {
        /// Expression whose value is bound to the new temporary.
        init: Box<Expr>,
        /// Expression evaluated with the new binding in scope.
        body: Box<Expr>,
    },
    /// `if cond { then_branch } else { else_branch }`.
    If {
        /// Boolean-typed condition expression.
        cond: Box<BoolExpr>,
        /// Branch evaluated when `cond` is true.
        then_branch: Box<Expr>,
        /// Branch evaluated when `cond` is false.
        else_branch: Box<Expr>,
    },
    /// `<inner>.resize::<W>()` -- zero/sign-extends or truncates to
    /// `to_width`. Preserves the kind (Bits stays Bits, Signed stays
    /// Signed). RHDL requires explicit `.resize()` when bridging widths.
    Resize {
        /// Inner expression to resize. May have any scalar width.
        inner: Box<Expr>,
        /// Target width. The resulting expression has the same kind as
        /// `inner` but with this width.
        to_width: u8,
    },
    /// Signedness change: `<inner>.as_signed()` or `<inner>.as_unsigned()`.
    /// `inner` and `to` must have the same width.
    Cast {
        /// Inner expression. Must be a scalar of width `to.width()`.
        inner: Box<Expr>,
        /// Target type. Differs from `inner`'s type only in signedness.
        to: Type,
    },
    /// Shift operation. The shift amount is unsigned `Bits<W>` where W
    /// matches `value`'s width; the result has the same type as `value`.
    /// `Shr` on a `Signed` operand performs an arithmetic (sign-extending)
    /// shift; on a `Bits` operand it performs a logical shift.
    Shift {
        /// Direction of the shift.
        kind: ShiftKind,
        /// Value being shifted. Result type matches this.
        value: Box<Expr>,
        /// Unsigned shift count.
        amount: Box<Expr>,
    },
    /// Tuple constructor: `(e0, e1, ...)`. The element expressions'
    /// types must match the surrounding context's `Type::Tuple(...)`.
    TupleCtor(Vec<Expr>),
    /// Tuple field projection: `inner.<index>`. The inner expression
    /// must have a `Type::Tuple` whose element at `index` is the
    /// expression's result type.
    TupleProj {
        /// Tuple-typed inner expression.
        inner: Box<Expr>,
        /// 0-based field index.
        index: usize,
    },
    /// Array constructor: `[e0, e1, ...]`. All element expressions must
    /// have the same type, matching the surrounding context's
    /// `Type::Array(elem, len)`.
    ArrayCtor(Vec<Expr>),
    /// Array-repeat constructor: `[value; len]`. Equivalent to
    /// `ArrayCtor` with `len` copies of `value`, but lowers to a
    /// distinct RHIF opcode (`OpCode::Repeat`).
    ArrayRepeat {
        /// Element expression, evaluated once at construction time;
        /// the result is replicated `len` times into the array.
        value: Box<Expr>,
        /// Number of repetitions. Must be positive.
        len: usize,
    },
    /// Array index: `inner[index]`. The inner expression has type
    /// `Type::Array(elem, _)`; the index has an unsigned `Bits<W>`
    /// type and must evaluate to a value < array length (the
    /// generator emits a literal in range).
    ArrayIndex {
        /// Array-typed inner expression.
        inner: Box<Expr>,
        /// Unsigned `Bits<W>` index expression.
        index: Box<Expr>,
    },
    /// Call another kernel by index in the surrounding
    /// `MultiKernelProgram::kernels` table. The argument types must
    /// match the callee's `arg_types`; the result has the callee's
    /// `ret_type`. Only meaningful in multi-kernel programs --
    /// `interpret` (single-kernel) panics if an `Expr::Call` is
    /// reached.
    Call {
        /// Index into the parent `MultiKernelProgram::kernels`.
        callee: usize,
        /// Argument expressions, one per callee `arg_types` entry.
        args: Vec<Expr>,
    },
    /// Construct a value of `Type::Struct(struct_id)`. The `fields`
    /// vector has one expression per declared field, in declaration
    /// order; each expression must produce a value of the matching
    /// field type.
    StructCtor { struct_id: usize, fields: Vec<Expr> },
    /// Access a field of a struct by positional index.
    FieldAccess {
        /// Struct-typed inner expression.
        inner: Box<Expr>,
        /// Index into the struct's `fields` array.
        field_idx: usize,
    },
    /// Struct update expression `S { f0: v0, ..base }`. Equivalent in
    /// observed value to a `StructCtor` whose unmentioned fields come
    /// from `base`, but lowers to `op_struct` with a non-None `rest`
    /// argument, which is a distinct compiler path from the no-rest
    /// form. `overrides` is a sparse list of `(field_idx, expr)`.
    StructUpdate {
        struct_id: usize,
        base: Box<Expr>,
        overrides: Vec<(usize, Expr)>,
    },
    /// Construct a value of `Type::Enum(enum_id)` by selecting variant
    /// `variant_idx` and supplying one expression per payload field
    /// declared by that variant (empty list for unit variants).
    EnumCtor {
        enum_id: usize,
        variant_idx: usize,
        payload: Vec<Expr>,
    },
    /// `match scrutinee { ... }` over an enum-typed value. Exhaustiveness
    /// is the caller's responsibility: every variant of the scrutinee's
    /// enum must appear exactly once across `arms`. Each arm's body sees
    /// the matched variant's payload as additional positional bindings
    /// appended to the binding stack (just like `let` bindings).
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// `{ stmt0; stmt1; ...; tail }` -- a block with a sequence of
    /// statements followed by a required tail expression. Statements may
    /// introduce `let`/`let mut` bindings (which extend the binding
    /// stack for the remainder of the block, including `tail`) or
    /// assign to an in-scope mutable binding.
    Block { stmts: Vec<Stmt>, tail: Box<Expr> },
    /// `for i in 0..range_hi { body }` -- iterates `body` exactly
    /// `range_hi` times. The loop variable `i` of type
    /// `Bits<loop_var_width>` is pushed onto the binding stack while
    /// `body` is evaluated. The for-expression itself has unit value;
    /// it is only useful inside a `Block` that has prior mutable
    /// bindings to update.
    For {
        /// Width of the loop variable's `Bits<W>` type. Must satisfy
        /// `2^loop_var_width >= range_hi`.
        loop_var_width: u8,
        /// Exclusive upper bound for the iteration. Iteration values are
        /// 0..range_hi as `Bits<loop_var_width>`.
        range_hi: u32,
        /// Loop body. Evaluated `range_hi` times with the loop variable
        /// in scope as a fresh binding.
        body: Box<Expr>,
    },
}

/// One arm of an `Expr::Match`. Patterns are structural: the arm
/// matches when the scrutinee's enum_id and variant_idx agree. The
/// arm's body is evaluated with the variant's payload pushed onto the
/// binding stack in payload-declaration order.
#[derive(Debug, Clone)]
pub struct MatchArm {
    /// Enum id this arm matches. Must equal the enclosing `Match`'s
    /// scrutinee's enum_id.
    pub enum_id: usize,
    /// Variant index this arm matches.
    pub variant_idx: usize,
    /// Body expression, evaluated with payload values bound as fresh
    /// temporaries. The body's result type is the `Match` expression's
    /// type and must be the same across all arms.
    pub body: Expr,
}

/// A statement inside an `Expr::Block` (M6). Each variant either
/// introduces a new binding (`Let`/`LetMut`), mutates an existing
/// binding (`Assign`), or evaluates an expression for its side
/// effects (`ExprStmt`).
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `let tN = init;` — push an immutable binding.
    Let(Expr),
    /// `let mut tN = init;` — push a mutable binding.
    LetMut(Expr),
    /// `tN = value;` — overwrite a previously-introduced mutable
    /// binding. `var_idx` must reference a binding marked mutable by
    /// `LetMut`. The value's type must match the binding's type.
    Assign { var_idx: u8, value: Expr },
}

/// Direction of a shift operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    /// `<<`
    Shl,
    /// `>>`
    Shr,
}

/// Boolean-typed expression. Bool stays ephemeral throughout step 5 --
/// produced by comparisons / bool literals / combinators, consumed by
/// `Expr::If` conditions, never bound to a variable.
#[derive(Debug, Clone)]
pub enum BoolExpr {
    /// `true` or `false` literal.
    Lit(bool),
    /// Bitwise/logical NOT.
    Not(Box<BoolExpr>),
    /// Short-circuit `&&`.
    And(Box<BoolExpr>, Box<BoolExpr>),
    /// Short-circuit `||`.
    Or(Box<BoolExpr>, Box<BoolExpr>),
    /// Comparison between two same-typed sub-expressions producing `bool`.
    Cmp(CmpKind, Box<Expr>, Box<Expr>),
    /// Reduction (`.any()`, `.all()`, `.xor()`) over a scalar
    /// (`Bits<W>` or `Signed<W>`) operand, producing `bool`.
    Reduction(ReductionKind, Box<Expr>),
}

/// Bit-reduction operators (apply to all bits of a scalar value,
/// produce `bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// `.any()` -- OR of all bits.
    Any,
    /// `.all()` -- AND of all bits.
    All,
    /// `.xor()` -- XOR of all bits (population-count parity).
    Xor,
}

/// Comparison operators producing `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpKind {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// Binary operators. All operate on operands of the same scalar type and
/// produce a value of that type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    /// Wrapping addition.
    Add,
    /// Wrapping subtraction.
    Sub,
    /// Wrapping multiplication.
    Mul,
    /// Bitwise AND.
    BitAnd,
    /// Bitwise OR.
    BitOr,
    /// Bitwise XOR.
    BitXor,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOpKind {
    /// Bitwise complement (Rust `!`). Valid on `Bits` and `Signed`.
    Not,
    /// Arithmetic negation (Rust `-`). Valid on `Signed` only.
    Neg,
}
