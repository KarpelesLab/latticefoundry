//! The **constant-time verifier** (`docs/ir-design.md` §6d): no secret-derived
//! value may influence control flow, an address, or the timing of a
//! variable-time instruction.
//!
//! Built on the secret-taint analysis ([`SecretTaint`]), it rejects any
//! function where a secret-derived value reaches:
//!
//! | use | why |
//! |---|---|
//! | a `cond_br` or `switch` condition | control flow (branch predictor, i-cache, timing) |
//! | the address of a `load`/`store`/atomic, the base or offset of a `ptr_add` | data-cache timing |
//! | a call target | control flow |
//! | either operand of `udiv`/`sdiv`/`urem`/`srem` | early-exit dividers on every target |
//! | any floating-point arithmetic, `fcmp`, or float conversion | subnormal slow paths; x86-64's `u64`↔float conversions branch |
//! | the size of a `dyn_alloca` | a stack-probe loop over the size |
//! | an operand of a syscall | it leaves the program |
//! | the value of an atomic store/rmw/cmpxchg, or an rmw/cmpxchg on memory that may hold a secret | retry loops compare the (secret) memory contents |
//! | a public parameter of a direct call, any argument of an indirect call, a variadic argument | secrecy is part of a callee's interface |
//! | the `ret` of a function whose return is not secret | ditto |
//! | an unflagged `store` to memory that is not a non-escaping stack slot or a secret global | secrets crossing functions through memory are declared (`store secret`) |
//! | a shift amount / a multiply operand, only under [`CtPolicy::STRICT`] | variable-latency shifters and multipliers on some small cores |
//!
//! Everything else is allowed on secrets: integer `add`/`sub`/`and`/`or`/`xor`,
//! shifts and `mul` (under [`CtPolicy::DEFAULT`]), `icmp`, integer casts,
//! `bitcast`, `fneg`, `freeze`, `select` (the backends lower it without a branch:
//! `cmov` on x86-64, `csel` on AArch64, a mask blend on RISC-V), block
//! arguments, stores to non-escaping stack slots and secret globals, and
//! `declassify`.
//!
//! [`verify_module`](super::verify_module) runs this check on every module that
//! declares a secret ([`Module::has_secrets`]); a module without one is
//! trivially constant-time and costs nothing.

use crate::analysis::secret::{MemRoot, Origin, SecretTaint};
use crate::ir::inst::{BinOp, CastOp, InstId, InstKind, ReduceOp};
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Module};
use crate::support::diagnostics::Diagnostic;

/// Which operations are variable-time on the target (the parts of the
/// constant-time rules that differ between CPUs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CtPolicy {
    /// Whether a secret shift *amount* is allowed. Barrel shifters are
    /// constant-time on every target LatticeFoundry has (x86-64 `shl r, cl`,
    /// AArch64 `lslv`, RISC-V `sll`); some small cores (e.g. without a barrel
    /// shifter) shift one bit per cycle.
    pub secret_shift_amount: bool,
    /// Whether a secret multiply operand is allowed. Constant-time on x86-64
    /// and AArch64 application cores; early-terminating multipliers exist on
    /// some microcontrollers and RISC-V cores without Zkt.
    pub secret_multiply: bool,
}

impl CtPolicy {
    /// The policy for x86-64, AArch64 and RV64: shifts and multiplies are
    /// constant-time.
    pub const DEFAULT: CtPolicy = CtPolicy { secret_shift_amount: true, secret_multiply: true };

    /// The conservative policy for cores with variable-time shifts or
    /// multiplies: neither may take a secret operand.
    pub const STRICT: CtPolicy = CtPolicy { secret_shift_amount: false, secret_multiply: false };
}

impl Default for CtPolicy {
    fn default() -> Self {
        CtPolicy::DEFAULT
    }
}

/// How a secret-derived value was misused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CtRole {
    /// The condition of a `cond_br`.
    BranchCondition,
    /// The scrutinee of a `switch`.
    SwitchCondition,
    /// The address of a memory access, or the base/offset of a `ptr_add`.
    Address,
    /// The callee operand of a `call`.
    CallTarget,
    /// Argument `index` of a direct call, passed to a public parameter.
    PublicParameter(usize),
    /// An argument of an indirect call (function types carry no secrecy).
    IndirectArgument,
    /// A variadic argument.
    VariadicArgument,
    /// The returned value of a function whose return is public.
    PublicReturn,
    /// An operand of an integer division or remainder.
    Division,
    /// An operand of a floating-point operation or conversion.
    FloatOperation,
    /// A shift amount (only under a policy that forbids it).
    ShiftAmount,
    /// A multiply operand (only under a policy that forbids it).
    Multiply,
    /// The size of a `dyn_alloca`.
    AllocaSize,
    /// An operand of a `syscall`.
    SyscallOperand,
    /// The value operand of an atomic store, rmw or cmpxchg.
    AtomicOperand,
    /// An atomic rmw or cmpxchg on memory that may hold a secret.
    AtomicOnSecretMemory,
    /// An unflagged store to memory that is neither a non-escaping stack slot
    /// nor a secret global.
    PublicStore,
}

impl CtRole {
    /// A short phrase for diagnostics.
    pub fn describe(self) -> String {
        match self {
            CtRole::BranchCondition => "is the condition of a `cond_br`".into(),
            CtRole::SwitchCondition => "is the scrutinee of a `switch`".into(),
            CtRole::Address => "is used as a memory address".into(),
            CtRole::CallTarget => "is the target of a `call`".into(),
            CtRole::PublicParameter(i) => {
                format!("is passed as argument {i} to a parameter that is not `secret`")
            }
            CtRole::IndirectArgument => {
                "is passed to an indirect call (function types carry no secrecy)".into()
            }
            CtRole::VariadicArgument => "is passed as a variadic argument".into(),
            CtRole::PublicReturn => "is returned from a function whose return is not `secret`".into(),
            CtRole::Division => "is an operand of a division or remainder (variable time)".into(),
            CtRole::FloatOperation => {
                "is an operand of a floating-point operation (variable time)".into()
            }
            CtRole::ShiftAmount => "is a shift amount (variable time on this target)".into(),
            CtRole::Multiply => "is a multiply operand (variable time on this target)".into(),
            CtRole::AllocaSize => "is the size of a `dyn_alloca`".into(),
            CtRole::SyscallOperand => "is an operand of a `syscall`".into(),
            CtRole::AtomicOperand => "is the value operand of an atomic operation".into(),
            CtRole::AtomicOnSecretMemory => {
                "is memory read by an atomic rmw/cmpxchg (its retry loop compares it)".into()
            }
            CtRole::PublicStore => {
                "is stored to memory that is not declared secret (use `store secret`)".into()
            }
        }
    }
}

/// One constant-time violation: the secret-derived `value` misused by
/// instruction `inst` in the role `role`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CtViolation {
    /// The function.
    pub func: FuncId,
    /// The offending instruction.
    pub inst: InstId,
    /// Its block.
    pub block: BlockId,
    /// The secret-derived operand (for [`CtRole::AtomicOnSecretMemory`], the
    /// address).
    pub value: ValueId,
    /// How it is misused.
    pub role: CtRole,
}

/// Every constant-time violation of function `func` under `policy`.
/// A declaration has none.
pub fn ct_violations(module: &Module, func: FuncId, policy: CtPolicy) -> Vec<CtViolation> {
    let f = module.function(func);
    if f.is_declaration() {
        return Vec::new();
    }
    let taint = SecretTaint::compute(module, func);
    ct_violations_with(module, func, &taint, policy)
}

/// [`ct_violations`] over an already computed [`SecretTaint`].
pub fn ct_violations_with(
    module: &Module,
    func: FuncId,
    taint: &SecretTaint,
    policy: CtPolicy,
) -> Vec<CtViolation> {
    let f = module.function(func);
    let mut out = Vec::new();
    if !taint.any_secret() {
        return out;
    }
    let secret_ret = module.func_attrs(func).secret_ret;
    for (bid, block) in f.blocks() {
        if !taint.is_reachable(bid) {
            continue;
        }
        let insts = block.insts().iter().copied().chain(block.terminator());
        for i in insts {
            let data = f.inst(i);
            let ops = data.operands();
            // Candidate (operand, role) pairs; only secret-derived operands are
            // violations.
            let mut uses: Vec<(ValueId, CtRole)> = Vec::new();
            let mut flag = |v: ValueId, role: CtRole| uses.push((v, role));
            match &data.kind {
                InstKind::CondBr { .. } => flag(ops[0], CtRole::BranchCondition),
                InstKind::Switch(_) => flag(ops[0], CtRole::SwitchCondition),
                InstKind::Ret => {
                    if !secret_ret && let Some(&v) = ops.first() {
                        flag(v, CtRole::PublicReturn);
                    }
                }
                InstKind::Bin(op) => match op {
                    BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                        flag(ops[0], CtRole::Division);
                        flag(ops[1], CtRole::Division);
                    }
                    BinOp::Shl | BinOp::LShr | BinOp::AShr if !policy.secret_shift_amount => {
                        flag(ops[1], CtRole::ShiftAmount);
                    }
                    BinOp::Mul if !policy.secret_multiply => {
                        flag(ops[0], CtRole::Multiply);
                        flag(ops[1], CtRole::Multiply);
                    }
                    op if op.is_float() => {
                        flag(ops[0], CtRole::FloatOperation);
                        flag(ops[1], CtRole::FloatOperation);
                    }
                    _ => {}
                },
                InstKind::FCmp(_) => {
                    flag(ops[0], CtRole::FloatOperation);
                    flag(ops[1], CtRole::FloatOperation);
                }
                // A reduction is a chain of its scalar op over the lanes (the
                // lane-wise vector ops go through the arms above).
                InstKind::Reduce(op) if op.is_float() => flag(ops[0], CtRole::FloatOperation),
                InstKind::Reduce(ReduceOp::Mul) if !policy.secret_multiply => {
                    flag(ops[0], CtRole::Multiply);
                }
                InstKind::Cast(
                    CastOp::FpTrunc
                    | CastOp::FpExt
                    | CastOp::FpToUi
                    | CastOp::FpToSi
                    | CastOp::UiToFp
                    | CastOp::SiToFp,
                ) => flag(ops[0], CtRole::FloatOperation),
                InstKind::PtrAdd { .. } => {
                    flag(ops[0], CtRole::Address);
                    flag(ops[1], CtRole::Address);
                }
                InstKind::Load { .. } | InstKind::AtomicLoad { .. } => flag(ops[0], CtRole::Address),
                InstKind::Store { secret, .. } => {
                    flag(ops[0], CtRole::Address);
                    let local = match taint.root_of(ops[0]) {
                        MemRoot::Stack(_) => true,
                        MemRoot::Global(g) => module.global_attrs(g).secret,
                        MemRoot::Unknown => false,
                    };
                    if !secret && !local {
                        flag(ops[1], CtRole::PublicStore);
                    }
                }
                InstKind::AtomicStore { .. } => {
                    flag(ops[0], CtRole::Address);
                    flag(ops[1], CtRole::AtomicOperand);
                }
                InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } => {
                    flag(ops[0], CtRole::Address);
                    for &v in &ops[1..] {
                        flag(v, CtRole::AtomicOperand);
                    }
                    if taint.memory_secret(taint.root_of(ops[0])) {
                        out.push(CtViolation {
                            func,
                            inst: i,
                            block: bid,
                            value: ops[0],
                            role: CtRole::AtomicOnSecretMemory,
                        });
                    }
                }
                InstKind::DynAlloca { .. } => flag(ops[0], CtRole::AllocaSize),
                InstKind::Syscall => {
                    for &v in ops {
                        flag(v, CtRole::SyscallOperand);
                    }
                }
                InstKind::Call => {
                    flag(ops[0], CtRole::CallTarget);
                    match f.value(ops[0]).def {
                        ValueDef::Func(callee) => {
                            let attrs = module.func_attrs(callee);
                            let nparams = match module.types().get(module.function(callee).sig) {
                                crate::ir::Type::Func(ft) => ft.params.len(),
                                _ => 0,
                            };
                            for (k, &a) in ops[1..].iter().enumerate() {
                                if k >= nparams {
                                    flag(a, CtRole::VariadicArgument);
                                } else if !attrs.is_param_secret(k) {
                                    flag(a, CtRole::PublicParameter(k));
                                }
                            }
                        }
                        _ => {
                            for &a in &ops[1..] {
                                flag(a, CtRole::IndirectArgument);
                            }
                        }
                    }
                }
                _ => {}
            }
            for (v, role) in uses {
                if taint.is_secret(v) {
                    out.push(CtViolation { func, inst: i, block: bid, value: v, role });
                }
            }
        }
    }
    out
}

/// The constant-time diagnostics of function `func` under `policy`: one error
/// per violation, naming the value (with its `.lf` print name), the use, and
/// the chain back to the secret it derives from.
pub fn verify_function_ct(module: &Module, func: FuncId, policy: CtPolicy) -> Vec<Diagnostic> {
    let f = module.function(func);
    if f.is_declaration() {
        return Vec::new();
    }
    let taint = SecretTaint::compute(module, func);
    let violations = ct_violations_with(module, func, &taint, policy);
    if violations.is_empty() {
        return Vec::new();
    }
    let names = crate::ir::text::value_print_names(f);
    let name = |v: ValueId| match names.get(&v) {
        Some(n) => format!("%{n}"),
        None => format!("value #{}", v.index()),
    };
    violations
        .iter()
        .map(|vi| {
            let op = opcode_name(&f.inst(vi.inst).kind);
            let mut msg = format!(
                "function #{}: constant-time violation: secret-derived {} {} (`{op}` in block ^{})",
                func.index(),
                name(vi.value),
                vi.role.describe(),
                vi.block.index()
            );
            let chain = taint.explain(module, func, vi.value);
            if !chain.is_empty() {
                msg.push_str("; it derives from ");
                let parts: Vec<String> = chain
                    .iter()
                    .map(|o| match *o {
                        Origin::Operand(_, p) | Origin::BlockArg(_, p) => name(p),
                        Origin::SecretParam(i) => format!("secret parameter {i}"),
                        Origin::SecretLoad(_) => "a `load secret`".into(),
                        Origin::SecretMemory(_, MemRoot::Stack(_)) => {
                            "a load of a stack slot holding a secret".into()
                        }
                        Origin::SecretMemory(_, MemRoot::Global(g)) => {
                            format!("a load of global #{} (secret)", g.index())
                        }
                        Origin::SecretMemory(_, MemRoot::Unknown) => {
                            "a load of memory that may hold a secret".into()
                        }
                        Origin::SecretReturn(_, callee) => {
                            format!("the secret return of function #{}", callee.index())
                        }
                    })
                    .collect();
                msg.push_str(&parts.join(" <- "));
            }
            if vi.role == CtRole::BranchCondition || vi.role == CtRole::SwitchCondition {
                msg.push_str(" (use `select`, or `declassify` a value that may be public)");
            }
            Diagnostic::error(msg)
        })
        .collect()
}

/// Run the constant-time verifier over every function of `module` under the
/// default policy. `Ok` for a module without secrets.
pub fn verify_module_ct(module: &Module) -> Result<(), Vec<Diagnostic>> {
    verify_module_ct_with(module, CtPolicy::DEFAULT)
}

/// [`verify_module_ct`] under an explicit policy.
pub fn verify_module_ct_with(module: &Module, policy: CtPolicy) -> Result<(), Vec<Diagnostic>> {
    if !module.has_secrets() {
        return Ok(());
    }
    let mut diags = Vec::new();
    for i in 0..module.function_count() {
        diags.extend(verify_function_ct(module, FuncId::from_index(i), policy));
    }
    if diags.is_empty() { Ok(()) } else { Err(diags) }
}

/// The `.lf` spelling of an opcode, for diagnostics.
fn opcode_name(kind: &InstKind) -> &'static str {
    match kind {
        InstKind::Bin(op) => match op {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::UDiv => "udiv",
            BinOp::SDiv => "sdiv",
            BinOp::URem => "urem",
            BinOp::SRem => "srem",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::LShr => "lshr",
            BinOp::AShr => "ashr",
            BinOp::FAdd => "fadd",
            BinOp::FSub => "fsub",
            BinOp::FMul => "fmul",
            BinOp::FDiv => "fdiv",
            BinOp::FRem => "frem",
            BinOp::SMin => "smin",
            BinOp::SMax => "smax",
            BinOp::UMin => "umin",
            BinOp::UMax => "umax",
            BinOp::SAddSat => "sadd_sat",
            BinOp::UAddSat => "uadd_sat",
            BinOp::SSubSat => "ssub_sat",
            BinOp::USubSat => "usub_sat",
        },
        InstKind::Unary(_) => "fneg",
        InstKind::ICmp(_) => "icmp",
        InstKind::FCmp(_) => "fcmp",
        InstKind::Cast(_) => "cast",
        InstKind::Alloca { .. } => "alloca",
        InstKind::DynAlloca { .. } => "dyn_alloca",
        InstKind::Load { .. } => "load",
        InstKind::Store { .. } => "store",
        InstKind::AtomicLoad { .. } => "atomic_load",
        InstKind::AtomicStore { .. } => "atomic_store",
        InstKind::AtomicRmw { .. } => "atomic_rmw",
        InstKind::CmpXchg { .. } => "cmpxchg",
        InstKind::Fence(_) => "fence",
        InstKind::PtrAdd { .. } => "ptr_add",
        InstKind::Select => "select",
        InstKind::Freeze => "freeze",
        InstKind::Declassify => "declassify",
        InstKind::ExtractElement { .. } => "extractelement",
        InstKind::InsertElement { .. } => "insertelement",
        InstKind::ShuffleVector(_) => "shufflevector",
        InstKind::Splat => "splat",
        InstKind::Reduce(_) => "reduce",
        InstKind::Call => "call",
        InstKind::Syscall => "syscall",
        InstKind::Ret => "ret",
        InstKind::Br(_) => "br",
        InstKind::CondBr { .. } => "cond_br",
        InstKind::Switch(_) => "switch",
        InstKind::Unreachable => "unreachable",
    }
}
#[cfg(test)]
mod tests;
