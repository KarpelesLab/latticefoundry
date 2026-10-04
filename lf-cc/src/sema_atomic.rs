//! The GNU atomic builtins: the C11-model `__atomic_*` family (which lf-cc's
//! `<stdatomic.h>` is written in) and the legacy `__sync_*` family, checked
//! into the typed atomic nodes [`TExprKind::AtomicLoad`] & co. Written from the
//! GCC manual's description of each builtin.
//!
//! The object operated on is `*p` for the first argument `p`, a pointer to an
//! integer, pointer, floating or `_Bool` object of 1, 2, 4 or 8 bytes (an
//! `_Atomic` qualifier on it is allowed and changes nothing). A memory-order
//! argument that is not a constant expression is treated as `__ATOMIC_SEQ_CST`
//! (the strongest order, always a correct choice).

use super::*;

/// What a builtin's object may be: any scalar (load, store, exchange,
/// compare-exchange), or an integer or pointer (the arithmetic and bitwise
/// read-modify-writes).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operand {
    Any,
    Arith,
}

impl Checker {
    /// Check a call to the atomic builtin `name` (an `__atomic_*` or `__sync_*`
    /// identifier that is not otherwise declared).
    pub(super) fn check_atomic_builtin(
        &mut self,
        ctx: &mut FnCtx,
        name: &str,
        callee_span: Span,
        args: &[Expr],
        span: Span,
    ) -> Option<TExpr> {
        let argc = |n: usize, this: &mut Self| -> Option<()> {
            if args.len() == n {
                Some(())
            } else {
                this.error(span, format!("'{name}' takes {n} argument(s), found {}", args.len()));
                None
            }
        };
        let rmw_op = |s: &str| -> Option<AtomicOp> {
            Some(match s {
                "add" => AtomicOp::Add,
                "sub" => AtomicOp::Sub,
                "and" => AtomicOp::And,
                "or" => AtomicOp::Or,
                "xor" => AtomicOp::Xor,
                "nand" => AtomicOp::Nand,
                _ => return None,
            })
        };
        if let Some(rest) = name.strip_prefix("__atomic_") {
            match rest {
                "load_n" => {
                    argc(2, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let order = self.memory_order(&args[1]);
                    return Some(TExpr::new(TExprKind::AtomicLoad { ptr: Box::new(ptr), order }, ty, span));
                }
                "store_n" => {
                    argc(3, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let v = self.check_rvalue(ctx, &args[1])?;
                    let value = Box::new(self.convert(v, &ty));
                    let order = self.memory_order(&args[2]);
                    let kind = TExprKind::AtomicStore { ptr: Box::new(ptr), value, order };
                    return Some(TExpr::new(kind, CType::Void, span));
                }
                "exchange_n" => {
                    argc(3, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let v = self.check_rvalue(ctx, &args[1])?;
                    let value = Box::new(self.convert(v, &ty));
                    let order = self.memory_order(&args[2]);
                    let kind =
                        TExprKind::AtomicRmw { op: AtomicOp::Xchg, ptr: Box::new(ptr), value, order, fetch_old: true };
                    return Some(TExpr::new(kind, ty, span));
                }
                "compare_exchange_n" => {
                    argc(6, self)?;
                    // A weak compare-exchange may fail spuriously; the strong
                    // form is a valid implementation of it.
                    return self.atomic_cas(ctx, name, &args[0], &args[1], &args[2], &args[4], &args[5], span);
                }
                // The generic forms move their values through pointers; they are
                // the `_n` forms with the indirections spelled out.
                "load" => {
                    argc(3, self)?;
                    let call = call_expr("__atomic_load_n", callee_span, vec![args[0].clone(), args[2].clone()]);
                    return self.check_void_assign_through(ctx, &args[1], call, span);
                }
                "store" => {
                    argc(3, self)?;
                    let call = call_expr(
                        "__atomic_store_n",
                        callee_span,
                        vec![args[0].clone(), deref_expr(&args[1]), args[2].clone()],
                    );
                    return self.check_expr(ctx, &call);
                }
                "exchange" => {
                    argc(4, self)?;
                    let call = call_expr(
                        "__atomic_exchange_n",
                        callee_span,
                        vec![args[0].clone(), deref_expr(&args[1]), args[3].clone()],
                    );
                    return self.check_void_assign_through(ctx, &args[2], call, span);
                }
                "compare_exchange" => {
                    argc(6, self)?;
                    let desired = deref_expr(&args[2]);
                    return self.atomic_cas(ctx, name, &args[0], &args[1], &desired, &args[4], &args[5], span);
                }
                "test_and_set" => {
                    argc(2, self)?;
                    // On the byte at `p`: set it to 1, yielding whether it was set.
                    let ptr = self.atomic_byte_ptr(ctx, &args[0], name)?;
                    let order = self.memory_order(&args[1]);
                    let u8t = CType::Int(IntTy::new(8, false));
                    let one = Box::new(TExpr::new(TExprKind::Const(1), u8t.clone(), span));
                    let kind =
                        TExprKind::AtomicRmw { op: AtomicOp::Xchg, ptr: Box::new(ptr), value: one, order, fetch_old: true };
                    let old = TExpr::new(kind, u8t, span);
                    return Some(self.convert(old, &CType::Bool));
                }
                "clear" => {
                    argc(2, self)?;
                    let ptr = self.atomic_byte_ptr(ctx, &args[0], name)?;
                    let order = self.memory_order(&args[1]);
                    let zero = Box::new(TExpr::new(TExprKind::Const(0), CType::Int(IntTy::new(8, false)), span));
                    let kind = TExprKind::AtomicStore { ptr: Box::new(ptr), value: zero, order };
                    return Some(TExpr::new(kind, CType::Void, span));
                }
                "thread_fence" => {
                    argc(1, self)?;
                    let order = self.memory_order(&args[0]);
                    return Some(TExpr::new(TExprKind::AtomicFence(order), CType::Void, span));
                }
                "signal_fence" => {
                    // Orders only against a signal handler on the same thread:
                    // a compiler barrier, which lf-cc's passes already respect
                    // at every call and atomic operation.
                    argc(1, self)?;
                    let _ = self.memory_order(&args[0]);
                    let zero = TExpr::new(TExprKind::Const(0), CType::int(), span);
                    return Some(TExpr::new(TExprKind::Convert(Box::new(zero)), CType::Void, span));
                }
                "always_lock_free" | "is_lock_free" => {
                    argc(2, self)?;
                    // The size is typically `sizeof *obj`, which only sema can type.
                    let size = self.check_rvalue(ctx, &args[0])?;
                    let _ = self.check_rvalue(ctx, &args[1])?;
                    let TExprKind::Const(size) = size.kind else {
                        self.error(args[0].span, format!("the size argument of '{name}' must be a constant"));
                        return None;
                    };
                    let free = matches!(size, 1 | 2 | 4 | 8);
                    return Some(TExpr::new(TExprKind::Const(i128::from(free)), CType::Bool, span));
                }
                _ => {}
            }
            // `__atomic_<op>_fetch` / `__atomic_fetch_<op>`.
            let (op, fetch_old) = match rest.strip_suffix("_fetch") {
                Some(op) => (rmw_op(op), false),
                None => (rest.strip_prefix("fetch_").and_then(rmw_op), true),
            };
            if let Some(op) = op {
                argc(3, self)?;
                let order = self.memory_order(&args[2]);
                return self.atomic_rmw(ctx, name, op, &args[0], &args[1], order, fetch_old, span);
            }
        }
        if let Some(rest) = name.strip_prefix("__sync_") {
            // Every `__sync` builtin is a full barrier; trailing arguments (a
            // list of protected variables) are accepted and ignored.
            let order = MemOrder::SeqCst;
            let at_least = |n: usize, this: &mut Self| -> Option<()> {
                if args.len() >= n {
                    Some(())
                } else {
                    this.error(span, format!("'{name}' takes at least {n} argument(s)"));
                    None
                }
            };
            match rest {
                "synchronize" => return Some(TExpr::new(TExprKind::AtomicFence(order), CType::Void, span)),
                "bool_compare_and_swap" | "val_compare_and_swap" => {
                    at_least(3, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let e = self.check_rvalue(ctx, &args[1])?;
                    let expected = Box::new(self.convert(e, &ty));
                    let d = self.check_rvalue(ctx, &args[2])?;
                    let desired = Box::new(self.convert(d, &ty));
                    let want_old = rest == "val_compare_and_swap";
                    let kind = TExprKind::AtomicCas {
                        ptr: Box::new(ptr),
                        expected,
                        desired,
                        by_ref: false,
                        success: order,
                        failure: order,
                        want_old,
                    };
                    return Some(TExpr::new(kind, if want_old { ty } else { CType::Bool }, span));
                }
                // An acquire exchange (GCC: "an acquire barrier").
                "lock_test_and_set" => {
                    at_least(2, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let v = self.check_rvalue(ctx, &args[1])?;
                    let value = Box::new(self.convert(v, &ty));
                    let kind = TExprKind::AtomicRmw {
                        op: AtomicOp::Xchg,
                        ptr: Box::new(ptr),
                        value,
                        order: MemOrder::Acquire,
                        fetch_old: true,
                    };
                    return Some(TExpr::new(kind, ty, span));
                }
                // A release store of 0.
                "lock_release" => {
                    at_least(1, self)?;
                    let (ptr, ty) = self.atomic_object(ctx, &args[0], name, Operand::Any)?;
                    let zero = TExpr::new(TExprKind::Const(0), CType::int(), span);
                    let value = Box::new(self.convert(zero, &ty));
                    let kind = TExprKind::AtomicStore { ptr: Box::new(ptr), value, order: MemOrder::Release };
                    return Some(TExpr::new(kind, CType::Void, span));
                }
                _ => {}
            }
            let (op, fetch_old) = match rest.strip_prefix("fetch_and_") {
                Some(op) => (rmw_op(op), true),
                None => (rest.strip_suffix("_and_fetch").and_then(rmw_op), false),
            };
            if let Some(op) = op {
                at_least(2, self)?;
                return self.atomic_rmw(ctx, name, op, &args[0], &args[1], order, fetch_old, span);
            }
        }
        self.error(span, format!("unknown atomic builtin '{name}'"));
        None
    }

    /// Check the object-pointer argument of an atomic builtin, returning it and
    /// the (unqualified) type of the object it points to.
    fn atomic_object(&mut self, ctx: &mut FnCtx, p: &Expr, name: &str, operand: Operand) -> Option<(TExpr, CType)> {
        let ptr = self.check_rvalue(ctx, p)?;
        let Some(pointee) = ptr.ty.pointee() else {
            self.error(p.span, format!("the first argument of '{name}' must be a pointer"));
            return None;
        };
        let ty = pointee.unqual().clone();
        let ok = match operand {
            Operand::Any => ty.is_scalar() && ty.unsupported_value().is_none(),
            Operand::Arith => matches!(ty, CType::Int(_) | CType::Pointer(_)),
        };
        let size = self.size_of(&ty);
        if !ok || !matches!(size, 1 | 2 | 4 | 8) {
            let what = if operand == Operand::Arith { "an integer or pointer" } else { "a scalar" };
            self.error(p.span, format!("'{name}' needs a pointer to {what} object of 1, 2, 4 or 8 bytes, found '{ty}'"));
            return None;
        }
        Some((ptr, ty))
    }

    /// The pointer argument of `__atomic_test_and_set`/`__atomic_clear`, as a
    /// pointer to the byte they operate on (any object pointer is accepted).
    fn atomic_byte_ptr(&mut self, ctx: &mut FnCtx, p: &Expr, name: &str) -> Option<TExpr> {
        let ptr = self.check_rvalue(ctx, p)?;
        if !ptr.ty.is_pointer() {
            self.error(p.span, format!("the first argument of '{name}' must be a pointer"));
            return None;
        }
        Some(self.convert(ptr, &CType::ptr_to(CType::Int(IntTy::new(8, false)))))
    }

    /// An arithmetic or bitwise read-modify-write of `*p` with `v`. On a
    /// pointer object the operand is a byte offset (GCC does not scale it by
    /// the pointee size).
    #[allow(clippy::too_many_arguments)]
    fn atomic_rmw(
        &mut self,
        ctx: &mut FnCtx,
        name: &str,
        op: AtomicOp,
        p: &Expr,
        v: &Expr,
        order: MemOrder,
        fetch_old: bool,
        span: Span,
    ) -> Option<TExpr> {
        let (ptr, ty) = self.atomic_object(ctx, p, name, Operand::Arith)?;
        let v = self.check_rvalue(ctx, v)?;
        let operand_ty = if ty.is_pointer() { CType::long() } else { ty.clone() };
        if !v.ty.is_integer() && !(v.ty.is_pointer() && ty.is_pointer()) {
            self.error(v.span, format!("the value operand of '{name}' must be an integer"));
            return None;
        }
        let value = Box::new(self.convert(v, &operand_ty));
        let kind = TExprKind::AtomicRmw { op, ptr: Box::new(ptr), value, order, fetch_old };
        Some(TExpr::new(kind, ty, span))
    }

    /// A strong compare-exchange of `*p`: the expected value is read through
    /// the pointer `expected` and, on failure, the value found is written back
    /// to it. Yields the `_Bool` success flag.
    #[allow(clippy::too_many_arguments)]
    fn atomic_cas(
        &mut self,
        ctx: &mut FnCtx,
        name: &str,
        p: &Expr,
        expected: &Expr,
        desired: &Expr,
        success: &Expr,
        failure: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let (ptr, ty) = self.atomic_object(ctx, p, name, Operand::Any)?;
        let e = self.check_rvalue(ctx, expected)?;
        if !e.ty.is_pointer() {
            self.error(expected.span, format!("the expected-value argument of '{name}' must be a pointer"));
            return None;
        }
        let e = self.convert(e, &CType::ptr_to(ty.clone()));
        let d = self.check_rvalue(ctx, desired)?;
        let d = self.convert(d, &ty);
        let success = self.memory_order(success);
        let failure = self.memory_order(failure);
        let kind = TExprKind::AtomicCas {
            ptr: Box::new(ptr),
            expected: Box::new(e),
            desired: Box::new(d),
            by_ref: true,
            success,
            failure,
            want_old: false,
        };
        Some(TExpr::new(kind, CType::Bool, span))
    }

    /// `*dst = value` evaluated for effect, typed `void` (the generic
    /// `__atomic_load`/`__atomic_exchange` forms).
    fn check_void_assign_through(&mut self, ctx: &mut FnCtx, dst: &Expr, value: Expr, span: Span) -> Option<TExpr> {
        let assign = Expr { kind: ExprKind::Assign(None, Box::new(deref_expr(dst)), Box::new(value)), span };
        let t = self.check_expr(ctx, &assign)?;
        Some(TExpr::new(TExprKind::Convert(Box::new(t)), CType::Void, span))
    }

    /// The memory order a builtin's order argument denotes: its constant value
    /// (`__ATOMIC_RELAXED` = 0 … `__ATOMIC_SEQ_CST` = 5), else `seq_cst`.
    fn memory_order(&mut self, e: &Expr) -> MemOrder {
        match self.const_eval(e) {
            Some(0) => MemOrder::Relaxed,
            Some(1) => MemOrder::Consume,
            Some(2) => MemOrder::Acquire,
            Some(3) => MemOrder::Release,
            Some(4) => MemOrder::AcqRel,
            _ => MemOrder::SeqCst,
        }
    }
}

/// The call expression `name(args...)`.
fn call_expr(name: &str, span: Span, args: Vec<Expr>) -> Expr {
    let callee = Expr { kind: ExprKind::Ident(name.to_owned()), span };
    Expr { kind: ExprKind::Call(Box::new(callee), args), span }
}

/// The expression `*e`.
fn deref_expr(e: &Expr) -> Expr {
    Expr { kind: ExprKind::Unary(UnaryOp::Deref, Box::new(e.clone())), span: e.span }
}
