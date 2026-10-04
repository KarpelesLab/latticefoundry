//! GCC vector extensions (`__attribute__((vector_size(N)))`), written from the
//! GCC manual's "Using Vector Instructions through Built-in Functions": the
//! element-wise operators, comparisons yielding a vector of `-1`/`0` lanes,
//! subscripting, scalar operands (broadcast to every lane), casts between
//! same-size vector and integer types, and the `__builtin_shufflevector`,
//! `__builtin_shuffle` and `__builtin_convertvector` builtins.

use super::*;

/// The type of comparing two vectors of type `vty`: a vector of signed
/// integers as wide as the elements, each lane `-1` (true) or `0` (false).
pub(super) fn vector_cmp_type(vty: &CType) -> CType {
    let (elem, n) = vty.vector_parts().expect("a vector type");
    let bits = match elem {
        CType::Float(f) => f.bits(),
        other => other.int_width().unwrap_or(32),
    };
    CType::Vector(Box::new(CType::Int(IntTy::new(bits, true))), n)
}

impl Checker {
    /// A binary operator with at least one vector operand. Both operands
    /// become the vector type (a scalar is converted to the element type and
    /// broadcast); `%`, the bitwise operators and shifts need integer lanes.
    pub(super) fn check_vector_binary(&mut self, op: BinaryOp, lt: TExpr, rt: TExpr, span: Span) -> Option<TExpr> {
        let vty = if lt.ty.is_vector() { lt.ty.clone() } else { rt.ty.clone() };
        let float = vty.vector_parts().is_some_and(|(e, _)| e.is_float());
        match op {
            BinaryOp::LAnd | BinaryOp::LOr => {
                self.error(span, "'&&' and '||' do not take vector operands");
                return None;
            }
            BinaryOp::Rem
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
                if float =>
            {
                self.error(span, "this operator needs integer vector elements");
                return None;
            }
            _ => {}
        }
        let l = self.vector_operand(lt, &vty, span)?;
        let r = self.vector_operand(rt, &vty, span)?;
        let (kind, ty) = match op {
            BinaryOp::Shl | BinaryOp::Shr => (TExprKind::Shift(op, Box::new(l), Box::new(r)), vty),
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                let cty = vector_cmp_type(&vty);
                (TExprKind::Cmp(op, Box::new(l), Box::new(r)), cty)
            }
            _ => (TExprKind::Arith(op, Box::new(l), Box::new(r)), vty),
        };
        Some(TExpr::new(kind, ty, span))
    }

    /// Bring an operand to the vector type `vty`: a vector of the same size
    /// and length is reinterpreted (elements differing only in signedness,
    /// as gcc allows); an arithmetic scalar is converted to the element type
    /// and broadcast to every lane.
    pub(super) fn vector_operand(&mut self, e: TExpr, vty: &CType, span: Span) -> Option<TExpr> {
        if &e.ty == vty {
            return Some(e);
        }
        let (elem, n) = vty.vector_parts().expect("a vector type");
        if let Some((eelem, en)) = e.ty.vector_parts() {
            if en == n && self.size_of(eelem) == self.size_of(elem) && eelem.is_float() == elem.is_float() {
                let span = e.span;
                return Some(TExpr::new(TExprKind::Convert(Box::new(e)), vty.clone(), span));
            }
            self.error(span, format!("incompatible vector types '{}' and '{vty}'", e.ty));
            return None;
        }
        if !e.ty.is_arithmetic() {
            self.error(span, format!("cannot use a '{}' as an operand of a vector operation", e.ty));
            return None;
        }
        let elem = elem.clone();
        let lane = self.convert(e, &elem);
        Some(TExpr::new(TExprKind::VecSplat(Box::new(lane)), vty.clone(), span))
    }

    /// `v[i]` on a vector: the `i`-th element, an lvalue when `v` is one
    /// (addressed through the vector's storage, as gcc treats it); an rvalue
    /// vector is held in a temporary first.
    pub(super) fn check_vector_index(&mut self, ctx: &mut FnCtx, v: TExpr, idx: TExpr, span: Span) -> Option<TExpr> {
        if !idx.ty.is_integer() {
            self.error(span, "a vector subscript must be an integer");
            return None;
        }
        let vty = v.ty.clone();
        let (elem, _) = vty.vector_parts().expect("a vector type");
        let elem = elem.clone();
        let (init, v) = if v.is_lvalue() {
            (None, v)
        } else {
            let (init, t) = self.bind_temp(ctx, v);
            (Some(init), t)
        };
        let quals = v.quals;
        let addr = TExpr::new(TExprKind::AddrOf(Box::new(v)), CType::ptr_to(vty.qualified(quals)), span);
        let elem_ptr = CType::ptr_to(elem.clone());
        let base = self.convert(addr, &elem_ptr);
        let index = Box::new(self.convert(idx, &CType::long()));
        let elem_size = self.size_of(&elem);
        let ptr = TExpr::new(TExprKind::PtrArith { ptr: Box::new(base), index, elem_size, sub: false }, elem_ptr, span);
        let lane = TExpr::new(TExprKind::Deref(Box::new(ptr)), elem.clone(), span).with_quals(quals);
        Some(match init {
            None => lane,
            Some(init) => TExpr::new(TExprKind::StmtExpr(vec![init], Some(Box::new(lane))), elem, span),
        })
    }

    /// `__builtin_shufflevector(a, b, i0, i1, ...)`: a vector of `a`'s element
    /// type whose lane `k` is lane `ik` of the concatenation `a ++ b`; each
    /// index is a constant (`-1`: any lane).
    pub(super) fn builtin_shufflevector(&mut self, ctx: &mut FnCtx, args: &[Expr], span: Span) -> Option<TExpr> {
        if args.len() < 3 {
            self.error(span, "__builtin_shufflevector takes two vectors and at least one index");
            return None;
        }
        let (a, b, vty) = self.shuffle_sources(ctx, &args[0], Some(&args[1]), span)?;
        let n = vty.vector_parts().map_or(0, |(_, n)| n);
        let mut mask = Vec::with_capacity(args.len() - 2);
        for idx in &args[2..] {
            match self.const_eval(idx) {
                Some(-1) => mask.push(0),
                Some(i) if (0..2 * i128::from(n)).contains(&i) => mask.push(i as u32),
                _ => {
                    self.error(idx.span, format!("a shuffle index must be a constant in -1..{}", 2 * n));
                    return None;
                }
            }
        }
        self.shuffle_result(a, b, &vty, mask, span)
    }

    /// GCC's `__builtin_shuffle(a, mask)` / `__builtin_shuffle(a, b, mask)`:
    /// lane `k` is lane `mask[k]` of `a` (or of `a ++ b`), the indices taken
    /// modulo the number of source lanes. Only a constant mask (a
    /// compound-literal vector of constants) is supported.
    pub(super) fn builtin_shuffle(&mut self, ctx: &mut FnCtx, args: &[Expr], span: Span) -> Option<TExpr> {
        let (mask_expr, b) = match args {
            [_, m] => (m, None),
            [_, b, m] => (m, Some(b)),
            _ => {
                self.error(span, "__builtin_shuffle takes two or three arguments");
                return None;
            }
        };
        let (a, b, vty) = self.shuffle_sources(ctx, &args[0], b, span)?;
        let n = vty.vector_parts().map_or(0, |(_, n)| n);
        let sources = if args.len() == 3 { 2 * n } else { n };
        let Some(lanes) = self.const_vector_lanes(mask_expr) else {
            self.error(mask_expr.span, "only a constant shuffle mask (a vector literal of constants) is supported");
            return None;
        };
        let mask = lanes.iter().map(|&i| (i.rem_euclid(i128::from(sources))) as u32).collect();
        self.shuffle_result(a, b, &vty, mask, span)
    }

    /// The source vectors of a shuffle: `a`, and `b` (or `a` again when absent
    /// — any lane of the second half is then never picked), of one vector type.
    fn shuffle_sources(
        &mut self,
        ctx: &mut FnCtx,
        a: &Expr,
        b: Option<&Expr>,
        span: Span,
    ) -> Option<(TExpr, TExpr, CType)> {
        let a = self.check_rvalue(ctx, a)?;
        if !a.ty.is_vector() {
            self.error(span, "a shuffle needs vector operands");
            return None;
        }
        let vty = a.ty.clone();
        let b = match b {
            Some(b) => {
                let b = self.check_rvalue(ctx, b)?;
                if b.ty != vty {
                    self.error(span, format!("shuffled vectors must have one type, found '{vty}' and '{}'", b.ty));
                    return None;
                }
                b
            }
            None => {
                let elem = vty.vector_parts().expect("a vector type").0.clone();
                let zero = TExpr::new(TExprKind::Const(0), CType::int(), span);
                let zero = self.convert(zero, &elem);
                TExpr::new(TExprKind::VecSplat(Box::new(zero)), vty.clone(), span)
            }
        };
        Some((a, b, vty))
    }

    fn shuffle_result(&mut self, a: TExpr, b: TExpr, vty: &CType, mask: Vec<u32>, span: Span) -> Option<TExpr> {
        if !mask.len().is_power_of_two() {
            self.error(span, "a shuffle must produce a power-of-two number of lanes");
            return None;
        }
        let (elem, _) = vty.vector_parts().expect("a vector type");
        let rty = CType::Vector(Box::new(elem.clone()), mask.len() as u32);
        Some(TExpr::new(TExprKind::VecShuffle(Box::new(a), Box::new(b), mask), rty, span))
    }

    /// The constant lanes of a vector literal `(T){c0, c1, ...}` (missing lanes
    /// are 0), through casts; `None` if `e` is not one.
    fn const_vector_lanes(&self, e: &Expr) -> Option<Vec<i128>> {
        match &e.kind {
            ExprKind::Cast(_, inner) => self.const_vector_lanes(inner),
            ExprKind::CompoundLiteral(ty, init) => {
                let (_, n) = ty.unqual().vector_parts()?;
                let Init::List(items) = &**init else { return None };
                let mut lanes = vec![0i128; n as usize];
                for (i, item) in items.iter().enumerate() {
                    if !item.designators.is_empty() || i >= lanes.len() {
                        return None;
                    }
                    let Init::Expr(x) = &item.init else { return None };
                    lanes[i] = self.const_eval(x)?;
                }
                Some(lanes)
            }
            _ => None,
        }
    }

    /// `__builtin_convertvector(v, T)`: each lane converted (as by a cast) to
    /// the element type of the vector type `T`, which has `v`'s lane count.
    pub(super) fn check_convertvector(&mut self, ctx: &mut FnCtx, v: &Expr, ty: &CType, span: Span) -> Option<TExpr> {
        let v = self.check_rvalue(ctx, v)?;
        let ty = ty.unqual().clone();
        match (v.ty.vector_parts(), ty.vector_parts()) {
            (Some((_, n)), Some((_, m))) if n == m => {
                if v.ty == ty {
                    return Some(v);
                }
                Some(TExpr::new(TExprKind::VecConvert(Box::new(v)), ty, span))
            }
            _ => {
                self.error(span, "__builtin_convertvector converts between vector types of the same length");
                None
            }
        }
    }
}
