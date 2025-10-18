from mpmath import mp
import sympy as sp
from sympy import symbols
from sympy.core.function import AppliedUndef

from slice_detector import build_slice_graph
from priority_rules import precedence_of, PRIORITY
from utils.trace_helpers import add_traceback


# Unevaluated node types that normally require .doit() or further processing
UNEVALUATED = (
    sp.Derivative,
    sp.Integral,
    sp.Limit,
    sp.Sum,
    sp.Product,
    sp.Subs,
)

PIECEWISE = sp.Piecewise

x = symbols('x')
g = sp.Function('g')(x)
f = sp.Function('f')(x)
x = sp.Symbol('x')

class SliceMixin:
    """
    Plug-in for any MathEngine subclass that wants structure-aware compute.
    Assumes the subclass provides:
        • _evaluate_atom(slice_text)   -> result (mp.mpf, SymPy, etc.)
        • _add_traceback(...)
    """

    # -------------------------------------------------------------- #
    # Trace helper
    # -------------------------------------------------------------- #
    def _add_traceback(self, step: str, info: str):
        add_traceback(self, step, info)

    @staticmethod
    def needs_more_work(expr: sp.Expr, *, require_numeric=False) -> bool:
        """Return True if expr needs more work to compute.
        - If require_numeric=True, also require the result to be a numeric value (no special functions left).
        """
        # 1) Any unevaluated calculus objects left?
        if expr.has(*UNEVALUATED):
            return True

        # 2) Any undefined functions like f(x)?
        if expr.has(AppliedUndef):
            return True

        # 3) Piecewise: mark dirty if any nontrivial conditions remain
        if expr.has(PIECEWISE):
            for pw in expr.atoms(PIECEWISE):
                # If any condition (other than True) still contains symbols, we consider it dirty
                if any((cond is not True) and cond.free_symbols for cond, _ in pw.args):
                    return True
            # If you want to treat any Piecewise as dirty regardless, uncomment:
            # return True

        # 4) Symbolic variables anywhere in the tree?
        if expr.free_symbols:  # more inclusive than isinstance(expr, sp.Symbol)
            return True

        # 5) Numeric requirement: if you demand a concrete numeric value (not just exact expression)
        if require_numeric:
            # If it’s a pure number, it’s fine
            if expr.is_Number:
                return False
            # SymPy numeric constants (pi, E, I) are NumberSymbols; treat as dirty if you need a float
            if expr.is_NumberSymbol:
                return True
            # Special functions of numeric arguments, e.g., sin(1), erf(2) – treat as dirty
            # if you want a float result rather than an exact form.
            if not expr.free_symbols and expr.has(sp.Function):
                return True

        return False

    # ──────────────────────────────────────────────────────────────
    #  Generic part-ordering helper (PEMDAS, left-to-right)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _set_part_order(parts, *, apply_at_start=True, apply_after_return=False):
        """
        Sort *parts* by operator-precedence (highest → lowest) and
        stable left-to-right within the same precedence tier.

        Parameters
        ----------
        parts : Iterable[Any]
            Anything that carries either an attribute ``.op`` **or**
            embeds the operator symbol in its ``str(part)``.
        apply_at_start / apply_after_return : bool
            Use the same helper before computation, after computation,
            or both – mirrors the original API so engines don’t break.
        """

        # Fast exit – keep order if no sort requested
        if not (apply_at_start or apply_after_return):
            return list(parts)

        def _precedence(part):
            """Return integer precedence; larger means tighter binding."""
            # 1) explicit attr .op set by engine
            op = getattr(part, 'op', None)
            # 2) best-effort scan in the textual representation
            if op is None:
                for sym in PRIORITY:
                    if sym in str(part):
                        op = sym
                        break
            return precedence_of(op) if op is not None else 0

        # Enumerate to preserve original position as a stable tiebreaker
        enumerated = list(enumerate(parts))
        enumerated.sort(key=lambda kv: (-_precedence(kv[1]), kv[0]))
        return [part for _, part in enumerated]

    @staticmethod
    def _linear_tokenize(flat_expr: str):
        """Simple left-to-right tokenizer for numbers and operators (no parens)."""
        out, cur = [], ''
        for ch in flat_expr:
            if ch.isdigit() or ch == '.':
                cur += ch
            elif ch in '+-*/^':
                if cur:
                    out.append(cur); cur = ''
                out.append(ch)
            elif ch == ' ':
                continue
            elif ch == ',':
                # Comma separates function arguments; we ignore it for
                # pure arithmetic tokenising.
                continue
            else:
                raise ValueError(f'Unexpected char {ch}')
        if cur:
            out.append(cur)
        return out

    # -------------------------------------------------------------- #
    # SliceMixin atom evaluator (needed but rarely used here)
    # -------------------------------------------------------------- #
    def _evaluate_atom(self, slice_text: str):
        # Defer to BasicArithmeticEngine style evaluation if someone
        # sends an arithmetic slice to us.
        tokens = self._linear_tokenize(slice_text)
        for op_level in [4, 3, 2]:
            i = 0
            while i < len(tokens):
                if precedence_of(tokens[i]) == op_level:
                    a = mp.mpf(tokens[i - 1]); b = mp.mpf(tokens[i + 1])
                    val = (mp.power if tokens[i] == '^'
                           else a.__mul__ if tokens[i] == '*'
                           else a.__truediv__ if tokens[i] == '/'
                           else a.__add__ if tokens[i] == '+'
                           else a.__sub__)(b)
                    tokens = tokens[:i - 1] + [str(val)] + tokens[i + 2:]
                else:
                    i += 1
        if len(tokens) != 1:
            raise ValueError(f'Could not resolve slice {tokens}')
        return mp.mpf(tokens[0])

    def _compute_slice_parallel(self, expr: str):
        graph = build_slice_graph(expr)
        leaves = graph.leaves()

        # 1.  Evaluate deepest slices (parallel not shown here; can reuse _compute_parts_parallel)
        slice_results = {}
        for node in leaves:
            res = self._evaluate_atom(node.text)
            slice_results[node.id] = res
            self._add_traceback('slice_eval', f'{node.id}: {node.text} -> {res}')

        # 2.  Iteratively fold upward
        depth_levels = sorted({n.depth for n in graph.nodes.values()}, reverse=True)
        for d in depth_levels:
            for n in [n for n in graph.nodes.values() if n.depth == d]:
                if not n.children:         # already done (leaf)
                    continue
                # Replace children text with their computed values
                folded = n.text
                for cid in n.children:
                    folded = folded.replace(f'({graph.nodes[cid].text})',
                                            str(slice_results[cid]))
                # Evaluate this combined slice
                slice_results[n.id] = self._evaluate_atom(folded)
                self._add_traceback('slice_fold',
                                    f'{n.id}: {folded} -> {slice_results[n.id]}')

        # The root slice is at depth 0
        root_id = [n.id for n in graph.nodes.values() if n.depth == 0][0]
        return slice_results[root_id]