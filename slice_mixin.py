from slice_detector import build_slice_graph
from priority_rules import precedence_of

class SliceMixin:
    """
    Plug-in for any MathEngine subclass that wants structure-aware compute.
    Assumes the subclass provides:
        • _evaluate_atom(slice_text)   -> result (mp.mpf, SymPy, etc.)
        • _add_traceback(...)
    """

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