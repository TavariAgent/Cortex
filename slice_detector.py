"""
SliceDetector: turn an expression string into a graph of ‘slices’.

Key ideas
=========

• A *slice* is either:
    1. a parenthesised sub-expression "( … )"
    2. or the full top-level expression when no outer parens exist.

• Each slice gets:
    - id        : "slice_{depth}_{index}"
    - text      : raw substring (without wrapping parens)
    - depth     : nesting level (0 == outermost)
    - start/end : indices in original string (optional; nice for highlights)

• The output is a SliceGraph:
      SliceNode {
          id, depth, text,
          children: [child_ids …],
          parent  : parent_id | None
      }

Deepest slices (max depth) are the ones we can evaluate first in parallel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SliceNode:
    id: str
    depth: int
    text: str
    parent: str | None
    children: List[str] = field(default_factory=list)
    # (Optional) byte-offsets in original string
    span: Tuple[int, int] | None = None


class SliceGraph:
    """Thin wrapper around {id: SliceNode} dict."""
    def __init__(self, nodes: Dict[str, SliceNode]):
        self.nodes = nodes

    def deepest_depth(self) -> int:
        return max(n.depth for n in self.nodes.values())

    def leaves(self) -> List[SliceNode]:
        max_d = self.deepest_depth()
        return [n for n in self.nodes.values() if n.depth == max_d]


def build_slice_graph(expr: str) -> SliceGraph:
    """Single-pass stack parser – O(n) time, O(n) space."""
    stack: List[Tuple[int, int]] = []  # [(start_idx, depth)]
    nodes: Dict[str, SliceNode] = {}
    depth = 0
    slice_index_at_depth: Dict[int, int] = {}

    def new_id(d: int) -> str:
        slice_index_at_depth[d] = slice_index_at_depth.get(d, 0) + 1
        return f"slice_{d}_{slice_index_at_depth[d]}"

    i = 0
    while i < len(expr):
        if expr[i] == '(':
            stack.append((i, depth))
            depth += 1
            i += 1
        elif expr[i] == ')':
            if not stack:
                raise ValueError("Unmatched ')' in expression")
            start, parent_depth = stack.pop()
            slice_depth = parent_depth  # depth before popping
            slice_text = expr[start + 1:i]
            sid = new_id(slice_depth)
            parent_id = new_id(slice_depth - 1) if slice_depth > 0 else None
            node = SliceNode(
                id=sid, depth=slice_depth, text=slice_text,
                parent=parent_id, span=(start, i)
            )
            nodes[sid] = node
            # Register parent <-> child
            if parent_id:
                nodes.setdefault(parent_id, SliceNode(
                    id=parent_id, depth=slice_depth - 1,
                    text='', parent=None)).children.append(sid)
            depth -= 1
            i += 1
        else:
            i += 1

    # If we never encountered '(', treat entire expr as depth-0 slice
    if not nodes:
        sid = "slice_0_1"
        nodes[sid] = SliceNode(id=sid, depth=0, text=expr, parent=None)
    return SliceGraph(nodes)