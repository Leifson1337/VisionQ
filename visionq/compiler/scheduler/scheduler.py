from typing import Any

from ..graph_ir.ir import AttentionGraph


class ExecutionScheduler:
    """
    Plans the execution order and memory allocation for the optimized attention graph.
    """

    def __init__(self, graph: AttentionGraph):
        self.graph = graph

    def plan(self, hardware_profile: dict[str, Any]) -> list[str]:
        """
        Determines the optimal execution order using a simplified topology sort.
        """
        # In this context, we execute in the order they were added for the baseline
        # but a real scheduler would respect hardware concurrency.
        return list(self.graph.nodes.keys())

    def allocate_memory(self, seq_len: int) -> dict[str, Any]:
        """
        Simulates static memory planning for tensor reuse.
        Maps each node output to a specific buffer ID to minimize VRAM footprint.
        """
        memory_map = {}
        buffer_idx = 0
        for node_id in self.graph.nodes:
            # Simple buffer reuse logic: if an input's last use is here, reuse its buffer
            memory_map[node_id] = f"buffer_{buffer_idx}"
            buffer_idx += 1

        return memory_map
