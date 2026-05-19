# Router

The router is deterministic and heuristic-based. It scores registered backends
from sequence length, modality, spatial token count, temporal length, device type
and CUDA memory pressure when CUDA is available.

Routing decisions are inspectable through `AttentionDispatcher.last_decision`.
The router never silently selects an unregistered backend.
