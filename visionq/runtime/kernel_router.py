from typing import Dict, Any, Tuple
from .feature_extractor import FeatureExtractor
from .policy_model import PolicyModel
from ..core.context import AttentionContext

class KernelRouter:
    """
    Policy-driven Execution Planner for GPU Attention.
    Decides algorithm, block structure, and sparsity strategy.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.policy_model = PolicyModel()

    def plan_execution(self, context: AttentionContext) -> Tuple[str, Dict[str, Any]]:
        """
        Plans the attention execution.
        Returns: (kernel_name, autotuned_parameters)
        """
        features = self.feature_extractor.extract(context)
        scores = self.policy_model.score_kernels(features)

        kernel_name = self.policy_model.get_best_kernel(scores)

        # Kernel Parameter Autotuning
        params = self.autotune(kernel_name, features)

        return kernel_name, params

    def autotune(self, kernel_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptively determines optimal block sizes and windows.
        """
        N = features["sequence_length"]

        # Defaults
        params = {
            "block_size": 32,
            "window_size": 7,
            "stride": 1
        }

        # Adaptive Tiling based on sequence length
        if N > 8192:
            params["block_size"] = 128
        elif N > 2048:
            params["block_size"] = 64

        # Hardware specific tuning
        if features["device_type"] == "cuda":
            capability = features["compute_capability"]
            if capability[0] >= 8: # Ampere+
                params["block_size"] = 128 # Larger shared memory

        return params
