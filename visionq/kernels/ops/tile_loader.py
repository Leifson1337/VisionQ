import torch


class TileLoader:
    """
    Abstraction for HBM -> SRAM / Register data movement.
    Designed to minimize bandwidth usage and maximize shared memory reuse.
    """

    @staticmethod
    def load_spatial_tile(
        x: torch.Tensor, tile_coords: tuple[int, int], tile_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Loads a 2D spatial block from a structured tensor.
        x: (B, T, H, W, C)
        """
        ty, tx = tile_coords
        th, tw = tile_size
        return x[:, :, ty : ty + th, tx : tx + tw, :]

    @staticmethod
    def load_temporal_window(x: torch.Tensor, center_t: int, window_size: int) -> torch.Tensor:
        """Loads a sliding window over the temporal dimension."""
        start = max(0, center_t - window_size // 2)
        end = min(x.shape[1], center_t + window_size // 2 + 1)
        return x[:, start:end, ...]
