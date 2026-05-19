import torch


def neighborhood_mask_cpu(T, H, W, window_size, device):
    """
    CPU-optimized neighborhood mask generation (baseline).
    """
    N = T * H * W
    mask = torch.full((N, N), float("-inf"), device=device)

    coords_t = torch.arange(T, device=device)
    coords_h = torch.arange(H, device=device)
    coords_w = torch.arange(W, device=device)
    grid_t, grid_h, grid_w = torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij")
    grid = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(N, 3)

    dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0))
    in_window = (
        (dist[:, :, 0] <= window_size // 2)
        & (dist[:, :, 1] <= window_size // 2)
        & (dist[:, :, 2] <= window_size // 2)
    )

    mask[in_window] = 0
    return mask
