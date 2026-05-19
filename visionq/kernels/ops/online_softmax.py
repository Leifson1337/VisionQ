import torch


class OnlineSoftmax:
    """
    Numerically stable streaming normalization (FlashAttention principle).
    Allows computing softmax over tiles without materializing the full NxN matrix.
    """

    @staticmethod
    def update(
        prev_max: torch.Tensor, prev_sum: torch.Tensor, current_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the running max and sum exponentials.

        Args:
            prev_max: Running max score (B, H, block_q, 1)
            prev_sum: Running exponential sum (B, H, block_q, 1)
            current_scores: New partial QK^T scores (B, H, block_q, block_k)

        Returns:
            new_max, new_sum, exp_weights
        """
        current_max = torch.max(current_scores, dim=-1, keepdim=True)[0]
        new_max = torch.max(prev_max, current_max)

        # Rescale previous sum and current exponentials to the new max
        # FlashAttention technique: exp(x - new_max) = exp(x - old_max) * exp(prev_max - new_max)
        exp_prev = torch.exp(prev_max - new_max)
        exp_curr = torch.exp(current_scores - new_max)

        new_sum = prev_sum * exp_prev + torch.sum(exp_curr, dim=-1, keepdim=True)

        return new_max, new_sum, exp_curr
