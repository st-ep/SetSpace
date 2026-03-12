import torch


def calculate_l2_relative_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    error_norm = torch.norm(y_pred - y_true, dim=1)
    true_norm = torch.norm(y_true, dim=1)
    return (error_norm / (true_norm + 1e-8)).mean()
