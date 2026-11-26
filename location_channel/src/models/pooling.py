import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: (B,T,C) or (B,T,F); mask: (B,T)
    mask = mask.unsqueeze(-1)
    x = x * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return x.sum(dim=1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    neg_inf = torch.full_like(x, -1e9)
    x = torch.where(mask > 0, x, neg_inf)
    return x.max(dim=1).values


def masked_logsumexp(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    neg_inf = torch.full_like(x, -1e9)
    x = torch.where(mask > 0, x, neg_inf)
    return torch.logsumexp(x, dim=1)


def pool(x: torch.Tensor, mask: torch.Tensor, mode: str = 'logsumexp') -> torch.Tensor:
    if mode == 'mean':
        return masked_mean(x, mask)
    if mode == 'max':
        return masked_max(x, mask)
    if mode == 'logsumexp':
        return masked_logsumexp(x, mask)
    raise ValueError(f"Unknown pooling mode: {mode}")
