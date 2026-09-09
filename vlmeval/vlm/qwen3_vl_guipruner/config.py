"""Configuration for the paper-faithful GUIPruner-reproduction."""

from __future__ import annotations

from dataclasses import dataclass


def resolve_current_keep_ratio(
    *, history_original_tokens: int, current_original_tokens: int,
    history_keep_ratio: float, overall_keep_ratio: float,
) -> float:
    """Derive current-frame SSP ``mu`` from history and global budgets.

    K_history=lambda*H and K_all=eta*(H+C), therefore
    mu=(K_all-K_history)/C.  Invalid pairs have no exact token allocation and
    are rejected rather than silently clipped.
    """
    if history_original_tokens < 0 or current_original_tokens <= 0:
        raise ValueError("Token counts must satisfy H >= 0 and C > 0.")
    history_kept = history_keep_ratio * history_original_tokens
    all_kept = overall_keep_ratio * (history_original_tokens + current_original_tokens)
    current_keep = all_kept - history_kept
    if current_keep < -1e-6 or current_keep > current_original_tokens + 1e-6:
        lower = history_kept / (history_original_tokens + current_original_tokens)
        upper = (history_kept + current_original_tokens) / (history_original_tokens + current_original_tokens)
        raise ValueError(
            "Infeasible GUIPruner global budget: overall_keep_ratio must lie in "
            f"[{lower:.6f}, {upper:.6f}] for the selected history_keep_ratio."
        )
    return min(1.0, max(0.0, current_keep / current_original_tokens))


@dataclass(frozen=True)
class GUIPrunerConfig:
    """Paper parameters; ``prune_layer=2`` is one-based, as in the paper."""

    history_steps: int = 4
    history_keep_ratio: float = 0.40  # lambda
    temporal_decay: float = 0.20  # gamma
    current_keep_ratio: float = 0.75  # mu
    background_saliency: float = 0.30  # rho
    overall_keep_ratio: float | None = None  # eta, history + current total
    prune_layer: int = 2  # one-based decoder layer, never zero-based

    def __post_init__(self) -> None:
        if not 0 <= self.history_steps <= 4:
            raise ValueError("GUIPruner-reproduction supports history_steps in [0, 4].")
        for name in ("history_keep_ratio", "temporal_decay", "current_keep_ratio", "background_saliency"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")
        if self.overall_keep_ratio is not None and not 0.0 < float(self.overall_keep_ratio) <= 1.0:
            raise ValueError("overall_keep_ratio must be in (0, 1] when provided.")
        if self.prune_layer != 2:
            raise ValueError("The paper-faithful GUIPruner-reproduction fixes prune_layer=2 (one-based).")
