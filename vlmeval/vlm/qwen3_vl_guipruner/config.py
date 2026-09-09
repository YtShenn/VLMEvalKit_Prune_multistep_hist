"""Configuration for the paper-faithful GUIPruner-reproduction."""

from __future__ import annotations

from dataclasses import dataclass


def resolve_current_keep_ratio(
    *, history_original_tokens: int, current_original_tokens: int,
    history_keep_ratio: float, overall_keep_ratio: float,
    history_tokens_after_tar: int | None = None,
    baseline_prompt_tokens: int | None = None,
    fixed_prompt_tokens_after_tar: int | None = None,
    infeasible_policy: str = "strict",
) -> float:
    """Derive current-frame SSP ``mu`` from TAR and an optional global budget.

    When exact prompt counts are supplied, ``eta`` has the experiment-facing
    end-to-end meaning: ``N_after / N_before_baseline``.  TAR leaves a fixed
    prefix ``F`` (text, special tokens, and TAR history) and SSP retains
    ``mu*C`` current visual tokens, hence ``mu=(eta*N_before_baseline-F)/C``.
    The visual-only fallback is retained for pure arithmetic callers.
    """
    if infeasible_policy not in {"strict", "clip"}:
        raise ValueError("infeasible_policy must be 'strict' or 'clip'.")
    if history_original_tokens < 0 or current_original_tokens <= 0:
        raise ValueError("Token counts must satisfy H >= 0 and C > 0.")
    if (baseline_prompt_tokens is None) != (fixed_prompt_tokens_after_tar is None):
        raise ValueError("baseline_prompt_tokens and fixed_prompt_tokens_after_tar must be provided together.")
    if baseline_prompt_tokens is not None:
        baseline = float(baseline_prompt_tokens)
        fixed = float(fixed_prompt_tokens_after_tar)
        if baseline <= 0 or fixed < 0:
            raise ValueError("Prompt token counts must satisfy baseline > 0 and fixed >= 0.")
        current_keep = overall_keep_ratio * baseline - fixed
        lower = fixed / baseline
        upper = (fixed + current_original_tokens) / baseline
    else:
        history_tar = (
            float(history_tokens_after_tar)
            if history_tokens_after_tar is not None
            else history_keep_ratio * history_original_tokens
        )
        if history_tar < 0:
            raise ValueError("history_tokens_after_tar must be >= 0.")
        stage_input = history_tar + current_original_tokens
        current_keep = overall_keep_ratio * stage_input - history_tar
        lower = history_tar / stage_input
        upper = 1.0
    if current_keep < -1e-6 or current_keep > current_original_tokens + 1e-6:
        if infeasible_policy == "clip":
            return min(1.0, max(0.0, current_keep / current_original_tokens))
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
    global_budget_policy: str = "strict"  # strict | clip, used only when eta is set
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
        if self.global_budget_policy not in {"strict", "clip"}:
            raise ValueError("global_budget_policy must be 'strict' or 'clip'.")
        if self.prune_layer != 2:
            raise ValueError("The paper-faithful GUIPruner-reproduction fixes prune_layer=2 (one-based).")
