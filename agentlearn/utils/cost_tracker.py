"""Tracks API spend per learning cycle."""

from typing import Optional

from .logging import get_logger

logger = get_logger("cost_tracker")

# Per-token pricing (USD) — input / output per 1M tokens
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    pricing = PRICING.get(model)
    if pricing is None:
        # Try prefix matching
        for key, val in PRICING.items():
            if model.startswith(key.rsplit("-", 1)[0]):
                pricing = val
                break
    if pricing is None:
        logger.warning(f"Unknown model for pricing: {model}. Using $0.00.")
        return 0.0
    input_cost, output_cost = pricing
    return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000


class CostTracker:
    """Tracks API spend across learning cycles."""

    def __init__(self, budget_per_day: Optional[float] = None):
        self.total_cost = 0.0
        self.budget = budget_per_day
        self.costs_by_component: dict[str, float] = {
            "analysis": 0.0,
            "validation": 0.0,
            "outcome_signal": 0.0,
            "injection": 0.0,
            "embedding": 0.0,
        }

    def record(self, component: str, cost: float) -> None:
        """Record a cost for a component."""
        self.total_cost += cost
        if component not in self.costs_by_component:
            self.costs_by_component[component] = 0.0
        self.costs_by_component[component] += cost

    def can_spend(self, estimated_cost: float) -> bool:
        """Check if we can afford this spend within the daily budget."""
        if self.budget is None:
            return True
        return (self.total_cost + estimated_cost) <= self.budget

    def summary(self) -> dict:
        """Return a cost breakdown."""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "budget_remaining": round(self.budget - self.total_cost, 6) if self.budget else None,
            "by_component": {k: round(v, 6) for k, v in self.costs_by_component.items() if v > 0},
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.total_cost = 0.0
        for key in self.costs_by_component:
            self.costs_by_component[key] = 0.0
