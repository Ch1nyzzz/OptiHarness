"""Local half-duplex agent base compatible with tau2-bench orchestration.

This intentionally mirrors only the small surface tau2's text orchestrator
uses.  Candidate snapshots can evolve this file together with the agent code
without modifying the external tau2-bench checkout.
"""

from __future__ import annotations

import random
from typing import Any, Generic, TypeVar


AgentState = TypeVar("AgentState")
ValidAgentInputMessage = Any


class AgentError(Exception):
    """Generic agent error."""


class HalfDuplexAgent(Generic[AgentState]):
    """Minimal turn-based agent contract used by tau2-bench."""

    def __init__(self, tools: list[Any], domain_policy: str) -> None:
        self.tools = tools
        self.domain_policy = domain_policy
        self.seed: int | None = None
        self.rng = random.Random()

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentState,
    ) -> tuple[Any, AgentState]:
        """Generate the next assistant message."""

        raise NotImplementedError

    def get_init_state(
        self,
        message_history: list[Any] | None = None,
    ) -> AgentState:
        """Return the initial agent state."""

        raise NotImplementedError

    def stop(
        self,
        message: ValidAgentInputMessage | None = None,
        state: AgentState | None = None,
    ) -> None:
        """Release resources held by the agent."""

        return None

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        """Return whether an assistant message asks the orchestrator to stop."""

        content = getattr(message, "content", None)
        return isinstance(content, str) and content.strip() == "###STOP###"

    def set_seed(self, seed: int) -> None:
        """Set deterministic local randomness for candidate logic."""

        self.seed = int(seed)
        self.rng.seed(self.seed)
