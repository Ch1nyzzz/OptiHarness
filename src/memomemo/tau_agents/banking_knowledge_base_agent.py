from __future__ import annotations

from typing import Optional

from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate
from tau_agent_runtime.base_agent import HalfDuplexAgent, ValidAgentInputMessage


SYSTEM_PROMPT = """\
You are a banking customer-service agent in the tau banking_knowledge domain.

Follow the domain policy exactly. The policy alone is not complete: most product,
fee, eligibility, procedure, and tool-use details are in the knowledge base.

Operating rules:
- Search the knowledge base before applying any product rule, eligibility rule,
  procedure, exception, promotion, fee, rate, or internal protocol.
- Before any write/mutating tool call, verify the customer identity, current
  account or transaction state, eligibility, blockers, and the documented tool
  procedure.
- If a tool is mentioned in retrieved documentation but is not obvious from the
  initial tool list, search for that tool's documentation before using it.
- Do not guess policy. If evidence is incomplete, retrieve more or ask the user
  for the missing required information.
- Keep user-facing replies concise and action-oriented.

<domain_policy>
{domain_policy}
</domain_policy>
"""

PLAN_PROMPT = """\
Privately decide the next step for this banking support turn.

Return a concise plan covering:
1. The user's active request(s).
2. Knowledge documents or tool docs still needed.
3. Customer/account/transaction facts still needed.
4. Whether the next action should be KB search, a read tool, a write tool, or a
   user-facing response.
"""

ACT_PROMPT = """\
Use the plan below to take exactly the next useful action.

If documentation or state is missing, call the appropriate search/read tool.
If a policy-compliant mutation is ready, call the correct write tool with the
documented arguments. If no tool is needed, respond to the user.

<private_plan>
{plan}
</private_plan>
"""


class BankingKnowledgeBaseAgentState:
    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
        plans: list[str],
    ) -> None:
        self.system_messages = system_messages
        self.messages = messages
        self.plans = plans


class BankingKnowledgeBaseAgent(HalfDuplexAgent[BankingKnowledgeBaseAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> BankingKnowledgeBaseAgentState:
        system = SYSTEM_PROMPT.format(domain_policy=self.domain_policy)
        return BankingKnowledgeBaseAgentState(
            system_messages=[SystemMessage(role="system", content=system)],
            messages=list(message_history) if message_history else [],
            plans=[],
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: BankingKnowledgeBaseAgentState,
    ) -> tuple[AssistantMessage, BankingKnowledgeBaseAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        plan = self._plan(state)
        state.plans.append(plan)
        response = self._act(state, plan)
        state.messages.append(response)
        return response, state

    def _plan(self, state: BankingKnowledgeBaseAgentState) -> str:
        messages = state.system_messages + state.messages + [
            UserMessage(role="user", content=PLAN_PROMPT)
        ]
        response = generate(
            model=self.llm,
            tools=[],
            messages=messages,
            call_name="banking_knowledge_plan",
            **self.llm_args,
        )
        return str(response.content or "")

    def _act(
        self,
        state: BankingKnowledgeBaseAgentState,
        plan: str,
    ) -> AssistantMessage:
        messages = state.system_messages + state.messages + [
            UserMessage(role="user", content=ACT_PROMPT.format(plan=plan))
        ]
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="banking_knowledge_act",
            **self.llm_args,
        )


def create_banking_knowledge_base_agent(tools, domain_policy, **kwargs):
    return BankingKnowledgeBaseAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args"),
    )
