"""Ordered routing rules. The highest-priority match wins."""

from dataclasses import dataclass

from agentflow.errors import RoutingError
from agentflow.loading import load_yaml
from agentflow.schemas import RoutingDecision


@dataclass
class RoutingRule:
    name: str
    priority: int
    domain: str
    keywords: list[str]
    workflow: str

    def matches(self, message: str, domain: str) -> bool:
        if self.domain not in {"*", domain}:
            return False
        if not self.keywords:
            return True
        text = message.lower()
        return any(keyword.lower() in text for keyword in self.keywords)


class PolicyRouter:
    """Strategy collection: each rule decides whether it owns a request."""

    def __init__(self, rules: list[RoutingRule]) -> None:
        self.rules = sorted(rules, key=lambda rule: rule.priority, reverse=True)

    @classmethod
    def from_package(cls) -> "PolicyRouter":
        loaded = load_yaml("policies.yaml")
        rules = [
            RoutingRule(
                name=item["name"],
                priority=int(item["priority"]),
                domain=item["domain"],
                keywords=list(item.get("keywords") or []),
                workflow=item["workflow"],
            )
            for item in loaded["rules"]
        ]
        return cls(rules)

    def route(self, message: str, domain: str) -> RoutingDecision:
        matched = [rule for rule in self.rules if rule.matches(message, domain)]
        if not matched:
            raise RoutingError(f"No routing rule matched domain {domain!r}")
        chosen = matched[0]
        if chosen.keywords:
            reason = f"Matched rule {chosen.name} (priority {chosen.priority}) from the message text."
        else:
            reason = f"No specific rule matched, so {chosen.name} was selected."
        return RoutingDecision(workflow=chosen.workflow, rule=chosen.name, reason=reason)
