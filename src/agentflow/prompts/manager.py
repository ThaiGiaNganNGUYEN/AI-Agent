"""Prompt versions, weighted A/B selection, and activation."""

import random
import secrets

from agentflow.errors import NotFoundError
from agentflow.loading import load_yaml


class PromptManager:
    def __init__(self, seed: int | None = None) -> None:
        loaded = load_yaml("prompts.yaml")
        self.prompts: dict = loaded["prompts"]
        self.active: dict[str, str] = {}
        self.seed = seed if seed is not None else secrets.randbits(32)

    def render_system(self, name: str, assignment_key: str = "") -> tuple[str, str]:
        spec = self.prompts.get(name)
        if spec is None:
            raise NotFoundError(f"Unknown prompt {name}")
        if name in self.active:
            version = self.active[name]
        else:
            version = _weighted(spec["versions"], random.Random(f"{self.seed}|{name}|{assignment_key}"))
        system = spec["versions"][version]["system"].strip()
        return system, version

    def activate(self, name: str, version: str) -> None:
        spec = self.prompts.get(name)
        if spec is None or version not in spec["versions"]:
            raise NotFoundError(f"Unknown prompt version {name}/{version}")
        self.active[name] = version

    def describe(self) -> list[dict]:
        described = []
        for name, spec in self.prompts.items():
            described.append(
                {
                    "name": name,
                    "active": self.active.get(name),
                    "versions": [
                        {"version": version, "weight": body.get("weight", 1)}
                        for version, body in spec["versions"].items()
                    ],
                }
            )
        return described


def _weighted(versions: dict, rng: random.Random) -> str:
    names = list(versions)
    weights = [float(versions[name].get("weight", 1)) for name in names]
    total = sum(weights) or 1
    pick = rng.random() * total
    running = 0.0
    for name, weight in zip(names, weights):
        running += weight
        if pick < running:
            return name
    return names[-1]
