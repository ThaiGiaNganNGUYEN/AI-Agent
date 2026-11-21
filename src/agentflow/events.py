"""In-process observer bus for agent decisions and execution paths."""

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger("agentflow.events")


@dataclass
class Event:
    type: str
    run_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Publish/subscribe hub.

    Handlers are synchronous and must return quickly. A failing handler is
    logged and does not stop the workflow that published the event.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        handlers = list(self._subscribers.get(event.type, []))
        handlers.extend(self._subscribers.get("*", []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("event handler failed for %s", event.type)
