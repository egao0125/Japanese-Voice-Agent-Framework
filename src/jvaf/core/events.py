"""Typed async event bus for cross-component communication."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Awaitable, Callable

EventHandler = Callable[..., Awaitable[None]]


class EventBus:
    """Async event bus for session-level events.

    Used for events that don't flow through the pipeline (e.g.,
    lifecycle hooks, metrics, session state changes).
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register a handler for an event."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Remove a handler."""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all registered handlers."""
        for handler in self._handlers.get(event, []):
            await handler(**kwargs)
