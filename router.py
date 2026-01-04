"""Simple command router for voice assistant.

Maps keyword intents to handler callables. Handlers can return a response string
that will be spoken via the TTS engine.
"""

from typing import Callable, Dict, Optional

class CommandRouter:
    def __init__(self):
        self.handlers: Dict[str, Callable[[str], Optional[str]]] = {}

    def register(self, intent: str, handler: Callable[[str], Optional[str]]) -> None:
        """Register a handler for a given intent keyword.

        The handler receives the full transcript and may return a response string
        that will be spoken back to the user.
        """
        self.handlers[intent.lower()] = handler

    def route(self, transcript: str) -> Optional[str]:
        lowered = transcript.lower()
        for intent, fn in self.handlers.items():
            if intent in lowered:
                return fn(transcript)
        return None
