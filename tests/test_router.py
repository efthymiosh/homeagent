"""Tests for the CommandRouter."""

import pytest
from homeagent.router import CommandRouter


def test_route_matches_intent():
    router = CommandRouter()
    called = {}

    def handler(text: str):
        called["text"] = text
        return "handled"

    router.register("light", handler)
    result = router.route("please turn on the LIGHT now")
    assert result == "handled"
    assert called["text"] == "please turn on the LIGHT now"


def test_route_no_match_returns_none():
    router = CommandRouter()
    assert router.route("nothing matches here") is None
