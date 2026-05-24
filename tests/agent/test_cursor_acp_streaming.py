"""Regression tests for live ACP chunk forwarding in CursorACPClient."""

from __future__ import annotations

import io
import json
import unittest

from agent.cursor_acp_client import CursorACPClient


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CursorACPStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CursorACPClient(acp_cwd="/tmp")
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.text_deltas: list[str] = []
        self.reasoning_deltas: list[str] = []
        self.first_delta_count = 0

        def on_first() -> None:
            self.first_delta_count += 1

        self.client.set_stream_callbacks(
            on_text_delta=self.text_deltas.append,
            on_reasoning_delta=self.reasoning_deltas.append,
            on_first_delta=on_first,
        )

    def _dispatch(self, message: dict) -> None:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd="/tmp",
            text_parts=self.text_parts,
            reasoning_parts=self.reasoning_parts,
        )
        self.assertTrue(handled)

    def test_message_chunk_forwards_to_callback(self) -> None:
        self._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "Hello "},
                    }
                },
            }
        )
        self._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "world"},
                    }
                },
            }
        )

        self.assertEqual(self.text_parts, ["Hello ", "world"])
        self.assertEqual(self.text_deltas, ["Hello ", "world"])
        self.assertEqual(self.first_delta_count, 1)

    def test_thought_chunk_forwards_to_reasoning_callback(self) -> None:
        self._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"text": "planning"},
                    }
                },
            }
        )

        self.assertEqual(self.reasoning_parts, ["planning"])
        self.assertEqual(self.reasoning_deltas, ["planning"])
        self.assertEqual(self.first_delta_count, 1)

    def test_clear_stream_callbacks_stops_delivery(self) -> None:
        self.client.clear_stream_callbacks()
        self._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "hidden"},
                    }
                },
            }
        )

        self.assertEqual(self.text_parts, ["hidden"])
        self.assertEqual(self.text_deltas, [])


class WireACPStreamCallbacksTests(unittest.TestCase):
    def test_wires_callbacks_when_consumer_registered(self) -> None:
        from types import SimpleNamespace

        from agent.chat_completion_helpers import _wire_acp_stream_callbacks

        client = CursorACPClient(acp_cwd="/tmp")
        fired: list[str] = []

        agent = SimpleNamespace(
            _has_stream_consumers=lambda: True,
            _fire_stream_delta=lambda text: fired.append(f"text:{text}"),
            _fire_reasoning_delta=lambda text: fired.append(f"reason:{text}"),
            _acp_on_first_delta=lambda: fired.append("first"),
        )

        _wire_acp_stream_callbacks(agent, client)
        client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "live"},
                    }
                },
            },
            process=_FakeProcess(),
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
        )

        self.assertEqual(fired, ["first", "text:live"])

    def test_skips_callbacks_without_consumer(self) -> None:
        from types import SimpleNamespace

        from agent.chat_completion_helpers import _wire_acp_stream_callbacks

        client = CursorACPClient(acp_cwd="/tmp")
        client.set_stream_callbacks(on_text_delta=lambda _t: (_ for _ in ()).throw(AssertionError("boom")))

        agent = SimpleNamespace(_has_stream_consumers=lambda: False)
        _wire_acp_stream_callbacks(agent, client)

        self.assertIsNone(client._on_text_delta)
        self.assertIsNone(client._on_reasoning_delta)


if __name__ == "__main__":
    unittest.main()
