import os
import unittest

os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder")

import langgraph_backend as backend
from langchain_core.messages import AIMessage, ToolMessage


class FakeToolCallingModel:
    async def ainvoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Tool routing works")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {
                        "first_number": 6,
                        "second_number": 7,
                        "operation": "multiply",
                    },
                    "id": "test-call",
                    "type": "tool_call",
                }
            ],
        )


class UnifiedBackendTest(unittest.TestCase):
    def test_calculator_rejects_division_by_zero(self):
        result = backend.calculator.invoke(
            {"first_number": 1, "second_number": 0, "operation": "divide"}
        )
        self.assertEqual("Division by zero is not allowed.", result["error"])

    def test_invalid_pdf_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "not a valid PDF"):
            backend.ingest_pdf(b"not-a-pdf", "invalid-pdf-thread")

    def test_mcp_exposes_only_expected_read_only_tools(self):
        self.assertEqual(
            {
                "convert_temperature",
                "calculate_percentage",
                "current_utc_time",
            },
            {tool.name for tool in backend.mcp_tools},
        )

    def test_thread_cleanup_removes_tracking_state(self):
        thread_id = "retention-test-thread"
        backend.touch_thread(thread_id)
        self.assertIn(thread_id, backend._THREAD_LAST_SEEN)
        backend.delete_thread_data(thread_id)
        self.assertNotIn(thread_id, backend._THREAD_LAST_SEEN)
        self.assertNotIn(thread_id, backend._PENDING_THREAD_DELETIONS)

    def test_langgraph_routes_a_tool_call(self):
        original_model = backend.llm_with_tools

        async def run_graph():
            events = []
            async for event in backend.chatbot.astream(
                {"messages": [], "thread_id": "unit-test-thread"},
                config={"configurable": {"thread_id": "unit-test-thread"}},
                stream_mode="updates",
            ):
                events.append(event)
            return events

        try:
            backend.llm_with_tools = FakeToolCallingModel()
            events = backend._run_async(run_graph(), timeout=20)
        finally:
            backend.llm_with_tools = original_model
            backend.delete_thread_data("unit-test-thread")

        self.assertEqual(["chat_node", "tools", "chat_node"], [next(iter(event)) for event in events])
        self.assertEqual(
            "Tool routing works",
            events[-1]["chat_node"]["messages"][0].content,
        )


if __name__ == "__main__":
    unittest.main()
