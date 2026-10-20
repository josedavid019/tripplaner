import unittest
from unittest.mock import AsyncMock, patch

from app.OrchestratorAgent import main as orchestrator


class RemoteAgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main.make_remote_agent")
    @patch("app.OrchestratorAgent.main.httpx.AsyncClient")
    async def test_remote_http_client_is_closed_after_success(self, async_client_class, make_remote_agent):
        http_client = AsyncMock()
        async_client_class.return_value.__aenter__.return_value = http_client
        remote_agent = AsyncMock()
        remote_agent.invoke_async.return_value = "response"
        make_remote_agent.return_value = remote_agent

        response = await orchestrator._invoke_remote_agent("runtime-arn", "request")

        self.assertEqual(response, "response")
        make_remote_agent.assert_called_once_with("runtime-arn", http_client)
        remote_agent.invoke_async.assert_awaited_once_with("request")
        async_client_class.return_value.__aexit__.assert_awaited_once()

    @patch("app.OrchestratorAgent.main.make_remote_agent")
    @patch("app.OrchestratorAgent.main.httpx.AsyncClient")
    async def test_remote_http_client_is_closed_after_failure(self, async_client_class, make_remote_agent):
        http_client = AsyncMock()
        async_client_class.return_value.__aenter__.return_value = http_client
        remote_agent = AsyncMock()
        remote_agent.invoke_async.side_effect = RuntimeError("remote failure")
        make_remote_agent.return_value = remote_agent

        with self.assertRaisesRegex(RuntimeError, "remote failure"):
            await orchestrator._invoke_remote_agent("runtime-arn", "request")

        async_client_class.return_value.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
