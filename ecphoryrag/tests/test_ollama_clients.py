import unittest
from unittest.mock import patch, MagicMock
import pytest

from ecphoryrag.src.ollama_clients import get_ollama_embedding, get_ollama_completion


class TestOllamaClients(unittest.TestCase):
    """测试Ollama客户端功能"""

    @patch("ecphoryrag.src.ollama_clients.embed")
    def test_get_ollama_embedding(self, mock_embed):
        """测试嵌入功能"""
        # 设置模拟返回值
        mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        # 调用函数
        result = get_ollama_embedding("test text", "test-model")
        
        # 验证结果
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_embed.assert_called_once_with(model="test-model", input="test text")
    
    @patch("ecphoryrag.src.ollama_clients.embed")
    def test_get_ollama_embedding_empty_input(self, mock_embed):
        """测试空输入处理"""
        result = get_ollama_embedding("", "test-model")
        self.assertEqual(result, [])
        mock_embed.assert_not_called()
    
    @patch("ecphoryrag.src.ollama_clients.asyncio")
    def test_get_ollama_completion(self, mock_asyncio):
        """测试完成功能"""
        # 设置模拟返回值
        mock_client = MagicMock()
        mock_response = {"message": {"content": "test completion"}}
        
        async def mock_chat(*args, **kwargs):
            return mock_response
        
        mock_client.chat = mock_chat
        mock_async_client = MagicMock(return_value=mock_client)
        
        with patch("ecphoryrag.src.ollama_clients.AsyncClient", mock_async_client):
            mock_asyncio.run.return_value = "test completion"
            
            # 调用函数
            result = get_ollama_completion("test prompt", "test-model")
            
            # 验证结果
            self.assertEqual(result, "test completion")
    
    @patch("ecphoryrag.src.ollama_clients.asyncio")
    def test_get_ollama_completion_empty_input(self, mock_asyncio):
        """测试空输入处理"""
        result = get_ollama_completion("", "test-model")
        self.assertEqual(result, "")
        mock_asyncio.run.assert_not_called()


if __name__ == "__main__":
    unittest.main() 