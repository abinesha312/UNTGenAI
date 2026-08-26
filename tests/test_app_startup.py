"""
Tests for app startup and OpenAI client integration.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, api_key, base_url, timeout):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.models = MockModels()
        self.chat = MockChat()


class MockModels:
    """Mock models endpoint."""
    
    def list(self):
        return MagicMock(data=[
            {"id": "google/gemma-3-27b-it", "object": "model"}
        ])


class MockChat:
    """Mock chat completions endpoint."""
    
    def __init__(self):
        self.completions = MockCompletions()


class MockCompletions:
    """Mock completions endpoint."""
    
    def create(self, model, messages, max_tokens, temperature, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "This is a test response from the mocked LLM."
        return response


class TestAppStartup:
    """Test suite for app startup with mocked OpenAI server."""
    
    @patch('src.app.OpenAI', return_value=MockOpenAIClient("EMPTY", "http://localhost:5000/v1", 10))
    def test_get_llm_client(self, mock_openai):
        """Test that get_llm_client creates proper client."""
        from src.app import get_llm_client
        
        client = get_llm_client()
        
        assert client is not None
        assert client.api_key == "EMPTY"
        assert client.base_url == "http://localhost:5000/v1"
    
    @patch('src.app.OpenAI', return_value=MockOpenAIClient("EMPTY", "http://localhost:5000/v1", 10))
    def test_verify_llm_server_success(self, mock_openai):
        """Test successful LLM server verification."""
        from src.app import verify_llm_server
        
        result = verify_llm_server()
        
        assert result is True
    
    @patch('src.app.OpenAI')
    def test_verify_llm_server_failure(self, mock_openai):
        """Test LLM server verification failure."""
        # Make the client raise an exception
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Connection refused")
        mock_openai.return_value = mock_client
        
        from src.app import verify_llm_server
        
        result = verify_llm_server()
        
        assert result is False
    
    @patch('src.app.OpenAI', return_value=MockOpenAIClient("EMPTY", "http://localhost:5000/v1", 10))
    def test_imports_work(self, mock_openai):
        """Test that all necessary imports work."""
        try:
            from src.app import (
                get_llm_client,
                verify_llm_server,
                get_session_context,
            )
            from src.agents.registry import agents, determine_agent_type
            from src.models.classification import AgentType
            from src.config.settings import (
                MODEL_ID,
                INFERENCE_SERVER_URL,
                MAX_TOKENS,
                TEMPERATURE,
            )
            
            # If we get here, imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_agent_registry_complete(self):
        """Test that all agent types are registered."""
        from src.agents.registry import agents
        from src.models.classification import AgentType
        
        # Check that all agent types have an implementation
        expected_agents = [
            AgentType.EMAIL,
            AgentType.RESEARCH,
            AgentType.ACADEMIC,
            AgentType.REDIRECT,
            AgentType.GENERAL,
            AgentType.PLANNER,
            AgentType.VISION,
        ]
        
        for agent_type in expected_agents:
            assert agent_type in agents, f"Missing agent: {agent_type}"
            assert agents[agent_type] is not None
    
    def test_settings_configuration(self):
        """Test that settings are properly configured."""
        from src.config.settings import (
            MODEL_ID,
            INFERENCE_SERVER_URL,
            MAX_TOKENS,
            TEMPERATURE,
            MODELS_BASE_PATH,
            DATA_BASE_PATH,
            VECTOR_DB_PATH,
        )
        
        # Check that critical settings exist and are non-empty
        assert MODEL_ID
        assert INFERENCE_SERVER_URL
        assert MAX_TOKENS > 0
        assert 0 <= TEMPERATURE <= 2
        assert MODELS_BASE_PATH
        assert DATA_BASE_PATH
        assert VECTOR_DB_PATH


class TestBaseAgent:
    """Test suite for BaseAgent with mocked OpenAI."""
    
    @pytest.mark.asyncio
    @patch('src.agents.base_agent.OpenAI', return_value=MockOpenAIClient("EMPTY", "http://localhost:5000/v1", 10))
    async def test_base_agent_get_response(self, mock_openai):
        """Test that BaseAgent can get response from mocked server."""
        from src.agents.specialized_agents import GeneralAgent
        
        agent = GeneralAgent()
        messages = [{"role": "user", "content": "Hello, what is AI?"}]
        
        response = await agent.get_response(messages)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test response" in response.lower()
    
    @pytest.mark.asyncio
    @patch('src.agents.base_agent.OpenAI')
    async def test_base_agent_connection_error(self, mock_openai):
        """Test that BaseAgent handles connection errors gracefully."""
        # Make the client raise an exception
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Connection refused")
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        mock_openai.return_value = mock_client
        
        from src.agents.specialized_agents import GeneralAgent
        
        agent = GeneralAgent()
        messages = [{"role": "user", "content": "Test"}]
        
        response = await agent.get_response(messages)
        
        # Should return an error message, not raise exception
        assert isinstance(response, str)
        assert "trouble" in response.lower() or "error" in response.lower()


class TestClassifier:
    """Test suite for prompt classifier."""
    
    def test_classifier_initialization(self):
        """Test that classifier initializes correctly."""
        from src.models.classification import PromptClassifier
        
        classifier = PromptClassifier()
        
        assert classifier is not None
        assert classifier.keywords
        assert classifier.vectorizer
    
    def test_classify_email_query(self):
        """Test classification of email-related queries."""
        from src.models.classification import PromptClassifier, AgentType
        
        classifier = PromptClassifier()
        
        result = classifier.classify_message("Help me compose an email to my professor")
        
        assert result.agent_type == AgentType.EMAIL
        assert result.confidence_score > 0
    
    def test_classify_academic_query(self):
        """Test classification of academic concept queries."""
        from src.models.classification import PromptClassifier, AgentType
        
        classifier = PromptClassifier()
        
        result = classifier.classify_message("Explain the theory of relativity")
        
        assert result.agent_type == AgentType.ACADEMIC
        assert result.confidence_score > 0
    
    def test_classify_research_query(self):
        """Test classification of research-related queries."""
        from src.models.classification import PromptClassifier, AgentType
        
        classifier = PromptClassifier()
        
        result = classifier.classify_message("Help me structure my research paper on climate change")
        
        assert result.agent_type == AgentType.RESEARCH
        assert result.confidence_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
