"""
Tests for the multi-agent orchestrator (PlannerAgent).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.planner_agent import PlannerAgent
from models.classification import AgentType


class TestPlannerAgent:
    """Test suite for PlannerAgent orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = PlannerAgent()
    
    def test_initialization(self):
        """Test that PlannerAgent initializes correctly."""
        assert self.planner.name == "Planner"
        assert self.planner.description == "Multi-agent orchestrator"
        assert self.planner.classifier is not None
    
    def test_decompose_single_task(self):
        """Test decomposition of a single simple task."""
        goal = "Explain quantum mechanics"
        tasks = self.planner._decompose_goal(goal)
        
        assert len(tasks) == 1
        assert tasks[0][0] == goal
        assert isinstance(tasks[0][1], AgentType)
    
    def test_decompose_multi_task_with_and(self):
        """Test decomposition of goals connected with 'and'."""
        goal = "Compose an email to my professor and explain the theory of relativity"
        tasks = self.planner._decompose_goal(goal)
        
        assert len(tasks) >= 2
        # Should have separate tasks for email and explanation
        assert any("email" in task.lower() for task, _ in tasks)
        assert any("relativity" in task.lower() for task, _ in tasks)
    
    def test_decompose_multiple_questions(self):
        """Test decomposition of multiple questions."""
        goal = "What is machine learning? How does it work? Where can I learn more?"
        tasks = self.planner._decompose_goal(goal)
        
        assert len(tasks) >= 2
        # Each question should be a separate task
        assert all("?" in task for task, _ in tasks)
    
    def test_no_planner_loops(self):
        """Test that decomposition doesn't create infinite planner loops."""
        goal = "Plan how to write a research paper"
        tasks = self.planner._decompose_goal(goal)
        
        # Should not route to PLANNER (should use GENERAL or other specialist)
        assert all(agent_type != AgentType.PLANNER for _, agent_type in tasks)
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        task = "Explain photosynthesis"
        agent_type = AgentType.ACADEMIC
        
        # Mock the agent
        with patch('agents.planner_agent.agents') as mock_agents:
            mock_agent = MagicMock()
            mock_agent.process_input.return_value = {"type": "final_response"}
            mock_agent.get_response = AsyncMock(return_value="Photosynthesis is the process...")
            mock_agents.__getitem__.return_value = mock_agent
            
            result = await self.planner._execute_task(task, agent_type)
            
            assert result["success"] is True
            assert result["agent"] == agent_type.value
            assert result["task"] == task
            assert "Photosynthesis" in result["response"]
    
    @pytest.mark.asyncio
    async def test_execute_task_error(self):
        """Test task execution with error handling."""
        task = "Test task"
        agent_type = AgentType.GENERAL
        
        # Mock the agent to raise an exception
        with patch('agents.planner_agent.agents') as mock_agents:
            mock_agent = MagicMock()
            mock_agent.process_input.side_effect = Exception("Test error")
            mock_agents.__getitem__.return_value = mock_agent
            
            result = await self.planner._execute_task(task, agent_type)
            
            assert result["success"] is False
            assert result["error"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel(self):
        """Test parallel execution of multiple tasks."""
        tasks = [
            ("Explain AI", AgentType.ACADEMIC),
            ("Write an email", AgentType.EMAIL),
        ]
        
        with patch('agents.planner_agent.agents') as mock_agents:
            mock_agent1 = MagicMock()
            mock_agent1.process_input.return_value = {"type": "final_response"}
            mock_agent1.get_response = AsyncMock(return_value="AI explanation")
            
            mock_agent2 = MagicMock()
            mock_agent2.process_input.return_value = {"type": "final_response"}
            mock_agent2.get_response = AsyncMock(return_value="Email content")
            
            mock_agents.__getitem__.side_effect = lambda x: mock_agent1 if x == AgentType.ACADEMIC else mock_agent2
            
            results = await self.planner._execute_tasks_parallel(tasks)
            
            assert len(results) == 2
            assert all(r["success"] for r in results)
    
    def test_merge_results_single(self):
        """Test merging a single result."""
        goal = "Explain photosynthesis"
        results = [{
            "success": True,
            "agent": "academic",
            "task": "Explain photosynthesis",
            "response": "Photosynthesis is..."
        }]
        
        merged = self.planner._merge_results(goal, results)
        
        assert "Photosynthesis is..." in merged
        assert "Part 1" not in merged  # Single result shouldn't be numbered
    
    def test_merge_results_multiple(self):
        """Test merging multiple results."""
        goal = "Explain AI and write an email"
        results = [
            {
                "success": True,
                "agent": "academic",
                "task": "Explain AI",
                "response": "AI is artificial intelligence..."
            },
            {
                "success": True,
                "agent": "email",
                "task": "Write an email",
                "response": "Dear Professor..."
            }
        ]
        
        merged = self.planner._merge_results(goal, results)
        
        assert "Part 1" in merged
        assert "Part 2" in merged
        assert "AI is artificial intelligence" in merged
        assert "Dear Professor" in merged
        assert "Summary" in merged
        assert "2/2" in merged
    
    def test_merge_results_with_errors(self):
        """Test merging results with some errors."""
        goal = "Multiple tasks"
        results = [
            {
                "success": True,
                "agent": "academic",
                "task": "Task 1",
                "response": "Success response"
            },
            {
                "success": False,
                "agent": "email",
                "task": "Task 2",
                "error": "Connection failed"
            }
        ]
        
        merged = self.planner._merge_results(goal, results)
        
        assert "Success response" in merged
        assert "Error" in merged
        assert "1/2" in merged
    
    @pytest.mark.asyncio
    async def test_get_response_integration(self):
        """Test full integration of orchestrator response."""
        goal = "Explain machine learning"
        
        with patch('agents.planner_agent.agents') as mock_agents:
            mock_agent = MagicMock()
            mock_agent.process_input.return_value = {"type": "final_response"}
            mock_agent.get_response = AsyncMock(return_value="Machine learning is a subset of AI...")
            mock_agents.__getitem__.return_value = mock_agent
            
            response = await self.planner.get_response(goal)
            
            assert isinstance(response, str)
            assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
