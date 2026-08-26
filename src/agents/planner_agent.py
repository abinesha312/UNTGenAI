from typing import Dict, Any, Optional, List, Tuple
import logging
import asyncio
from .base_agent import BaseAgent
from models.query_models import QueryResponse
from models.classification import PromptClassifier, AgentType

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Multi-agent orchestrator that decomposes complex goals into sub-tasks,
    routes each to appropriate specialists, executes them (sequentially or in parallel),
    and merges results into a comprehensive response.
    """

    def __init__(self):
        super().__init__(name="Planner", description="Multi-agent orchestrator")
        self.system_prompt = (
            "You are an advanced AI planning and orchestration assistant. Given a user's complex goal, you:\n"
            "1. Break it into logical sub-tasks\n"
            "2. Identify which specialized agent should handle each sub-task\n"
            "3. Execute tasks efficiently (parallel when independent, sequential when dependent)\n"
            "4. Merge all results into a coherent, comprehensive response\n"
            "Available specialists: Email, Research, Academic Concepts, UNT Resources, General, Vision"
        )
        self.required_inputs = []  # No upfront inputs needed
        self.classifier = PromptClassifier()

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def _decompose_goal(self, goal: str) -> List[Tuple[str, AgentType]]:
        """
        Decompose a complex goal into sub-tasks with assigned agent types.
        
        Returns:
            List of (sub_task, agent_type) tuples
        """
        # Use simple heuristics to detect multi-part goals
        tasks = []
        
        # Check for explicit multi-part indicators
        goal_lower = goal.lower()
        
        # Pattern 1: "and" connector
        if " and " in goal_lower and any(kw in goal_lower for kw in ["email", "research", "explain", "find"]):
            parts = goal.split(" and ")
            for part in parts:
                part = part.strip()
                if part:
                    classification = self.classifier.classify_message(part)
                    tasks.append((part, classification.agent_type))
        
        # Pattern 2: Multiple questions
        elif "?" in goal and goal.count("?") > 1:
            parts = [p.strip() + "?" for p in goal.split("?") if p.strip()]
            for part in parts:
                classification = self.classifier.classify_message(part)
                tasks.append((part, classification.agent_type))
        
        # Pattern 3: Numbered or bulleted list
        elif any(indicator in goal for indicator in ["\n1.", "\n2.", "\n-", "\n*"]):
            # Split by common list indicators
            import re
            parts = re.split(r'\n[\d\-\*]+\.?\s+', goal)
            for part in parts:
                part = part.strip()
                if part and len(part) > 10:  # Skip very short fragments
                    classification = self.classifier.classify_message(part)
                    tasks.append((part, classification.agent_type))
        
        # Pattern 4: Complex query requiring multiple domains
        elif any(kw in goal_lower for kw in ["steps", "guide", "process", "how do i", "walkthrough"]):
            # For procedural queries, break into logical steps
            # For now, route to a single agent but flag as multi-step
            classification = self.classifier.classify_message(goal)
            tasks.append((goal, classification.agent_type))
        
        # If no decomposition patterns found, treat as single task
        if not tasks:
            classification = self.classifier.classify_message(goal)
            tasks.append((goal, classification.agent_type))
        
        # Avoid planner loops - replace PLANNER with GENERAL
        tasks = [(task, AgentType.GENERAL if agent_type == AgentType.PLANNER else agent_type) 
                 for task, agent_type in tasks]
        
        return tasks

    async def _execute_task(self, task: str, agent_type: AgentType, attachments=None) -> Dict[str, Any]:
        """
        Execute a single task with the specified agent.
        
        Returns:
            Dict with 'success', 'agent', 'task', 'response', and optional 'error'
        """
        from agents.registry import agents
        
        try:
            agent = agents[agent_type]
            agent.reset()
            
            # Process the task
            result = agent.process_input(task)
            
            # Handle input requests (for now, skip interactive prompts in orchestrator)
            if result["type"] == "input_request":
                logger.warning(f"Agent {agent_type} requires additional input, using best-effort response")
            
            # Get the response
            response = await agent.get_response(task, attachments)
            
            return {
                "success": True,
                "agent": agent_type.value,
                "task": task,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error executing task with {agent_type}: {e}", exc_info=True)
            return {
                "success": False,
                "agent": agent_type.value,
                "task": task,
                "response": None,
                "error": str(e)
            }

    async def _execute_tasks_parallel(self, tasks: List[Tuple[str, AgentType]], attachments=None) -> List[Dict]:
        """Execute independent tasks in parallel."""
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        coroutines = [self._execute_task(task, agent_type, attachments) for task, agent_type in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task, agent_type = tasks[i]
                processed_results.append({
                    "success": False,
                    "agent": agent_type.value,
                    "task": task,
                    "response": None,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def _execute_tasks_sequential(self, tasks: List[Tuple[str, AgentType]], attachments=None) -> List[Dict]:
        """Execute dependent tasks sequentially."""
        logger.info(f"Executing {len(tasks)} tasks sequentially")
        results = []
        for task, agent_type in tasks:
            result = await self._execute_task(task, agent_type, attachments)
            results.append(result)
        return results

    def _merge_results(self, goal: str, results: List[Dict]) -> str:
        """Merge results from multiple agents into a comprehensive response."""
        if not results:
            return "I couldn't process your request. Please try again."
        
        # Check if all tasks failed
        if all(not r["success"] for r in results):
            error_summary = "\n".join([f"- {r['agent']}: {r['error']}" for r in results if not r["success"]])
            return (
                f"I encountered errors while processing your request:\n\n"
                f"{error_summary}\n\n"
                f"Please try simplifying your request or breaking it into smaller parts."
            )
        
        # Build merged response
        if len(results) == 1:
            # Single task - return direct response
            result = results[0]
            if result["success"]:
                return result["response"]
            else:
                return f"Error: {result['error']}"
        
        # Multiple tasks - structured response
        merged = f"# Response to: {goal}\n\n"
        merged += "I've broken down your request and addressed each part:\n\n"
        
        for i, result in enumerate(results, 1):
            merged += f"## Part {i}: {result['task'][:80]}{'...' if len(result['task']) > 80 else ''}\n"
            merged += f"**Handled by**: {result['agent'].title()} specialist\n\n"
            
            if result["success"]:
                merged += f"{result['response']}\n\n"
            else:
                merged += f"❌ **Error**: {result['error']}\n\n"
            
            merged += "---\n\n"
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        merged += f"### Summary\n"
        merged += f"Successfully completed {successful}/{len(results)} tasks.\n"
        
        return merged

    async def get_response(self, messages, attachments=None):
        """
        Orchestrate multi-agent response to a complex goal.
        
        Process:
        1. Decompose goal into sub-tasks
        2. Determine if tasks can run in parallel or must be sequential
        3. Execute tasks
        4. Merge and return results
        """
        # Extract the user's goal from messages
        if isinstance(messages, str):
            goal = messages
        elif isinstance(messages, list) and messages:
            goal = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        else:
            return "Please provide a goal or question."
        
        logger.info(f"Orchestrating response for goal: {goal[:100]}...")
        
        # Step 1: Decompose into sub-tasks
        tasks = self._decompose_goal(goal)
        logger.info(f"Decomposed into {len(tasks)} sub-tasks: {[(t[:50], a.value) for t, a in tasks]}")
        
        # Step 2: Determine execution strategy
        # For now, use parallel execution if tasks use different agents (likely independent)
        # Use sequential if they use the same agent (might be dependent)
        agent_types = [agent_type for _, agent_type in tasks]
        use_parallel = len(set(agent_types)) == len(agent_types)  # All different agents
        
        # Step 3: Execute tasks
        if use_parallel and len(tasks) > 1:
            results = await self._execute_tasks_parallel(tasks, attachments)
        else:
            results = await self._execute_tasks_sequential(tasks, attachments)
        
        # Step 4: Merge and return results
        merged_response = self._merge_results(goal, results)
        
        return merged_response 