from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..experience.buffer import ReplayBuffer
from ..experience.types import Episode
from .task_schema import Task

__all__ = ["TaskSetter"]


class TaskSetter:
    """
    The TaskSetter component from the SIMA architecture that proposes new tasks
    with estimated rewards based on the agent's experience.
    
    This class maintains a curriculum of tasks and adaptively selects which
    tasks to propose based on:
    - Agent's historical performance on similar tasks
    - Learning progress (improvement over time)
    - Task diversity to ensure broad skill development
    
    In production, this could be enhanced with:
    - LLM-based task generation from natural language
    - Hierarchical task decomposition
    - Human feedback integration
    """

    def __init__(self, experience_buffer: Optional[ReplayBuffer] = None):
        """
        Initialize the task setter.
        
        Args:
            experience_buffer: Buffer containing agent's past experience
        """
        self.experience_buffer = experience_buffer
        self._task_templates = self._create_default_tasks()
        self._task_performance: Dict[str, List[float]] = {}
        self._task_counts: Dict[str, int] = {}
        self._template_counts: Dict[str, int] = {}

    def propose_task(self) -> Task:
        """
        Propose a task for the agent to attempt.
        
        Uses a simple heuristic that balances:
        - Tasks where agent is making progress but not perfect
        - Exploration of new/underexplored tasks
        - Appropriate difficulty based on current skill level
        
        Returns:
            Task for the agent to attempt
        """
        # Get task performance statistics
        task_stats = self._analyze_task_performance()
        
        # Select task using adaptive strategy
        task_template = self._select_task_template(task_stats)
        
        # Create specific task instance
        task = self._instantiate_task(task_template)
        
        # Track that we've proposed this task
        self._task_counts[task.id] = self._task_counts.get(task.id, 0) + 1
        template_id = task.metadata.get("template", task.id)
        self._template_counts[template_id] = self._template_counts.get(template_id, 0) + 1
        
        return task

    def update_from_episode(self, episode: Episode) -> None:
        """
        Update task selection strategy based on episode outcome.
        
        Args:
            episode: Completed episode to learn from
        """
        task_id = episode.task_id
        final_reward = episode.final_reward
        
        # Track performance for this task type
        if task_id not in self._task_performance:
            self._task_performance[task_id] = []
        
        self._task_performance[task_id].append(final_reward)
        
        # Keep only recent performance (sliding window)
        max_history = 20
        if len(self._task_performance[task_id]) > max_history:
            self._task_performance[task_id] = self._task_performance[task_id][-max_history:]

    def get_task_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about task performance and selection.
        
        Returns:
            Dictionary mapping task IDs to performance statistics
        """
        stats = {}
        
        for task_id, rewards in self._task_performance.items():
            if rewards:
                stats[task_id] = {
                    "avg_reward": sum(rewards) / len(rewards),
                    "recent_reward": rewards[-1] if rewards else 0.0,
                    "num_attempts": len(rewards),
                    "success_rate": sum(1 for r in rewards if r > 5.0) / len(rewards),
                    "improvement": rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
                    "times_proposed": self._task_counts.get(task_id, 0)
                }
        
        return stats

    def _create_default_tasks(self) -> List[Dict[str, Any]]:
        """
        Create a set of default task templates.
        
        In production, this would be replaced with:
        - LLM-generated tasks based on game analysis
        - Human-authored task libraries
        - Procedurally generated task variants
        
        Returns:
            List of task template dictionaries
        """
        return [
            {
                "id_prefix": "reach_goal",
                "description_template": "Navigate to the goal location",
                "base_reward": 8.0,
                "max_steps": 50,
                "difficulty": "easy"
            },
            {
                "id_prefix": "avoid_obstacles",
                "description_template": "Reach the goal while avoiding obstacles",
                "base_reward": 6.0,
                "max_steps": 75,
                "difficulty": "medium"
            },
            {
                "id_prefix": "collect_items",
                "description_template": "Collect all items before reaching the goal",
                "base_reward": 4.0,
                "max_steps": 100,
                "difficulty": "hard"
            },
            {
                "id_prefix": "time_limit",
                "description_template": "Complete the objective within time limit",
                "base_reward": 5.0,
                "max_steps": 30,
                "difficulty": "medium"
            },
            {
                "id_prefix": "exploration",
                "description_template": "Explore and discover hidden areas",
                "base_reward": 3.0,
                "max_steps": 150,
                "difficulty": "hard"
            },
            # Minecraft-specific tasks
            {
                "id_prefix": "collect_wood",
                "description_template": "Find and collect wood blocks by punching trees",
                "base_reward": 6.0,
                "max_steps": 200,
                "difficulty": "easy"
            },
            {
                "id_prefix": "look_around",
                "description_template": "Look around and observe the environment",
                "base_reward": 2.0,
                "max_steps": 50,
                "difficulty": "easy"
            },
            {
                "id_prefix": "build_pillar",
                "description_template": "Build a pillar using collected blocks",
                "base_reward": 8.0,
                "max_steps": 300,
                "difficulty": "medium"
            },
            {
                "id_prefix": "craft_tools",
                "description_template": "Craft basic tools from collected materials",
                "base_reward": 10.0,
                "max_steps": 400,
                "difficulty": "hard"
            },
            {
                "id_prefix": "survive_night",
                "description_template": "Survive until dawn while avoiding monsters",
                "base_reward": 12.0,
                "max_steps": 500,
                "difficulty": "hard"
            },
            {
                "id_prefix": "find_diamonds",
                "description_template": "Mine down and find diamond ore",
                "base_reward": 15.0,
                "max_steps": 600,
                "difficulty": "hard"
            }
        ]

    def _analyze_task_performance(self) -> Dict[str, float]:
        """
        Analyze performance across different task types.
        
        Returns:
            Dictionary mapping task types to learning potential scores
        """
        task_scores = {}
        
        for template in self._task_templates:
            task_prefix = template["id_prefix"]
            
            # Find performance data for this task type
            matching_performance = []
            for task_id, rewards in self._task_performance.items():
                if task_id.startswith(task_prefix):
                    matching_performance.extend(rewards)
            
            if not matching_performance:
                # New task - give it a chance
                task_scores[task_prefix] = 0.8
            else:
                # Calculate learning potential based on:
                # - Current performance (not too easy/hard)
                # - Recent improvement trend
                # - Exploration bonus for underexplored tasks
                
                recent_performance = sum(matching_performance[-5:]) / min(5, len(matching_performance))
                attempt_count = len(matching_performance)
                
                # Prefer tasks where agent performs moderately well (can learn)
                performance_score = 1.0 - abs(recent_performance - 5.0) / 10.0  # Peak at reward=5
                
                # Exploration bonus for less-attempted tasks
                exploration_bonus = 0.3 / (1.0 + attempt_count * 0.1)
                
                # Improvement trend bonus
                if len(matching_performance) >= 3:
                    recent_avg = sum(matching_performance[-3:]) / 3
                    older_avg = sum(matching_performance[:-3]) / max(1, len(matching_performance) - 3)
                    improvement_bonus = max(0, (recent_avg - older_avg) / 10.0)
                else:
                    improvement_bonus = 0
                
                task_scores[task_prefix] = performance_score + exploration_bonus + improvement_bonus
        
        return task_scores

    def _select_task_template(self, task_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Select a task template based on learning potential scores.
        
        Args:
            task_scores: Scores indicating learning potential for each task type
            
        Returns:
            Selected task template
        """
        # Weighted random selection based on scores
        templates = []
        weights = []
        
        for template in self._task_templates:
            task_prefix = template["id_prefix"]
            score = task_scores.get(task_prefix, 0.5)
            templates.append(template)
            weights.append(max(0.1, score))  # Minimum weight to ensure all tasks possible
        
        # Weighted random choice
        total_weight = sum(weights)
        rand_val = random.random() * total_weight
        
        cumulative_weight = 0
        for template, weight in zip(templates, weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return template
                
        return templates[-1]  # Fallback

    def _instantiate_task(self, template: Dict[str, Any]) -> Task:
        """
        Create a specific task instance from a template.
        
        Args:
            template: Task template dictionary
            
        Returns:
            Concrete Task instance
        """
        # Generate unique task ID
        task_count = self._template_counts.get(template["id_prefix"], 0)
        task_id = f"{template['id_prefix']}_{task_count + 1:03d}"
        
        # Estimate reward based on agent's current capabilities
        base_reward = template["base_reward"]
        
        # Adjust based on recent performance
        recent_performance = self._get_recent_performance(template["id_prefix"])
        if recent_performance is not None:
            # Slightly adjust estimated reward based on recent success
            adjustment = (recent_performance - 5.0) * 0.2
            estimated_reward = base_reward + adjustment
        else:
            estimated_reward = base_reward
            
        # Ensure reasonable bounds
        estimated_reward = max(0.0, min(10.0, estimated_reward))
        
        return Task(
            id=task_id,
            description=template["description_template"],
            estimated_reward=estimated_reward,
            max_steps=template["max_steps"],
            metadata={
                "template": template["id_prefix"],
                "difficulty": template["difficulty"],
                "generation": task_count + 1
            }
        )

    def _get_recent_performance(self, task_prefix: str) -> Optional[float]:
        """
        Get recent average performance for a task type.
        
        Args:
            task_prefix: Task type prefix
            
        Returns:
            Recent average reward, or None if no data
        """
        recent_rewards = []
        
        for task_id, rewards in self._task_performance.items():
            if task_id.startswith(task_prefix):
                recent_rewards.extend(rewards[-3:])  # Last 3 attempts
                
        if recent_rewards:
            return sum(recent_rewards) / len(recent_rewards)
        else:
            return None


# TODO: LLM-enhanced task generation would look like:
# 
# class LLMTaskSetter(TaskSetter):
#     def __init__(self, llm_client, experience_buffer=None):
#         super().__init__(experience_buffer)
#         self.llm_client = llm_client
#         
#     def generate_creative_task(self, context: str) -> Task:
#         """Use LLM to generate novel tasks based on agent's experience."""
#         prompt = f"""
#         Based on the agent's recent experience: {context}
#         
#         Generate a new challenging but achievable task that will help 
#         the agent develop new skills. The task should be:
#         - Specific and measurable
#         - Connected to previous successes/failures
#         - Appropriately difficult for current skill level
#         
#         Return as JSON with: id, description, estimated_reward, max_steps
#         """
#         
#         response = self.llm_client.generate(prompt)
#         task_data = json.loads(response)
#         return Task.from_dict(task_data)
