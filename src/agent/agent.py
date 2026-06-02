from __future__ import annotations

from typing import Dict, List

from ..env.base_env import Action, GameEnv, Observation
from ..env.vision import ObservationEncoder
from ..experience.types import Episode, Transition
from ..reward.reward_model import RewardModel
from ..tasks.task_schema import Task
from ..utils.logging_utils import get_logger
from .policy import Policy

logger = get_logger(__name__)

__all__ = ["Agent"]


class Agent:
    """
    The central Agent orchestrator in the SIMA architecture.
    
    This class coordinates all the components needed for an agent to:
    1. Perceive the environment through observation encoding
    2. Make decisions through the policy
    3. Collect experience as episodes
    4. Learn from that experience over time
    
    The Agent serves as the main interface between the high-level training loop
    and the lower-level environment interactions.
    """

    def __init__(self, 
                 env: GameEnv, 
                 policy: Policy, 
                 reward_model: RewardModel, 
                 encoder: ObservationEncoder,
                 success_threshold: float = 5.0):
        """
        Initialize the agent with its core components.
        
        Args:
            env: Game environment to interact with
            policy: Policy for action selection
            reward_model: Model for scoring episodes
            encoder: Observation encoder for feature extraction
        """
        self.env = env
        self.policy = policy
        self.reward_model = reward_model
        self.encoder = encoder
        self.success_threshold = success_threshold
        
        # Internal state
        self._current_episode_data: List[Dict] = []

    def run_episode(self, task: Task) -> Episode:
        """
        Run one complete episode attempting the given task.
        
        This is the core interaction loop where the agent:
        1. Resets the environment 
        2. Encodes observations
        3. Selects actions via policy
        4. Collects transitions
        5. Evaluates performance via reward model
        
        Args:
            task: Task for the agent to attempt
            
        Returns:
            Complete Episode object with all transitions and final evaluation
        """
        # Reset environment and policy
        obs = self.env.reset()
        self.policy.reset()
        
        # Initialize episode tracking
        transitions = []
        episode_rewards = []
        step_count = 0
        
        # Main interaction loop
        while step_count < task.max_steps:
            # Encode observation
            obs_vector = self.encoder.encode(obs)
            
            # Select action
            action = self.policy.act(obs_vector)
            
            # Execute action in environment
            next_obs, env_reward, done, info = self.env.step(action)
            
            # Create transition
            transition = Transition(
                obs=obs,
                action=action,
                reward=env_reward,
                done=done,
                info=info.copy()
            )
            transitions.append(transition)
            episode_rewards.append(env_reward)
            
            # Store for policy learning
            self._current_episode_data.append({
                "obs_vector": obs_vector,
                "action": action,
                "reward": env_reward,
                "done": done,
                "next_obs_vector": self.encoder.encode(next_obs) if not done else None
            })
            
            # Update for next step
            obs = next_obs
            step_count += 1
            
            if done:
                break
                
        # Create episode object
        episode = Episode(
            task_id=task.id,
            task_description=task.description,
            estimated_reward=task.estimated_reward,
            transitions=transitions,
            final_reward=0.0,  # Will be set by reward model
            success=False,     # Will be set based on reward model
            metadata={
                "template": task.metadata.get("template", "unknown"),
                "steps_taken": step_count,
                "env_reward_total": sum(episode_rewards),
                "task_max_steps": task.max_steps,
                "truncated": step_count >= task.max_steps and not done
            }
        )
        
        # Score episode using reward model
        final_reward = self.reward_model.score_episode(task, episode)
        episode.final_reward = final_reward
        
        # Determine success (threshold can be adjusted)
        episode.success = (
            final_reward >= self.success_threshold and episode.reached_goal
        )
        
        return episode

    def improve_from_episodes(self, episodes: List[Episode]) -> None:
        """
        Improve the agent's policy using completed episodes.
        
        This method converts episodes into the format expected by the policy's
        update method and triggers learning.
        
        Args:
            episodes: List of episodes to learn from
        """
        if not episodes:
            return
            
        # Convert episodes to format expected by policy
        episodes_data = []
        
        for episode in episodes:
            episode_data = {
                "task_id": episode.task_id,
                "success": episode.success,
                "final_reward": episode.final_reward,
                "transitions": [],
                "rewards": []
            }
            
            # Extract transitions and rewards
            for i, transition in enumerate(episode.transitions):
                # Encode observation for policy learning
                obs_vector = self.encoder.encode(transition.obs)
                
                # Get next observation if available
                next_obs_vector = None
                if i + 1 < len(episode.transitions):
                    next_obs = episode.transitions[i + 1].obs
                    next_obs_vector = self.encoder.encode(next_obs)
                
                transition_data = {
                    "obs_vector": obs_vector,
                    "action": transition.action,
                    "reward": transition.reward,
                    "done": transition.done,
                    "next_obs_vector": next_obs_vector
                }
                
                episode_data["transitions"].append(transition_data)
                episode_data["rewards"].append(transition.reward)
            
            episodes_data.append(episode_data)
        
        # Update policy
        try:
            self.policy.update(episodes_data)
        except Exception as e:
            print(f"Warning: Policy update failed: {e}")
            # Continue execution - some policies might not support updates

    def evaluate_on_task(self, task: Task, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate agent performance on a specific task.
        
        Args:
            task: Task to evaluate on
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        episodes = []
        
        for _ in range(num_episodes):
            episode = self.run_episode(task)
            episodes.append(episode)
            
        # Compute evaluation metrics
        success_rate = sum(ep.success for ep in episodes) / len(episodes)
        avg_reward = sum(ep.final_reward for ep in episodes) / len(episodes)
        avg_steps = sum(ep.length for ep in episodes) / len(episodes)
        goal_reach_rate = sum(ep.reached_goal for ep in episodes) / len(episodes)
        
        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "goal_reach_rate": goal_reach_rate,
            "num_episodes": num_episodes
        }

    def get_agent_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the agent's internal state.
        
        Returns:
            Dictionary with agent statistics
        """
        stats = {
            "policy_type": type(self.policy).__name__,
            "encoder_type": type(self.encoder).__name__,
            "reward_model_type": type(self.reward_model).__name__,
            "environment_type": type(self.env).__name__,
        }
        
        # Add policy-specific stats if available
        if hasattr(self.policy, 'get_statistics'):
            stats["policy_stats"] = self.policy.get_statistics()
            
        # Add encoder stats if available
        try:
            stats["encoder_feature_dim"] = self.encoder.feature_dim
        except:
            stats["encoder_feature_dim"] = "unknown"
            
        return stats

    def get_performance_stats(self) -> Dict[str, any]:
        """
        Get performance statistics for the agent.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "agent_type": type(self).__name__,
            "creation_time": getattr(self, '_creation_time', None),
            "components": self.get_agent_statistics()
        }

    def save_agent_state(self, filepath: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            filepath: Path to save agent state
        """
        from pathlib import Path
        import json
        import pickle
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save different components
        agent_data = {
            'stats': self.get_performance_stats(),
            'creation_time': getattr(self, '_creation_time', None),
            'version': '0.1.0'
        }
        
        # Save basic agent data as JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        # Try to save policy if it supports checkpointing
        if hasattr(self.policy, 'save_checkpoint'):
            try:
                policy_path = filepath.with_name(f"{filepath.stem}_policy.pt")
                self.policy.save_checkpoint(str(policy_path))
                agent_data['policy_checkpoint'] = str(policy_path)
                logger.info(f"Saved policy checkpoint to {policy_path}")
            except Exception as e:
                logger.warning(f"Failed to save policy checkpoint: {e}")
        
        # Save reward model if it has state
        if hasattr(self.reward_model, '__dict__'):
            try:
                reward_path = filepath.with_name(f"{filepath.stem}_reward.pkl")
                with open(reward_path, 'wb') as f:
                    pickle.dump(self.reward_model.__dict__, f)
                agent_data['reward_model_state'] = str(reward_path)
                logger.info(f"Saved reward model state to {reward_path}")
            except Exception as e:
                logger.warning(f"Failed to save reward model: {e}")
        
        # Update JSON with checkpoint paths
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        logger.info(f"Agent state saved to {filepath.with_suffix('.json')}")

    def load_agent_state(self, filepath: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            filepath: Path to load agent state from
        """
        from pathlib import Path
        import json
        import pickle
        
        filepath = Path(filepath).with_suffix('.json')
        
        if not filepath.exists():
            logger.error(f"Agent state file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                agent_data = json.load(f)
            
            logger.info(f"Loading agent state from {filepath}")
            logger.info(f"Saved stats: {agent_data.get('stats', {})}")
            
            # Load policy checkpoint if available
            if 'policy_checkpoint' in agent_data and hasattr(self.policy, 'load_checkpoint'):
                policy_path = Path(agent_data['policy_checkpoint'])
                if policy_path.exists():
                    self.policy.load_checkpoint(str(policy_path))
                    logger.info(f"Loaded policy checkpoint from {policy_path}")
                else:
                    logger.warning(f"Policy checkpoint not found: {policy_path}")
            
            # Load reward model if available
            if 'reward_model_state' in agent_data:
                reward_path = Path(agent_data['reward_model_state'])
                if reward_path.exists():
                    with open(reward_path, 'rb') as f:
                        reward_state = pickle.load(f)
                    self.reward_model.__dict__.update(reward_state)
                    logger.info(f"Loaded reward model state from {reward_path}")
                else:
                    logger.warning(f"Reward model state not found: {reward_path}")
            
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")

    def reset_learning_state(self) -> None:
        """Reset any internal learning state."""
        self.policy.reset()
        self._current_episode_data.clear()
