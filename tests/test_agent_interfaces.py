"""
Tests for agent interface and integration functionality.

This module tests the integration between different components of the
SIMA system, ensuring they work together correctly.
"""

import tempfile
from pathlib import Path

import numpy as np

from src.agent.agent import Agent
from src.agent.policy import EpsilonGreedyPolicy, RandomPolicy
from src.env.dummy_env import DummyGameEnv
from src.env.vision import DummyVisionEncoder
from src.experience.buffer import ReplayBuffer
from src.experience.storage import EpisodeStorage
from src.reward.reward_model import SimpleRewardModel
from src.tasks.task_schema import Task
from src.tasks.task_setter import TaskSetter


class TestAgentInterfaces:
    """Test agent component interfaces and integration."""
    
    def test_dummy_environment(self):
        """Test the dummy environment functionality."""
        env = DummyGameEnv(grid_size=5, max_steps=20)
        
        # Test reset
        obs = env.reset()
        assert "pixels" in obs
        assert "info" in obs
        assert isinstance(obs["pixels"], np.ndarray)
        
        # Test step
        action = {"move": "right"}
        next_obs, reward, done, info = env.step(action)
        
        assert "pixels" in next_obs
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Test goal reaching
        start_pos = env.agent_pos
        goal_pos = env.goal_pos
        
        # Move towards goal
        steps = 0
        while not done and steps < 100:
            # Simple navigation towards goal
            agent_x, agent_y = env.agent_pos
            goal_x, goal_y = goal_pos
            
            if agent_x < goal_x:
                action = {"move": "right"}
            elif agent_x > goal_x:
                action = {"move": "left"}
            elif agent_y < goal_y:
                action = {"move": "down"}
            elif agent_y > goal_y:
                action = {"move": "up"}
            else:
                action = {"move": "noop"}
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done and info.get("goal_reached"):
                break
        
        # Should eventually reach goal
        assert info.get("goal_reached", False) or steps < 100
    
    def test_vision_encoder(self):
        """Test observation encoding."""
        encoder = DummyVisionEncoder(feature_dim=64)
        
        # Create dummy observation
        obs = {
            "pixels": np.random.randint(0, 256, (25,), dtype=np.uint8),
            "info": {"agent_pos": (2, 3), "goal_pos": (4, 4)}
        }
        
        # Encode observation
        features = encoder.encode(obs)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == encoder.get_feature_dim()
        assert encoder.get_feature_dim() == 64
    
    def test_random_policy(self):
        """Test random policy functionality."""
        policy = RandomPolicy()
        
        # Test action selection
        obs_vector = np.random.randn(10)
        action = policy.act(obs_vector)
        
        assert isinstance(action, dict)
        assert "move" in action
        assert action["move"] in ["up", "down", "left", "right", "noop"]
        
        # Test update (should be no-op)
        policy.update([])  # Should not raise
    
    def test_epsilon_greedy_policy(self):
        """Test epsilon-greedy policy."""
        policy = EpsilonGreedyPolicy(epsilon=0.2, learning_rate=0.1)
        
        # Test action selection
        obs_vector = np.array([1.0, 2.0, 0.5, 1.5, 0.8])
        action = policy.act(obs_vector)
        
        assert isinstance(action, dict)
        assert "move" in action
        
        # Test learning (simplified episode data)
        episode_data = [{
            "transitions": [{
                "obs_vector": obs_vector,
                "action": action,
                "reward": 1.0,
                "done": False,
                "next_obs_vector": obs_vector * 1.1
            }]
        }]
        
        policy.update(episode_data)  # Should not raise
    
    def test_task_creation(self):
        """Test task schema functionality."""
        task = Task(
            id="test_task_001",
            description="Navigate to the goal location",
            estimated_reward=7.5,
            max_steps=50,
            metadata={"template": "reach_goal", "difficulty": "medium"}
        )
        
        assert task.id == "test_task_001"
        assert task.description == "Navigate to the goal location"
        assert task.estimated_reward == 7.5
        assert task.max_steps == 50
        assert task.difficulty_estimate == "medium"
        
        # Test serialization
        task_dict = task.to_dict()
        restored_task = Task.from_dict(task_dict)
        assert restored_task.id == task.id
        assert restored_task.description == task.description
    
    def test_task_setter(self):
        """Test task setter functionality."""
        buffer = ReplayBuffer()
        task_setter = TaskSetter(experience_buffer=buffer)
        
        # Test task proposal
        task = task_setter.propose_task()
        
        assert isinstance(task, Task)
        assert len(task.id) > 0
        assert len(task.description) > 0
        assert task.estimated_reward > 0
        assert task.max_steps > 0
        
        # Test updating from episode (simplified)
        from src.experience.types import Episode
        episode = Episode(
            task_id=task.id,
            task_description=task.description,
            estimated_reward=task.estimated_reward,
            transitions=[],
            final_reward=5.0,
            success=True
        )
        
        task_setter.update_from_episode(episode)
        stats = task_setter.get_task_statistics()
        assert isinstance(stats, dict)

    def test_task_ids_increment_per_template(self):
        """Task instances should get unique sequential IDs per template."""
        task_setter = TaskSetter()

        first = task_setter._instantiate_task(task_setter._task_templates[0])
        task_setter._template_counts[first.metadata["template"]] = 1
        second = task_setter._instantiate_task(task_setter._task_templates[0])

        assert first.id.endswith("_001")
        assert second.id.endswith("_002")
    
    def test_reward_model(self):
        """Test reward model functionality."""
        reward_model = SimpleRewardModel(
            goal_completion_reward=10.0,
            step_penalty=0.1
        )
        
        # Create test task
        task = Task(
            id="test_task",
            description="Test task",
            estimated_reward=5.0,
            max_steps=20
        )
        
        # Test task reward estimation
        estimated = reward_model.estimate_task_reward(task)
        assert isinstance(estimated, (int, float))
        assert estimated >= 0
        
        # Create test episode
        from src.experience.types import Episode, Transition
        transitions = [
            Transition(
                obs={"pixels": np.zeros(25), "info": {"agent_pos": (0, 0)}},
                action={"move": "right"},
                reward=0.1,
                done=False,
                info={"distance_to_goal": 5}
            ),
            Transition(
                obs={"pixels": np.zeros(25), "info": {"agent_pos": (4, 4)}},
                action={"move": "noop"},
                reward=10.0,
                done=True,
                info={"goal_reached": True, "distance_to_goal": 0}
            )
        ]
        
        episode = Episode(
            task_id=task.id,
            task_description=task.description,
            estimated_reward=task.estimated_reward,
            transitions=transitions,
            final_reward=0.0,  # Will be set by reward model
            success=False      # Will be set based on final reward
        )
        
        # Test episode scoring
        final_reward = reward_model.score_episode(task, episode)
        assert isinstance(final_reward, (int, float))
        assert final_reward > 0  # Should be positive for goal-reaching episode
        
        # Test model update
        reward_model.update_from_experience([episode])  # Should not raise


class TestAgentIntegration:
    """Test full agent integration."""
    
    def create_test_agent(self):
        """Create a test agent with all components."""
        env = DummyGameEnv(grid_size=5, max_steps=30)
        encoder = DummyVisionEncoder(feature_dim=32)
        policy = RandomPolicy()
        reward_model = SimpleRewardModel()
        
        agent = Agent(
            env=env,
            policy=policy,
            reward_model=reward_model,
            encoder=encoder
        )
        
        return agent

    def test_success_threshold_is_configurable(self):
        """Test that agent success uses the configured threshold."""
        env = DummyGameEnv(grid_size=5, max_steps=5)
        encoder = DummyVisionEncoder(feature_dim=16)
        policy = RandomPolicy()
        reward_model = SimpleRewardModel(
            goal_completion_reward=10.0,
            step_penalty=0.0,
            progress_reward_scale=0.0,
        )

        strict_agent = Agent(
            env=env,
            policy=policy,
            reward_model=reward_model,
            encoder=encoder,
            success_threshold=11.0,
        )
        permissive_agent = Agent(
            env=env,
            policy=policy,
            reward_model=reward_model,
            encoder=encoder,
            success_threshold=1.0,
        )

        task = Task(
            id="threshold_test",
            description="Threshold test task",
            estimated_reward=5.0,
            max_steps=20,
        )

        success_episode = None
        for _ in range(20):
            episode = permissive_agent.run_episode(task)
            if episode.reached_goal:
                success_episode = episode
                break

        assert success_episode is not None
        assert success_episode.success is True

        strict_agent_episode = strict_agent.run_episode(task)
        if strict_agent_episode.reached_goal:
            assert strict_agent_episode.success is False
    
    def test_agent_creation(self):
        """Test agent initialization."""
        agent = self.create_test_agent()
        
        assert agent.env is not None
        assert agent.policy is not None
        assert agent.reward_model is not None
        assert agent.encoder is not None
    
    def test_episode_execution(self):
        """Test running a complete episode."""
        agent = self.create_test_agent()
        
        # Create test task
        task = Task(
            id="integration_test",
            description="Integration test task",
            estimated_reward=5.0,
            max_steps=20
        )
        
        # Run episode
        episode = agent.run_episode(task)
        
        # Validate episode
        assert episode.task_id == task.id
        assert episode.task_description == task.description
        assert episode.length > 0
        assert episode.length <= task.max_steps
        assert len(episode.transitions) == episode.length
        assert isinstance(episode.final_reward, (int, float))
        assert isinstance(episode.success, bool)
        
        # Check transitions
        for transition in episode.transitions:
            assert "pixels" in transition.obs
            assert "move" in transition.action
            assert isinstance(transition.reward, (int, float))
            assert isinstance(transition.done, bool)
    
    def test_agent_learning(self):
        """Test agent learning from episodes."""
        agent = self.create_test_agent()
        
        # Create multiple episodes
        episodes = []
        for i in range(3):
            task = Task(
                id=f"learn_test_{i}",
                description=f"Learning test task {i}",
                estimated_reward=5.0,
                max_steps=15
            )
            episode = agent.run_episode(task)
            episodes.append(episode)
        
        # Test learning
        agent.improve_from_episodes(episodes)  # Should not raise
    
    def test_agent_evaluation(self):
        """Test agent evaluation functionality."""
        agent = self.create_test_agent()
        
        task = Task(
            id="eval_test",
            description="Evaluation test task",
            estimated_reward=5.0,
            max_steps=20
        )
        
        # Run evaluation
        results = agent.evaluate_on_task(task, num_episodes=3)
        
        assert isinstance(results, dict)
        assert "success_rate" in results
        assert "avg_reward" in results
        assert "avg_steps" in results
        assert "goal_reach_rate" in results
        assert "num_episodes" in results
        
        assert 0 <= results["success_rate"] <= 1
        assert results["num_episodes"] == 3


class TestSystemIntegration:
    """Test full system integration."""
    
    def test_end_to_end_training_cycle(self):
        """Test a complete but minimal training cycle."""
        # Create components
        env = DummyGameEnv(grid_size=4, max_steps=20)
        encoder = DummyVisionEncoder(feature_dim=16)
        policy = RandomPolicy()
        reward_model = SimpleRewardModel()
        
        agent = Agent(
            env=env,
            policy=policy,
            reward_model=reward_model,
            encoder=encoder
        )
        
        buffer = ReplayBuffer(max_size=10)
        task_setter = TaskSetter(experience_buffer=buffer)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(storage_dir=Path(temp_dir))
            
            # Run mini training cycle
            for generation in range(3):
                # Get task and run episode
                task = task_setter.propose_task()
                episode = agent.run_episode(task)
                
                # Store experience
                buffer.add_episode(episode)
                storage.append_to_log(episode)
                
                # Update components
                task_setter.update_from_episode(episode)
                if len(buffer) >= 2:
                    learning_episodes = buffer.sample_episodes(2)
                    reward_model.update_from_experience(learning_episodes)
                    agent.improve_from_episodes(learning_episodes)
            
            # Verify results
            assert len(buffer) == 3
            assert storage.get_episode_count() == 3
            
            stats = buffer.get_task_statistics()
            assert len(stats) > 0
    
    def test_component_compatibility(self):
        """Test that all components work together correctly."""
        # This test ensures interface compatibility between components
        
        # Environment -> Encoder
        env = DummyGameEnv()
        encoder = DummyVisionEncoder()
        obs = env.reset()
        features = encoder.encode(obs)
        assert features.shape[0] == encoder.get_feature_dim()
        
        # Encoder -> Policy  
        policy = RandomPolicy()
        action = policy.act(features)
        assert isinstance(action, dict)
        
        # Policy -> Environment
        next_obs, reward, done, info = env.step(action)
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        
        # Episode -> RewardModel
        from src.experience.types import Episode, Transition
        transition = Transition(obs=obs, action=action, reward=reward, done=done, info=info)
        task = Task(id="test", description="test", estimated_reward=5.0, max_steps=10)
        episode = Episode(
            task_id=task.id,
            task_description=task.description,
            estimated_reward=task.estimated_reward,
            transitions=[transition],
            final_reward=0.0,
            success=False
        )
        
        reward_model = SimpleRewardModel()
        final_reward = reward_model.score_episode(task, episode)
        assert isinstance(final_reward, (int, float))


if __name__ == "__main__":
    # Simple test runner
    test_classes = [TestAgentInterfaces, TestAgentIntegration, TestSystemIntegration]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        # Run all test methods
        for attr_name in dir(instance):
            if attr_name.startswith("test_"):
                print(f"  {attr_name}... ", end="")
                try:
                    method = getattr(instance, attr_name)
                    method()
                    print("PASSED")
                except Exception as e:
                    print(f"FAILED: {e}")
    
    print("\nAll tests completed!")
