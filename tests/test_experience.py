"""
Tests for the experience system components.

This module tests the core experience management functionality including
episode storage, replay buffer operations, and serialization.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest
import numpy as np

from src.env.base_env import Action, Observation
from src.experience.buffer import ReplayBuffer
from src.experience.storage import EpisodeStorage, load_episodes, save_episode
from src.experience.types import Episode, Transition


def create_test_observation() -> Observation:
    """Create a test observation."""
    return Observation(
        pixels=np.random.randint(0, 256, (5, 5), dtype=np.uint8).flatten(),
        info={"agent_pos": (2, 3), "goal_pos": (4, 4), "step_count": 10}
    )


def create_test_action() -> Action:
    """Create a test action."""
    return {"move": "up", "type": "discrete"}


def create_test_transition() -> Transition:
    """Create a test transition."""
    return Transition(
        obs=create_test_observation(),
        action=create_test_action(),
        reward=1.5,
        done=False,
        info={"distance_to_goal": 3}
    )


def create_test_episode(task_id: str = "test_task", success: bool = True, num_transitions: int = 5) -> Episode:
    """Create a test episode with specified parameters."""
    transitions = [create_test_transition() for _ in range(num_transitions)]
    
    # Make last transition done
    if transitions:
        transitions[-1].done = True
        if success:
            transitions[-1].info["goal_reached"] = True
    
    return Episode(
        task_id=task_id,
        task_description=f"Test task {task_id}",
        estimated_reward=5.0,
        transitions=transitions,
        final_reward=8.0 if success else 2.0,
        success=success,
        created_at=datetime.utcnow(),
        metadata={"test": True, "template": "test"}
    )


class TestTransition:
    """Test the Transition dataclass."""
    
    def test_transition_creation(self):
        """Test basic transition creation."""
        transition = create_test_transition()
        
        assert isinstance(transition.obs, dict)
        assert "pixels" in transition.obs
        assert isinstance(transition.action, dict)
        assert isinstance(transition.reward, (int, float))
        assert isinstance(transition.done, bool)
        assert isinstance(transition.info, dict)
    
    def test_transition_validation(self):
        """Test transition validation."""
        # Invalid observation
        with pytest.raises(ValueError):
            Transition(
                obs={"invalid": "no pixels"},
                action={"move": "up"},
                reward=1.0,
                done=False
            )
        
        # Invalid action
        with pytest.raises(ValueError):
            Transition(
                obs=create_test_observation(),
                action="not a dict",
                reward=1.0,
                done=False
            )
        
        # Invalid reward
        with pytest.raises(ValueError):
            Transition(
                obs=create_test_observation(),
                action={"move": "up"},
                reward="not a number",
                done=False
            )


class TestEpisode:
    """Test the Episode dataclass."""
    
    def test_episode_creation(self):
        """Test basic episode creation."""
        episode = create_test_episode()
        
        assert episode.task_id == "test_task"
        assert episode.task_description == "Test task test_task"
        assert episode.estimated_reward == 5.0
        assert len(episode.transitions) == 5
        assert episode.final_reward == 8.0
        assert episode.success is True
        assert isinstance(episode.created_at, datetime)
        assert episode.metadata["test"] is True
    
    def test_episode_properties(self):
        """Test episode computed properties."""
        episode = create_test_episode(num_transitions=3)
        
        assert episode.length == 3
        assert episode.total_environment_reward == sum(t.reward for t in episode.transitions)
        assert episode.reached_goal is True  # Set in create_test_episode
        assert episode.was_truncated is False
    
    def test_episode_validation(self):
        """Test episode validation."""
        # Empty task ID
        with pytest.raises(ValueError):
            Episode(
                task_id="",
                task_description="Test",
                estimated_reward=5.0,
                transitions=[],
                final_reward=0.0,
                success=False
            )
        
        # Invalid transitions
        with pytest.raises(ValueError):
            Episode(
                task_id="test",
                task_description="Test",
                estimated_reward=5.0,
                transitions=["not a transition"],
                final_reward=0.0,
                success=False
            )
    
    def test_episode_serialization(self):
        """Test episode to_dict and from_dict."""
        episode = create_test_episode()
        
        # Serialize to dict
        data = episode.to_dict()
        assert isinstance(data, dict)
        assert data["task_id"] == episode.task_id
        assert len(data["transitions"]) == len(episode.transitions)
        
        # Deserialize from dict
        restored_episode = Episode.from_dict(data)
        assert restored_episode.task_id == episode.task_id
        assert restored_episode.task_description == episode.task_description
        assert restored_episode.length == episode.length
        assert restored_episode.success == episode.success


class TestReplayBuffer:
    """Test the ReplayBuffer class."""
    
    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(max_size=10, auto_evict=True)
        
        assert len(buffer) == 0
        assert buffer.max_size == 10
        assert buffer.auto_evict is True
    
    def test_add_episodes(self):
        """Test adding episodes to buffer."""
        buffer = ReplayBuffer()
        
        episode1 = create_test_episode("task1")
        episode2 = create_test_episode("task2", success=False)
        
        buffer.add_episode(episode1)
        buffer.add_episode(episode2)
        
        assert len(buffer) == 2
        assert buffer[0] == episode1
        assert buffer[1] == episode2
    
    def test_buffer_size_limit(self):
        """Test buffer size limit and auto-eviction."""
        buffer = ReplayBuffer(max_size=3, auto_evict=True)
        
        # Add episodes up to limit
        for i in range(5):
            episode = create_test_episode(f"task_{i}")
            buffer.add_episode(episode)
        
        # Should only keep the 3 most recent
        assert len(buffer) == 3
        assert buffer[0].task_id == "task_2"  # Oldest remaining
        assert buffer[2].task_id == "task_4"  # Most recent
    
    def test_sampling(self):
        """Test episode sampling."""
        buffer = ReplayBuffer()
        
        # Add test episodes
        episodes = [create_test_episode(f"task_{i}") for i in range(10)]
        for episode in episodes:
            buffer.add_episode(episode)
        
        # Sample without replacement
        sample = buffer.sample_episodes(5, replace=False)
        assert len(sample) == 5
        assert len(set(ep.task_id for ep in sample)) == 5  # All unique
        
        # Sample with replacement
        sample = buffer.sample_episodes(5, replace=True)
        assert len(sample) == 5
    
    def test_task_filtering(self):
        """Test sampling by task type."""
        buffer = ReplayBuffer()
        
        # Add episodes with different tasks
        for i in range(5):
            buffer.add_episode(create_test_episode("task_a", success=(i % 2 == 0)))
            buffer.add_episode(create_test_episode("task_b", success=True))
        
        # Sample by task
        task_a_episodes = buffer.sample_by_task("task_a")
        assert len(task_a_episodes) == 5
        assert all(ep.task_id == "task_a" for ep in task_a_episodes)
        
        # Sample successful episodes
        successful = buffer.sample_successful_episodes(10)
        assert all(ep.success for ep in successful)
        
        # Sample failed episodes
        failed = buffer.sample_failed_episodes(10)
        assert all(not ep.success for ep in failed)
    
    def test_statistics(self):
        """Test buffer statistics."""
        buffer = ReplayBuffer()
        
        # Add mixed episodes
        for i in range(10):
            success = i < 7  # 70% success rate
            buffer.add_episode(create_test_episode(f"task_{i % 3}", success=success))
        
        stats = buffer.get_task_statistics()
        
        assert len(stats) == 3  # 3 unique task types
        for task_stats in stats.values():
            assert "total_episodes" in task_stats
            assert "successful_episodes" in task_stats
            assert "success_rate" in task_stats
            assert 0 <= task_stats["success_rate"] <= 1


class TestEpisodeStorage:
    """Test the EpisodeStorage class."""
    
    def test_storage_creation(self):
        """Test storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(
                storage_dir=Path(temp_dir),
                format="jsonl"
            )
            
            assert storage.storage_dir == Path(temp_dir)
            assert storage.format == "jsonl"
            assert storage.storage_dir.exists()
    
    def test_episode_saving_loading(self):
        """Test saving and loading episodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save episode to file
            episode = create_test_episode()
            file_path = Path(temp_dir) / "test_episode.json"
            
            save_episode(episode, file_path, format="json")
            assert file_path.exists()
            
            # Load episode back
            loaded_episodes = load_episodes(file_path, format="json")
            assert len(loaded_episodes) == 1
            
            loaded_episode = loaded_episodes[0]
            assert loaded_episode.task_id == episode.task_id
            assert loaded_episode.length == episode.length
            assert loaded_episode.success == episode.success
    
    def test_append_to_log(self):
        """Test appending episodes to log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(storage_dir=Path(temp_dir))
            
            # Add episodes
            episodes = [create_test_episode(f"task_{i}") for i in range(5)]
            for episode in episodes:
                storage.append_to_log(episode)
            
            # Load all episodes
            loaded = storage.load_all_episodes()
            assert len(loaded) == 5
            
            # Check episode count
            assert storage.get_episode_count() == 5
    
    def test_storage_statistics(self):
        """Test storage statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(storage_dir=Path(temp_dir))
            
            # Add test episodes
            for i in range(10):
                episode = create_test_episode(f"task_{i % 3}", success=(i < 6))
                storage.append_to_log(episode)
            
            stats = storage.get_storage_stats()
            
            assert stats["total_episodes"] == 10
            assert stats["unique_tasks"] == 3
            assert 0 <= stats["overall_success_rate"] <= 1
            assert "task_distribution" in stats
    
    def test_cleanup(self):
        """Test episode cleanup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(storage_dir=Path(temp_dir))
            
            # Add many episodes
            for i in range(20):
                episode = create_test_episode(f"task_{i}")
                storage.append_to_log(episode)
            
            assert storage.get_episode_count() == 20
            
            # Cleanup to keep only 10 recent
            removed_count = storage.cleanup_old_episodes(keep_recent=10)
            
            assert removed_count == 10
            assert storage.get_episode_count() == 10


if __name__ == "__main__":
    pytest.main([__file__])
