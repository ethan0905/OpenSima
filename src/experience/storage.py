from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

from .types import Episode

__all__ = ["save_episode", "load_episodes", "append_episode", "EpisodeStorage"]


def save_episode(episode: Episode, path: Path, format: str = "json") -> None:
    """
    Save a single episode to disk.
    
    Args:
        episode: Episode to save
        path: File path to save to
        format: Storage format ("json" or "pickle")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(path, "w") as f:
            json.dump(episode.to_dict(), f, indent=2)
    elif format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(episode, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_episodes(path: Path, format: str = "json") -> List[Episode]:
    """
    Load episodes from a file.
    
    Args:
        path: Path to file containing episodes
        format: Storage format ("json", "pickle", or "jsonl")
        
    Returns:
        List of loaded episodes
    """
    if not path.exists():
        return []
        
    episodes = []
    
    if format == "json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                episodes = [Episode.from_dict(ep_data) for ep_data in data]
            else:
                episodes = [Episode.from_dict(data)]
                
    elif format == "jsonl":
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    episodes.append(Episode.from_dict(data))
                    
    elif format == "pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):
                episodes = data
            else:
                episodes = [data]
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    return episodes


def append_episode(path: Path, episode: Episode, format: str = "jsonl") -> None:
    """
    Append an episode to an existing file.
    
    Args:
        episode: Episode to append
        path: File path to append to
        format: Storage format (only "jsonl" supported for appending)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(path, "a") as f:
            json.dump(episode.to_dict(), f)
            f.write("\n")
    else:
        raise ValueError(f"Appending only supported for jsonl format, got: {format}")


class EpisodeStorage:
    """
    Persistent storage manager for episodes representing "Self-Generated Experience".
    
    This class manages the disk-based storage of episodes, which serves as the
    persistent component of the experience system in the SIMA architecture.
    
    The storage supports:
    - Individual episode files
    - Batched episode collections  
    - Append-only logs for streaming
    - Multiple serialization formats
    """

    def __init__(self, storage_dir: Path, format: str = "jsonl", auto_create: bool = True):
        """
        Initialize episode storage.
        
        Args:
            storage_dir: Directory to store episode files
            format: Default storage format ("json", "jsonl", or "pickle")
            auto_create: Whether to create storage directory if it doesn't exist
        """
        self.storage_dir = Path(storage_dir)
        self.format = format
        
        if auto_create:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
        # Main episode log file
        self.episode_log = self.storage_dir / "episodes.jsonl"
        
    def save_episode(self, episode: Episode, filename: Optional[str] = None) -> Path:
        """
        Save a single episode to storage.
        
        Args:
            episode: Episode to save
            filename: Optional custom filename. If None, auto-generated.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = episode.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{episode.task_id}_{timestamp}.{self.format}"
            
        file_path = self.storage_dir / filename
        save_episode(episode, file_path, self.format)
        return file_path
        
    def append_to_log(self, episode: Episode) -> None:
        """
        Append episode to the main episode log.
        
        Args:
            episode: Episode to append to log
        """
        append_episode(self.episode_log, episode, "jsonl")
        
    def load_all_episodes(self) -> List[Episode]:
        """
        Load all episodes from the main log file.
        
        Returns:
            List of all stored episodes
        """
        return load_episodes(self.episode_log, "jsonl")
        
    def load_episodes_by_task(self, task_id: str) -> List[Episode]:
        """
        Load all episodes for a specific task.
        
        Args:
            task_id: Task ID to filter by
            
        Returns:
            List of episodes for the specified task
        """
        all_episodes = self.load_all_episodes()
        return [ep for ep in all_episodes if ep.task_id == task_id]
        
    def get_episode_count(self) -> int:
        """
        Get the total number of stored episodes.
        
        Returns:
            Number of episodes in storage
        """
        try:
            with open(self.episode_log, "r") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0
            
    def get_storage_stats(self) -> dict:
        """
        Get statistics about stored episodes.
        
        Returns:
            Dictionary with storage statistics
        """
        episodes = self.load_all_episodes()
        
        if not episodes:
            return {
                "total_episodes": 0,
                "unique_tasks": 0,
                "task_distribution": {},
                "overall_success_rate": 0.0,
                "average_reward": 0.0,
                "storage_size_bytes": self._get_storage_size(),
            }
            
        task_counts = {}
        success_count = 0
        total_reward = 0.0
        
        for episode in episodes:
            task_counts[episode.task_id] = task_counts.get(episode.task_id, 0) + 1
            if episode.success:
                success_count += 1
            total_reward += episode.final_reward
            
        return {
            "total_episodes": len(episodes),
            "unique_tasks": len(task_counts),
            "task_distribution": task_counts,
            "overall_success_rate": success_count / len(episodes),
            "average_reward": total_reward / len(episodes),
            "storage_size_bytes": self._get_storage_size()
        }
        
    def cleanup_old_episodes(self, keep_recent: int = 1000) -> int:
        """
        Remove old episodes to manage storage size.
        
        Args:
            keep_recent: Number of most recent episodes to keep
            
        Returns:
            Number of episodes removed
        """
        episodes = self.load_all_episodes()
        
        if len(episodes) <= keep_recent:
            return 0
            
        # Keep most recent episodes
        episodes_to_keep = sorted(episodes, key=lambda ep: ep.created_at)[-keep_recent:]
        
        # Rewrite the log file with only recent episodes
        backup_path = self.episode_log.with_suffix(".backup")
        self.episode_log.rename(backup_path)
        
        try:
            for episode in episodes_to_keep:
                self.append_to_log(episode)
            backup_path.unlink()  # Remove backup if successful
            return len(episodes) - keep_recent
        except Exception:
            # Restore backup if something went wrong
            backup_path.rename(self.episode_log)
            raise
            
    def export_episodes(self, output_path: Path, format: str = "json", task_id: Optional[str] = None) -> None:
        """
        Export episodes to a different format or location.
        
        Args:
            output_path: Path to export to
            format: Export format
            task_id: Optional task ID filter
        """
        if task_id:
            episodes = self.load_episodes_by_task(task_id)
        else:
            episodes = self.load_all_episodes()
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump([ep.to_dict() for ep in episodes], f, indent=2)
        elif format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(episodes, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _get_storage_size(self) -> int:
        """Get total storage size in bytes."""
        total_size = 0
        for file_path in self.storage_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
