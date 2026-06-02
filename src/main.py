#!/usr/bin/env python3
"""
SIMA-like Agent Training System

Main entry point for training and running the self-improvement agent.
Provides a CLI interface for different modes of operation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Core imports
from .config.config import SIMAConfig, get_baseline_config, load_config, save_config
from .utils.logging_utils import setup_logging, get_logger
from .utils.seed import set_random_seed

# Component imports
from .env.dummy_env import DummyGameEnv
from .env.minecraft_env import MinecraftEnv
from .env.vision import DummyVisionEncoder, FlattenEncoder, MinecraftVisionEncoder
from .agent.policy import RandomPolicy, EpsilonGreedyPolicy, MLPPolicy
from .agent.agent import Agent
from .tasks.task_setter import TaskSetter
from .reward.reward_model import SimpleRewardModel
from .experience.buffer import ReplayBuffer
from .experience.storage import EpisodeStorage

# Training
from .training.self_improvement_loop import self_improvement_cycle

__version__ = "0.1.0"


def create_components(config: SIMAConfig):
    """
    Create and configure all system components based on configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Tuple of (agent, task_setter, reward_model, experience_buffer, storage)
    """
    logger = get_logger(__name__)
    
    # Create environment
    logger.info(f"Creating environment: {config.environment.env_type}")
    if config.environment.env_type == "dummy":
        env = DummyGameEnv(
            grid_size=config.environment.grid_size,
            max_steps=config.environment.max_steps,
            goal_reward=config.environment.goal_reward
        )
    elif config.environment.env_type == "minecraft":
        env = MinecraftEnv(config=config.minecraft)
    else:
        raise ValueError(f"Unsupported environment type: {config.environment.env_type}")
    
    # Create observation encoder
    logger.info(f"Creating encoder: {config.agent.encoder_type}")
    if config.agent.encoder_type == "dummy":
        encoder = DummyVisionEncoder(
            feature_dim=config.agent.encoder_feature_dim,
            add_noise=config.agent.encoder_add_noise
        )
    elif config.agent.encoder_type == "flatten":
        encoder = FlattenEncoder(normalize_pixels=config.environment.normalize_observations)
    elif config.agent.encoder_type == "minecraft":
        encoder = MinecraftVisionEncoder(
            target_size=(config.minecraft.frame_height, config.minecraft.frame_width),
            feature_dim=config.agent.encoder_feature_dim,
            use_cnn=True  # Default to CNN for Minecraft
        )
    else:
        raise ValueError(f"Unsupported encoder type: {config.agent.encoder_type}")
    
    # Create policy
    logger.info(f"Creating policy: {config.agent.policy_type}")
    if config.agent.policy_type == "random":
        policy = RandomPolicy(action_space=config.agent.action_space)
    elif config.agent.policy_type == "epsilon_greedy":
        policy = EpsilonGreedyPolicy(
            action_space=config.agent.action_space,
            epsilon=config.agent.epsilon,
            learning_rate=config.agent.learning_rate,
            discount_factor=config.agent.discount_factor
        )
    elif config.agent.policy_type == "mlp":
        policy = MLPPolicy(
            obs_dim=config.agent.encoder_feature_dim,
            hidden_dims=config.agent.hidden_dims,
            action_space=config.agent.action_space,
            learning_rate=config.agent.mlp_learning_rate
        )
    else:
        raise ValueError(f"Unsupported policy type: {config.agent.policy_type}")
    
    # Create reward model
    logger.info(f"Creating reward model: {config.training.reward_model_type}")
    if config.training.reward_model_type == "simple":
        reward_model = SimpleRewardModel(
            goal_completion_reward=config.training.goal_completion_reward,
            step_penalty=config.training.step_penalty,
            progress_reward_scale=config.training.progress_reward_scale
        )
    else:
        raise ValueError(f"Unsupported reward model type: {config.training.reward_model_type}")
    
    # Create experience components
    logger.info("Creating experience management components")
    experience_buffer = ReplayBuffer(
        max_size=config.experience.max_buffer_size,
        auto_evict=config.experience.auto_evict_old
    )
    
    storage = EpisodeStorage(
        storage_dir=Path(config.experience.storage_dir),
        format=config.experience.storage_format
    )
    
    # Create task setter
    task_setter = TaskSetter(experience_buffer=experience_buffer)
    
    # Create agent
    agent = Agent(
        env=env,
        policy=policy,
        reward_model=reward_model,
        encoder=encoder,
        success_threshold=config.training.success_threshold,
    )
    
    logger.info("All components created successfully")
    return agent, task_setter, reward_model, experience_buffer, storage


def train_mode(config: SIMAConfig) -> None:
    """
    Run the agent in training mode.
    
    Args:
        config: Training configuration
    """
    logger = get_logger(__name__)
    logger.info(f"Starting training mode: {config.experiment_name}")
    
    # Create components
    agent, task_setter, reward_model, experience_buffer, storage = create_components(config)
    
    # Set up metrics path
    experiment_path = Path("experiments") / config.experiment_name
    metrics_path = experiment_path / "metrics.json"
    
    # Run self-improvement cycle
    metrics = self_improvement_cycle(
        agent=agent,
        task_setter=task_setter,
        reward_model=reward_model,
        experience_buffer=experience_buffer,
        storage=storage,
        num_generations=config.training.num_generations,
        episodes_per_generation=config.training.episodes_per_generation,
        learning_frequency=config.training.learning_frequency,
        evaluation_frequency=config.training.evaluation_frequency,
        metrics_path=str(metrics_path)
    )
    
    # Save agent checkpoint
    agent_checkpoint_path = experiment_path / "agent_checkpoint"
    agent.save_agent_state(str(agent_checkpoint_path))
    
    logger.info("Training completed successfully")


def play_once_mode(config: SIMAConfig, task_id: Optional[str] = None) -> None:
    """
    Run a single episode for demonstration/testing.
    
    Args:
        config: System configuration
        task_id: Optional specific task ID to run
    """
    logger = get_logger(__name__)
    logger.info("Running single episode demonstration")
    
    # Create components
    agent, task_setter, reward_model, experience_buffer, storage = create_components(config)
    
    # Get a task
    if task_id:
        # Create a specific task (simplified for demo)
        from .tasks.task_schema import Task
        task = Task(
            id=task_id,
            description=f"Demo task: {task_id}",
            estimated_reward=5.0,
            max_steps=config.environment.max_steps
        )
    else:
        task = task_setter.propose_task()
    
    logger.info(f"Running task: {task.id} - {task.description}")
    
    # Run episode
    episode = agent.run_episode(task)
    
    # Display results
    logger.info("Episode Results:")
    logger.info(f"  Task: {episode.task_id}")
    logger.info(f"  Description: {episode.task_description}")
    logger.info(f"  Steps taken: {episode.length}")
    logger.info(f"  Final reward: {episode.final_reward:.2f}")
    logger.info(f"  Success: {episode.success}")
    logger.info(f"  Goal reached: {episode.reached_goal}")
    logger.info(f"  Estimated vs actual: {episode.estimated_reward:.2f} vs {episode.final_reward:.2f}")
    
    # Show some transitions
    if episode.transitions:
        logger.info("First few actions:")
        for i, transition in enumerate(episode.transitions[:5]):
            action = transition.action.get("move", "unknown")
            reward = transition.reward
            logger.info(f"  Step {i+1}: {action} (reward: {reward:.2f})")
    
    logger.info("Single episode completed")


def inspect_buffer_mode(config: SIMAConfig) -> None:
    """
    Inspect stored experience data.
    
    Args:
        config: System configuration
    """
    logger = get_logger(__name__)
    logger.info("Inspecting stored experience buffer")
    
    # Load storage
    storage = EpisodeStorage(
        storage_dir=Path(config.experience.storage_dir),
        format=config.experience.storage_format
    )
    
    # Get storage statistics
    storage_stats = storage.get_storage_stats()
    
    logger.info("Storage Statistics:")
    logger.info(f"  Total episodes: {storage_stats['total_episodes']}")
    logger.info(f"  Unique tasks: {storage_stats['unique_tasks']}")
    logger.info(f"  Overall success rate: {storage_stats.get('overall_success_rate', 0.0):.2%}")
    logger.info(f"  Average reward: {storage_stats.get('average_reward', 0.0):.2f}")
    logger.info(f"  Storage size: {storage_stats['storage_size_bytes'] / 1024:.1f} KB")
    
    # Show task distribution
    if 'task_distribution' in storage_stats:
        logger.info("Task Distribution:")
        for task_type, count in storage_stats['task_distribution'].items():
            logger.info(f"  {task_type}: {count} episodes")
    
    # Load recent episodes
    try:
        all_episodes = storage.load_all_episodes()
        if all_episodes:
            logger.info("Recent Episodes:")
            for episode in all_episodes[-5:]:  # Show last 5
                logger.info(f"  {episode.task_id}: {episode.success} (reward: {episode.final_reward:.2f})")
    except Exception as e:
        logger.warning(f"Could not load episodes: {e}")
    
    logger.info("Buffer inspection completed")


def view_metrics_mode(config: SIMAConfig) -> None:
    """
    View training metrics and generate a report.
    
    Args:
        config: System configuration  
    """
    from .utils.metrics import TrainingMetrics, generate_training_report
    
    logger = get_logger(__name__)
    logger.info("Viewing training metrics")
    
    # Try to load metrics file
    experiment_path = Path("experiments") / config.experiment_name
    metrics_path = experiment_path / "metrics.json"
    
    if not metrics_path.exists():
        logger.error(f"No metrics file found at {metrics_path}")
        logger.info("Run training first to generate metrics.")
        return
    
    # Load metrics
    metrics = TrainingMetrics.load_from_file(str(metrics_path))
    
    # Generate and display report
    report = generate_training_report(metrics, str(experiment_path))
    print("\n" + report)
    
    # Display summary statistics
    summary = metrics.get_summary()
    if summary.get("status") != "no_data":
        logger.info("Summary Statistics:")
        logger.info(f"  Training Progress: {summary['total_generations']} generations")
        logger.info(f"  Performance: {summary['overall_success_rate']:.1%} success rate")
        logger.info(f"  Best Episode: {summary['best_episode_reward']:.2f} reward")
        logger.info(f"  Task Variety: {summary['task_count']} unique tasks")
        
        if summary.get('most_common_task'):
            logger.info(f"  Focus: {summary['most_common_task']} (most attempted)")
        
        if summary.get('best_task'):
            logger.info(f"  Strength: {summary['best_task']} (highest success)")
    
    logger.info("Metrics viewing completed")


def main() -> int:
    """
    Main entry point with command-line interface.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="SIMA-like Self-Improving Agent Training System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "play-once", "inspect-buffer", "view-metrics"],
        default="train",
        help="Operation mode"
    )
    
    # Environment selection
    parser.add_argument(
        "--env", "--game",
        choices=["dummy", "minecraft"],
        default="dummy",
        help="Environment type to use"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (JSON/YAML)"
    )
    
    # Training parameters
    parser.add_argument(
        "--generations", "--num-generations",
        type=int,
        default=20,
        help="Number of training generations"
    )
    
    parser.add_argument(
        "--episodes-per-gen",
        type=int,
        default=5,
        help="Episodes per generation"
    )
    
    # Single episode parameters
    parser.add_argument(
        "--task-id",
        help="Specific task ID for play-once mode"
    )
    
    # System parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--experiment-name",
        default="sima_experiment",
        help="Name for this experiment"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"SIMA Agent v{__version__}"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config and args.config.exists():
            config = load_config(args.config)
            print(f"Loaded configuration from {args.config}")
        else:
            config = get_baseline_config()
            print("Using baseline configuration")
        
        # Override config with command line arguments
        if args.env:
            config.environment.env_type = args.env
        if args.generations:
            config.training.num_generations = args.generations
        if args.episodes_per_gen:
            config.training.episodes_per_generation = args.episodes_per_gen
        if args.seed is not None:
            config.random_seed = args.seed
        if args.log_level:
            config.log_level = args.log_level
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        
        # Set up logging
        log_file = Path(config.log_file) if config.log_file else None
        setup_logging(log_level=config.log_level, log_file=log_file)
        
        # Set random seed
        if config.random_seed is not None:
            set_random_seed(config.random_seed)
        
        logger = get_logger(__name__)
        logger.info(f"SIMA Agent v{__version__} starting")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Experiment: {config.experiment_name}")
        
        # Save current configuration
        config_save_path = Path(f"experiments/{config.experiment_name}/config.json")
        save_config(config, config_save_path)
        logger.info(f"Configuration saved to {config_save_path}")
        
        # Run selected mode
        if args.mode == "train":
            train_mode(config)
        elif args.mode == "play-once":
            play_once_mode(config, args.task_id)
        elif args.mode == "inspect-buffer":
            inspect_buffer_mode(config)
        elif args.mode == "view-metrics":
            view_metrics_mode(config)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        logger.info("Program completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
