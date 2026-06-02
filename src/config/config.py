from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "EnvironmentConfig", "AgentConfig", "TrainingConfig", 
    "ExperienceConfig", "MinecraftConfig", "SIMAConfig", "load_config", "save_config"
]


@dataclass
class EnvironmentConfig:
    """Configuration for the game environment."""
    
    # Environment type
    env_type: str = "dummy"  # "dummy", "minecraft", "real_game", etc.
    
    # Dummy environment settings
    grid_size: int = 5
    max_steps: int = 50
    goal_reward: float = 10.0
    
    # Real game settings (for future use)
    game_executable_path: Optional[str] = None
    screen_resolution: tuple = (640, 480)
    input_delay_ms: int = 50
    
    # Observation settings
    observation_type: str = "pixels"  # "pixels", "features", "hybrid"
    normalize_observations: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass
class MinecraftConfig:
    """Configuration for Minecraft environment."""
    
    # Connection method
    use_minerl: bool = True           # if False, fall back to raw keyboard/mouse control
    env_id: str = "MineRLTreechop-v0" # MineRL environment ID
    
    # Frame settings
    frame_width: int = 160
    frame_height: int = 120
    frame_skip: int = 4               # number of game frames per agent step
    
    # Episode settings
    max_episode_steps: int = 1000
    
    # Raw control settings (when use_minerl=False)
    game_window_title: str = "Minecraft"
    control_sensitivity: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")


@dataclass
class AgentConfig:
    """Configuration for the agent and its policy."""
    
    # Policy settings
    policy_type: str = "random"  # "random", "epsilon_greedy", "mlp"
    
    # Random policy settings
    action_space: List[str] = field(default_factory=lambda: ["up", "down", "left", "right", "noop"])
    
    # Epsilon-greedy settings
    epsilon: float = 0.1
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    
    # MLP policy settings
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    mlp_learning_rate: float = 0.001
    
    # Vision/encoding settings
    encoder_type: str = "dummy"  # "dummy", "flatten", "cnn"
    encoder_feature_dim: int = 128
    encoder_add_noise: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    
    # Main training loop
    num_generations: int = 20
    episodes_per_generation: int = 5
    learning_frequency: int = 1  # Update every N generations
    evaluation_frequency: int = 5  # Evaluate every N generations
    
    # Task setter settings
    task_templates: List[str] = field(default_factory=lambda: [
        "reach_goal", "avoid_obstacles", "collect_items", "time_limit", "exploration"
    ])
    
    # Reward model settings
    reward_model_type: str = "simple"  # "simple", "learned", "llm"
    goal_completion_reward: float = 10.0
    step_penalty: float = 0.1
    progress_reward_scale: float = 2.0
    success_threshold: float = 5.0
    
    # Evaluation settings
    eval_episodes_per_task: int = 3
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive")
        if self.episodes_per_generation <= 0:
            raise ValueError("episodes_per_generation must be positive")


@dataclass
class ExperienceConfig:
    """Configuration for experience storage and replay."""
    
    # Replay buffer settings
    max_buffer_size: Optional[int] = 1000  # None for unlimited
    auto_evict_old: bool = True
    
    # Storage settings
    storage_dir: str = "./experience_data"
    storage_format: str = "jsonl"  # "json", "jsonl", "pickle"
    auto_backup: bool = True
    backup_frequency: int = 100  # Backup every N episodes
    
    # Sampling settings
    learning_sample_size: int = 50
    priority_sampling: bool = False  # For future use
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_buffer_size is not None and self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive or None")


@dataclass
class SIMAConfig:
    """
    Main configuration class that combines all component configurations.
    
    This represents the complete configuration for a SIMA-like training run,
    including environment, agent, training, and experience settings.
    """
    
    # Component configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
    minecraft: MinecraftConfig = field(default_factory=MinecraftConfig)
    
    # Global settings
    experiment_name: str = "sima_experiment"
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    log_file: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Save every N generations
    
    # Hardware settings
    device: str = "cpu"  # "cpu", "cuda", "mps"
    num_workers: int = 1  # For future distributed training
    
    def __post_init__(self) -> None:
        """Validate configuration and set up derived paths."""
        # Set up logging file path
        if self.log_file is None:
            self.log_file = f"logs/{self.experiment_name}.log"
            
        # Set up storage directory
        if not self.experience.storage_dir.startswith("/"):
            # Relative path - make it relative to experiment
            self.experience.storage_dir = f"experiments/{self.experiment_name}/{self.experience.storage_dir}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.__dict__,
            "agent": self.agent.__dict__,
            "training": self.training.__dict__,
            "experience": self.experience.__dict__,
            "minecraft": self.minecraft.__dict__,
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_frequency": self.checkpoint_frequency,
            "device": self.device,
            "num_workers": self.num_workers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SIMAConfig":
        """Create configuration from dictionary."""
        env_config = EnvironmentConfig(**data.get("environment", {}))
        agent_config = AgentConfig(**data.get("agent", {}))
        training_config = TrainingConfig(**data.get("training", {}))
        experience_config = ExperienceConfig(**data.get("experience", {}))
        minecraft_config = MinecraftConfig(**data.get("minecraft", {}))
        
        return cls(
            environment=env_config,
            agent=agent_config,
            training=training_config,
            experience=experience_config,
            minecraft=minecraft_config,
            experiment_name=data.get("experiment_name", "sima_experiment"),
            random_seed=data.get("random_seed", 42),
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
            save_checkpoints=data.get("save_checkpoints", True),
            checkpoint_frequency=data.get("checkpoint_frequency", 10),
            device=data.get("device", "cpu"),
            num_workers=data.get("num_workers", 1)
        )


def load_config(config_path: Path) -> SIMAConfig:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Loaded configuration object
    """
    import json
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r") as f:
            data = json.load(f)
    elif config_path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return SIMAConfig.from_dict(data)


def save_config(config: SIMAConfig, config_path: Path) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration object to save
        config_path: Path to save configuration file
    """
    import json
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    
    if config_path.suffix.lower() == ".json":
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
    elif config_path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


# Predefined configuration templates

def get_baseline_config() -> SIMAConfig:
    """Get a baseline configuration for testing and development."""
    return SIMAConfig(
        experiment_name="baseline_dummy_env",
        environment=EnvironmentConfig(
            env_type="dummy",
            grid_size=5,
            max_steps=50
        ),
        agent=AgentConfig(
            policy_type="random",
            encoder_type="dummy"
        ),
        training=TrainingConfig(
            num_generations=10,
            episodes_per_generation=3
        )
    )


def get_learning_config() -> SIMAConfig:
    """Get a configuration for actual learning experiments."""
    return SIMAConfig(
        experiment_name="learning_experiment",
        environment=EnvironmentConfig(
            env_type="dummy",
            grid_size=7,
            max_steps=75
        ),
        agent=AgentConfig(
            policy_type="epsilon_greedy",
            epsilon=0.2,
            learning_rate=0.01,
            encoder_type="dummy"
        ),
        training=TrainingConfig(
            num_generations=50,
            episodes_per_generation=5,
            learning_frequency=2
        )
    )


def get_advanced_config() -> SIMAConfig:
    """Get a configuration for advanced experiments with neural policies."""
    return SIMAConfig(
        experiment_name="advanced_mlp_experiment",
        environment=EnvironmentConfig(
            env_type="dummy",
            grid_size=10,
            max_steps=100
        ),
        agent=AgentConfig(
            policy_type="mlp",
            hidden_dims=[256, 128, 64],
            mlp_learning_rate=0.0003,
            encoder_type="dummy",
            encoder_feature_dim=256
        ),
        training=TrainingConfig(
            num_generations=100,
            episodes_per_generation=10,
            learning_frequency=1,
            evaluation_frequency=5
        ),
        experience=ExperienceConfig(
            max_buffer_size=2000,
            learning_sample_size=100
        )
    )


# Example usage:
# 
# # Create and save a configuration
# config = get_learning_config()
# save_config(config, Path("experiments/my_experiment/config.json"))
#
# # Load configuration
# config = load_config(Path("config.json"))
#
# # Modify configuration programmatically
# config.training.num_generations = 100
# config.agent.learning_rate = 0.005
