# SIMA Minecraft Agent

Two powerful approaches to AI agents for Minecraft:

## 🎮 Conversational Minecraft Agent (NEW!)
**Chat with an AI that plays Minecraft for you!**

A conversational assistant that:
- 💬 **Chats in natural language**: "Go mine that tree" or "Build a house here"
- 🎮 **Controls real Minecraft**: Direct keyboard/mouse automation
- 🤖 **Executes complex tasks**: Understands multi-step instructions
- 🧠 **Learns from conversation**: Remembers context and preferences

**Perfect for:** Playing Minecraft with an AI companion, automating repetitive tasks, learning Minecraft mechanics

### Quick Start
```bash
# Install dependencies for chat agent
pip install loguru mss pyautogui pynput openai

# Check installation
python minecraft_chat.py --check-deps

# Start the conversational agent
python minecraft_chat.py
```

### Usage Examples
```bash
# With OpenAI API key (recommended)
export OPENAI_API_KEY="your-api-key"
python minecraft_chat.py

# Basic mode (no API key needed)
python minecraft_chat.py
```

### Example Conversation
```
👤 You: Go forward and mine some wood
🤖 Agent: I'll help you mine those trees! Moving forward and breaking the wood blocks.

👤 You: Now build a small house with that wood
🤖 Agent: Perfect! I'll use the wood we collected to build a cozy house. Starting with a 5x5 foundation...
```

### Available Actions
- **Movement**: "go forward", "turn left", "walk backward" 
- **Mining**: "break this block", "mine that tree", "dig down"
- **Building**: "place a block", "build a wall", "make a platform"
- **Complex**: "build a house", "find diamonds", "make a farm"

## 🔬 Research Agent (Original)
**Self-improving agent for AI research**

A modular research framework inspired by Google's SIMA 2:
- **Task Setter**: Proposes tasks with estimated rewards
- **Agent**: Executes tasks in game environments  
- **Reward Model**: Evaluates episode performance
- **Self-Generated Experience**: Stores and manages episode data
- **Self-Improvement Loop**: Orchestrates the learning cycle

**Perfect for:** AI research, reinforcement learning experiments, academic studies

## 🚀 Quick Start Options

### Option 1: Conversational Agent (Recommended for most users)
```bash
# Install dependencies
pip install loguru mss pyautogui pynput openai

# Start chatting with your Minecraft assistant  
python minecraft_chat.py

# Example conversation:
# You: "go forward and mine some wood"
# Agent: "I'll move forward and break those wood blocks for you!"
```

### Option 2: Research Agent (For AI researchers)  
```bash
# Basic installation
pip install -e .

# Train the research agent
python -m src.main --mode train --env dummy --generations 5

# Train with Minecraft (requires MineRL)
pip install minerl
python -m src.main --mode train --env minecraft --generations 3
```

## 📚 Detailed Documentation

- **[Conversational Agent Setup](CONVERSATIONAL_AGENT.md)** - Complete guide for the chat-based Minecraft assistant
- **[Research Agent Setup](SETUP.md)** - Detailed setup for AI research applications  

## 🏗️ Architecture (Research Agent)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Task Setter │────│    Agent     │────│ Game Env    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │
       │            ┌──────▼──────┐
       │            │  Episodes   │
       │            └──────┬──────┘
       │                   │
       ▼            ┌──────▼──────┐
┌─────────────┐     │ Self-Gen.   │     ┌─────────────┐
│Reward Model │◄────│ Experience  │────►│   Storage   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

### Basic Installation
```bash
pip install -e .
```

### For Development
```bash
pip install -e ".[dev]"
```

### For Minecraft Support (Optional)
For MineRL environment support:
```bash
pip install minerl
```

Note: MineRL has additional dependencies including Java JDK 8. See the [MineRL installation guide](https://minerl.readthedocs.io/en/latest/tutorials/index.html) for detailed setup instructions.

For raw Minecraft control (experimental):
```bash
pip install pyautogui pynput mss pillow
```

## Quick Start

### Train the agent (dummy environment)
```bash
python -m src.main --mode train --generations 10 --episodes-per-gen 5
```

### Train the agent in Minecraft (MineRL)
```bash
python -m src.main --mode train --env minecraft --generations 5 --episodes-per-gen 3
```

### Run a single episode (dummy environment)
```bash
python -m src.main --mode play-once --task-id "reach_goal"
```

### Run a single Minecraft episode
```bash
python -m src.main --mode play-once --env minecraft --task-id "collect_wood"
```

### Inspect stored experience
```bash
python -m src.main --mode inspect-buffer
```

### View saved metrics
```bash
python -m src.main --mode view-metrics
```

## Project Structure

```
sima_like_agent/
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py                    # Entry point and CLI
│   ├── config/
│   │   └── config.py             # Configuration classes
│   ├── env/
│   │   ├── base_env.py           # Abstract game environment
│   │   ├── dummy_env.py          # Simple test environment
│   │   ├── minecraft_env.py      # Minecraft environment wrapper ⭐
│   │   ├── io_controller.py      # Game I/O interfaces
│   │   └── vision.py             # Observation encoding
│   ├── agent/
│   │   ├── policy.py             # Policy implementations
│   │   └── agent.py              # Main agent orchestrator
│   ├── tasks/
│   │   ├── task_schema.py        # Task data structures
│   │   └── task_setter.py        # Task generation logic ⭐
│   ├── reward/
│   │   └── reward_model.py       # Episode scoring ⭐
│   ├── experience/
│   │   ├── types.py              # Core data types
│   │   ├── buffer.py             # In-memory experience store
│   │   └── storage.py            # Persistent experience store
│   ├── training/
│   │   └── self_improvement_loop.py  # Main training loop
│   └── utils/
│       ├── logging_utils.py      # Structured logging
│       └── seed.py               # Random seed management
└── tests/
    ├── test_experience.py        # Experience system tests
    └── test_agent_interfaces.py  # Agent integration tests
```

## Key Components

### Environment Interface
The `GameEnv` abstract base class provides a clean interface for any video game. Includes:
- `DummyGameEnv` for testing and development
- `MinecraftEnv` for Minecraft gameplay via MineRL or raw control

### Minecraft Support
The system now supports Minecraft through two modes:
- **MineRL**: Uses the MineRL research platform for structured Minecraft gameplay
- **Raw Control**: Direct keyboard/mouse control of Minecraft client (experimental)

Minecraft-specific features:
- Specialized action space for movement, building, mining, crafting
- Minecraft vision encoder for processing pixel observations
- Task templates for common Minecraft objectives (collect wood, build structures, etc.)
- Reward model understanding Minecraft progress indicators

### Experience System
Episodes are stored both in-memory (`ReplayBuffer`) and persistently (`storage.py`) as the "Self-Generated Experience" that drives learning.

### Task Generation
The `TaskSetter` proposes new tasks based on previous performance, with hooks for LLM-based task generation.

### Reward Learning
The `RewardModel` scores episodes, with clear interfaces for plugging in learned reward functions or LLM-based evaluation. Now includes Minecraft-specific scoring based on inventory changes, health preservation, and exploration.

## Minecraft Configuration

The Minecraft environment is configured through `MinecraftConfig`:

```python
from src.config.config import MinecraftConfig

minecraft_config = MinecraftConfig(
    use_minerl=True,
    env_id="MineRLTreechop-v0",
    frame_width=160,
    frame_height=120,
    frame_skip=4,
    max_episode_steps=1000,
)
```

### MineRL Environments
Supported MineRL environments include:
- `MineRLNavigateDense-v0`: Navigation with dense rewards
- `MineRLTreechop-v0`: Tree chopping tasks
- `MineRLObtainDiamond-v0`: Complex diamond obtaining task

### Raw Control Mode
For direct Minecraft control:
1. Start Minecraft client
2. Set `use_minerl=False` in config
3. Ensure Minecraft window is focused and accessible

**Note**: Raw control mode is experimental and requires additional setup.

## Development

### Running Tests
```bash
pip install -e ".[dev]"
python -m pytest tests/
```

### Code Quality
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## TODOs for Production Use

### Core System
1. **RL Algorithms**: Implement proper policy gradient/Q-learning in `policy.py`
2. **Scalability**: Add distributed training and experience storage
3. **Vision Models**: Plug in CNN/transformer backbones in `vision.py`

### Environment Integration
1. **More Games**: Add support for other games beyond Minecraft
2. **Real Input Control**: Complete the raw Minecraft controller implementation
3. **MineRL Integration**: Test and optimize MineRL environment performance

### Intelligence Components  
1. **Reward Learning**: Train neural reward models or integrate LLM evaluation
2. **Task Generation**: Add LLM-based creative task generation
3. **Hierarchical Tasks**: Implement task decomposition and sub-goals

### Minecraft-Specific Enhancements
1. **Advanced Actions**: Implement crafting, building, combat mechanics
2. **State Detection**: Add inventory tracking, health monitoring, environment analysis
3. **Long-Term Goals**: Add support for complex, multi-step Minecraft objectives

## License

MIT License - see LICENSE file for details.
