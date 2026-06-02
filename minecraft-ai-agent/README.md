# Minecraft AI Agent MVP

A local Python + Mineflayer MVP where a user chats with an OpenAI-powered planner. The Python app reads current Minecraft state from a Node.js Mineflayer agent, asks OpenAI for a strict JSON action plan, validates the plan with Pydantic and extra safety rules, then sends only validated whitelisted actions to the Mineflayer HTTP API for execution.

## Requirements

- Python 3.11+
- Node.js 20+
- Minecraft Java Edition
- Local or remote Minecraft server
- OpenAI API key

## Install

```bash
cd minecraft-ai-agent
npm install
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Defaults:

```bash
OPENAI_MODEL=gpt-4.1-mini
MINECRAFT_HOST=localhost
MINECRAFT_PORT=25565
MINECRAFT_AGENT_USERNAME=AI_Agent
MINECRAFT_OPS=
AGENT_API_PORT=3001
AGENT_VIEWER_PORT=3002
AGENT_VIEWER_DISTANCE=6
AGENT_VIEWER_ENABLED=true
REASONING_MAX_ITERATIONS=5
REASONING_MAX_STEPS_PER_PLAN=4
REASONING_VERBOSE=true
```

## Run

Start your Minecraft Java server manually first, or use the included local helper:

```bash
cd minecraft-ai-agent
ACCEPT_MINECRAFT_EULA=true ./scripts/start_minecraft_server.sh
```

The helper downloads the official latest release server jar, writes `minecraft-server/server.properties`, and starts the server on `25565`. By default it sets `online-mode=false` for local bot testing. Do not use that setting for a public server.

If the latest server requires a newer Java runtime, pin a known Minecraft version:

```bash
MINECRAFT_VERSION=1.21.8 ACCEPT_MINECRAFT_EULA=true ./scripts/start_minecraft_server.sh
```

To start the local server in creative mode:

```bash
MINECRAFT_VERSION=1.21.8 MINECRAFT_GAMEMODE=creative ACCEPT_MINECRAFT_EULA=true ./scripts/start_minecraft_server.sh
```

To allow `/gamemode`, `/tp`, `/give`, and other slash commands from Minecraft chat, make your Minecraft username an operator on the local server:

```bash
MINECRAFT_VERSION=1.21.8 MINECRAFT_OPS=YourMinecraftName ACCEPT_MINECRAFT_EULA=true ./scripts/start_minecraft_server.sh
```

For multiple operators, use a comma-separated list:

```bash
MINECRAFT_OPS=YourMinecraftName,AI_Agent ACCEPT_MINECRAFT_EULA=true ./scripts/start_minecraft_server.sh
```

The name must match the player name shown in the server player list. Restart the server after changing `MINECRAFT_OPS`. `enable-command-block` only affects command blocks; chat commands are controlled by operator permissions.

Make sure the server accepts the configured bot username and connection mode.

In one terminal:

```bash
cd minecraft-ai-agent
npm run agent
```

Open the realtime dashboard in a browser:

```bash
http://localhost:3001/dashboard
```

The dashboard shows the agent's live Prismarine Viewer perspective plus health, food, position, held item, inventory, nearby entities, and nearby tracked blocks. The viewer itself runs on `AGENT_VIEWER_PORT`, which defaults to `3002`.

In another terminal:

```bash
cd minecraft-ai-agent
python python/main.py
```

Example chat requests:

- `follow me`
- `collect 10 oak logs`
- `build a basic shelter`
- `eat if you are hungry`
- `stop`

## Troubleshooting

The Mineflayer bot is a separate Minecraft player named `AI_Agent`. It does not control your player camera, keyboard, mouse, or single-player world.

To see the bot:

1. Start the local server.
2. Join `localhost:25565` from Minecraft Java using Multiplayer or Direct Connection.
3. Start the bot with `npm run agent`.
4. Look for `AI_Agent` in the player list and in chat.

If actions return success but you see nothing happen, check that your Minecraft client is connected to the same server, not a single-player world. You can also inspect state directly:

```bash
curl http://localhost:3001/state
```

## Flow

1. User enters a natural-language objective in the Python CLI.
2. Python starts a reasoning loop with a default budget of 5 iterations.
3. Each iteration reads current state from `GET http://localhost:3001/state`.
4. OpenAI plans the next small action batch, limited by `REASONING_MAX_STEPS_PER_PLAN`.
5. Python validates the batch with Pydantic and safety rules, then sends it to `POST http://localhost:3001/action-plan`.
6. Mineflayer executes the whitelisted steps and returns per-step results.
7. Python reads fresh state, asks OpenAI to verify progress, and continues until the objective is done, blocked, cancelled, or the iteration budget is reached.

## Safety Model

- The LLM never directly controls Mineflayer.
- The LLM cannot send arbitrary Minecraft commands.
- Python validates every plan before execution.
- Node.js validates again before running actions.
- Mineflayer only executes known whitelisted actions.
- Suspicious command-like strings such as `/give`, `/tp`, `/op`, and `/execute` are rejected.
- Action plans are limited to 12 steps.
- Reasoning loop tasks are bounded by `REASONING_MAX_ITERATIONS`.
- Counts and movement distances are bounded.
- High-risk plans require confirmation.

## Supported Actions

- `move_to`
- `follow_player`
- `collect_block`
- `craft_item`
- `place_block`
- `build_structure`
- `attack_entity`
- `eat_food`
- `stop`

`build_structure` currently implements only `basic_shelter` robustly. Other structure names are accepted by schema for future expansion but rejected by the Node handler until implemented.

## Known Limitations

- No authentication.
- No database or memory.
- No screen control, keyboard/mouse automation, or computer vision.
- Nearby block scanning is intentionally compact and may miss resources outside the scan radius.
- Crafting is basic and depends on Mineflayer recipe availability and nearby crafting tables.
- Combat is simple and not a full survival strategy.
- Building is intentionally small and may partially complete if placement geometry or inventory is insufficient.
- A real Minecraft server is required to fully test Mineflayer execution.

## Next Improvements

- Add richer action-specific Pydantic models.
- Add retries for explicitly safe actions like nearby collection.
- Add persistent task state and cancellation.
- Add better shelter and structure blueprints.
- Add tool-use telemetry and structured logs.
- Add integration tests with a disposable Minecraft server.
