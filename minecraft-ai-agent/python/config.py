import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    agent_api_port: int
    agent_base_url: str


def get_settings() -> Settings:
    port = int(os.getenv("AGENT_API_PORT", "3001"))
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        agent_api_port=port,
        agent_base_url=f"http://localhost:{port}",
    )
