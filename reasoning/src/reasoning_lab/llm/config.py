from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class that loads environment variables from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI API configuration
    openai_api_key: str

    gemini_api_key: str
    open_router_api_key: str
    anthropic_api_key: str

    # General settings
    max_retries: int = 3
    timeout: int = 60


# Create a global config instance
cfg = Config()


class LLMApiConfig(BaseModel):
    """Configuration class for LLM API settings."""

    gemini_2_5_flash: dict = {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": cfg.gemini_api_key,
    }

    qwen3_14b: dict = {"model": "qwen3:14b", "provider": "ollama"}

    gpt_oss_120b: dict = {
        "model": "gpt-oss-120b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": cfg.open_router_api_key,
    }  #

    claude_opus_4_5: dict = {
        "model": "claude-opus-4-5-20251101",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": cfg.anthropic_api_key,
    }


if __name__ == "__main__":
    print(cfg.model_dump())
