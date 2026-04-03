from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: str = "./models/phi-4-mini-Q4_K_M.gguf"
    port: int = 8000
    n_ctx: int = 4096
    n_gpu_layers: int = 0


settings = Settings()
