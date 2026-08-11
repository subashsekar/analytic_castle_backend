from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AnalyticCastle"
    APP_ENV: str = "local"
    DEBUG: bool = False
    DATABASE_URL: str


settings = Settings()
