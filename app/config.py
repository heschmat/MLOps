from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "app/model/rf_pipeline.joblib"
    app_name: str = "Heart Risk Prediction API"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
