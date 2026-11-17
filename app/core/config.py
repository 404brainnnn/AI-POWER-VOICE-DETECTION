import os


class Settings:
    def __init__(self) -> None:
        self.app_env = os.getenv("APP_ENV", "development")
        self.api_key = os.getenv("API_KEY")
        self.default_sample_rate = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
        self.window_seconds = float(os.getenv("WINDOW_SECONDS", "3.0"))
        self.hop_seconds = float(os.getenv("HOP_SECONDS", "1.0"))
        self.vad_threshold_db = float(os.getenv("VAD_THRESHOLD_DB", "-45"))
        self.enable_mock_detector = os.getenv("ENABLE_MOCK_DETECTOR", "false").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.mongodb_database = os.getenv("MONGODB_DATABASE", "voice_clone_detector")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
        self.model_checkpoint = os.getenv("MODEL_CHECKPOINT")
        self.model_config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model.yaml")
        self.thresholds_path = os.getenv("THRESHOLDS_PATH", "artifacts/calibration/thresholds.json")
        self.model_device = os.getenv("MODEL_DEVICE", "cuda")
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
        self.max_batch_wait_ms = int(os.getenv("MAX_BATCH_WAIT_MS", "20"))
        self.hide_raw_scores = os.getenv("HIDE_RAW_SCORES", "false").lower() == "true"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


settings = Settings()
