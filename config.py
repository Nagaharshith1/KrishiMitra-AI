import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration settings for the KRISHIMITRA AI application.
    Loads environment variables and provides access to various API keys,
    database URIs, and other application settings.
    """

    # --- Database Settings ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/krishimitra_db")

    # --- AI/ML Service API Keys ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-your-openai-api-key")
    GOOGLE_CLOUD_SPEECH_KEY_PATH: str = os.getenv("GOOGLE_CLOUD_SPEECH_KEY_PATH", "path/to/google_speech_key.json")
    GOOGLE_CLOUD_TRANSLATE_KEY_PATH: str = os.getenv("GOOGLE_CLOUD_TRANSLATE_KEY_PATH", "path/to/google_translate_key.json")
    GOOGLE_CLOUD_TTS_KEY_PATH: str = os.getenv("GOOGLE_CLOUD_TTS_KEY_PATH", "path/to/google_tts_key.json")
    # Example for a custom ML model service (e.g., for Crop Doctor image analysis)
    CROP_DOCTOR_MODEL_API_URL: str = os.getenv("CROP_DOCTOR_MODEL_API_URL", "http://localhost:8001/predict")
    CROP_DOCTOR_API_KEY: str = os.getenv("CROP_DOCTOR_API_KEY", "your-crop-doctor-api-key")

    # --- External Integrations API Keys ---
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "your-weather-api-key")
    WEATHER_API_URL: str = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5/weather")
    MARKET_DATA_API_KEY: str = os.getenv("MARKET_DATA_API_KEY", "your-market-data-api-key")
    MARKET_DATA_API_URL: str = os.getenv("MARKET_DATA_API_URL", "https://api.example.com/marketdata")
    GOVT_SCHEMES_API_URL: str = os.getenv("GOVT_SCHEMES_API_URL", "https://api.example.com/govtschemes")

    # --- Application Settings ---
    APP_NAME: str = "KRISHIMITRA AI"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    ALLOWED_LANGUAGES: list[str] = ["en", "hi", "te", "ta"] # English, Hindi, Telugu, Tamil
    DEFAULT_LANGUAGE: str = "en"
    AUDIO_UPLOAD_DIR: str = os.getenv("AUDIO_UPLOAD_DIR", "/tmp/audio_uploads")
    IMAGE_UPLOAD_DIR: str = os.getenv("IMAGE_UPLOAD_DIR", "/tmp/image_uploads")

    # --- Security Settings ---
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-for-jwt-or-csrf") # Example for future authentication

    # --- Logging Settings ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "krishimitra.log")

    # --- Voice Pipeline Settings ---
    # Max audio duration for STT in seconds
    MAX_AUDIO_DURATION_SEC: int = int(os.getenv("MAX_AUDIO_DURATION_SEC", "30"))
    # Audio sample rate for STT
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    # --- AI Safety Settings ---
    SAFETY_THRESHOLD: float = float(os.getenv("SAFETY_THRESHOLD", "0.7")) # Confidence threshold for AI responses

    def __init__(self) -> None:
        """
        Initializes the Config class and ensures necessary directories exist.
        """
        self._create_upload_dirs()

    def _create_upload_dirs(self) -> None:
        """
        Creates necessary upload directories if they don't exist.
        """
        os.makedirs(self.AUDIO_UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.IMAGE_UPLOAD_DIR, exist_ok=True)

# Instantiate the configuration
settings = Config()

if __name__ == "__main__":
    # Example usage and verification
    print("--- KRISHIMITRA AI Configuration ---")
    print(f"App Name: {settings.APP_NAME}")
    print(f"Debug Mode: {settings.DEBUG_MODE}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"OpenAI API Key (first 5 chars): {settings.OPENAI_API_KEY[:5]}...")
    print(f"Weather API Key (first 5 chars): {settings.WEATHER_API_KEY[:5]}...")
    print(f"Allowed Languages: {settings.ALLOWED_LANGUAGES}")
    print(f"Audio Upload Dir: {settings.AUDIO_UPLOAD_DIR} (exists: {os.path.exists(settings.AUDIO_UPLOAD_DIR)})")
    print(f"Image Upload Dir: {settings.IMAGE_UPLOAD_DIR} (exists: {os.path.exists(settings.IMAGE_UPLOAD_DIR)})")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Crop Doctor Model URL: {settings.CROP_DOCTOR_MODEL_API_URL}")