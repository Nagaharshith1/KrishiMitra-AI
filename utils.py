import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec

from config import Config

# --- Custom Logging Setup ---

class CustomFormatter(logging.Formatter):
    """
    Custom formatter for log messages to include timestamp, log level, module,
    and message in a structured format.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging(name: str = "krishimitra_ai") -> logging.Logger:
    """
    Sets up a custom logger for the application.

    Args:
        name: The name of the logger.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Prevent adding multiple handlers if called multiple times
        logger.setLevel(logging.DEBUG if Config.DEBUG_MODE else logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if Config.DEBUG_MODE else logging.INFO)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        # Optional: File handler (for production deployments)
        # if not Config.DEBUG_MODE:
        #     log_dir = "logs"
        #     os.makedirs(log_dir, exist_ok=True)
        #     fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        #     fh.setLevel(logging.INFO)
        #     fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        #     logger.addHandler(fh)

    return logger

logger = setup_logging()

# --- Error Handling Helpers ---

class AppError(Exception):
    """
    Base exception class for application-specific errors.
    """
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        logger.error(f"AppError caught: {message} (Status: {status_code}, Details: {details})", exc_info=True)

class ExternalAPIError(AppError):
    """
    Exception raised when an external API call fails.
    """
    def __init__(self, api_name: str, message: str, status_code: int = 500, original_error: Optional[Exception] = None):
        super().__init__(f"Error from {api_name} API: {message}", status_code)
        self.api_name = api_name
        self.original_error = original_error

class AIProcessingError(AppError):
    """
    Exception raised when there's an issue with AI model processing.
    """
    def __init__(self, model_name: str, message: str, status_code: int = 500, original_error: Optional[Exception] = None):
        super().__init__(f"AI processing error in {model_name}: {message}", status_code)
        self.model_name = model_name
        self.original_error = original_error

class DatabaseError(AppError):
    """
    Exception raised when there's an issue with database operations.
    """
    def __init__(self, operation: str, message: str, status_code: int = 500, original_error: Optional[Exception] = None):
        super().__init__(f"Database error during {operation}: {message}", status_code)
        self.operation = operation
        self.original_error = original_error

P = ParamSpec("P")
R = TypeVar("R")

def handle_errors(
    logger_instance: logging.Logger = logger,
    reraise_exceptions: tuple[type[Exception], ...] = (),
    default_status_code: int = 500
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    A decorator to wrap functions and handle common exceptions, logging them
    and raising a unified AppError.

    Args:
        logger_instance: The logger to use for logging exceptions.
        reraise_exceptions: A tuple of exception types that should be re-raised
                            as their original type, without wrapping in AppError.
        default_status_code: The default HTTP status code for AppError if not specified.

    Returns:
        A decorator that wraps the target function.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        async def wrapper(*args: P.args, **kwargs: P.wkwargs) -> R:
            try:
                return await func(*args, **kwargs)
            except reraise_exceptions as e:
                # Re-raise specific exceptions without wrapping
                raise e
            except AppError as e:
                # If it's already an AppError, just re-raise
                raise e
            except Exception as e:
                error_message = f"An unexpected error occurred in {func.__name__}: {e}"
                logger_instance.exception(error_message) # Log the full traceback
                raise AppError(
                    message=error_message,
                    status_code=default_status_code,
                    details={"function": func.__name__, "original_error_type": type(e).__name__}
                ) from e
        return wrapper
    return decorator

# --- General Purpose Functions ---

def generate_unique_id(prefix: str = "") -> str:
    """
    Generates a unique ID using UUID4 and an optional prefix.

    Args:
        prefix: An optional string prefix for the ID.

    Returns:
        A unique string ID.
    """
    return f"{prefix}_{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex

def get_current_timestamp_utc() -> datetime:
    """
    Returns the current UTC timestamp.

    Returns:
        A datetime object representing the current UTC time.
    """
    return datetime.utcnow()

def sanitize_text(text: str) -> str:
    """
    Sanitizes input text by removing leading/trailing whitespace and
    normalizing common whitespace characters.

    Args:
        text: The input string to sanitize.

    Returns:
        The sanitized string.
    """
    return ' '.join(text.strip().split())

def is_valid_language_code(lang_code: str) -> bool:
    """
    Checks if a given language code is among the allowed languages configured.

    Args:
        lang_code: The language code (e.g., 'en', 'hi', 'te').

    Returns:
        True if the language code is allowed, False otherwise.
    """
    return lang_code in Config.ALLOWED_LANGUAGES

def validate_audio_format(file_content: bytes, expected_format: str) -> bool:
    """
    A very basic check to validate audio format based on common magic numbers.
    This is not robust and should be augmented with proper libraries in production.

    Args:
        file_content: The raw bytes content of the audio file.
        expected_format: The expected format (e.g., "wav", "mp3", "flac").

    Returns:
        True if the format matches a known pattern, False otherwise.
    """
    # Simplified checks for common audio formats
    if expected_format.lower() == "wav":
        # WAV files start with 'RIFF' and then 'WAVE'
        return file_content.startswith(b'RIFF') and file_content[8:12] == b'WAVE'
    elif expected_format.lower() == "mp3":
        # MP3 frames often start with 0xFFFB or 0xFFF3
        return file_content.startswith(b'\xFF\xFB') or file_content.startswith(b'\xFF\xF3')
    elif expected_format.lower() == "flac":
        # FLAC files start with 'fLaC'
        return file_content.startswith(b'fLaC')
    elif expected_format.lower() == "ogg" or expected_format.lower() == "opus":
        # OGG/Opus files start with 'OggS'
        return file_content.startswith(b'OggS')
    
    logger.warning(f"Unsupported or unknown audio format check requested: {expected_format}")
    return False

# Example usage (for testing purposes only, remove in final production code if not needed)
if __name__ == "__main__":
    test_logger = setup_logging("test_utils")
    test_logger.info("Utils module started for testing.")
    test_logger.debug("Debug message from utils.")
    test_logger.warning("Warning message from utils.")

    try:
        raise ValueError("Something went wrong here!")
    except Exception as e:
        test_logger.error("An error occurred during testing: %s", e, exc_info=True)

    print(f"Generated unique ID: {generate_unique_id('user')}")
    print(f"Current UTC timestamp: {get_current_timestamp_utc()}")
    print(f"Sanitized text: '{sanitize_text('  Hello   World!  ')}'")
    print(f"Is 'hi' a valid language? {is_valid_language_code('hi')}")
    print(f"Is 'fr' a valid language? {is_valid_language_code('fr')}")

    @handle_errors(logger_instance=test_logger)
    async def my_test_function_success():
        test_logger.info("Test function ran successfully.")
        return "Success!"

    @handle_errors(logger_instance=test_logger)
    async def my_test_function_failure():
        test_logger.info("Test function intentionally failing.")
        raise ValueError("This is a simulated failure.")

    @handle_errors(logger_instance=test_logger, reraise_exceptions=(KeyError,))
    async def my_test_function_reraise():
        test_logger.info("Test function raising KeyError to be re-raised.")
        raise KeyError("This key is missing.")

    import asyncio

    async def run_tests():
        try:
            result = await my_test_function_success()
            print(f"my_test_function_success result: {result}")
        except AppError as e:
            print(f"my_test_function_success caught AppError: {e.message}")

        try:
            await my_test_function_failure()
        except AppError as e:
            print(f"my_test_function_failure caught AppError: {e.message}")

        try:
            await my_test_function_reraise()
        except AppError as e:
            print(f"my_test_function_reraise caught AppError (should not happen if KeyError is re-raised): {e.message}")
        except KeyError as e:
            print(f"my_test_function_reraise caught original KeyError as expected: {e}")

    asyncio.run(run_tests())

    # Test audio format validation (placeholder, as we don't have actual audio files here)
    mock_wav_header = b'RIFF\x00\x00\x00\x00WAVE'
    mock_mp3_header = b'\xFF\xFB'
    mock_flac_header = b'fLaC'
    mock_ogg_header = b'OggS'
    mock_unknown_header = b'ABCD'

    print(f"Validate WAV: {validate_audio_format(mock_wav_header, 'wav')}")
    print(f"Validate MP3: {validate_audio_format(mock_mp3_header, 'mp3')}")
    print(f"Validate FLAC: {validate_audio_format(mock_flac_header, 'flac')}")
    print(f"Validate OGG: {validate_audio_format(mock_ogg_header, 'ogg')}")
    print(f"Validate unknown as WAV: {validate_audio_format(mock_unknown_header, 'wav')}")
    print(f"Validate unknown format type: {validate_audio_format(mock_wav_header, 'm4a')}")