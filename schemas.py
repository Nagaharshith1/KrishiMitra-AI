from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import date, datetime

# --- Voice Pipeline Schemas ---

class VoiceCommandRequest(BaseModel):
    """
    Request model for a voice command.
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data of the user's voice command.")
    language_code: Optional[str] = Field(None, description="Optional: User's preferred language for the command (e.g., 'en', 'hi', 'te', 'ta'). If not provided, language detection will be used.")

class VoiceCommandResponse(BaseModel):
    """
    Response model after processing a voice command.
    """
    original_text: str = Field(..., description="The recognized text from the user's audio, in the detected/specified language.")
    translated_text: str = Field(..., description="The translated text (usually to English) for NLU processing.")
    response_audio_base64: str = Field(..., description="Base64 encoded audio data of the AI's spoken response.")
    response_text: str = Field(..., description="The text of the AI's spoken response (in the target language).")
    language_code: str = Field(..., description="The language code of the AI's response (e.g., 'en', 'hi', 'te', 'ta').")
    intent: str = Field(..., description="The identified intent of the user's command.")
    entities: Dict[str, Any] = Field(..., description="Extracted entities from the user's command.")

class TTSRequest(BaseModel):
    """
    Request model for Text-to-Speech conversion.
    """
    text: str = Field(..., description="The text to convert to speech.")
    language_code: str = Field(..., description="Target language code for TTS (e.g., 'en', 'hi', 'te', 'ta').")

class TTSResponse(BaseModel):
    """
    Response model for Text-to-Speech conversion.
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data of the synthesized speech.")
    language_code: str = Field(..., description="The language code of the synthesized speech.")

# --- NLU Schemas ---

class NLURequest(BaseModel):
    """
    Request model for Natural Language Understanding.
    """
    text: str = Field(..., description="The text input for NLU processing.")
    original_language_code: str = Field(..., description="The original language code of the text.")

class NLUParsingResult(BaseModel):
    """
    Result model for NLU processing.
    """
    intent: str = Field(..., description="Identified intent of the user's query (e.g., 'diagnose_crop', 'get_weather', 'ask_sowing_advice').")
    entities: Dict[str, Any] = Field(..., description="Extracted entities and their values (e.g., {'crop_type': 'paddy', 'symptom': 'yellow leaves'}).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the identified intent.")
    original_text: str = Field(..., description="The original text processed by NLU.")
    original_language_code: str = Field(..., description="The original language code of the text.")

# --- Crop Doctor Schemas ---

class ImageAnalysisRequest(BaseModel):
    """
    Request model for crop image analysis.
    """
    image_base64: str = Field(..., description="Base64 encoded image data of the crop.")
    language_code: str = Field(..., description="Target language for the AI response (e.g., 'en', 'hi', 'te', 'ta').")
    crop_type: Optional[str] = Field(None, description="Optional: Specific crop type for more accurate diagnosis.")

class DiseaseDiagnosis(BaseModel):
    """
    Model for a single disease diagnosis.
    """
    name: str = Field(..., description="Name of the diagnosed disease/pest/deficiency.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this specific diagnosis.")
    description: str = Field(..., description="Detailed description of the condition.")
    symptoms: List[str] = Field(..., description="List of common symptoms.")
    recommendations: List[str] = Field(..., description="Recommended actions for treatment.")
    prevention: List[str] = Field(..., description="Preventive measures.")
    image_url: Optional[HttpUrl] = Field(None, description="Optional URL to a reference image of the condition.")

class CropDoctorResponse(BaseModel):
    """
    Response model for crop image analysis.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the diagnosis.")
    message: str = Field(..., description="A summary message for the user.")
    diagnoses: List[DiseaseDiagnosis] = Field(..., description="List of potential diagnoses with details.")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence of the analysis.")
    language_code: str = Field(..., description="The language code of the response.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")

# --- Crop Management & Advisor Schemas ---

class CropAdviceRequest(BaseModel):
    """
    Request model for general crop advice.
    """
    intent: Literal["sowing_advice", "planning_advice", "harvest_advice", "rotation_advice"] = Field(..., description="Specific type of crop advice requested.")
    crop_type: str = Field(..., description="Type of crop.")
    location: str = Field(..., description="Farm location (e.g., 'Hyderabad, Telangana').")
    season: Optional[str] = Field(None, description="Current or target season (e.g., 'Kharif', 'Rabi', 'summer').")
    soil_type: Optional[str] = Field(None, description="Optional: Soil type (e.g., 'loamy', 'clayey').")
    water_availability: Optional[str] = Field(None, description="Optional: Water availability (e.g., 'high', 'moderate', 'low').")
    language_code: str = Field(..., description="Target language for the AI response.")

class CropAdviceResponse(BaseModel):
    """
    Response model for crop advice.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the advice generation.")
    advice_text: str = Field(..., description="Detailed advice for the user.")
    recommendations: List[str] = Field(..., description="Key recommendations from the advice.")
    language_code: str = Field(..., description="The language code of the response.")

class WeatherRequest(BaseModel):
    """
    Request model for weather information.
    """
    location: str = Field(..., description="Farm location (e.g., 'Hyderabad, Telangana').")
    date: Optional[date] = Field(None, description="Specific date for weather forecast. Defaults to today if not provided.")
    forecast_days: Optional[int] = Field(1, ge=1, le=7, description="Number of days for forecast, up to 7.")
    language_code: str = Field(..., description="Target language for the AI response.")

class DailyWeather(BaseModel):
    """
    Model for daily weather data.
    """
    date: date = Field(..., description="Date of the weather data.")
    condition: str = Field(..., description="General weather condition (e.g., 'Sunny', 'Partly Cloudy', 'Rain').")
    temperature_celsius_min: float = Field(..., description="Minimum temperature in Celsius.")
    temperature_celsius_max: float = Field(..., description="Maximum temperature in Celsius.")
    humidity_percent: float = Field(..., ge=0.0, le=100.0, description="Average humidity in percent.")
    precipitation_mm: float = Field(..., ge=0.0, description="Precipitation in millimeters.")
    wind_speed_kph: float = Field(..., ge=0.0, description="Wind speed in kilometers per hour.")
    farming_impact: str = Field(..., description="Impact of weather on farming activities.")

class WeatherResponse(BaseModel):
    """
    Response model for weather information.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the weather query.")
    location: str = Field(..., description="Location for which weather data is provided.")
    current_weather: Optional[DailyWeather] = Field(None, description="Current weather conditions if available and requested for today.")
    forecast: List[DailyWeather] = Field(..., description="List of daily weather forecasts.")
    ai_response_text: str = Field(..., description="The generated AI response text summarizing the weather.")
    language_code: str = Field(..., description="The language code of the response.")

class WaterManagementRequest(BaseModel):
    """
    Request model for water management advice.
    """
    crop_type: str = Field(..., description="Type of crop.")
    soil_type: str = Field(..., description="Soil type (e.g., 'loamy', 'sandy', 'clayey').")
    current_weather_condition: Optional[str] = Field(None, description="Optional: Current weather, e.g., 'sunny', 'rainy'.")
    last_irrigated_days_ago: Optional[int] = Field(None, ge=0, description="Days since last irrigation.")
    language_code: str = Field(..., description="Target language for the AI response.")

class WaterManagementResponse(BaseModel):
    """
    Response model for water management advice.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the advice generation.")
    irrigation_recommendation: str = Field(..., description="Specific irrigation recommendation (e.g., 'Irrigate today', 'Irrigate in 2 days').")
    method_guidance: str = Field(..., description="Guidance on irrigation methods (drip, sprinkler, flood).")
    water_saving_tips: List[str] = Field(..., description="Tips for efficient water usage.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

class SoilTestResult(BaseModel):
    """
    Model for a single soil test parameter.
    """
    parameter: str = Field(..., description="Name of the soil parameter (e.g., 'pH', 'Nitrogen', 'Phosphorus').")
    value: Union[float, str] = Field(..., description="Value of the parameter.")
    unit: Optional[str] = Field(None, description="Unit of the value (e.g., 'mg/kg', 'ppm', 'pH unit').")

class SoilAIRequest(BaseModel):
    """
    Request model for Soil AI.
    """
    soil_test_results: List[SoilTestResult] = Field(..., description="List of soil test results.")
    crop_type: Optional[str] = Field(None, description="Optional: Target crop type for tailored recommendations.")
    language_code: str = Field(..., description="Target language for the AI response.")

class SoilAIResponse(BaseModel):
    """
    Response model for Soil AI.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the soil analysis.")
    interpretation: str = Field(..., description="Explanation of the soil test results.")
    recommendations: List[str] = Field(..., description="Recommendations based on soil analysis (e.g., 'add lime', 'fertilize with urea').")
    crop_suitability: Optional[str] = Field(None, description="Optional: Advice on crop suitability based on soil type.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

class FertilizerRequest(BaseModel):
    """
    Request model for Fertilizer Assistant.
    """
    crop_type: str = Field(..., description="Type of crop.")
    soil_test_results: Optional[List[SoilTestResult]] = Field(None, description="Optional: List of soil test results for precise recommendations.")
    growth_stage: Optional[str] = Field(None, description="Optional: Current growth stage of the crop (e.g., 'vegetative', 'flowering').")
    location: Optional[str] = Field(None, description="Optional: Farm location for regional specific fertilizer recommendations.")
    language_code: str = Field(..., description="Target language for the AI response.")

class FertilizerRecommendation(BaseModel):
    """
    Model for a single fertilizer recommendation.
    """
    fertilizer_name: str = Field(..., description="Name of the fertilizer (e.g., 'Urea', 'DAP').")
    composition: str = Field(..., description="N-P-K composition or other key components.")
    quantity: str = Field(..., description="Recommended quantity (e.g., '100 kg per acre').")
    application_method: str = Field(..., description="Recommended application method (e.g., 'broadcasting', 'foliar spray').")
    timing: str = Field(..., description="Recommended timing for application.")

class FertilizerResponse(BaseModel):
    """
    Response model for Fertilizer Assistant.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the fertilizer recommendation.")
    recommendations: List[FertilizerRecommendation] = Field(..., description="List of fertilizer recommendations.")
    general_advice: str = Field(..., description="General advice on fertilizer use.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

class PestManagementRequest(BaseModel):
    """
    Request model for Pest Management.
    """
    crop_type: str = Field(..., description="Type of crop affected.")
    pest_or_symptom: str = Field(..., description="Name of the pest or description of symptoms.")
    location: Optional[str] = Field(None, description="Optional: Farm location for regional specific pest information.")
    language_code: str = Field(..., description="Target language for the AI response.")

class PestManagementResponse(BaseModel):
    """
    Response model for Pest Management.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the pest management advice.")
    pest_identification: str = Field(..., description="Identification of the pest or condition.")
    control_measures: List[str] = Field(..., description="List of recommended control measures (biological, chemical, cultural).")
    prevention_strategies: List[str] = Field(..., description="List of preventive strategies.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

class FarmPlannerEvent(BaseModel):
    """
    Model for a single farm planner event/reminder.
    """
    event_id: Optional[str] = Field(None, description="Unique ID for the event (auto-generated for new events).")
    event_type: Literal["sowing", "irrigation", "fertilization", "pest_control", "harvest", "other"] = Field(..., description="Type of farm event.")
    crop_type: str = Field(..., description="Crop associated with the event.")
    date: date = Field(..., description="Scheduled date for the event.")
    description: str = Field(..., description="Detailed description of the event or task.")
    status: Literal["scheduled", "completed", "cancelled"] = Field("scheduled", description="Status of the event.")
    remind_at: Optional[datetime] = Field(None, description="Optional: Specific datetime for a voice reminder.")

class FarmPlannerRequest(BaseModel):
    """
    Request model for Farm Planner actions.
    """
    action: Literal["add", "update", "delete", "list"] = Field(..., description="Action to perform on the farm planner.")
    event: Optional[FarmPlannerEvent] = Field(None, description="Event details for 'add' or 'update' actions.")
    event_id: Optional[str] = Field(None, description="Event ID for 'update' or 'delete' actions.")
    language_code: str = Field(..., description="Target language for the AI response.")
    filter_date: Optional[date] = Field(None, description="Optional: Filter events by date for 'list' action.")
    filter_crop_type: Optional[str] = Field(None, description="Optional: Filter events by crop type for 'list' action.")

class FarmPlannerResponse(BaseModel):
    """
    Response model for Farm Planner actions.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the farm planner operation.")
    message: str = Field(..., description="A message describing the result of the operation.")
    events: Optional[List[FarmPlannerEvent]] = Field(None, description="List of events, typically returned for 'list' action.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

# --- Market & Government Schemas ---

class MarketDataRequest(BaseModel):
    """
    Request model for market data.
    """
    crop_name: str = Field(..., description="Name of the crop to get market data for.")
    location: Optional[str] = Field(None, description="Optional: Specific market location (e.g., 'Hyderabad, Telangana').")
    date: Optional[date] = Field(None, description="Optional: Specific date for market price. Defaults to latest if not provided.")
    language_code: str = Field(..., description="Target language for the AI response.")

class MarketPrice(BaseModel):
    """
    Model for a single market price entry.
    """
    market_name: str = Field(..., description="Name of the market.")
    crop_name: str = Field(..., description="Name of the crop.")
    price_per_unit: float = Field(..., ge=0.0, description="Price of the crop per unit.")
    unit: str = Field(..., description="Unit of the price (e.g., 'INR/quintal', 'INR/kg').")
    date: date = Field(..., description="Date of the price data.")
    grade: Optional[str] = Field(None, description="Optional: Grade of the crop (e.g., 'Grade A', 'Normal').")

class MarketDataResponse(BaseModel):
    """
    Response model for market data.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the market data query.")
    crop_name: str = Field(..., description="Crop for which data is provided.")
    prices: List[MarketPrice] = Field(..., description="List of market prices for the crop.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

class GovernmentSchemeRequest(BaseModel):
    """
    Request model for government schemes.
    """
    topic: Optional[str] = Field(None, description="Optional: Specific topic for schemes (e.g., 'subsidies', 'loans', 'irrigation').")
    state: Optional[str] = Field(None, description="Optional: Specific Indian state for state-specific schemes.")
    language_code: str = Field(..., description="Target language for the AI response.")

class GovernmentScheme(BaseModel):
    """
    Model for a single government scheme.
    """
    name: str = Field(..., description="Name of the scheme.")
    description: str = Field(..., description="Brief description of the scheme.")
    eligibility: str = Field(..., description="Eligibility criteria for the scheme.")
    benefits: List[str] = Field(..., description="Key benefits of the scheme.")
    how_to_apply: Optional[str] = Field(None, description="Optional: Steps or link for application.")
    official_link: Optional[HttpUrl] = Field(None, description="Optional: Official website link for more details.")

class GovernmentSchemeResponse(BaseModel):
    """
    Response model for government schemes.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the government schemes query.")
    schemes: List[GovernmentScheme] = Field(..., description="List of relevant government schemes.")
    ai_response_text: str = Field(..., description="The generated AI response text for the user.")
    language_code: str = Field(..., description="The language code of the response.")

# --- Database Models (SQLAlchemy ORM models will be defined in database.py, but Pydantic versions for API/internal use here) ---

class UserBase(BaseModel):
    """Base Pydantic model for User."""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = Field(None, pattern=r"^[^@]+@[^@]+\.[^@]+$") # Basic email regex

class UserCreate(UserBase):
    """Pydantic model for creating a User."""
    password: str = Field(..., min_length=6)

class User(UserBase):
    """Pydantic model for reading a User (includes ID)."""
    id: int
    is_active: bool = True

    class Config:
        orm_mode = True # Enable ORM mode for SQLAlchemy integration

class FarmProfileBase(BaseModel):
    """Base Pydantic model for Farm Profile."""
    user_id: int
    location: str = Field(..., min_length=3)
    soil_type: Optional[str] = None
    average_rainfall_mm: Optional[float] = None
    primary_crops: List[str] = Field(default_factory=list)

class FarmProfileCreate(FarmProfileBase):
    """Pydantic model for creating a Farm Profile."""
    pass

class FarmProfile(FarmProfileBase):
    """Pydantic model for reading a Farm Profile (includes ID)."""
    id: int

    class Config:
        orm_mode = True

class ReminderBase(BaseModel):
    """Base Pydantic model for Reminder."""
    user_id: int
    event_id: str # Link to a FarmPlannerEvent
    reminder_time: datetime
    message: str
    is_sent: bool = False

class ReminderCreate(ReminderBase):
    """Pydantic model for creating a Reminder."""
    pass

class Reminder(ReminderBase):
    """Pydantic model for reading a Reminder (includes ID)."""
    id: int

    class Config:
        orm_mode = True

# --- General API Response/Error Schemas ---

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    status_code: int = Field(..., description="HTTP status code of the error.")
    code: str = Field(..., description="Internal error code (e.g., 'NLU_ERROR', 'IMAGE_PROCESSING_FAILED').")
    message: str = Field(..., description="A user-friendly error message.")
    details: Optional[Any] = Field(None, description="Optional: Additional details about the error.")

class HealthCheckResponse(BaseModel):
    """
    Response model for API health check.
    """
    status: str = Field(..., description="Status of the application (e.g., 'healthy', 'degraded').")
    timestamp: datetime = Field(..., description="Current server time.")
    dependencies: Dict[str, str] = Field(..., description="Status of key dependencies (e.g., 'database', 'AI models').")

# --- AI Router / Orchestrator Internal Schemas ---

class AgricultureAIServiceRequest(BaseModel):
    """
    Unified request schema for internal AI services.
    """
    intent: str = Field(..., description="The determined intent from NLU.")
    entities: Dict[str, Any] = Field(..., description="Extracted entities from the user's query.")
    user_id: Optional[int] = Field(None, description="Optional: ID of the user making the request.")
    language_code: str = Field(..., description="Target language for the AI's response.")
    original_user_text: Optional[str] = Field(None, description="Optional: Original text query from the user.")
    image_base64: Optional[str] = Field(None, description="Optional: Base64 encoded image data for image analysis intents.")

class AgricultureAIServiceResponse(BaseModel):
    """
    Unified response schema from internal AI services back to the orchestrator.
    """
    status: Literal["success", "error"] = Field(..., description="Status of the AI service operation.")
    response_text: str = Field(..., description="The main AI-generated text response in the target language.")
    language_code: str = Field(..., description="The language code of the response.")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Optional: Raw structured data returned by the service (e.g., list of diagnoses, weather forecast).")
    safety_flagged: bool = Field(False, description="True if the response was flagged by safety service.")
    safety_message: Optional[str] = Field(None, description="Reason for safety flag.")

# --- Demo specific schemas ---

class DemoResponse(BaseModel):
    """
    Response model for the hackathon demo scenario.
    """
    original_audio_text: str = Field(..., description="Text from the initial Telugu voice command.")
    detected_language: str = Field(..., description="Detected language of the voice command.")
    nlu_intent: str = Field(..., description="Identified NLU intent.")
    nlu_entities: Dict[str, Any] = Field(..., description="Extracted NLU entities.")
    image_analysis_summary: str = Field(..., description="Summary of the crop doctor image analysis.")
    image_analysis_details: Optional[CropDoctorResponse] = Field(None, description="Full details of the crop doctor response.")
    final_ai_response_text: str = Field(..., description="The final AI voice response text (multilingual).")
    final_ai_response_audio_base64: str = Field(..., description="Base64 encoded audio of the final AI voice response.")
    final_response_language: str = Field(..., description="Language of the final AI response.")