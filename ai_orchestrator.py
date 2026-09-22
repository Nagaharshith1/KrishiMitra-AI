import logging
import base64
from typing import Dict, Any, Optional

from config import Config
from schemas import (
    VoiceCommandRequest, VoiceCommandResponse,
    CropDoctorRequest, CropDoctorResponse,
    NLUParsingResult, Intent, GenericAIResponse,
    SafetyCheckRequest, SafetyCheckResponse,
    CropAdvisorRequest, CropAdvisorResponse,
    WeatherRequest, WeatherResponse,
    MarketDataRequest, MarketDataResponse,
    GovernmentSchemesRequest, GovernmentSchemesResponse,
    FarmPlannerRequest, FarmPlannerResponse,
    SoilAIRequest, SoilAIResponse,
    WaterManagementRequest, WaterManagementResponse,
    FertilizerAssistantRequest, FertilizerAssistantResponse,
    PestManagementRequest, PestManagementResponse,
    ImageAnalysisResponse,
)
from utils import setup_logging, KrishiMitraException
from ai_safety import AgricultureSafetyService
from voice_pipeline import VoicePipelineService
from nlu_engine import NLUEngine
from ai_router import AIRouter
from crop_doctor import CropDoctorService # Imported here for type hinting, though routed through AIRouter
from crop_management import ( # Imported here for type hinting, though routed through AIRouter
    CropAdvisorService, WaterManagementService, SoilAIService,
    FertilizerAssistantService, PestManagementService, FarmPlannerService
)
from external_integrations import ( # Imported here for type hinting, though routed through AIRouter
    WeatherIntegrationService, MarketIntegrationService, GovernmentSchemesIntegrationService
)

logger = setup_logging(__name__)

class AgricultureAIService:
    """
    The core AgricultureAIService layer orchestrating the entire AI pipeline:
    receiving voice input, routing through NLU, dispatching to relevant services,
    and synthesizing multilingual voice responses.
    """

    def __init__(self,
                 voice_pipeline: VoicePipelineService,
                 nlu_engine: NLUEngine,
                 ai_router: AIRouter,
                 ai_safety_service: AgricultureSafetyService):
        """
        Initializes the AgricultureAIService with required dependencies.

        Args:
            voice_pipeline: An instance of VoicePipelineService for speech processing.
            nlu_engine: An instance of NLUEngine for natural language understanding.
            ai_router: An instance of AIRouter for dispatching intents to specific services.
            ai_safety_service: An instance of AgricultureSafetyService for ensuring safe responses.
        """
        self.voice_pipeline = voice_pipeline
        self.nlu_engine = nlu_engine
        self.ai_router = ai_router
        self.ai_safety_service = ai_safety_service
        logger.info("AgricultureAIService initialized.")

    async def process_voice_command(self, request: VoiceCommandRequest) -> VoiceCommandResponse:
        """
        Processes a complete voice command from speech input to a spoken AI response.

        The pipeline includes:
        1. Speech-to-Text (STT) and Language Detection.
        2. Translation to English for NLU (if not already English).
        3. Natural Language Understanding (NLU) to identify intent and entities.
        4. Routing the intent to the appropriate specialized AI service.
        5. Generating a response from the specialized AI service.
        6. Performing safety checks on the AI's response.
        7. Translating the response back to the user's original language (if necessary).
        8. Text-to-Speech (TTS) conversion of the response.

        Args:
            request: A VoiceCommandRequest containing the base64 encoded audio.

        Returns:
            A VoiceCommandResponse containing the processed text, AI's response,
            and base64 encoded audio of the AI's response.

        Raises:
            KrishiMitraException: If any step in the pipeline fails.
        """
        logger.info(f"Processing voice command. Audio size: {len(request.audio_base64)} bytes.")

        try:
            # 1. Speech-to-Text (STT) and Language Detection
            stt_result = await self.voice_pipeline.speech_to_text(request.audio_base64, request.language_code)
            detected_language = stt_result.language_code
            original_text = stt_result.text
            logger.info(f"STT Result: Lang='{detected_language}', Text='{original_text[:50]}...'")

            # 2. Translate to English for NLU (if not English)
            translated_text = original_text
            if detected_language != "en":
                translation_result = await self.voice_pipeline.translate_text(
                    text=original_text,
                    source_lang=detected_language,
                    target_lang="en"
                )
                translated_text = translation_result.translated_text
                logger.info(f"Translated to English for NLU: '{translated_text[:50]}...'")

            # 3. Natural Language Understanding (NLU)
            nlu_request = NLUParsingResult(text=translated_text)
            nlu_result = await self.nlu_engine.parse_text(nlu_request)
            intent = nlu_result.intent.value
            entities = nlu_result.entities
            logger.info(f"NLU Result: Intent='{intent}', Entities='{entities}'")

            # 4. Route intent to appropriate specialized AI service and get response
            ai_response: GenericAIResponse
            if intent == Intent.DIAGNOSE_CROP_ISSUE:
                # If intent is DIAGNOSE_CROP_ISSUE, we need an image.
                # This path is typically handled by `process_image_command` or assumes an image was provided separately.
                # For a voice-only command leading to diagnosis, we would need to prompt the user to upload an image.
                # In this specific scenario, we'll respond indicating an image is needed or route to a generic prompt.
                logger.warning(f"Voice command identified intent '{Intent.DIAGNOSE_CROP_ISSUE}' but no image was provided.")
                response_text_en = "You asked to diagnose a crop issue. Please upload an image of your crop for analysis."
            else:
                ai_response = await self.ai_router.route_request(
                    intent=intent,
                    entities=entities,
                    user_language=detected_language
                )
                response_text_en = ai_response.response_text_en # Assume all routed services return this
                logger.info(f"AI Router Response (EN): '{response_text_en[:50]}...'")

            # 5. Safety check on the AI's response
            safety_request = SafetyCheckRequest(response_text=response_text_en, intent=intent)
            safety_result = await self.ai_safety_service.check_response_safety(safety_request)
            if not safety_result.is_safe:
                logger.error(f"Safety check failed for response: {response_text_en}")
                raise KrishiMitraException("AI response deemed unsafe by safety service.")

            final_response_text_en = safety_result.safe_response_text
            logger.info(f"Safety Service applied disclaimers: '{final_response_text_en[:50]}...'")

            # 6. Translate response back to user's original language (if necessary)
            response_text_target_lang = final_response_text_en
            if detected_language != "en":
                translation_result = await self.voice_pipeline.translate_text(
                    text=final_response_text_en,
                    source_lang="en",
                    target_lang=detected_language
                )
                response_text_target_lang = translation_result.translated_text
                logger.info(f"Translated response to {detected_language}: '{response_text_target_lang[:50]}...'")
            
            # 7. Text-to-Speech (TTS) conversion
            tts_result = await self.voice_pipeline.text_to_speech(
                text=response_text_target_lang,
                language_code=detected_language
            )
            response_audio_base64 = tts_result.audio_base64
            logger.info(f"TTS completed for response. Audio size: {len(response_audio_base64)} bytes.")

            return VoiceCommandResponse(
                original_text=original_text,
                translated_text=translated_text,
                response_audio_base64=response_audio_base64,
                response_text=response_text_target_lang,
                language_code=detected_language,
                intent=intent,
                entities=entities
            )

        except KrishiMitraException as e:
            logger.error(f"KrishiMitraException in process_voice_command: {e}")
            # Attempt to provide a spoken error message
            error_msg_en = f"I encountered an issue processing your request: {e.detail}. Please try again."
            return await self._handle_error_response(error_msg_en, request.language_code)
        except Exception as e:
            logger.exception("An unexpected error occurred during voice command processing.")
            # Attempt to provide a spoken error message
            error_msg_en = "I'm sorry, an unexpected error occurred. Please try again or contact support."
            return await self._handle_error_response(error_msg_en, request.language_code)

    async def process_image_command(self, request: CropDoctorRequest) -> ImageAnalysisResponse:
        """
        Processes an image command for crop analysis.

        The pipeline includes:
        1. Routing the image to the CropDoctorService.
        2. Getting the diagnosis, recommendations, and prevention strategies.
        3. Performing safety checks on the AI's response.
        4. Translating the response back to the user's preferred language (if necessary).
        5. Synthesizing a multilingual voice response.

        Args:
            request: A CropDoctorRequest containing the base64 encoded image and preferred language.

        Returns:
            An ImageAnalysisResponse containing the diagnosis, confidence, recommendations,
            prevention, and base64 encoded audio of the AI's spoken response.

        Raises:
            KrishiMitraException: If any step in the pipeline fails.
        """
        logger.info(f"Processing image command. Image size: {len(request.image_base64)} bytes. Lang: {request.language_code}")

        try:
            # 1. Route the image to the CropDoctorService
            # The AI router will handle dispatching to CropDoctorService internally.
            # We assume CropDoctor is available via the router for consistent flow.
            # Directly calling crop_doctor_service.analyze_crop_image can also be done.
            crop_doctor_response: CropDoctorResponse = await self.ai_router.route_request(
                intent=Intent.DIAGNOSE_CROP_ISSUE.value,
                entities={"crop_issue_description": "image analysis"}, # Placeholder entity
                image_base64=request.image_base64,
                user_language=request.language_code
            )

            # Extract primary diagnosis text in English
            diagnosis_text_en = f"Diagnosis: {crop_doctor_response.diagnosis.name}. " \
                                f"Confidence: {crop_doctor_response.confidence_score:.2f}. " \
                                f"Symptoms: {crop_doctor_response.diagnosis.description}. " \
                                f"Recommendations: {' '.join([rec.description for rec in crop_doctor_response.recommendations])}. " \
                                f"Prevention: {' '.join([prev.description for prev in crop_doctor_response.prevention_strategies])}."

            # 2. Safety check on the AI's response
            safety_request = SafetyCheckRequest(response_text=diagnosis_text_en, intent=Intent.DIAGNOSE_CROP_ISSUE.value)
            safety_result = await self.ai_safety_service.check_response_safety(safety_request)
            if not safety_result.is_safe:
                logger.error(f"Safety check failed for image diagnosis response: {diagnosis_text_en}")
                raise KrishiMitraException("AI image diagnosis response deemed unsafe by safety service.")

            final_response_text_en = safety_result.safe_response_text
            logger.info(f"Safety Service applied disclaimers to image diagnosis: '{final_response_text_en[:50]}...'")

            # 3. Translate response back to user's preferred language (if necessary)
            response_text_target_lang = final_response_text_en
            if request.language_code and request.language_code != "en":
                translation_result = await self.voice_pipeline.translate_text(
                    text=final_response_text_en,
                    source_lang="en",
                    target_lang=request.language_code
                )
                response_text_target_lang = translation_result.translated_text
                logger.info(f"Translated image diagnosis response to {request.language_code}: '{response_text_target_lang[:50]}...'")

            # 4. Text-to-Speech (TTS) conversion
            tts_result = await self.voice_pipeline.text_to_speech(
                text=response_text_target_lang,
                language_code=request.language_code if request.language_code else "en" # Use 'en' as default if not specified
            )
            response_audio_base64 = tts_result.audio_base64
            logger.info(f"TTS completed for image diagnosis response. Audio size: {len(response_audio_base64)} bytes.")

            return ImageAnalysisResponse(
                diagnosis=crop_doctor_response.diagnosis,
                confidence_score=crop_doctor_response.confidence_score,
                recommendations=crop_doctor_response.recommendations,
                prevention_strategies=crop_doctor_response.prevention_strategies,
                response_audio_base64=response_audio_base64,
                response_text=response_text_target_lang,
                language_code=request.language_code if request.language_code else "en"
            )

        except KrishiMitraException as e:
            logger.error(f"KrishiMitraException in process_image_command: {e}")
            return await self._handle_error_image_response(e.detail, request.language_code)
        except Exception as e:
            logger.exception("An unexpected error occurred during image command processing.")
            return await self._handle_error_image_response("An unexpected error occurred during image analysis. Please try again.", request.language_code)

    async def _handle_error_response(self, error_message_en: str, target_language: Optional[str] = None) -> VoiceCommandResponse:
        """
        Helper to generate a spoken error response.
        """
        logger.warning(f"Handling error response: {error_message_en}")
        lang_code = target_language if target_language and target_language in Config.ALLOWED_LANGUAGES else "en"
        
        translated_error_msg = error_message_en
        if lang_code != "en":
            try:
                translation_result = await self.voice_pipeline.translate_text(
                    text=error_message_en,
                    source_lang="en",
                    target_lang=lang_code
                )
                translated_error_msg = translation_result.translated_text
            except Exception as e:
                logger.error(f"Failed to translate error message to {lang_code}: {e}")
                translated_error_msg = "Sorry, I could not translate the error message. Please try again."

        try:
            tts_result = await self.voice_pipeline.text_to_speech(
                text=translated_error_msg,
                language_code=lang_code
            )
            audio_base64 = tts_result.audio_base64
        except Exception as e:
            logger.error(f"Failed to generate TTS for error message: {e}")
            audio_base64 = "" # Return empty audio if TTS fails

        return VoiceCommandResponse(
            original_text="Error during processing",
            translated_text=error_message_en,
            response_audio_base64=audio_base64,
            response_text=translated_error_msg,
            language_code=lang_code,
            intent="ERROR",
            entities={}
        )

    async def _handle_error_image_response(self, error_message_en: str, target_language: Optional[str] = None) -> ImageAnalysisResponse:
        """
        Helper to generate a spoken error response for image analysis.
        """
        logger.warning(f"Handling image error response: {error_message_en}")
        lang_code = target_language if target_language and target_language in Config.ALLOWED_LANGUAGES else "en"
        
        translated_error_msg = error_message_en
        if lang_code != "en":
            try:
                translation_result = await self.voice_pipeline.translate_text(
                    text=error_message_en,
                    source_lang="en",
                    target_lang=lang_code
                )
                translated_error_msg = translation_result.translated_text
            except Exception as e:
                logger.error(f"Failed to translate image error message to {lang_code}: {e}")
                translated_error_msg = "Sorry, I could not translate the error message for image analysis. Please try again."

        try:
            tts_result = await self.voice_pipeline.text_to_speech(
                text=translated_error_msg,
                language_code=lang_code
            )
            audio_base64 = tts_result.audio_base64
        except Exception as e:
            logger.error(f"Failed to generate TTS for image error message: {e}")
            audio_base64 = "" # Return empty audio if TTS fails

        # Return a default/empty CropDoctorResponse for error, with the error message in audio/text
        return ImageAnalysisResponse(
            diagnosis={"name": "Error", "description": translated_error_msg},
            confidence_score=0.0,
            recommendations=[],
            prevention_strategies=[],
            response_audio_base64=audio_base64,
            response_text=translated_error_msg,
            language_code=lang_code
        )