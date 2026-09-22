import logging
import base64
import httpx
from typing import Dict, Any, Optional

from config import Config
from schemas import CropDoctorRequest, CropDoctorResponse, Diagnosis, ConfidenceLevel, Recommendation, Prevention
from utils import setup_logging
from ai_safety import AgricultureSafetyService

logger = setup_logging(__name__)

class CropDoctorService:
    """
    Provides AI image analysis for crop diseases, pests, and nutrient deficiencies.
    It integrates with an external ML model API to get diagnosis, processes the results,
    and formats them into structured responses for the KRISHIMITRA AI application.
    """

    def __init__(self, ai_safety_service: AgricultureSafetyService):
        """
        Initializes the CropDoctorService with configuration and an AI safety service.

        Args:
            ai_safety_service: An instance of AgricultureSafetyService for safety checks.
        """
        self.model_api_url = Config.CROP_DOCTOR_MODEL_API_URL
        self.api_key = Config.CROP_DOCTOR_API_KEY
        self.ai_safety_service = ai_safety_service
        self._http_client = httpx.AsyncClient(timeout=30.0) # Increased timeout for image processing
        logger.info(f"CropDoctorService initialized. Model API: {self.model_api_url}")

    async def analyze_crop_image(self, request: CropDoctorRequest) -> CropDoctorResponse:
        """
        Analyzes a crop image to identify diseases, pests, or nutrient deficiencies.

        Args:
            request: A CropDoctorRequest object containing the base64 encoded image and optional language.

        Returns:
            A CropDoctorResponse object with diagnosis, confidence, recommendations, and prevention strategies.

        Raises:
            Exception: If there's an issue communicating with the ML model or processing the image.
        """
        logger.info(f"Received image for analysis. Image size: {len(request.image_base64)} bytes.")

        try:
            # Prepare payload for the external ML model API
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "image_base64": request.image_base64,
                "language_code": request.language_code or "en" # Default to English for model input if not specified
            }

            logger.debug(f"Sending request to Crop Doctor model API: {self.model_api_url}")
            response = await self._http_client.post(self.model_api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses

            model_response = response.json()
            logger.debug(f"Received response from Crop Doctor model API: {model_response}")

            # Process the model's raw response into structured schemas
            diagnosis_data = model_response.get("diagnosis", {})
            recommendations_data = model_response.get("recommendations", [])
            prevention_data = model_response.get("prevention", [])
            confidence_score = diagnosis_data.get("confidence", 0.0)

            # Map confidence score to ConfidenceLevel enum
            if confidence_score >= 0.9:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence_score >= 0.7:
                confidence_level = ConfidenceLevel.MEDIUM
            elif confidence_score >= 0.5:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.VERY_LOW

            diagnosis = Diagnosis(
                issue=diagnosis_data.get("issue", "Unknown"),
                description=diagnosis_data.get("description", "Could not determine the specific issue."),
                confidence=confidence_score,
                confidence_level=confidence_level
            )

            recommendations = [
                Recommendation(step=str(i+1), description=rec.get("description", ""))
                for i, rec in enumerate(recommendations_data)
            ]

            prevention_strategies = [
                Prevention(method=str(i+1), description=prev.get("description", ""))
                for i, prev in enumerate(prevention_data)
            ]

            crop_doctor_response = CropDoctorResponse(
                diagnosis=diagnosis,
                recommendations=recommendations,
                prevention_strategies=prevention_strategies,
                image_id="img_" + str(hash(request.image_base64)) # A simple, non-unique ID for demo
            )

            # Apply safety checks to the generated response
            safety_check_request = {
                "text": f"Diagnosis: {diagnosis.issue}. Description: {diagnosis.description}. "
                        f"Recommendations: {' '.join([r.description for r in recommendations])}. "
                        f"Prevention: {' '.join([p.description for p in prevention_strategies])}",
                "confidence": diagnosis.confidence,
                "context": "crop_doctor_diagnosis",
                "original_request": request.dict()
            }
            safety_response = self.ai_safety_service.perform_safety_check(safety_check_request)

            if not safety_response.is_safe:
                logger.warning(f"Safety check failed for Crop Doctor response: {safety_response.reason}")
                # Modify the response to include safety disclaimer or generic advice
                crop_doctor_response.diagnosis.issue = "Safety Warning / Re-evaluation Recommended"
                crop_doctor_response.diagnosis.description = (
                    f"Due to a safety concern or low confidence, we recommend re-evaluating the image or consulting an expert. "
                    f"Original diagnosis might be unsafe or unreliable. {safety_response.reason}"
                )
                crop_doctor_response.recommendations = []
                crop_doctor_response.prevention_strategies = []

            # If confidence is low, add a general disclaimer
            if crop_doctor_response.diagnosis.confidence < self.ai_safety_service._safety_policies["max_confidence_threshold_for_diagnosis"]:
                logger.info(f"Low confidence diagnosis ({crop_doctor_response.diagnosis.confidence}). Adding disclaimer.")
                disclaimer = self.ai_safety_service._safety_policies["disclaimer_for_low_confidence"]
                if crop_doctor_response.recommendations:
                    crop_doctor_response.recommendations.append(Recommendation(step="Disclaimer", description=disclaimer))
                else:
                    crop_doctor_response.recommendations = [Recommendation(step="1", description=disclaimer)]


            return crop_doctor_response

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error communicating with Crop Doctor model API: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Failed to get diagnosis from AI model. Status: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Network error communicating with Crop Doctor model API: {e}")
            raise Exception("Network error connecting to the Crop Doctor AI model. Please check connectivity.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during image analysis: {e}", exc_info=True)
            raise Exception(f"An error occurred while analyzing the crop image: {e}")

if __name__ == "__main__":
    # This block is for demonstrating the CropDoctorService in isolation.
    # In a real application, it would be instantiated and used by FastAPI.

    from dotenv import load_dotenv
    load_dotenv() # Load .env variables for testing

    # Mock AgricultureSafetyService for local testing
    class MockAgricultureSafetyService:
        _safety_policies = {
            "max_confidence_threshold_for_diagnosis": 0.6,
            "disclaimer_for_low_confidence": "Please consult a local agricultural expert for a more accurate diagnosis (mock safety service).",
            "prohibited_keywords": [],
            "required_disclaimers": []
        }

        def perform_safety_check(self, request: Dict[str, Any]) -> schemas.SafetyCheckResponse:
            logger.info(f"Mock Safety Check: {request.get('text', '')[:50]}...")
            is_safe = True
            reason = ""

            if request.get("confidence", 0.0) < self._safety_policies["max_confidence_threshold_for_diagnosis"]:
                is_safe = False
                reason = "Confidence too low for reliable advice."

            return schemas.SafetyCheckResponse(is_safe=is_safe, reason=reason, checked_text=request.get("text", ""))

    async def demo_crop_doctor():
        mock_safety_service = MockAgricultureSafetyService()
        crop_doctor_service = CropDoctorService(ai_safety_service=mock_safety_service)

        # Create a dummy base64 image (a tiny red dot in a 1x1 png for example, or any small image)
        # For a real test, replace with a base64 encoded image string
        # A 1x1 transparent PNG: `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=`
        # This is a very small, non-representative image, but serves the purpose of being valid base64
        dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

        # Test case 1: High confidence (mocked)
        print("\n--- Demo: Crop Doctor Service (High Confidence Mock) ---")
        request_high_conf = CropDoctorRequest(
            image_base64=dummy_image_base64,
            language_code="en"
        )
        try:
            # Mock the external API call to simulate a high confidence response
            async def mock_post(*args, **kwargs):
                class MockResponse:
                    def __init__(self, json_data, status_code=200):
                        self._json_data = json_data
                        self.status_code = status_code
                        self.text = str(json_data)
                    def json(self): return self._json_data
                    def raise_for_status(self):
                        if self.status_code >= 400: raise httpx.HTTPStatusError(f"Bad status {self.status_code}", request=httpx.Request("GET", "http://mock.url"), response=self)
                return MockResponse({
                    "diagnosis": {"issue": "Nitrogen Deficiency", "description": "Leaves are turning yellow, especially older ones, due to lack of nitrogen.", "confidence": 0.95},
                    "recommendations": [{"description": "Apply urea or composted manure."}],
                    "prevention": [{"description": "Ensure regular soil testing."}]
                })
            crop_doctor_service._http_client.post = mock_post
            response = await crop_doctor_service.analyze_crop_image(request_high_conf)
            print(f"Diagnosis: {response.diagnosis.issue}")
            print(f"Confidence: {response.diagnosis.confidence_level.value} ({response.diagnosis.confidence:.2f})")
            print(f"Recommendations: {[r.description for r in response.recommendations]}")
            print(f"Prevention: {[p.description for p in response.prevention_strategies]}")
            assert response.diagnosis.issue == "Nitrogen Deficiency"
            assert response.diagnosis.confidence_level == ConfidenceLevel.HIGH
            print("High Confidence Test Passed.")
        except Exception as e:
            print(f"High Confidence Test Failed: {e}")

        # Test case 2: Low confidence (mocked)
        print("\n--- Demo: Crop Doctor Service (Low Confidence Mock) ---")
        request_low_conf = CropDoctorRequest(
            image_base64=dummy_image_base64,
            language_code="en"
        )
        try:
            # Mock the external API call to simulate a low confidence response
            async def mock_post_low_conf(*args, **kwargs):
                class MockResponse:
                    def __init__(self, json_data, status_code=200):
                        self._json_data = json_data
                        self.status_code = status_code
                        self.text = str(json_data)
                    def json(self): return self._json_data
                    def raise_for_status(self):
                        if self.status_code >= 400: raise httpx.HTTPStatusError(f"Bad status {self.status_code}", request=httpx.Request("GET", "http://mock.url"), response=self)
                return MockResponse({
                    "diagnosis": {"issue": "Unclear Issue", "description": "Image quality too low or symptoms are ambiguous.", "confidence": 0.45},
                    "recommendations": [{"description": "Improve lighting for the photo."}],
                    "prevention": []
                })
            crop_doctor_service._http_client.post = mock_post_low_conf
            response = await crop_doctor_service.analyze_crop_image(request_low_conf)
            print(f"Diagnosis: {response.diagnosis.issue}")
            print(f"Confidence: {response.diagnosis.confidence_level.value} ({response.diagnosis.confidence:.2f})")
            print(f"Recommendations: {[r.description for r in response.recommendations]}")
            print(f"Prevention: {[p.description for p in response.prevention_strategies]}")
            assert response.diagnosis.confidence_level == ConfidenceLevel.VERY_LOW
            assert any("consult a local agricultural expert" in r.description for r in response.recommendations)
            print("Low Confidence Test Passed with Disclaimer.")
        except Exception as e:
            print(f"Low Confidence Test Failed: {e}")

        await crop_doctor_service._http_client.aclose()

    import asyncio
    asyncio.run(demo_crop_doctor())