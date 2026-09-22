import logging
from typing import Dict, Any, List
from schemas import VoiceCommandResponse, CropDoctorResponse, CropAdvisorResponse, SafetyCheckRequest, SafetyCheckResponse

# Configure logging for the module
logger = logging.getLogger(__name__)

class AgricultureSafetyService:
    """
    Implements the AgricultureSafetyService, ensuring AI responses are safe, factual, and non-fabricated.
    This service is crucial for providing reliable agricultural advice, preventing misinformation,
    and ensuring responses adhere to ethical guidelines.
    """

    def __init__(self):
        """
        Initializes the AgricultureSafetyService.
        In a real-world scenario, this might load safety policies, access content filters,
        or connect to an external safety API.
        """
        logger.info("AgricultureSafetyService initialized.")
        # Placeholder for safety policies or configuration
        self._safety_policies = {
            "max_confidence_threshold_for_diagnosis": 0.6,  # If diagnosis confidence is below this, suggest re-evaluation.
            "disclaimer_for_low_confidence": "Please consult a local agricultural expert or re-upload clearer images for a more accurate diagnosis.",
            "prohibited_keywords": ["harmful chemicals", "illegal practices", "dangerous advice"],
            "required_disclaimers": [
                "Always verify information with local agricultural experts.",
                "Weather forecasts are estimates and can change.",
                "Market prices are subject to fluctuation."
            ]
        }

    def _check_for_prohibited_content(self, text: str) -> bool:
        """
        Checks the given text for any prohibited keywords or phrases.
        """
        text_lower = text.lower()
        for keyword in self._safety_policies["prohibited_keywords"]:
            if keyword in text_lower:
                logger.warning(f"Prohibited content detected: '{keyword}' in '{text}'")
                return True
        return False

    def _add_required_disclaimers(self, text: str, context: str = "") -> str:
        """
        Adds required disclaimers to the response text based on the context.
        """
        disclaimers_to_add = []
        if "weather" in context.lower():
            disclaimers_to_add.append(self._safety_policies["required_disclaimers"][1]) # Weather forecast disclaimer
        if "market" in context.lower():
            disclaimers_to_add.append(self._safety_policies["required_disclaimers"][2]) # Market prices disclaimer
        
        # Always add general disclaimer if not already present and it's a primary response
        if self._safety_policies["required_disclaimers"][0] not in text and not disclaimers_to_add:
             disclaimers_to_add.append(self._safety_policies["required_disclaimers"][0])


        if disclaimers_to_add:
            # Avoid duplicating disclaimers if already present in the text
            unique_disclaimers = [d for d in disclaimers_to_add if d.lower() not in text.lower()]
            if unique_disclaimers:
                disclaimer_string = " ".join(unique_disclaimers)
                # Ensure a period if not already at the end of the main text
                if not text.endswith(('.', '!', '?')):
                    text += ". "
                else:
                    text += " "
                text += f"({disclaimer_string})"
                logger.debug(f"Added disclaimers: {unique_disclaimers}")
        return text

    def _check_fabrication(self, generated_text: str, source_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        A placeholder for a more sophisticated fabrication check.
        In a real system, this would involve comparing generated_text against
        known factual databases, cross-referencing information, or using
        an LLM specifically tuned for factual verification.
        For now, it's a basic check.
        """
        # Simple heuristic: If source_data is provided, check if key entities are mentioned.
        # This is a highly simplistic approach.
        if source_data:
            for key, value in source_data.items():
                if isinstance(value, str) and value.lower() in generated_text.lower():
                    # A very weak check, implies some source data was used.
                    return False
        
        # More advanced: Check if the text contains placeholder-like or overly generic phrasing
        # that might indicate fabrication without real data.
        if "i cannot provide specific details" in generated_text.lower() and source_data:
            return True # Potentially fabricating lack of data
        
        # Assume not fabricated if no strong indicators
        return False

    def _assess_confidence_and_adjust_response(self, response_data: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Assesses confidence scores (e.g., from Crop Doctor) and adjusts the response
        by adding disclaimers or modifying recommendations if confidence is low.
        """
        if context == "crop_doctor" and "confidence" in response_data:
            confidence = response_data["confidence"]
            if confidence < self._safety_policies["max_confidence_threshold_for_diagnosis"]:
                logger.warning(f"Low confidence diagnosis detected ({confidence:.2f}). Adding disclaimer.")
                response_data["recommendations"] = (
                    f"Due to lower confidence ({confidence*100:.1f}%), {self._safety_policies['disclaimer_for_low_confidence']} "
                    f"Original recommendation: {response_data.get('recommendations', 'No specific recommendations were provided.')}"
                )
                response_data["diagnosis"] = f"Potential {response_data.get('diagnosis', 'issue')}. " + self._safety_policies['disclaimer_for_low_confidence']
        return response_data

    def perform_safety_check(
        self,
        text: str,
        context: str,
        response_data: Optional[Dict[str, Any]] = None,
        source_data: Optional[Dict[str, Any]] = None
    ) -> SafetyCheckResponse:
        """
        Performs a comprehensive safety check on the AI's generated response text
        and associated data.

        Args:
            text (str): The primary text of the AI's response.
            context (str): The context of the AI's response (e.g., "crop_doctor", "crop_advisor", "weather").
            response_data (Optional[Dict[str, Any]]): Additional structured data from the AI service
                                                       (e.g., confidence scores from Crop Doctor).
            source_data (Optional[Dict[str, Any]]): Any source data used to generate the response,
                                                     useful for fabrication checks.

        Returns:
            SafetyCheckResponse: An object indicating safety status and potentially
                                 an adjusted response.
        """
        logger.info(f"Performing safety check for context: {context}")
        is_safe = True
        safety_concerns: List[str] = []
        adjusted_text = text
        adjusted_response_data = response_data if response_data is not None else {}

        # 1. Prohibited Content Check
        if self._check_for_prohibited_content(adjusted_text):
            is_safe = False
            safety_concerns.append("Prohibited content detected.")
            adjusted_text = "I cannot provide advice containing harmful or inappropriate content."

        # 2. Fabrication Check (Simplified)
        if self._check_fabrication(adjusted_text, source_data):
            is_safe = False
            safety_concerns.append("Potential fabrication or lack of factual basis detected.")
            adjusted_text = "I need more information or cannot verify this advice at the moment. Please consult a local expert."

        # 3. Confidence Assessment and Adjustment
        if adjusted_response_data:
            adjusted_response_data = self._assess_confidence_and_adjust_response(adjusted_response_data, context)
            # If the response data was adjusted, we need to ensure the text reflects it.
            # This part is highly dependent on how response_data translates to text.
            # For Crop Doctor, if 'recommendations' or 'diagnosis' changed, the text should be rebuilt.
            if context == "crop_doctor":
                if "diagnosis" in adjusted_response_data and "recommendations" in adjusted_response_data:
                    adjusted_text = (f"Diagnosis: {adjusted_response_data['diagnosis']} "
                                     f"Recommendations: {adjusted_response_data['recommendations']}")
                elif "diagnosis" in adjusted_response_data:
                    adjusted_text = f"Diagnosis: {adjusted_response_data['diagnosis']}"
        
        # 4. Add Required Disclaimers
        adjusted_text = self._add_required_disclaimers(adjusted_text, context)

        final_response = SafetyCheckResponse(
            is_safe=is_safe,
            adjusted_response_text=adjusted_text,
            safety_concerns=safety_concerns,
            adjusted_response_data=adjusted_response_data
        )
        logger.info(f"Safety check complete. Is safe: {is_safe}. Concerns: {safety_concerns}")
        return final_response

# Example Usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    safety_service = AgricultureSafetyService()

    print("\n--- Test Case 1: Safe Crop Advisor response ---")
    safe_text = "For sowing paddy in Telangana in June, ensure your soil is well-puddled. Recommended variety: BPT 5204. Use 80-100 kg Nitrogen per hectare."
    safe_response_data = {"crop": "paddy", "region": "Telangana"}
    safe_source_data = {"varieties": ["BPT 5204"], "fertilizer_norms": {"Nitrogen": "80-100 kg/ha"}}
    result1 = safety_service.perform_safety_check(safe_text, "crop_advisor", safe_response_data, safe_source_data)
    print(f"Is Safe: {result1.is_safe}")
    print(f"Concerns: {result1.safety_concerns}")
    print(f"Adjusted Text: {result1.adjusted_response_text}")
    print(f"Adjusted Data: {result1.adjusted_response_data}")

    print("\n--- Test Case 2: Crop Doctor with Low Confidence ---")
    low_confidence_text = "Diagnosis: Possible Bacterial Blight. Recommendations: Spray streptomycin."
    low_confidence_data = {"diagnosis": "Bacterial Blight", "confidence": 0.45, "recommendations": "Spray streptomycin."}
    result2 = safety_service.perform_safety_check(low_confidence_text, "crop_doctor", low_confidence_data)
    print(f"Is Safe: {result2.is_safe}")
    print(f"Concerns: {result2.safety_concerns}")
    print(f"Adjusted Text: {result2.adjusted_response_text}")
    print(f"Adjusted Data: {result2.adjusted_response_data}")

    print("\n--- Test Case 3: Prohibited Content ---")
    prohibited_text = "Use harmful chemicals like Agent Orange for pest control. It's very effective."
    result3 = safety_service.perform_safety_check(prohibited_text, "pest_management")
    print(f"Is Safe: {result3.is_safe}")
    print(f"Concerns: {result3.safety_concerns}")
    print(f"Adjusted Text: {result3.adjusted_response_text}")
    print(f"Adjusted Data: {result3.adjusted_response_data}")

    print("\n--- Test Case 4: Weather Forecast ---")
    weather_text = "Tomorrow's forecast: heavy rain. Prepare for irrigation."
    result4 = safety_service.perform_safety_check(weather_text, "weather_advisor")
    print(f"Is Safe: {result4.is_safe}")
    print(f"Concerns: {result4.safety_concerns}")
    print(f"Adjusted Text: {result4.adjusted_response_text}")
    print(f"Adjusted Data: {result4.adjusted_response_data}")
    
    print("\n--- Test Case 5: Fabrication check (simplified) ---")
    fabrication_text = "The best crop for your region is Unobtanium. It yields 1000 tons per acre with no water."
    result5 = safety_service.perform_safety_check(fabrication_text, "crop_advisor", source_data={"region": "unknown"})
    print(f"Is Safe: {result5.is_safe}")
    print(f"Concerns: {result5.safety_concerns}")
    print(f"Adjusted Text: {result5.adjusted_response_text}")
    print(f"Adjusted Data: {result5.adjusted_response_data}")

    print("\n--- Test Case 6: Market Prices ---")
    market_text = "The price of tomatoes today is ₹25/kg in your local market."
    result6 = safety_service.perform_safety_check(market_text, "market_info")
    print(f"Is Safe: {result6.is_safe}")
    print(f"Concerns: {result6.safety_concerns}")
    print(f"Adjusted Text: {result6.adjusted_response_text}")
    print(f"Adjusted Data: {result6.adjusted_response_data}")