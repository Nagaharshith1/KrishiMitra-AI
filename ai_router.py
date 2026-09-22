from typing import Any, Dict, List, Optional, Type

from schemas import (
    CropAdvisorRequest, CropAdvisorResponse, CropDoctorRequest, CropDoctorResponse,
    MarketDataRequest, MarketDataResponse, SoilAIRequest, SoilAIResponse,
    WeatherRequest, WeatherResponse, GenericAIResponse,
    FarmPlannerRequest, FarmPlannerResponse, WaterManagementRequest, WaterManagementResponse,
    FertilizerAssistantRequest, FertilizerAssistantResponse, PestManagementRequest, PestManagementResponse,
    GovernmentSchemesRequest, GovernmentSchemesResponse
)
from nlu_engine import NLUResult
from utils import logger
from ai_safety import AgricultureSafetyService
from crop_doctor import CropDoctorService
from crop_management import (
    CropAdvisorService, WaterManagementService, SoilAIService,
    FertilizerAssistantService, PestManagementService, FarmPlannerService
)
from external_integrations import (
    WeatherIntegrationService, MarketIntegrationService, GovernmentSchemesIntegrationService
)


class AIRouter:
    """
    A centralized AI router that directs parsed user intents from the NLU engine
    to the appropriate specialized AI service.
    """

    def __init__(self,
                 safety_service: AgricultureSafetyService,
                 crop_doctor_service: CropDoctorService,
                 crop_advisor_service: CropAdvisorService,
                 water_management_service: WaterManagementService,
                 soil_ai_service: SoilAIService,
                 fertilizer_assistant_service: FertilizerAssistantService,
                 pest_management_service: PestManagementService,
                 farm_planner_service: FarmPlannerService,
                 weather_integration_service: WeatherIntegrationService,
                 market_integration_service: MarketIntegrationService,
                 government_schemes_integration_service: GovernmentSchemesIntegrationService):
        """
        Initializes the AIRouter with instances of various AI and integration services.

        Args:
            safety_service: An instance of AgricultureSafetyService.
            crop_doctor_service: An instance of CropDoctorService.
            crop_advisor_service: An instance of CropAdvisorService.
            water_management_service: An instance of WaterManagementService.
            soil_ai_service: An instance of SoilAIService.
            fertilizer_assistant_service: An instance of FertilizerAssistantService.
            pest_management_service: An instance of PestManagementService.
            farm_planner_service: An instance of FarmPlannerService.
            weather_integration_service: An instance of WeatherIntegrationService.
            market_integration_service: An instance of MarketIntegrationService.
            government_schemes_integration_service: An instance of GovernmentSchemesIntegrationService.
        """
        self._safety_service = safety_service
        self._crop_doctor_service = crop_doctor_service
        self._crop_advisor_service = crop_advisor_service
        self._water_management_service = water_management_service
        self._soil_ai_service = soil_ai_service
        self._fertilizer_assistant_service = fertilizer_assistant_service
        self._pest_management_service = pest_management_service
        self._farm_planner_service = farm_planner_service
        self._weather_integration_service = weather_integration_service
        self._market_integration_service = market_integration_service
        self._government_schemes_integration_service = government_schemes_integration_service
        
        self.intent_to_service_map: Dict[str, Any] = {
            "crop_doctor_image_analysis": self._crop_doctor_service, # Special case for image analysis
            "diagnose_crop_problem": self._crop_doctor_service,
            "crop_advisor": self._crop_advisor_service,
            "weather_forecast": self._weather_integration_service,
            "water_management": self._water_management_service,
            "soil_ai": self._soil_ai_service,
            "fertilizer_assistant": self._fertilizer_assistant_service,
            "pest_management": self._pest_management_service,
            "farm_planner": self._farm_planner_service,
            "market_data": self._market_integration_service,
            "government_schemes": self._government_schemes_integration_service,
            # Add more intents as they are defined in NLU engine
        }
        
        # Mapping intents to the expected method name for dispatch
        self.intent_to_method_map: Dict[str, str] = {
            "diagnose_crop_problem": "get_diagnosis_from_description",
            "crop_doctor_image_analysis": "analyze_image_for_problems", # Will be called directly from ai_orchestrator
            "crop_advisor": "get_advice",
            "weather_forecast": "get_weather_forecast",
            "water_management": "get_irrigation_advice",
            "soil_ai": "explain_soil_test_results",
            "fertilizer_assistant": "get_fertilizer_recommendation",
            "pest_management": "get_pest_control_advice",
            "farm_planner": "get_or_update_calendar",
            "market_data": "get_market_prices",
            "government_schemes": "get_government_schemes",
        }
        
        # Mapping intents to expected request schema type
        self.intent_to_request_schema: Dict[str, Type[Any]] = {
            "diagnose_crop_problem": CropDoctorRequest,
            "crop_doctor_image_analysis": CropDoctorRequest, # For future direct use, though image path is handled
            "crop_advisor": CropAdvisorRequest,
            "weather_forecast": WeatherRequest,
            "water_management": WaterManagementRequest,
            "soil_ai": SoilAIRequest,
            "fertilizer_assistant": FertilizerAssistantRequest,
            "pest_management": PestManagementRequest,
            "farm_planner": FarmPlannerRequest,
            "market_data": MarketDataRequest,
            "government_schemes": GovernmentSchemesRequest,
        }

    async def route_request(self, nlu_result: NLUResult, image_data_base64: Optional[str] = None) -> GenericAIResponse:
        """
        Routes the user's request based on the NLU result to the appropriate AI service.

        Args:
            nlu_result: The result from the Natural Language Understanding engine.
            image_data_base64: Optional base64 encoded image data if the request involves image analysis.

        Returns:
            A GenericAIResponse containing the service's output.

        Raises:
            ValueError: If the intent is not recognized or a required parameter is missing.
        """
        intent = nlu_result.intent
        entities = nlu_result.entities
        
        logger.info(f"Routing request for intent: '{intent}' with entities: {entities}")

        if intent == "unrecognized_intent":
            response_text = "I couldn't understand your request. Please try rephrasing."
            return GenericAIResponse(
                response_text=response_text,
                original_intent="unrecognized_intent",
                processed_data={},
                confidence=0.0
            )

        # Handle image analysis as a special case, potentially overriding other intents if image is present
        if image_data_base64 and intent != "crop_doctor_image_analysis":
            # If an image is provided, and the NLU didn't explicitly identify an image analysis intent,
            # assume it's for crop doctor image analysis. This might need refinement based on UX flows.
            logger.warning(f"Image data provided, overriding intent '{intent}' to 'crop_doctor_image_analysis'.")
            intent = "crop_doctor_image_analysis"
            
        if intent not in self.intent_to_service_map:
            error_msg = f"No service mapped for intent: {intent}"
            logger.error(error_msg)
            return GenericAIResponse(
                response_text=f"I apologize, but I don't have the capability to handle requests about '{intent}'. "
                              "Please ask me something related to crop health, weather, or farm management.",
                original_intent=intent,
                processed_data={},
                confidence=0.0
            )

        service = self.intent_to_service_map[intent]
        method_name = self.intent_to_method_map.get(intent)
        
        if not method_name:
            error_msg = f"No method defined for intent: {intent} in intent_to_method_map."
            logger.error(error_msg)
            return GenericAIResponse(
                response_text=f"An internal error occurred while processing your request (no method mapping).",
                original_intent=intent,
                processed_data={},
                confidence=0.0
            )

        # Special handling for Crop Doctor image analysis
        if intent == "crop_doctor_image_analysis":
            if not image_data_base64:
                error_msg = "Crop Doctor image analysis intent received but no image data provided."
                logger.error(error_msg)
                return GenericAIResponse(
                    response_text="Please provide an image for crop disease detection.",
                    original_intent=intent,
                    processed_data={},
                    confidence=0.0
                )
            try:
                request_model = CropDoctorRequest(
                    image_base64=image_data_base64,
                    problem_description=entities.get("problem_description") # Use description if available
                )
                response: CropDoctorResponse = await self._crop_doctor_service.analyze_image_for_problems(request_model)
                response_text = (
                    f"Diagnosis: {response.diagnosis}. "
                    f"Confidence: {response.confidence_score:.0%}. "
                    f"Recommendation: {response.recommendations}. "
                    f"Prevention: {response.prevention_strategies}."
                )
                return self._safety_service.ensure_safe_response(
                    GenericAIResponse(
                        response_text=response_text,
                        original_intent=intent,
                        processed_data=response.dict(),
                        confidence=response.confidence_score
                    )
                )
            except Exception as e:
                logger.error(f"Error processing Crop Doctor image analysis: {e}", exc_info=True)
                return GenericAIResponse(
                    response_text="I encountered an error while analyzing the image. Please try again.",
                    original_intent=intent,
                    processed_data={},
                    confidence=0.0
                )
        
        # For other intents, dynamically call the respective service method
        try:
            request_schema = self.intent_to_request_schema.get(intent)
            if not request_schema:
                raise ValueError(f"No request schema defined for intent: {intent}")
            
            # Prepare request data using entities
            request_data = {k: v for k, v in entities.items() if k in request_schema.__fields__}
            
            # Add original text as problem_description if available for text-based crop diagnosis
            if intent == "diagnose_crop_problem" and "problem_description" not in request_data and nlu_result.original_text:
                request_data["problem_description"] = nlu_result.original_text

            # Create Pydantic request model
            request_model = request_schema(**request_data)
            
            service_method = getattr(service, method_name)
            raw_response = await service_method(request_model)
            
            # Convert raw_response (which is a Pydantic model) to a dictionary for GenericAIResponse
            processed_data = raw_response.dict()
            
            # Construct a human-readable response text based on the raw_response
            response_text = self._format_response_text(intent, raw_response)
            
            return self._safety_service.ensure_safe_response(
                GenericAIResponse(
                    response_text=response_text,
                    original_intent=intent,
                    processed_data=processed_data,
                    confidence=nlu_result.confidence # Use NLU confidence for overall response, or service-specific if available
                )
            )

        except Exception as e:
            logger.error(f"Error routing request for intent '{intent}': {e}", exc_info=True)
            return GenericAIResponse(
                response_text=f"I'm sorry, an unexpected error occurred while processing your request for '{intent}'. Please try again.",
                original_intent=intent,
                processed_data={},
                confidence=0.0
            )

    def _format_response_text(self, intent: str, response_model: Any) -> str:
        """
        Formats the service response model into a human-readable text string.

        Args:
            intent: The recognized intent.
            response_model: The Pydantic response model from the specific service.

        Returns:
            A formatted string of the AI's response.
        """
        if isinstance(response_model, CropDoctorResponse):
            return (f"Based on your description, the diagnosis is: {response_model.diagnosis}. "
                    f"Confidence: {response_model.confidence_score:.0%}. "
                    f"Recommendations: {response_model.recommendations}. "
                    f"Prevention: {response_model.prevention_strategies}.")
        elif isinstance(response_model, CropAdvisorResponse):
            return f"Here's some advice for your crop: {response_model.advice}"
        elif isinstance(response_model, WeatherResponse):
            return f"The weather forecast for {response_model.location} is: {response_model.forecast}. Temperature: {response_model.temperature_celsius}°C. Humidity: {response_model.humidity}%."
        elif isinstance(response_model, WaterManagementResponse):
            return f"Regarding water management: {response_model.advice}. Suggested irrigation frequency: {response_model.irrigation_frequency}."
        elif isinstance(response_model, SoilAIResponse):
            return f"Your soil analysis shows: {response_model.explanation}. Recommendations: {response_model.recommendations}."
        elif isinstance(response_model, FertilizerAssistantResponse):
            return f"For fertilizer, I recommend: {response_model.recommendation}. Application details: {response_model.application_guidance}."
        elif isinstance(response_model, PestManagementResponse):
            return f"For pest management, consider: {response_model.advice}. Suggested products: {response_model.suggested_products}."
        elif isinstance(response_model, FarmPlannerResponse):
            if response_model.event_type == "reminder":
                return f"Reminder set for {response_model.event_name} on {response_model.event_date.strftime('%Y-%m-%d')}. Details: {response_model.details}."
            else:
                return f"Your farm calendar has been updated with: {response_model.event_name} on {response_model.event_date.strftime('%Y-%m-%d')}. Details: {response_model.details}."
        elif isinstance(response_model, MarketDataResponse):
            prices_str = ", ".join([f"{item.crop}: ₹{item.price_per_unit}/{item.unit}" for item in response_model.market_prices])
            return f"Latest market prices for {response_model.location}: {prices_str}. Data as of {response_model.as_of_date.strftime('%Y-%m-%d')}."
        elif isinstance(response_model, GovernmentSchemesResponse):
            schemes_str = "; ".join([f"{s.name}: {s.description}. Eligibility: {s.eligibility_criteria}. Link: {s.application_link}" for s in response_model.schemes])
            return f"Here are some government schemes: {schemes_str}"
        else:
            # Default fallback for unknown response types or generic responses
            return str(response_model.dict()) # Convert Pydantic model to string representation of its dict


if __name__ == "__main__":
    # This block is for testing purposes only and requires mocked dependencies
    # In a real scenario, these services would be properly instantiated and injected.
    print("Running AIRouter test scenario...")

    # Mock Services for testing
    class MockSafetyService(AgricultureSafetyService):
        def __init__(self):
            super().__init__()
        def ensure_safe_response(self, response: GenericAIResponse) -> GenericAIResponse:
            logger.info(f"Safety check passed for intent: {response.original_intent}")
            return response

    class MockCropDoctorService(CropDoctorService):
        def __init__(self):
            # For testing, we don't need real models, just mock the responses
            pass 
        async def analyze_image_for_problems(self, request: CropDoctorRequest) -> CropDoctorResponse:
            logger.info(f"Mock Crop Doctor: Analyzing image for '{request.problem_description}'...")
            return CropDoctorResponse(
                diagnosis="Early blight",
                confidence_score=0.85,
                recommendations="Apply fungicide, remove infected leaves.",
                prevention_strategies="Ensure proper spacing, use disease-resistant varieties.",
                image_analysis_url="http://mock-analysis-url.com"
            )
        async def get_diagnosis_from_description(self, request: CropDoctorRequest) -> CropDoctorResponse:
            logger.info(f"Mock Crop Doctor: Diagnosing from description '{request.problem_description}'...")
            return CropDoctorResponse(
                diagnosis="Nutrient deficiency (likely Nitrogen)",
                confidence_score=00.75,
                recommendations="Apply nitrogen-rich fertilizer, check soil pH.",
                prevention_strategies="Regular soil testing, balanced fertilization.",
                image_analysis_url=None
            )

    class MockCropAdvisorService(CropAdvisorService):
        def __init__(self): pass
        async def get_advice(self, request: CropAdvisorRequest) -> CropAdvisorResponse:
            logger.info(f"Mock Crop Advisor: Getting advice for crop '{request.crop_type}' in {request.location}...")
            return CropAdvisorResponse(advice=f"For {request.crop_type} in {request.location}, consider {request.season} sowing. Ensure proper soil preparation.", crop_type=request.crop_type)

    class MockWeatherIntegrationService(WeatherIntegrationService):
        def __init__(self): pass
        async def get_weather_forecast(self, request: WeatherRequest) -> WeatherResponse:
            logger.info(f"Mock Weather: Getting forecast for {request.location}...")
            return WeatherResponse(
                location=request.location,
                forecast="Partly cloudy with chances of light rain.",
                temperature_celsius=28.5,
                humidity=70,
                wind_speed_kmph=15
            )

    class MockWaterManagementService(WaterManagementService):
        def __init__(self): pass
        async def get_irrigation_advice(self, request: WaterManagementRequest) -> WaterManagementResponse:
            logger.info(f"Mock Water Management: Getting advice for {request.crop_type}...")
            return WaterManagementResponse(
                crop_type=request.crop_type,
                advice=f"For {request.crop_type}, use drip irrigation system. Irrigate every other day.",
                irrigation_frequency="Every 2 days",
                method_guidance="Drip irrigation recommended."
            )

    class MockSoilAIService(SoilAIService):
        def __init__(self): pass
        async def explain_soil_test_results(self, request: SoilAIRequest) -> SoilAIResponse:
            logger.info(f"Mock Soil AI: Explaining soil results for '{request.soil_test_results_description}'...")
            return SoilAIResponse(
                explanation=f"Your soil has a pH of {request.soil_test_results_description.get('ph', 6.5)}. Nitrogen is low.",
                recommendations="Add organic matter and a balanced NPK fertilizer."
            )

    class MockFertilizerAssistantService(FertilizerAssistantService):
        def __init__(self): pass
        async def get_fertilizer_recommendation(self, request: FertilizerAssistantRequest) -> FertilizerAssistantResponse:
            logger.info(f"Mock Fertilizer: Getting recommendation for '{request.crop_type}'...")
            return FertilizerAssistantResponse(
                crop_type=request.crop_type,
                recommendation="Use Urea (46-0-0) for nitrogen boost.",
                application_guidance="Apply 50kg/acre split into two doses."
            )

    class MockPestManagementService(PestManagementService):
        def __init__(self): pass
        async def get_pest_control_advice(self, request: PestManagementRequest) -> PestManagementResponse:
            logger.info(f"Mock Pest: Getting advice for '{request.pest_problem_description}'...")
            return PestManagementResponse(
                pest_problem_description=request.pest_problem_description,
                advice="For aphids, use neem oil spray. Apply in early morning.",
                suggested_products=["Neem Oil"],
                prevention_strategies=["Crop rotation", "Natural predators"]
            )

    class MockFarmPlannerService(FarmPlannerService):
        def __init__(self): pass
        async def get_or_update_calendar(self, request: FarmPlannerRequest) -> FarmPlannerResponse:
            logger.info(f"Mock Farm Planner: Creating event '{request.event_name}' for {request.event_date}...")
            return FarmPlannerResponse(
                event_type="planting",
                event_name=request.event_name,
                event_date=request.event_date,
                details=f"Planting {request.details} for this season."
            )

    class MockMarketIntegrationService(MarketIntegrationService):
        def __init__(self): pass
        async def get_market_prices(self, request: MarketDataRequest) -> MarketDataResponse:
            logger.info(f"Mock Market Data: Getting prices for {request.crop_type} in {request.location}...")
            return MarketDataResponse(
                location=request.location,
                market_prices=[
                    {"crop": request.crop_type, "price_per_unit": 35.0, "unit": "kg"},
                    {"crop": "Tomato", "price_per_unit": 20.0, "unit": "kg"}
                ],
                as_of_date="2023-10-27"
            )

    class MockGovernmentSchemesIntegrationService(GovernmentSchemesIntegrationService):
        def __init__(self): pass
        async def get_government_schemes(self, request: GovernmentSchemesRequest) -> GovernmentSchemesResponse:
            logger.info(f"Mock Govt Schemes: Getting schemes for {request.category} in {request.location}...")
            return GovernmentSchemesResponse(
                category=request.category,
                location=request.location,
                schemes=[
                    {"name": "PM KISAN", "description": "Income support for farmers.", "eligibility_criteria": "Small and marginal farmers.", "application_link": "http://pmkisan.gov.in"},
                    {"name": "Fasal Bima Yojana", "description": "Crop insurance scheme.", "eligibility_criteria": "Farmers growing notified crops.", "application_link": "http://pmfby.gov.in"}
                ]
            )


    async def run_router_tests():
        safety_service = MockSafetyService()
        crop_doctor = MockCropDoctorService()
        crop_advisor = MockCropAdvisorService()
        water_management = MockWaterManagementService()
        soil_ai = MockSoilAIService()
        fertilizer_assistant = MockFertilizerAssistantService()
        pest_management = MockPestManagementService()
        farm_planner = MockFarmPlannerService()
        weather_integration = MockWeatherIntegrationService()
        market_integration = MockMarketIntegrationService()
        government_schemes_integration = MockGovernmentSchemesIntegrationService()

        router = AIRouter(
            safety_service=safety_service,
            crop_doctor_service=crop_doctor,
            crop_advisor_service=crop_advisor,
            water_management_service=water_management,
            soil_ai_service=soil_ai,
            fertilizer_assistant_service=fertilizer_assistant,
            pest_management_service=pest_management,
            farm_planner_service=farm_planner,
            weather_integration_service=weather_integration,
            market_integration_service=market_integration,
            government_schemes_integration_service=government_schemes_integration
        )

        print("\n--- Test 1: Crop Doctor (image analysis with problem description) ---")
        nlu_result_crop_image = NLUResult(
            original_text="My tomato leaves are yellowing and have spots.",
            translated_text="My tomato leaves are yellowing and have spots.",
            intent="crop_doctor_image_analysis",
            entities={"crop_type": "tomato", "problem_description": "yellowing leaves with spots"},
            confidence=0.9
        )
        # In a real scenario, this would be actual base64 image data
        mock_image_data_base64 = "mock_base64_image_data_tomato_leaf" 
        response = await router.route_request(nlu_result_crop_image, image_data_base64=mock_image_data_base64)
        print(f"Response: {response.response_text}\n")

        print("\n--- Test 2: Crop Doctor (text description only) ---")
        nlu_result_crop_text = NLUResult(
            original_text="నా పంట ఆకులు పసుపుగా మారుతున్నాయి.", # Telugu for "My crop leaves are turning yellow."
            translated_text="My crop leaves are turning yellow.",
            intent="diagnose_crop_problem",
            entities={"problem_description": "leaves turning yellow"},
            confidence=0.8
        )
        response = await router.route_request(nlu_result_crop_text)
        print(f"Response: {response.response_text}\n")


        print("\n--- Test 3: Crop Advisor ---")
        nlu_result_advisor = NLUResult(
            original_text="What should I plant this rabi season in Punjab?",
            translated_text="What should I plant this rabi season in Punjab?",
            intent="crop_advisor",
            entities={"season": "rabi", "location": "Punjab", "crop_type": "unspecified"},
            confidence=0.95
        )
        response = await router.route_request(nlu_result_advisor)
        print(f"Response: {response.response_text}\n")

        print("\n--- Test 4: Weather Forecast ---")
        nlu_result_weather = NLUResult(
            original_text="What's the weather like in Hyderabad tomorrow?",
            translated_text="What's the weather like in Hyderabad tomorrow?",
            intent="weather_forecast",
            entities={"location": "Hyderabad", "date": "tomorrow"},
            confidence=0.88
        )
        response = await router.route_request(nlu_result_weather)
        print(f"Response: {response.response_text}\n")

        print("\n--- Test 5: Unrecognized Intent ---")
        nlu_result_unrecognized = NLUResult(
            original_text="Tell me a joke about farming.",
            translated_text="Tell me a joke about farming.",
            intent="tell_joke",
            entities={},
            confidence=0.3
        )
        response = await router.route_request(nlu_result_unrecognized)
        print(f"Response: {response.response_text}\n")
        
        print("\n--- Test 6: Market Data ---")
        nlu_result_market = NLUResult(
            original_text="What are the current potato prices in Bengaluru?",
            translated_text="What are the current potato prices in Bengaluru?",
            intent="market_data",
            entities={"crop_type": "Potato", "location": "Bengaluru"},
            confidence=0.9
        )
        response = await router.route_request(nlu_result_market)
        print(f"Response: {response.response_text}\n")

        print("\n--- Test 7: Government Schemes ---")
        nlu_result_gov_schemes = NLUResult(
            original_text="Are there any government schemes for farmers?",
            translated_text="Are there any government schemes for farmers?",
            intent="government_schemes",
            entities={"category": "farmers", "location": "India"},
            confidence=0.85
        )
        response = await router.route_request(nlu_result_gov_schemes)
        print(f"Response: {response.response_text}\n")

        print("\n--- Test 8: Farm Planner (Set reminder) ---")
        from datetime import date
        nlu_result_farm_planner = NLUResult(
            original_text="Remind me to fertilize the wheat field next Monday.",
            translated_text="Remind me to fertilize the wheat field next Monday.",
            intent="farm_planner",
            entities={
                "event_name": "Fertilize Wheat",
                "event_date": date.fromisoformat("2023-10-30"), # Mock next Monday
                "details": "wheat field",
                "event_type": "reminder"
            },
            confidence=0.92
        )
        response = await router.route_request(nlu_result_farm_planner)
        print(f"Response: {response.response_text}\n")


    import asyncio
    asyncio.run(run_router_tests())