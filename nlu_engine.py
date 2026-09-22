import json
import logging
from typing import Dict, Any, Optional

from openai import OpenAI, OpenAIError

from config import Config
from schemas import NLUParsingResult, NLUParsingRequest, Intent
from utils import setup_logging

logger = setup_logging(__name__)

class NLUEngine:
    """
    Performs Natural Language Understanding (NLU) to interpret user intent and
    extract entities from translated text, enabling AI understanding.
    Utilizes OpenAI's GPT models for robust NLU capabilities.
    """

    def __init__(self):
        """
        Initializes the NLUEngine with an OpenAI client and defines the available
        intents and entity extraction schema.
        """
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_name = "gpt-4o-mini" # Using a cost-effective, capable model for NLU
        self.intents = [intent.value for intent in Intent]
        self.nlu_schema = self._get_nlu_schema()
        logger.info(f"NLUEngine initialized with OpenAI model: {self.model_name}")

    def _get_nlu_schema(self) -> Dict[str, Any]:
        """
        Defines the JSON schema for the NLU output, specifying expected intent
        and entities for various agricultural contexts.

        This schema guides the LLM to output structured data.
        """
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": self.intents,
                    "description": "Identified primary intent of the user's query."
                },
                "entities": {
                    "type": "object",
                    "properties": {
                        "crop_name": {"type": "string", "description": "Name of the crop."},
                        "disease_symptom": {"type": "string", "description": "Description of disease symptom."},
                        "location": {"type": "string", "description": "Geographic location (city, district, state)."},
                        "season": {"type": "string", "description": "Farming season (e.g., Kharif, Rabi, Zaid, Monsoon, Winter, Summer)."},
                        "soil_type": {"type": "string", "description": "Type of soil (e.g., black, red, sandy, loamy)."},
                        "water_source": {"type": "string", "description": "Source of water (e.g., borewell, river, canal, rain)."},
                        "fertilizer_type": {"type": "string", "description": "Type of fertilizer (e.g., Urea, DAP, NPK)."},
                        "pest_name": {"type": "string", "description": "Name of the pest or description of pest issue."},
                        "government_scheme": {"type": "string", "description": "Name or topic of a government scheme."},
                        "market_commodity": {"type": "string", "description": "Agricultural commodity (e.g., wheat, rice, tomato)."},
                        "duration": {"type": "string", "description": "Time duration or frequency."},
                        "date": {"type": "string", "format": "date", "description": "Specific date related to the query."},
                        "time": {"type": "string", "format": "time", "description": "Specific time related to the query."},
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Severity of a problem (e.g., disease)."},
                        "query_type": {"type": "string", "enum": ["diagnosis", "prevention", "recommendation", "information"], "description": "Type of information being sought."},
                        "reminder_details": {"type": "string", "description": "Details for a reminder (e.g., 'water the plants', 'apply fertilizer')."},
                        "crop_stage": {"type": "string", "description": "Current growth stage of the crop (e.g., 'sowing', 'flowering', 'harvest')."},
                    },
                    "additionalProperties": True # Allow for other relevant entities not explicitly defined
                }
            },
            "required": ["intent", "entities"]
        }

    async def parse_text(self, request: NLUParsingRequest) -> NLUParsingResult:
        """
        Parses a given text to identify the user's intent and extract relevant entities
        using an OpenAI LLM.

        Args:
            request: NLUParsingRequest containing the text to be parsed.

        Returns:
            NLUParsingResult: An object containing the identified intent and extracted entities.

        Raises:
            OpenAIError: If there's an issue with the OpenAI API call.
            ValueError: If the NLU response from the LLM is malformed or invalid.
        """
        user_query = request.text
        logger.debug(f"Parsing user query: '{user_query}'")

        system_prompt = (
            "You are an expert agricultural AI assistant (KRISHIMITRA AI) specialized in understanding farmer queries. "
            "Your task is to identify the user's primary intent and extract all relevant entities "
            "from the provided text, strictly following the given JSON schema. "
            "Prioritize agricultural contexts for intent and entity identification. "
            "If the intent is unclear, default to 'general_query'. "
            "If no specific entities are found for an intent, the 'entities' object should be empty. "
            "Do NOT include any explanatory text or pleasantries, only the JSON output."
            f"Available intents are: {', '.join(self.intents)}."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "extract_nlu_data",
                            "description": "Extract intent and entities from an agricultural query.",
                            "parameters": self.nlu_schema,
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "extract_nlu_data"}}
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("LLM did not return tool calls for NLU parsing.")

            function_args = tool_calls[0].function.arguments
            parsed_data = json.loads(function_args)

            # Validate the parsed data against the schema (Pydantic will handle this)
            nlu_result = NLUParsingResult(**parsed_data)
            logger.info(f"Successfully parsed NLU for '{user_query}': Intent='{nlu_result.intent}', Entities={nlu_result.entities}")
            return nlu_result

        except OpenAIError as e:
            logger.error(f"OpenAI API error during NLU parsing: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}. Raw response: {function_args}")
            raise ValueError(f"Malformed JSON from NLU model: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during NLU parsing: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    import asyncio
    from schemas import Intent

    async def test_nlu():
        nlu_engine = NLUEngine()

        # Test cases
        test_queries = [
            ("నా పంట ఆకులు పసుపుగా మారుతున్నాయి. సహాయం కావాలి.", Intent.DIAGNOSE_CROP_PROBLEM), # Telugu for "My crop leaves are turning yellow. I need help."
            ("How to protect rice from pests?", Intent.PEST_MANAGEMENT),
            ("धान को कीटों से कैसे बचाएं?", Intent.PEST_MANAGEMENT), # Hindi for "How to protect rice from pests?"
            ("What is the weather like in Hyderabad tomorrow?", Intent.GET_WEATHER),
            ("பெங்களூரில் நாளை வானிலை எப்படி இருக்கும்?", Intent.GET_WEATHER), # Tamil for "What is the weather like in Bangalore tomorrow?"
            ("Suggest best practices for tomato cultivation.", Intent.CROP_ADVICE),
            ("Tell me about the PM Kisan Samman Nidhi scheme.", Intent.GET_GOVT_SCHEME_INFO),
            ("What is the current market price of potatoes in Delhi?", Intent.GET_MARKET_DATA),
            ("Set a reminder to water plants at 7 AM daily.", Intent.SET_REMINDER),
            ("My soil test results show low nitrogen. What should I do?", Intent.SOIL_ADVICE),
            ("मुझे पानी प्रबंधन के लिए कुछ सुझाव चाहिए।", Intent.WATER_MANAGEMENT), # Hindi for "I need some suggestions for water management."
            ("Tell me something general about farming.", Intent.GENERAL_QUERY)
        ]

        for query, expected_intent in test_queries:
            print(f"\n--- Testing Query: '{query}' ---")
            try:
                request = NLUParsingRequest(text=query)
                result = await nlu_engine.parse_text(request)
                print(f"Recognized Intent: {result.intent}")
                print(f"Extracted Entities: {result.entities}")
                assert result.intent == expected_intent, f"Intent mismatch for '{query}'. Expected {expected_intent}, got {result.intent}"
            except Exception as e:
                print(f"Error parsing '{query}': {e}")
            print("-" * 40)

    asyncio.run(test_nlu())