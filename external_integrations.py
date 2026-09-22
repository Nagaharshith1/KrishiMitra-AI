import httpx
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, date

from config import Config
from schemas import (
    WeatherForecast, WeatherAlert, WeatherCondition,
    MarketDataPoint, MarketPriceResponse,
    GovernmentScheme, GovernmentSchemeResponse
)
from utils import setup_logging, KrishiMitraException

logger = setup_logging(__name__)

class ExternalIntegrationsService:
    """
    Manages integrations with external APIs for weather data, market prices,
    and government schemes.
    """

    def __init__(self):
        self.weather_api_key = Config.WEATHER_API_KEY
        self.weather_api_url = Config.WEATHER_API_URL
        self.market_data_api_key = Config.MARKET_DATA_API_KEY
        self.market_data_api_url = Config.MARKET_DATA_API_URL
        self.govt_schemes_api_url = Config.GOVT_SCHEMES_API_URL
        self.client = httpx.AsyncClient(timeout=30.0) # Using AsyncClient for FastAPI async compatibility

    async def _make_request(self, url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Internal helper to make an asynchronous HTTP GET request.

        Args:
            url: The URL to make the request to.
            params: Dictionary of query parameters.
            headers: Optional dictionary of request headers.

        Returns:
            JSON response as a dictionary.

        Raises:
            KrishiMitraException: If the API request fails or returns an error.
        """
        try:
            logger.debug(f"Making external API request to: {url} with params: {params}")
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raises HTTPStatusError for bad responses (4xx, 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"API request to {url} failed with status {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise KrishiMitraException(error_msg, error_code=e.response.status_code)
        except httpx.RequestError as e:
            error_msg = f"An error occurred while requesting {e.request.url!r}: {e}"
            logger.error(error_msg)
            raise KrishiMitraException(error_msg, error_code=500)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON response from {url}: {e}"
            logger.error(error_msg)
            raise KrishiMitraException(error_msg, error_code=500)
        except Exception as e:
            error_msg = f"An unexpected error occurred during API request to {url}: {e}"
            logger.error(error_msg)
            raise KrishiMitraException(error_msg, error_code=500)

    async def get_weather_data(self, latitude: float, longitude: float) -> WeatherForecast:
        """
        Fetches current weather and forecast data from an external weather API.
        Uses OpenWeatherMap API for demonstration.

        Args:
            latitude: The latitude of the location.
            longitude: The longitude of the location.

        Returns:
            A WeatherForecast object containing current and forecast weather details.

        Raises:
            KrishiMitraException: If weather data cannot be fetched.
        """
        if not self.weather_api_key or self.weather_api_key == "your-weather-api-key":
            logger.warning("Weather API key not configured. Returning mock data.")
            return self._get_mock_weather_data(latitude, longitude)

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": self.weather_api_key,
            "units": "metric",  # For Celsius
            "exclude": "minutely,hourly" # To reduce data for forecast demo
        }
        url = f"{self.weather_api_url}"
        try:
            data = await self._make_request(url, params)
            logger.debug(f"Raw weather data received: {data}")

            current_weather_data = data.get("current", {})
            daily_forecast_data = data.get("daily", [])

            # Parse current weather
            current_condition = WeatherCondition(
                timestamp=datetime.fromtimestamp(current_weather_data.get("dt", datetime.now().timestamp())),
                temperature=current_weather_data.get("temp"),
                feels_like_temperature=current_weather_data.get("feels_like"),
                pressure=current_weather_data.get("pressure"),
                humidity=current_weather_data.get("humidity"),
                dew_point=current_weather_data.get("dew_point"),
                uvi=current_weather_data.get("uvi"),
                clouds=current_weather_data.get("clouds"),
                visibility=current_weather_data.get("visibility"),
                wind_speed=current_weather_data.get("wind_speed"),
                wind_deg=current_weather_data.get("wind_deg"),
                weather_main=current_weather_data.get("weather", [{}])[0].get("main"),
                weather_description=current_weather_data.get("weather", [{}])[0].get("description"),
                pop=None # Not available in current for OpenWeatherMap
            )

            # Parse daily forecast (first 5 days for simplicity)
            forecast_conditions: List[WeatherCondition] = []
            for day_data in daily_forecast_data[:5]:
                forecast_conditions.append(WeatherCondition(
                    timestamp=datetime.fromtimestamp(day_data.get("dt", datetime.now().timestamp())),
                    temperature=day_data.get("temp", {}).get("day"),
                    feels_like_temperature=day_data.get("feels_like", {}).get("day"),
                    pressure=day_data.get("pressure"),
                    humidity=day_data.get("humidity"),
                    dew_point=day_data.get("dew_point"),
                    uvi=day_data.get("uvi"),
                    clouds=day_data.get("clouds"),
                    wind_speed=day_data.get("wind_speed"),
                    wind_deg=day_data.get("wind_deg"),
                    weather_main=day_data.get("weather", [{}])[0].get("main"),
                    weather_description=day_data.get("weather", [{}])[0].get("description"),
                    pop=day_data.get("pop")
                ))

            # No specific alert data in OpenWeatherMap One Call API by default without additional subscriptions
            # Mocking alerts for demo purposes
            alerts: List[WeatherAlert] = []
            if current_condition.weather_main == "Rain":
                alerts.append(WeatherAlert(
                    severity="Moderate",
                    event="Heavy Rain Advisory",
                    description="Expect moderate to heavy rainfall in the next 24 hours. Prepare for waterlogging.",
                    start_time=datetime.now(),
                    end_time=datetime.now() # Mock end time
                ))
            elif current_condition.temperature is not None and current_condition.temperature > 35:
                alerts.append(WeatherAlert(
                    severity="High",
                    event="Heatwave Warning",
                    description="High temperatures expected. Advise minimal outdoor activity, especially during peak hours.",
                    start_time=datetime.now(),
                    end_time=datetime.now() # Mock end time
                ))


            return WeatherForecast(
                latitude=latitude,
                longitude=longitude,
                current_condition=current_condition,
                forecast_conditions=forecast_conditions,
                alerts=alerts
            )
        except KrishiMitraException:
            raise # Re-raise custom exceptions directly
        except Exception as e:
            logger.error(f"Error processing weather data for {latitude}, {longitude}: {e}")
            raise KrishiMitraException(f"Failed to fetch or parse weather data: {e}", error_code=500)

    def _get_mock_weather_data(self, latitude: float, longitude: float) -> WeatherForecast:
        """
        Generates mock weather data for demonstration when API key is missing.
        """
        logger.info("Returning mock weather data.")
        current_time = datetime.now()
        tomorrow = current_time.date() + timedelta(days=1)
        day_after_tomorrow = current_time.date() + timedelta(days=2)

        return WeatherForecast(
            latitude=latitude,
            longitude=longitude,
            current_condition=WeatherCondition(
                timestamp=current_time,
                temperature=28.5,
                feels_like_temperature=32.0,
                pressure=1012,
                humidity=75,
                dew_point=24.1,
                uvi=7.2,
                clouds=40,
                visibility=10000,
                wind_speed=5.1,
                wind_deg=220,
                weather_main="Clouds",
                weather_description="partly cloudy",
                pop=0.1
            ),
            forecast_conditions=[
                WeatherCondition(
                    timestamp=datetime.combine(tomorrow, time(12, 0)),
                    temperature=30.0,
                    feels_like_temperature=34.5,
                    pressure=1010,
                    humidity=70,
                    dew_point=25.0,
                    uvi=8.0,
                    clouds=20,
                    wind_speed=4.5,
                    wind_deg=200,
                    weather_main="Clear",
                    weather_description="clear sky",
                    pop=0.05
                ),
                WeatherCondition(
                    timestamp=datetime.combine(day_after_tomorrow, time(12, 0)),
                    temperature=27.0,
                    feels_like_temperature=30.0,
                    pressure=1015,
                    humidity=80,
                    dew_point=24.5,
                    uvi=6.5,
                    clouds=70,
                    wind_speed=6.0,
                    wind_deg=250,
                    weather_main="Rain",
                    weather_description="light rain",
                    pop=0.7
                ),
            ],
            alerts=[
                WeatherAlert(
                    severity="Moderate",
                    event="Monsoon Rains Expected",
                    description="Light to moderate monsoon showers expected in the next 48 hours. Ensure proper drainage.",
                    start_time=current_time,
                    end_time=datetime.now() + timedelta(days=2)
                )
            ]
        )


    async def get_market_prices(self, commodity: str, location: str, date_from: Optional[date] = None, date_to: Optional[date] = None) -> MarketPriceResponse:
        """
        Fetches market prices for a specified commodity and location.
        Mocks an external API for demonstration.

        Args:
            commodity: The agricultural commodity (e.g., "Tomato", "Wheat").
            location: The market location (e.g., "Hyderabad", "Delhi").
            date_from: Optional start date for price history.
            date_to: Optional end date for price history.

        Returns:
            A MarketPriceResponse object with current and historical prices.

        Raises:
            KrishiMitraException: If market data cannot be fetched.
        """
        if not self.market_data_api_key or self.market_data_api_key == "your-market-data-api-key":
            logger.warning("Market Data API key not configured. Returning mock data.")
            return self._get_mock_market_prices(commodity, location, date_from, date_to)

        params = {
            "commodity": commodity,
            "location": location,
            "api_key": self.market_data_api_key,
        }
        if date_from:
            params["date_from"] = date_from.isoformat()
        if date_to:
            params["date_to"] = date_to.isoformat()

        # This is a placeholder for a real API endpoint
        url = self.market_data_api_url

        try:
            # For a real API, uncomment below:
            # data = await self._make_request(url, params)
            # market_data_points = [
            #     MarketDataPoint(
            #         date=datetime.strptime(dp["date"], "%Y-%m-%d").date(),
            #         min_price=dp["min_price"],
            #         max_price=dp["max_price"],
            #         avg_price=dp["avg_price"],
            #         unit=dp.get("unit", "INR/quintal")
            #     ) for dp in data.get("prices", [])
            # ]
            # current_price = market_data_points[-1] if market_data_points else None

            # For demo, using mock data temporarily
            return self._get_mock_market_prices(commodity, location, date_from, date_to)

        except KrishiMitraException:
            raise
        except Exception as e:
            logger.error(f"Error processing market data for {commodity}, {location}: {e}")
            raise KrishiMitraException(f"Failed to fetch market data: {e}", error_code=500)

    def _get_mock_market_prices(self, commodity: str, location: str, date_from: Optional[date] = None, date_to: Optional[date] = None) -> MarketPriceResponse:
        """
        Generates mock market price data.
        """
        logger.info(f"Returning mock market data for {commodity} in {location}.")
        current_date = date.today()
        mock_prices: List[MarketDataPoint] = []
        for i in range(5):
            day = current_date - timedelta(days=i)
            if (date_from is None or day >= date_from) and (date_to is None or day <= date_to):
                base_price = 2500 if commodity.lower() == "wheat" else 1500 if commodity.lower() == "rice" else 800 if commodity.lower() == "tomato" else 1000
                mock_prices.append(
                    MarketDataPoint(
                        date=day,
                        min_price=base_price - 100 + i*10,
                        max_price=base_price + 200 + i*10,
                        avg_price=base_price + 50 + i*10,
                        unit="INR/quintal"
                    )
                )
        mock_prices.reverse() # Order by date ascending

        current_price = mock_prices[-1] if mock_prices else MarketDataPoint(date=current_date, min_price=0, max_price=0, avg_price=0, unit="INR/quintal")

        return MarketPriceResponse(
            commodity=commodity,
            location=location,
            current_price=current_price,
            historical_prices=mock_prices
        )

    async def get_government_schemes(self, language_code: str, region: Optional[str] = None, crop_type: Optional[str] = None) -> GovernmentSchemeResponse:
        """
        Fetches relevant government schemes for farmers.
        Mocks an external API for demonstration.

        Args:
            language_code: The preferred language for scheme descriptions (e.g., 'en', 'hi').
            region: Optional region/state to filter schemes.
            crop_type: Optional crop type to filter schemes.

        Returns:
            A GovernmentSchemeResponse object containing a list of schemes.

        Raises:
            KrishiMitraException: If government schemes cannot be fetched.
        """
        # No specific API key check for government schemes as it's often public data,
        # but a real API might require one. Using mock data for now.
        if self.govt_schemes_api_url == "https://api.example.com/govtschemes":
            logger.warning("Government Schemes API URL is default placeholder. Returning mock data.")
            return self._get_mock_government_schemes(language_code, region, crop_type)

        params: Dict[str, Any] = {
            "lang": language_code
        }
        if region:
            params["region"] = region
        if crop_type:
            params["crop_type"] = crop_type

        # This is a placeholder for a real API endpoint
        url = self.govt_schemes_api_url

        try:
            # For a real API, uncomment below:
            # data = await self._make_request(url, params)
            # schemes = [
            #     GovernmentScheme(
            #         name=s["name"],
            #         description=s["description"],
            #         eligibility=s.get("eligibility"),
            #         benefits=s.get("benefits"),
            #         application_process=s.get("application_process"),
            #         website_url=HttpUrl(s["website_url"]) if s.get("website_url") else None,
            #         contact_info=s.get("contact_info")
            #     ) for s in data.get("schemes", [])
            # ]
            # return GovernmentSchemeResponse(schemes=schemes)

            # For demo, using mock data temporarily
            return self._get_mock_government_schemes(language_code, region, crop_type)

        except KrishiMitraException:
            raise
        except Exception as e:
            logger.error(f"Error processing government schemes for lang '{language_code}', region '{region}', crop '{crop_type}': {e}")
            raise KrishiMitraException(f"Failed to fetch government schemes: {e}", error_code=500)

    def _get_mock_government_schemes(self, language_code: str, region: Optional[str] = None, crop_type: Optional[str] = None) -> GovernmentSchemeResponse:
        """
        Generates mock government scheme data.
        """
        logger.info(f"Returning mock government schemes for lang {language_code}, region {region}, crop {crop_type}.")
        schemes: List[GovernmentScheme] = []

        # Example schemes (simplified for different languages)
        if language_code == "hi":
            schemes.extend([
                GovernmentScheme(
                    name="प्रधान मंत्री फसल बीमा योजना",
                    description="यह योजना किसानों को प्राकृतिक आपदाओं, कीटों और बीमारियों के कारण होने वाली फसल हानि से सुरक्षा प्रदान करती है।",
                    eligibility="सभी किसान जो अधिसूचित फसलों की खेती करते हैं।",
                    benefits="फसल हानि के खिलाफ वित्तीय सहायता।",
                    application_process="नजदीकी बैंक या कृषि विभाग कार्यालय के माध्यम से।",
                    website_url="https://pmfby.gov.in/pmfby/",
                    contact_info="टोल-फ्री नंबर: 1800-103-0061"
                ),
                GovernmentScheme(
                    name="किसान क्रेडिट कार्ड (केसीसी)",
                    description="किसानों को कृषि और संबद्ध गतिविधियों की वित्तीय आवश्यकताओं को पूरा करने के लिए समय पर और पर्याप्त ऋण सहायता प्रदान करना।",
                    eligibility="व्यक्तिगत किसान, संयुक्त उधारकर्ता, स्वयं सहायता समूह (एसएचजी), संयुक्त देयता समूह (जेएलजी)।",
                    benefits="आसान और सस्ती दर पर ऋण।",
                    application_process="बैंकों के माध्यम से आवेदन करें।",
                    website_url=None, # Mock example
                    contact_info=None
                )
            ])
        elif language_code == "te":
            schemes.extend([
                GovernmentScheme(
                    name="ప్రధానమంత్రి ఫసల్ బీమా యోజన",
                    description="సహజ విపత్తులు, తెగుళ్లు మరియు వ్యాధుల కారణంగా పంట నష్టాల నుండి రైతులకు రక్షణ కల్పిస్తుంది.",
                    eligibility="నోటిఫై చేయబడిన పంటలను సాగు చేసే రైతులందరూ.",
                    benefits="పంట నష్టానికి ఆర్థిక సహాయం.",
                    application_process="సమీప బ్యాంక్ లేదా వ్యవసాయ శాఖ కార్యాలయం ద్వారా.",
                    website_url="https://pmfby.gov.in/pmfby/",
                    contact_info="టోల్-ఫ్రీ నంబర్: 1800-103-0061"
                ),
                GovernmentScheme(
                    name="రైతు బంధు పథకం (తెలంగాణ)",
                    description="ఈ పథకం పెట్టుబడి మద్దతుగా ప్రతి సీజన్‌లో రైతులకు ఆర్థిక సహాయం అందిస్తుంది.",
                    eligibility="తెలంగాణలోని భూమి యజమాని రైతులు.",
                    benefits="ఖరీఫ్ మరియు రబీ సీజన్‌లకు ప్రతి ఎకరాకు ₹5,000.",
                    application_process="ఆన్‌లైన్ లేదా గ్రామ సేవ కేంద్రాల ద్వారా.",
                    website_url="https://rythubandhu.telangana.gov.in/",
                    contact_info=None
                )
            ])
        elif language_code == "ta":
             schemes.extend([
                GovernmentScheme(
                    name="பிரதம மந்திரி பசல் பீமா யோஜனா",
                    description="இயற்கை சீற்றங்கள், பூச்சிகள் மற்றும் நோய்களால் ஏற்படும் பயிர் இழப்பிலிருந்து விவசாயிகளுக்கு பாதுகாப்பு அளிக்கிறது.",
                    eligibility="அறிவிக்கப்பட்ட பயிர்களை பயிரிடும் அனைத்து விவசாயிகளும்.",
                    benefits="பயிர் இழப்பிற்கு நிதி உதவி.",
                    application_process="அருகிலுள்ள வங்கி அல்லது வேளாண்மை துறை அலுவலகம் மூலம்.",
                    website_url="https://pmfby.gov.in/pmfby/",
                    contact_info="கட்டணமில்லா எண்: 1800-103-0061"
                ),
                GovernmentScheme(
                    name="உழவர் பாதுகாப்பு திட்டம் (தமிழ்நாடு)",
                    description="தமிழ்நாடு விவசாயிகளுக்கு பல்வேறு சமூக பாதுகாப்பு நன்மைகளை வழங்குகிறது.",
                    eligibility="தமிழ்நாட்டில் உள்ள விவசாயிகள்.",
                    benefits="விபத்து, மரணம், மற்றும் கல்வி உதவித்தொகை போன்ற பலன்கள்.",
                    application_process="மாவட்ட ஆட்சியர் அலுவலகங்கள் மூலம்.",
                    website_url=None, # Mock example
                    contact_info=None
                )
            ])
        else: # Default to English
            schemes.extend([
                GovernmentScheme(
                    name="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                    description="Provides insurance coverage to farmers for crop losses due to natural calamities, pests, and diseases.",
                    eligibility="All farmers cultivating notified crops.",
                    benefits="Financial assistance against crop loss.",
                    application_process="Through nearest bank or agriculture department office.",
                    website_url="https://pmfby.gov.in/pmfby/",
                    contact_info="Toll-free number: 1800-103-0061"
                ),
                GovernmentScheme(
                    name="Kisan Credit Card (KCC)",
                    description="Aims to provide timely and adequate credit support to farmers for their agricultural and allied activities.",
                    eligibility="Individual farmers, joint borrowers, Self Help Groups (SHGs), Joint Liability Groups (JLGs).",
                    benefits="Easy and affordable credit.",
                    application_process="Apply through banks.",
                    website_url=None, # Mock example
                    contact_info=None
                )
            ])

        # Simple filtering logic for mock data
        filtered_schemes = []
        for scheme in schemes:
            match = True
            if region and region.lower() not in scheme.description.lower() and region.lower() not in scheme.name.lower() \
                    and ("telangana" in scheme.name.lower() or "tamil nadu" in scheme.name.lower()): # Example specific region check
                 match = False
            if crop_type and crop_type.lower() not in scheme.description.lower() and crop_type.lower() not in scheme.name.lower():
                # This is a very basic mock filter. A real system would need structured tags.
                match = False
            if match:
                filtered_schemes.append(scheme)

        return GovernmentSchemeResponse(schemes=filtered_schemes)

    async def close(self):
        """Closes the HTTP client session."""
        await self.client.aclose()


from datetime import timedelta, time # Needed for mock data generation
if __name__ == "__main__":
    import asyncio

    async def demo_external_integrations():
        print("--- Demoing External Integrations Service ---")
        service = ExternalIntegrationsService()

        # Mock location for demo (e.g., Hyderabad)
        demo_lat, demo_lon = 17.3850, 78.4867

        try:
            # 1. Weather Data
            print("\nFetching weather data...")
            weather_forecast = await service.get_weather_data(demo_lat, demo_lon)
            print(f"Current weather in {demo_lat}, {demo_lon}:")
            print(f"  Temperature: {weather_forecast.current_condition.temperature}°C")
            print(f"  Description: {weather_forecast.current_condition.weather_description}")
            if weather_forecast.alerts:
                print("  Alerts:")
                for alert in weather_forecast.alerts:
                    print(f"    - {alert.event} ({alert.severity}): {alert.description}")
            print("\nForecast for next few days:")
            for fc in weather_forecast.forecast_conditions:
                print(f"  Date: {fc.timestamp.date()}, Temp: {fc.temperature}°C, Desc: {fc.weather_description}, POP: {fc.pop}")

            # 2. Market Prices
            print("\nFetching market prices for Tomato in Hyderabad...")
            market_prices = await service.get_market_prices("Tomato", "Hyderabad")
            print(f"Current Tomato price in Hyderabad: Avg {market_prices.current_price.avg_price} {market_prices.current_price.unit} on {market_prices.current_price.date}")
            print("Historical prices (last 5 days):")
            for dp in market_prices.historical_prices:
                print(f"  Date: {dp.date}, Min: {dp.min_price}, Max: {dp.max_price}, Avg: {dp.avg_price} {dp.unit}")

            print("\nFetching market prices for Wheat in Delhi...")
            market_prices_wheat = await service.get_market_prices("Wheat", "Delhi")
            print(f"Current Wheat price in Delhi: Avg {market_prices_wheat.current_price.avg_price} {market_prices_wheat.current_price.unit} on {market_prices_wheat.current_price.date}")


            # 3. Government Schemes
            print("\nFetching government schemes (English)...")
            schemes_en = await service.get_government_schemes(language_code="en")
            for scheme in schemes_en.schemes:
                print(f"  - {scheme.name}: {scheme.description[:100]}...")

            print("\nFetching government schemes (Hindi) for crop 'wheat'...")
            schemes_hi = await service.get_government_schemes(language_code="hi", crop_type="wheat")
            for scheme in schemes_hi.schemes:
                print(f"  - {scheme.name}: {scheme.description[:100]}...")

            print("\nFetching government schemes (Telugu) for region 'Telangana'...")
            schemes_te = await service.get_government_schemes(language_code="te", region="Telangana")
            for scheme in schemes_te.schemes:
                print(f"  - {scheme.name}: {scheme.description[:100]}...")

        except KrishiMitraException as e:
            print(f"KrishiMitra Error during demo: {e.detail}")
        except Exception as e:
            print(f"An unexpected error occurred during demo: {e}")
        finally:
            await service.close()
            print("\n--- Demo Complete ---")

    asyncio.run(demo_external_integrations())