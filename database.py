import os
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey,
    Float, Boolean, UniqueConstraint
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship

from config import Config
from schemas import (
    DiagnosisResult, Recommendation, PreventionMeasure,
    CropCalendarEvent, FarmLocation, MarketDataEntry, GovtSchemeDetail
)

# Define a base class for declarative models
Base = declarative_base()

class User(Base):
    """
    SQLAlchemy model for storing user information.
    """
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    phone_number: str = Column(String, unique=True, index=True, nullable=False)
    preferred_language: str = Column(String, default="en", nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    farm_locations = relationship("FarmLocationORM", back_populates="user")
    crop_calendar_events = relationship("CropCalendarEventORM", back_populates="user")
    ai_interactions = relationship("AIInteractionLog", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, phone_number='{self.phone_number}')>"


class FarmLocationORM(Base):
    """
    SQLAlchemy model for storing farm location details associated with a user.
    """
    __tablename__ = "farm_locations"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    name: str = Column(String, nullable=False)
    latitude: float = Column(Float, nullable=False)
    longitude: float = Column(Float, nullable=False)
    soil_type: Optional[str] = Column(String)
    soil_ph: Optional[float] = Column(Float)
    water_source: Optional[str] = Column(String) # e.g., borewell, canal, rain-fed
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="farm_locations")

    __table_args__ = (UniqueConstraint('user_id', 'name', name='_user_location_uc'),)

    def to_pydantic(self) -> FarmLocation:
        return FarmLocation(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            soil_type=self.soil_type,
            soil_ph=self.soil_ph,
            water_source=self.water_source
        )

    def __repr__(self):
        return f"<FarmLocation(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class CropCalendarEventORM(Base):
    """
    SQLAlchemy model for personalized crop calendar events.
    """
    __tablename__ = "crop_calendar_events"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    farm_location_id: Optional[int] = Column(Integer, ForeignKey("farm_locations.id"))
    crop_name: str = Column(String, nullable=False)
    event_type: str = Column(String, nullable=False)  # e.g., 'Sowing', 'Harvest', 'Fertilizer Application', 'Pest Spray'
    event_date: datetime = Column(DateTime, nullable=False)
    description: Optional[str] = Column(Text)
    is_completed: bool = Column(Boolean, default=False, nullable=False)
    reminder_set: bool = Column(Boolean, default=False, nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="crop_calendar_events")
    farm_location = relationship("FarmLocationORM")

    def to_pydantic(self) -> CropCalendarEvent:
        return CropCalendarEvent(
            id=self.id,
            user_id=self.user_id,
            farm_location_id=self.farm_location_id,
            crop_name=self.crop_name,
            event_type=self.event_type,
            event_date=self.event_date,
            description=self.description,
            is_completed=self.is_completed,
            reminder_set=self.reminder_set
        )

    def __repr__(self):
        return (f"<CropCalendarEvent(id={self.id}, crop='{self.crop_name}', "
                f"type='{self.event_type}', date='{self.event_date.strftime('%Y-%m-%d')}')>")


class CropDiagnosisLog(Base):
    """
    SQLAlchemy model for logging Crop Doctor diagnosis results.
    """
    __tablename__ = "crop_diagnosis_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_url: str = Column(String, nullable=False)
    detected_crop: Optional[str] = Column(String)
    diagnosis: str = Column(String, nullable=False)
    confidence: float = Column(Float, nullable=False)
    recommendations_json: dict = Column(Text, nullable=False)  # Stored as JSON string
    prevention_json: dict = Column(Text, nullable=False)      # Stored as JSON string
    diagnosis_date: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")

    def to_pydantic(self) -> DiagnosisResult:
        return DiagnosisResult(
            diagnosis=self.diagnosis,
            confidence=self.confidence,
            recommendations=[Recommendation(**r) for r in self.recommendations_json] if self.recommendations_json else [],
            prevention=[PreventionMeasure(**p) for p in self.prevention_json] if self.prevention_json else [],
            image_url=self.image_url,
            detected_crop=self.detected_crop
        )

    def __repr__(self):
        return (f"<CropDiagnosisLog(id={self.id}, user_id={self.user_id}, "
                f"diagnosis='{self.diagnosis}', confidence={self.confidence})>")


class AIInteractionLog(Base):
    """
    SQLAlchemy model for logging all AI interactions (voice commands, NLU, responses).
    """
    __tablename__ = "ai_interaction_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: Optional[int] = Column(Integer, ForeignKey("users.id"))
    input_text: str = Column(Text, nullable=False)
    input_language: str = Column(String(10), nullable=False)
    translated_text: Optional[str] = Column(Text) # Often English for NLU
    detected_intent: Optional[str] = Column(String)
    extracted_entities: Optional[dict] = Column(Text) # Stored as JSON string
    response_text: str = Column(Text, nullable=False)
    response_language: str = Column(String(10), nullable=False)
    interaction_timestamp: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_successful: bool = Column(Boolean, default=True, nullable=False)
    error_message: Optional[str] = Column(Text)
    related_diagnosis_id: Optional[int] = Column(Integer, ForeignKey("crop_diagnosis_logs.id"))

    user = relationship("User", back_populates="ai_interactions")
    related_diagnosis = relationship("CropDiagnosisLog")

    def __repr__(self):
        return (f"<AIInteractionLog(id={self.id}, user_id={self.user_id}, "
                f"intent='{self.detected_intent}', timestamp='{self.interaction_timestamp}')>")


class MarketDataORM(Base):
    """
    SQLAlchemy model for storing fetched market data.
    """
    __tablename__ = "market_data"

    id: int = Column(Integer, primary_key=True, index=True)
    crop_name: str = Column(String, index=True, nullable=False)
    market_name: str = Column(String, index=True, nullable=False)
    price: float = Column(Float, nullable=False)
    unit: str = Column(String, default="INR/Quintal", nullable=False)
    date_recorded: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint('crop_name', 'market_name', 'date_recorded', name='_market_data_uc'),)

    def to_pydantic(self) -> MarketDataEntry:
        return MarketDataEntry(
            crop_name=self.crop_name,
            market_name=self.market_name,
            price=self.price,
            unit=self.unit,
            date_recorded=self.date_recorded
        )

    def __repr__(self):
        return (f"<MarketData(id={self.id}, crop='{self.crop_name}', market='{self.market_name}', "
                f"price={self.price}, date='{self.date_recorded.strftime('%Y-%m-%d')}')>")


class GovtSchemeORM(Base):
    """
    SQLAlchemy model for storing government schemes.
    """
    __tablename__ = "govt_schemes"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, unique=True, nullable=False)
    description: str = Column(Text, nullable=False)
    eligibility: Optional[str] = Column(Text)
    benefits: Optional[str] = Column(Text)
    how_to_apply: Optional[str] = Column(Text)
    link: Optional[str] = Column(String)
    last_updated: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_pydantic(self) -> GovtSchemeDetail:
        return GovtSchemeDetail(
            name=self.name,
            description=self.description,
            eligibility=self.eligibility,
            benefits=self.benefits,
            how_to_apply=self.how_to_apply,
            link=self.link
        )

    def __repr__(self):
        return f"<GovtScheme(id={self.id}, name='{self.name}')>"


# Configure the asynchronous database engine
engine = create_async_engine(Config.DATABASE_URL, echo=Config.DEBUG_MODE)

# Configure an asynchronous sessionmaker
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False # This is important for async contexts
)

async def init_db() -> None:
    """
    Initializes the database by creating all tables defined in Base.
    This function should be called once on application startup.
    """
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Use with caution for development
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created or already exist.")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injector for FastAPI to provide an asynchronous database session.
    Yields a session which is then automatically closed.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

if __name__ == "__main__":
    import asyncio

    async def main():
        """
        Main function to run database initialization when database.py is executed directly.
        """
        print("Running database initialization...")
        await init_db()
        print("Database initialization complete.")

        # Example: Add a user and a farm location
        async with AsyncSessionLocal() as session:
            try:
                # Add a user
                user = User(phone_number="9876543210", preferred_language="te")
                session.add(user)
                await session.commit()
                await session.refresh(user)
                print(f"Added user: {user}")

                # Add a farm location for the user
                farm_location = FarmLocationORM(
                    user_id=user.id,
                    name="My Test Farm",
                    latitude=17.3850,
                    longitude=78.4867,
                    soil_type="Black",
                    soil_ph=7.2,
                    water_source="Borewell"
                )
                session.add(farm_location)
                await session.commit()
                await session.refresh(farm_location)
                print(f"Added farm location: {farm_location}")

                # Add a crop calendar event
                event = CropCalendarEventORM(
                    user_id=user.id,
                    farm_location_id=farm_location.id,
                    crop_name="Rice",
                    event_type="Sowing",
                    event_date=datetime(2024, 6, 15),
                    description="Sowing paddy variety BPT 5204",
                    reminder_set=True
                )
                session.add(event)
                await session.commit()
                await session.refresh(event)
                print(f"Added crop calendar event: {event}")

                # Add a dummy market data entry
                market_data = MarketDataORM(
                    crop_name="Rice",
                    market_name="Hyderabad Market",
                    price=2500.00,
                    unit="INR/Quintal",
                    date_recorded=datetime.now()
                )
                session.add(market_data)
                await session.commit()
                await session.refresh(market_data)
                print(f"Added market data: {market_data}")

            except Exception as e:
                print(f"Error during example data insertion: {e}")
                await session.rollback()

    asyncio.run(main())