from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from services.real_data import imd_weather, mandi_prices, government_schemes, DataUnavailable

app = FastAPI(title="KrishiMitra AI", version="2.0.0", description="Voice-first agriculture assistant with real-data-first integrations.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
async def health():
    return {"status":"ok", "mode":"real-data-first", "synthetic_agriculture_data":False}

@app.get("/api/weather")
async def weather(lat: float = Query(...), lon: float = Query(...)):
    try: return await imd_weather(lat, lon)
    except DataUnavailable as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(502, f"IMD source unavailable: {e}")

@app.get("/api/market")
async def market(commodity: str, state: str|None=None, district: str|None=None):
    try: return await mandi_prices(commodity, state, district)
    except DataUnavailable as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(502, f"Government market source unavailable: {e}")

@app.get("/api/schemes")
async def schemes(state: str|None=None):
    try: return await government_schemes(state)
    except DataUnavailable as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(502, f"Government scheme source unavailable: {e}")

app.mount("/static", StaticFiles(directory="frontend"), name="static")
@app.get("/")
async def home(): return FileResponse("frontend/index.html")
