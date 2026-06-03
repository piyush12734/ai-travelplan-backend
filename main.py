from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import build_travel_graph
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Allows Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

travel_graph = build_travel_graph()

class TripRequest(BaseModel):
    city: str
    days: int

@app.get("/")
def home():
    return {"message": "Welcome to AI TravelPlan API — Multi Agent System!"}

@app.get("/plan")
def plan_trip_get(city: str, days: int):
    try:
        result = travel_graph.invoke({
            "city": city,
            "days": days,
            "research": "",
            "activities": "",
            "final_plan": {}
        })
        return result["final_plan"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan")
def plan_trip_post(request: TripRequest):
    try:
        result = travel_graph.invoke({
            "city": request.city,
            "days": request.days,
            "research": "",
            "activities": "",
            "final_plan": {}
        })
        return result["final_plan"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)