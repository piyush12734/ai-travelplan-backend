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
    month: str = ""
    budget: str = ""


@app.get("/")
def home():
    return {"message": "Welcome to AI TravelPlan API — Multi Agent System!"}


def run_trip(city: str, days: int, month: str = "", budget: str = ""):
    result = travel_graph.invoke({
        "city": city,
        "days": days,
        "month": month,
        "budget": budget,
        "research": "",
        "activities": "",
        "weather": {},
        "final_plan": {}
    })
    final_plan = result["final_plan"]
    final_plan["weather"] = result["weather"]
    return final_plan


@app.get("/plan")
def plan_trip_get(city: str, days: int, month: str = "", budget: str = ""):
    try:
        return run_trip(city, days, month, budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan")
def plan_trip_post(request: TripRequest):
    try:
        return run_trip(request.city, request.days, request.month, request.budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
