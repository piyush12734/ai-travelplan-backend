from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import TypedDict
from datetime import date
import calendar
import os
import json
import requests

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

MONTH_NAME_TO_NUM = {name: num for num, name in enumerate(calendar.month_name) if name}


class TravelState(TypedDict):
    city: str
    days: int
    month: str
    budget: str
    research: str
    activities: str
    weather: dict
    final_plan: dict


def research_agent(state: TravelState) -> TravelState:
    print(f"🔍 Research Agent: Researching {state['city']}...")
    month_context = f" The traveler is planning to visit in {state['month']}." if state.get("month") else ""
    prompt = f"""
    You are a travel researcher.
    Research the city {state['city']} and provide:
    - Brief overview of the city
    - Best time to visit
    - Local culture and customs
    - Currency and language
    {month_context}

    Keep it concise, 4-5 sentences max.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["research"] = response.content
    print(f"✅ Research done!")
    return state


def activities_agent(state: TravelState) -> TravelState:
    print(f"🎯 Activities Agent: Finding activities for {state['city']}...")
    month_context = f" They will be visiting in {state['month']}, so favor activities well-suited to that time of year." if state.get("month") else ""
    prompt = f"""
    You are a travel activities expert.
    Based on this research about {state['city']}:
    {state['research']}

    Suggest the top 5 activities for a {state['days']} day trip.{month_context}
    Also suggest 2 local foods to try.
    Keep each activity to one sentence.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["activities"] = response.content
    print(f"✅ Activities done!")
    return state


def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Unknown")


def _next_occurrence(month_name: str) -> date:
    """Returns the date of the 1st of the requested month — this year if it
    hasn't started yet or is the current month, otherwise next year."""
    today = date.today()
    month_num = MONTH_NAME_TO_NUM.get(month_name)
    if not month_num:
        return today
    if month_num == today.month:
        return today
    year = today.year if month_num > today.month else today.year + 1
    return date(year, month_num, 1)


def _live_forecast(city: str, days: int, skip_days: int) -> dict:
    geo_res = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    geo_data = geo_res.json()
    if not geo_data.get("results"):
        return {"error": f"Could not find location data for {city}"}

    location = geo_data["results"][0]
    lat, lon = location["latitude"], location["longitude"]
    total_needed = min(skip_days + days, 16)

    forecast_res = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": total_needed,
        },
        timeout=10,
    )
    forecast_data = forecast_res.json()
    daily = forecast_data.get("daily", {})

    forecast_list = []
    times = daily.get("time", [])[skip_days:]
    for offset, date_str in enumerate(times):
        i = skip_days + offset
        forecast_list.append({
            "date": date_str,
            "high_c": daily["temperature_2m_max"][i],
            "low_c": daily["temperature_2m_min"][i],
            "precipitation_chance": daily.get("precipitation_probability_max", [None])[i],
            "condition": _weather_code_to_text(daily["weathercode"][i]),
        })

    return {
        "location": location.get("name", city),
        "type": "live_forecast",
        "note": "Live short-range forecast.",
        "forecast": forecast_list,
    }


def _typical_climate(city: str, month: str, days: int, research: str) -> dict:
    """Falls back to an LLM-estimated typical climate when the trip is more
    than ~16 days out, since real forecasts aren't available that far ahead."""
    start = _next_occurrence(month)
    prompt = f"""
    You are a climate data assistant. Based on typical historical weather
    patterns (not a live forecast), estimate what the weather is usually
    like in {city} during {month}.

    Context: {research}

    Return ONLY a JSON array (no extra text, no backticks) with exactly
    {min(days, 7)} entries, one per day starting {start.isoformat()}, in
    this shape:
    [
      {{"date": "YYYY-MM-DD", "high_c": 28, "low_c": 19, "precipitation_chance": 20, "condition": "Partly cloudy"}}
    ]
    Use realistic, varied but consistent values for the season. Dates must
    be sequential starting from {start.isoformat()}.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip().replace("```json", "").replace("```", "").strip()
        forecast_list = json.loads(text)
    except Exception as e:
        print(f"⚠️ Typical climate generation failed: {e}")
        return {"error": "Could not estimate typical weather for that month."}

    return {
        "location": city,
        "type": "typical_climate",
        "note": f"Typical seasonal weather for {month}, based on historical patterns — a live forecast isn't available this far ahead.",
        "forecast": forecast_list,
    }


def weather_agent(state: TravelState) -> TravelState:
    city = state["city"]
    month = state.get("month", "")
    days = state.get("days", 5) or 5
    print(f"🌤️ Weather Agent: Checking weather for {city} in {month or 'the near future'}...")
    try:
        if month:
            target_start = _next_occurrence(month)
            delta_days = (target_start - date.today()).days
        else:
            delta_days = 0

        if 0 <= delta_days <= 13:
            state["weather"] = _live_forecast(city, days, skip_days=max(delta_days, 0))
        else:
            state["weather"] = _typical_climate(city, month, days, state.get("research", ""))
        print(f"✅ Weather done!")
    except Exception as e:
        print(f"⚠️ Weather fetch failed: {e}")
        state["weather"] = {"error": "Weather information unavailable right now."}
    return state


def recommendation_agent(state: TravelState) -> TravelState:
    print(f"💡 Recommendation Agent: Creating final plan for {state['city']}...")
    budget_context = ""
    if state.get("budget"):
        budget_context = f"""
    The traveler's total budget for this trip is ₹{state['budget']} for {state['days']} days.
    Tailor daily_estimate_usd and total_estimate_usd (these fields hold INR amounts,
    e.g. "2500" or "2000-3000") and the breakdown so they realistically fit close to
    this budget. In the note, mention plainly whether this budget is comfortable, tight,
    or unrealistic for {state['city']}, and how to make it work if it's tight.
    """
    month_context = f" The trip is planned for {state['month']}." if state.get("month") else ""

    prompt = f"""
    You are a travel recommendation expert.
    Create a final structured travel plan using this information:

    City: {state['city']}
    Days: {state['days']}{month_context}
    Research: {state['research']}
    Activities: {state['activities']}
    {budget_context}

    Return ONLY a JSON object, no extra text, no backticks:
    {{
      "name": "{state['city']}",
      "days": {state['days']},
      "summary": "2 sentence city overview",
      "activities": ["activity 1", "activity 2", "activity 3", "activity 4", "activity 5"],
      "local_food": ["food 1", "food 2"],
      "tip": "most important travel tip",
      "best_season": "best time to visit",
      "budget": {{
        "daily_estimate_usd": "INR amount per day, e.g. 2500 or 2000-3000",
        "total_estimate_usd": "INR total for {state['days']} days, e.g. 15000-20000",
        "breakdown": {{
          "accommodation": "INR estimate, e.g. ₹1200-2000/night",
          "food": "INR estimate, e.g. ₹600-1000/day",
          "transport": "INR estimate, e.g. ₹300-600/day",
          "activities": "INR estimate, e.g. ₹400-800/day"
        }},
        "note": "one or two sentence tip on money, referencing the traveler's stated budget if one was given"
      }}
    }}
    """

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        # Some models still wrap the JSON with stray text — try to isolate
        # the outermost {...} block before parsing.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            state["final_plan"] = json.loads(text)
            print(f"✅ Final plan ready!")
            return state
        except json.JSONDecodeError as e:
            last_error = e
            print(f"⚠️ Recommendation JSON parse failed (attempt {attempt + 1}): {e}")
            print(f"Raw response was: {text[:500]}")

    # All retries failed — fall back to a minimal valid plan instead of crashing
    print(f"❌ Giving up on JSON parsing after {max_retries + 1} attempts: {last_error}")
    state["final_plan"] = {
        "name": state["city"],
        "days": state["days"],
        "summary": "We had trouble generating a detailed plan for this trip. Please try again.",
        "activities": [],
        "local_food": [],
        "tip": "",
        "best_season": "",
        "budget": {},
    }
    return state


def build_travel_graph():
    graph = StateGraph(TravelState)
    graph.add_node("research", research_agent)
    graph.add_node("activities", activities_agent)
    graph.add_node("weather", weather_agent)
    graph.add_node("recommendation", recommendation_agent)

    graph.set_entry_point("research")
    graph.add_edge("research", "activities")
    graph.add_edge("activities", "weather")
    graph.add_edge("weather", "recommendation")
    graph.add_edge("recommendation", END)

    return graph.compile()


if __name__ == "__main__":
    travel_graph = build_travel_graph()
    result = travel_graph.invoke({
        "city": "Tokyo",
        "days": 5,
        "month": "December",
        "budget": "40000",
        "research": "",
        "activities": "",
        "weather": {},
        "final_plan": {}
    })
    result["final_plan"]["weather"] = result["weather"]
    print("\n========== FINAL PLAN ==========")
    print(json.dumps(result["final_plan"], indent=2))
