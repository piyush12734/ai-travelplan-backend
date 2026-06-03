from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import TypedDict
import os
import json

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

class TravelState(TypedDict):
    city: str
    days: int
    research: str
    activities: str
    final_plan: dict

def research_agent(state: TravelState) -> TravelState:
    print(f"🔍 Research Agent: Researching {state['city']}...")
    prompt = f"""
    You are a travel researcher. 
    Research the city {state['city']} and provide:
    - Brief overview of the city
    - Best time to visit
    - Local culture and customs
    - Currency and language
    Keep it concise, 4-5 sentences max.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["research"] = response.content
    print(f"✅ Research done!")
    return state

def activities_agent(state: TravelState) -> TravelState:
    print(f"🎯 Activities Agent: Finding activities for {state['city']}...")
    prompt = f"""
    You are a travel activities expert.
    Based on this research about {state['city']}:
    {state['research']}
    Suggest the top 5 activities for a {state['days']} day trip.
    Also suggest 2 local foods to try.
    Keep each activity to one sentence.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["activities"] = response.content
    print(f"✅ Activities done!")
    return state

def recommendation_agent(state: TravelState) -> TravelState:
    print(f"💡 Recommendation Agent: Creating final plan for {state['city']}...")
    prompt = f"""
    You are a travel recommendation expert.
    Create a final structured travel plan using this information:
    City: {state['city']}
    Days: {state['days']}
    Research: {state['research']}
    Activities: {state['activities']}
    Return ONLY a JSON object, no extra text, no backticks:
    {{
        "name": "{state['city']}",
        "days": {state['days']},
        "summary": "2 sentence city overview",
        "activities": ["activity 1", "activity 2", "activity 3", "activity 4", "activity 5"],
        "local_food": ["food 1", "food 2"],
        "tip": "most important travel tip",
        "best_season": "best time to visit"
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    state["final_plan"] = json.loads(text)
    print(f"✅ Final plan ready!")
    return state

def build_travel_graph():
    graph = StateGraph(TravelState)
    graph.add_node("research", research_agent)
    graph.add_node("activities", activities_agent)
    graph.add_node("recommendation", recommendation_agent)
    graph.set_entry_point("research")
    graph.add_edge("research", "activities")
    graph.add_edge("activities", "recommendation")
    graph.add_edge("recommendation", END)
    return graph.compile()

if __name__ == "__main__":
    travel_graph = build_travel_graph()
    result = travel_graph.invoke({
        "city": "Tokyo",
        "days": 5,
        "research": "",
        "activities": "",
        "final_plan": {}
    })
    print("\n========== FINAL PLAN ==========")
    print(json.dumps(result["final_plan"], indent=2))