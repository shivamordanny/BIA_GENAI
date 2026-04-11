# Session 9: Mini-Project Sprint
## Build Your First Multi-Tool Travel Planner Agent

---

## 1. Quick Recap: From Session 8 to Session 9

| Aspect | Session 8 | Session 9 |
|--------|-----------|----------|
| **Focus** | Agent self-reflection, memory, planning | Building practical multi-tool agents |
| **Agent Type** | Single agent with self-correction | Single agent → Multi-agent (Manager/Worker) |
| **Tools** | Thought/Action/Observation loop | Multiple tools, real tool invocation |
| **Complexity** | Moderate | Progressive (simple → advanced) |
| **Output** | Plans, self-reflection | Actual tool calls, results |

### Why This Transition Matters
Session 8 taught you how agents think and reason. Session 9 teaches you how agents *act* — by selecting and using the right tools to solve real-world problems. By the end of this session, you'll have built a working travel-planning system that makes actual API calls (or uses realistic mock data).

---

## 2. Project Overview: The Travel Planner

### What We're Building
A **Travel Planning Assistant** that helps users plan trips by:
- Checking weather at their destination
- Finding flights
- Searching for hotels
- Converting currencies

### Architecture Overview
```
┌─────────────────────────────────────────┐
│   User Input: "Plan a trip to Bali"     │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼──────────┐
        │  Manager Agent  │
        │ (Router/Super)  │
        └──────┬──────────┘
               │
     ┌─────────┼─────────┬────────────┐
     │         │         │            │
  ┌──▼──┐  ┌──▼──┐  ┌───▼──┐  ┌─────▼────┐
  │Weather│ │Flight│ │Hotel │  │ Currency │
  │Worker │ │Worker│ │Worker│  │ Worker   │
  └──────┘  └──────┘  └──────┘  └──────────┘
     │         │         │            │
     └─────────┼─────────┼────────────┘
               │
        ┌──────▼──────────┐
        │  Consolidated   │
        │    Response     │
        └─────────────────┘
```

### What You'll Learn
- How to design tools for LLM agents
- Building single-agent systems with multiple tools
- Splitting into specialized multi-agent systems
- Calling real APIs (with mock data fallback)

---

## 3. Tool Selection & Design

### What Are Tools in LangChain?

Tools are functions that an AI agent can **call** to gather information or perform actions. Think of them like superpowers for your agent — without tools, an agent can only talk. With tools, it can **act**.

### The @tool Decorator (Simplest Approach)

In LangChain, the simplest way to create a tool is with the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def my_tool(input_param: str) -> str:
    """Clear description of what this tool does."""
    # Implementation here
    return "result"
```

**Key points:**
- Use `@tool` decorator (not `BaseTool` subclass — too complex for beginners)
- Add a docstring (the agent reads this to understand when to use the tool)
- Keep parameters simple: `str`, `int`, `float` (no Pydantic models)
- Return a string or simple dict

### Tool Naming Best Practices

| Good Tool Name | Bad Tool Name | Why? |
|---|---|---|
| `get_weather` | `weather` | Verbs are clearer |
| `search_flights` | `find_stuff` | Specific, not vague |
| `convert_currency` | `currency` | Action is obvious |
| `search_hotels` | `hotel_tool` | Describes what it does |

### Designing Tool Inputs

**✓ Good:**
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Simple string input
```

**✗ Avoid:**
```python
@tool
def get_weather(request: WeatherRequest) -> dict:
    """Get weather."""
    # Complex Pydantic model — agent struggles with this
```

**Why?** LLM agents understand simple inputs (strings, numbers). Complex schemas confuse them. Keep it simple!

---

## 4. Building Mock Tools

Start with mock tools — no API keys, no rate limits, fast iteration. Once the system works, swap in real APIs.

### Tool 1: Get Weather

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    Returns temperature, condition, and wind info.
    """
    # Mock weather data
    weather_data = {
        "Bali": "28°C, Sunny, Tropical Breeze",
        "Tokyo": "15°C, Clear Sky",
        "Paris": "12°C, Partly Cloudy",
        "New York": "10°C, Overcast",
        "Bangkok": "32°C, Hot & Humid, Occasional Showers",
    }
    return weather_data.get(city, f"Weather for {city}: 20°C, Pleasant conditions")
```

### Tool 2: Search Flights

```python
@tool
def search_flights(origin: str, destination: str) -> str:
    """
    Search for flights between two cities.
    Returns available flight options with prices and duration.
    """
    # Mock flight data
    flights = {
        ("New York", "Bali"): "Flight 1: 23h 45m, $850 (2 stops) | Flight 2: 25h 20m, $720 (1 stop)",
        ("Tokyo", "Bali"): "Flight 1: 7h 15m, $320 | Flight 2: 9h 30m, $280",
        ("Paris", "Tokyo"): "Flight 1: 12h 5m, $650 | Flight 2: 14h 20m, $580",
    }
    key = (origin, destination)
    return flights.get(key, f"Flights from {origin} to {destination}: Standard routes available, avg $500-800")
```

### Tool 3: Search Hotels

```python
@tool
def search_hotels(city: str, num_nights: int) -> str:
    """
    Search for hotel options in a city.
    Returns hotel names, ratings, and price per night.
    """
    # Mock hotel data
    hotel_options = {
        "Bali": f"1. Sunset Beach Resort ⭐⭐⭐⭐⭐ - $150/night\n2. Jungle Retreat - $75/night\n3. Budget Hostel - $25/night\nTotal for {num_nights} nights: ${150*num_nights} - ${25*num_nights}",
        "Tokyo": f"1. Imperial Palace Hotel ⭐⭐⭐⭐⭐ - $280/night\n2. Comfort Inn - $120/night\n3. Youth Hostel - $40/night\nTotal for {num_nights} nights: ${280*num_nights} - ${40*num_nights}",
        "Paris": f"1. Le Marais Suite ⭐⭐⭐⭐⭐ - $220/night\n2. Budget Hotel - $90/night\nTotal for {num_nights} nights: ${220*num_nights} - ${90*num_nights}",
    }
    return hotel_options.get(city, f"Hotels in {city}: Options from $50-300/night available")
```

### Tool 4: Convert Currency

```python
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert amount from one currency to another.
    Returns converted amount with exchange rate.
    """
    # Mock exchange rates
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("USD", "IDR"): 16500,
        ("EUR", "USD"): 1.09,
    }

    rate = rates.get((from_currency, to_currency), 1.0)
    converted = amount * rate
    return f"{amount} {from_currency} = {converted:.2f} {to_currency} (Rate: {rate})"
```

### Why Mock Tools First?

1. **No API keys needed** — start immediately, no setup delays
2. **Fast iteration** — no network latency, test the agent logic
3. **Predictable output** — easier to debug
4. **Focus on architecture** — learn agent patterns without API complexity
5. **Real APIs later** — swap in one tool at a time

---

## 5. Your First Single Agent

### Creating a ReAct Agent

Use LangGraph's `create_react_agent()` — it handles the Thought → Action → Observation loop for you.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define your tools (get_weather, search_flights, search_hotels, convert_currency)
# ... [insert tools from Section 4 here] ...

# Create the agent with all tools
tools = [get_weather, search_flights, search_hotels, convert_currency]
agent = create_react_agent(model, tools)

# Test it
query = "I'm planning a trip to Bali. What's the weather like? Find me a flight from New York and a hotel for 5 nights."

# Use the agent
from langchain_core.messages import HumanMessage

result = agent.invoke({"messages": [HumanMessage(content=query)]})

# Print the response
print(result["messages"][-1].content)
```

### Understanding the Loop

The agent automatically:
1. **Reads** your query
2. **Thinks** about what tools to use
3. **Calls** the right tools in sequence
4. **Observes** the results
5. **Synthesizes** a response

You don't code the loop — `create_react_agent` does it for you!

---

## 6. The Manager/Worker Pattern

### Why Split Into Multiple Agents?

As systems grow:
- A single agent with too many tools becomes confused
- Specialists work better than generalists
- Different tasks need different expertise

### Simple Manager/Worker Design

```
Manager (Router):
├─ "What's the weather?" → Send to Weather Worker
├─ "Find flights" → Send to Flight Worker
├─ "Find hotels" → Send to Hotel Worker
└─ "Convert money" → Send to Currency Worker
```

### Implementation

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ====== WORKER 1: Weather Specialist ======
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "Bali": "28°C, Sunny",
        "Tokyo": "15°C, Clear",
        "Paris": "12°C, Cloudy",
    }
    return weather_data.get(city, f"Weather for {city}: 20°C, Pleasant")

weather_agent = create_react_agent(model, [get_weather])

# ====== WORKER 2: Flight Specialist ======
@tool
def search_flights(origin: str, destination: str) -> str:
    """Search for flights between two cities."""
    flights = {
        ("New York", "Bali"): "Flight: 24h, $850",
        ("Tokyo", "Bali"): "Flight: 7h, $320",
    }
    key = (origin, destination)
    return flights.get(key, f"Flights available from {origin} to {destination}")

flight_agent = create_react_agent(model, [search_flights])

# ====== WORKER 3: Hotel Specialist ======
@tool
def search_hotels(city: str) -> str:
    """Search for hotels in a city."""
    hotels = {
        "Bali": "Sunset Resort: $150/night | Budget Hotel: $50/night",
        "Tokyo": "Imperial Hotel: $280/night | Budget Inn: $100/night",
    }
    return hotels.get(city, f"Hotels available in {city}")

hotel_agent = create_react_agent(model, [search_hotels])

# ====== MANAGER: Routes queries to workers ======
def route_to_worker(query: str):
    """
    Simple router: decide which worker to call based on keywords.
    In production, you'd use an LLM to make this decision.
    """
    query_lower = query.lower()

    if "weather" in query_lower:
        print("→ Routing to Weather Worker")
        result = weather_agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content

    elif "flight" in query_lower or "fly" in query_lower:
        print("→ Routing to Flight Worker")
        result = flight_agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content

    elif "hotel" in query_lower or "accommodation" in query_lower:
        print("→ Routing to Hotel Worker")
        result = hotel_agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content

    else:
        return "I'm not sure how to help with that. Try asking about weather, flights, or hotels."

# ====== TEST THE SYSTEM ======
print("Q1: What's the weather in Bali?")
print(route_to_worker("What's the weather in Bali?"))
print()

print("Q2: Find me flights from New York to Bali")
print(route_to_worker("Find me flights from New York to Bali"))
print()

print("Q3: Where can I stay in Tokyo?")
print(route_to_worker("Where can I stay in Tokyo?"))
```

### Key Insights

- **Each worker is a small, focused agent** — better at its job
- **The manager is simple** — just routes based on keywords
- **Easier to maintain** — add workers independently
- **Scales better** — each worker can be upgraded separately

---

## 7. Connecting Real APIs (Optional)

### Open-Meteo Weather API

Open-Meteo provides **free weather data** — no API key needed!

```python
import requests
from langchain_core.tools import tool

@tool
def get_weather_real(city: str) -> str:
    """
    Get real weather data from Open-Meteo API.
    Works for any major city worldwide.
    """
    try:
        # Step 1: Get coordinates for the city (using geocoding)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_response = requests.get(geo_url, params=geo_params)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find weather data for {city}"

        # Step 2: Extract latitude and longitude
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]

        # Step 3: Get weather for those coordinates
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "temperature_unit": "celsius"
        }
        weather_response = requests.get(weather_url, params=weather_params)
        weather_data = weather_response.json()

        # Step 4: Format the response
        current = weather_data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]

        return f"Weather in {city}: {temp}°C, Humidity: {humidity}%"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"
```

### How to Swap In the Real Tool

Replace the mock tool with the real one:

```python
# OLD (mock)
tools = [get_weather, search_flights, search_hotels, convert_currency]

# NEW (real weather, mock others)
tools = [get_weather_real, search_flights, search_hotels, convert_currency]

agent = create_react_agent(model, tools)
```

That's it! The agent doesn't care if it's mock or real — it just calls the function.

---

## 8. Exercises

### Exercise 1: Add a "Local Restaurants" Tool

**Task:** Add a new tool to the single agent that finds local restaurants.

**Requirements:**
- Use `@tool` decorator
- Accept `city` as input
- Return a string with 3-4 mock restaurant options
- Each restaurant should have a name, cuisine type, and rating

**Hint:**
```python
@tool
def find_restaurants(city: str) -> str:
    """Find popular local restaurants in a city."""
    # Your code here
    pass

# Add to agent
tools = [..., find_restaurants]
agent = create_react_agent(model, tools)
```

**Test:**
```
"What restaurants are there in Bali?"
"I want to eat Italian in Paris"
```

---

### Exercise 2: Add an "Activities" Worker

**Task:** Create a new worker agent specialized in finding activities.

**Requirements:**
- Create an `activities_agent` with `search_activities()` tool
- Tool should accept `city` as input
- Return mock activity suggestions (museums, hiking, beaches, etc.)
- Add to the router (modify `route_to_worker()`)

**Hint:**
```python
@tool
def search_activities(city: str) -> str:
    """Find activities and attractions in a city."""
    # Your code here
    pass

activities_agent = create_react_agent(model, [search_activities])

# In route_to_worker, add:
if "activity" in query_lower or "things to do" in query_lower:
    print("→ Routing to Activities Worker")
    result = activities_agent.invoke(...)
    return result["messages"][-1].content
```

**Test:**
```
"What activities are there in Bali?"
"Things to do in Tokyo"
```

---

## 9. Quick Reference Card

### Key Imports

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import requests  # For real APIs
```

### Tool Creation Pattern

```python
@tool
def my_tool(input1: str, input2: int) -> str:
    """Clear description of what this tool does."""
    # Implementation
    return "result as string"
```

### Single Agent Pattern

```python
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [tool1, tool2, tool3]
agent = create_react_agent(model, tools)

result = agent.invoke({"messages": [HumanMessage(content="Your query")]})
print(result["messages"][-1].content)
```

### Multi-Agent Manager/Worker Pattern

```python
# Create specialized agents
weather_agent = create_react_agent(model, [get_weather])
flight_agent = create_react_agent(model, [search_flights])
hotel_agent = create_react_agent(model, [search_hotels])

# Router function
def route_to_worker(query: str):
    if "weather" in query.lower():
        return weather_agent.invoke({"messages": [HumanMessage(content=query)]})["messages"][-1].content
    elif "flight" in query.lower():
        return flight_agent.invoke({"messages": [HumanMessage(content=query)]})["messages"][-1].content
    elif "hotel" in query.lower():
        return hotel_agent.invoke({"messages": [HumanMessage(content=query)]})["messages"][-1].content
    else:
        return "Query not understood"

# Use it
response = route_to_worker("Find flights from NYC to Bali")
```

### Real vs. Mock Tools

| Aspect | Mock | Real |
|--------|------|------|
| **Setup** | Define dict/logic in code | Call external API |
| **Speed** | Instant | Network latency |
| **Reliability** | Always works | Depends on API |
| **Testing** | Easy | Harder (rate limits) |
| **Production** | Not suitable | Yes |

Start with mock. Swap in real when ready.

---

## Summary

You've now learned:

1. **Tool Design** — Create simple, focused tools with clear inputs/outputs
2. **Single Agent Systems** — Give one agent multiple tools for different tasks
3. **Multi-Agent Systems** — Split into specialized workers with a manager router
4. **Real APIs** — Replace mock tools with actual data sources (Open-Meteo example)
5. **Practical Patterns** — Manager/Worker, routing, tool chaining

Your travel planner can now:
- Check weather at destinations
- Search for flights and hotels
- Convert currencies
- Find restaurants and activities

**Next steps:** Enhance the system by adding more tools, improving the router with LLM-based routing, and connecting more real APIs.

Happy building!
