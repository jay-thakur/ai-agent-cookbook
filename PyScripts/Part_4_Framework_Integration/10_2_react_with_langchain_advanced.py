from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# --- Define Custom Tools ---

@tool
def get_distance(origin: str, destination: str) -> str:
    """Gets the distance between two locations using a placeholder."""
    print(f"Getting distance from {origin} to {destination}")
    # Replace with a real distance API (e.g., Google Maps Distance Matrix API) for production
    if "paris airport" in origin.lower() and "london" in destination.lower():
        return "The distance from Paris Airport to central London is approximately 344 kilometers. (Placeholder)"
    elif "paris" in origin.lower() and "rome" in destination.lower():
        return "The distance from Paris to Rome is approximately 1420 kilometers. (Placeholder)"
    else:
        return f"Distance between {origin} and {destination} not found. (Placeholder)" # Explicitly flagged placeholder

@tool
def get_weather(location: str) -> str:
    """Retrieves current weather data from OpenWeatherMap API (External Tool)."""
    print(f"Getting weather for: {location}")
    api_key = OPENWEATHERMAP_API_KEY
    if not api_key:
        return "Error: OPENWEATHERMAP_API_KEY not set in .env file."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "main" not in data: # More robust API response checking
            return f"Weather information not found for {location} (Invalid API Response)."
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The weather in {location} is {description} with a temperature of {temp:.1f}°C."
    except requests.exceptions.RequestException as e:
        if response.status_code == 401:
            return "Error: Invalid OpenWeatherMap API key." # Explicit API key error handling
        return f"Error: Could not retrieve weather data: {e}"
    except (KeyError, IndexError):
        return "Error: Invalid weather data received."


tools = [get_distance, get_weather]

# --- Define Advanced Prompt Engineering and Custom Chains ---

class MemoryPromptTemplate(StringPromptTemplate):
    memory_key: str

    def format(self, **kwargs) -> str:
        memory = kwargs.pop(self.memory_key)
        kwargs["memory"] = memory
        return self.template.format(**kwargs)

plan_prompt = MemoryPromptTemplate(
    input_variables=["goal", "memory"],
    memory_key="chat_history",
    template="""
You are a helpful travel assistant.

Previous conversation:
{memory}

Goal: {goal}

Create a plan to achieve the goal using available tools. If the goal is already answered in the memory, simply return the answer.

Plan:"""
)
plan_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), prompt=plan_prompt, output_key="plan")

execution_prompt = StringPromptTemplate(
    template="""
Execute the following plan, using the provided memory if needed.

Plan: {plan}
Memory: {memory}

Execution Result:
"""
)

travel_prompt = StringPromptTemplate(
    template="""
You are a helpful travel assistant. You have access to tools that can get distances between cities and weather information.

Previous conversation:
{memory}

Goal: {goal}

Create a plan to achieve the goal by using available tools. Explicitly mention the tool and its arguments in your plan. If the goal is already answered in the memory, simply return the answer from the memory.

Example Plan:
1. Use get_distance with origin="London" and destination="Paris".
2. Use get_weather with location="Paris".

Plan:
"""
)
execution_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), prompt=execution_prompt, output_key="execution_result")

custom_chain = SimpleSequentialChain(chains=[plan_chain, execution_chain], verbose=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    prompt=travel_prompt,
    tools=tools,
    memory=memory,
    verbose=True,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# --- Agent Execution Function ---

def run_agent(goal):
    print(f"\nGoal: {goal}")
    response = agent_executor.invoke({"input": goal})
    print(f"Agent's Response: {response['output']}")
    print("\nMemory Content:")
    for message in memory.buffer:
        print(f"{message.type}: {message.content}")
    print("-" * 20)

# --- Run Agent with Goals ---

goals = [
    "I'm arriving at Paris Airport. How far is it to central London?",
    "What's the weather like in Paris right now?",
    "I'm then going to Rome for a football match. How far is it from Paris to Rome?",
    "What was the first destination I mentioned after arriving at the airport?",
    "What is the current weather in my arrival city?",
    "Remind me of my overall trip plan.",
    "Can you summarize my trip plan in one sentence?"
]

for goal in goals:
    run_agent(goal)