from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import StringPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Define Custom Tools ---

@tool
def search(query: str) -> str:
    """Searches for information online."""
    print(f"Searching for: {query}")  # Detailed explanation
    if "population of london" in query.lower():
        return "The population of London is approximately 9 million."
    elif "capital of france" in query.lower():
        return "The capital of France is Paris."
    elif "current weather in london" in query.lower(): # Extended Search Tool
        return "The current weather in London is cloudy with a chance of rain."
    else:
        return "I couldn't find information on that."

@tool
def calculator(expression: str) -> str:
    """Calculates a mathematical expression."""
    print(f"Calculating: {expression}")  # Detailed explanation
    try:
        result = eval(expression)  # Use safer alternatives in production!
        return str(result)
    except ZeroDivisionError: # Introduce Errors
        return "Error: Division by zero is not allowed."
    except (SyntaxError, TypeError, NameError):
        return "Error: Invalid calculation."

# --- Define Prompt and Tools ---

search_prompt = StringPromptTemplate(
    template="""
    Here are the tools available to me:
    * Search the web for information.
    * Perform basic mathematical calculations.

    What can I help you with today?

    Goal: {goal}

    Use the available tools or ask me anything you are curious about.

    Conversation History:
    {memory}
    """
)

tools = [search, calculator]

# --- Initialize Agent and Memory ---

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    prompt=search_prompt,
    tools=tools,
    memory=memory,
    verbose=True,
)

# --- Agent Execution Function ---

def run_agent(goal):
    print(f"\nGoal: {goal}")
    agent_executor = AgentExecutor(agent=agent)
    response = agent_executor.invoke({"goal": goal})
    print(f"Agent's Response: {response}")
    print("\nMemory Content:")
    for message in memory.buffer:
        print(f"{message.type}: {message.content}")
    print("-" * 20)

# --- Run Agent with Goals ---

goals = [
    "What is the population of London divided by 2?",
    "What is the capital of France?",
    "What is 10 / 0?", # Introduce Errors
    "What is 10 * 5 + 3?",
    "What is the current weather in London?", # Extended Search Tool
    "What was my first question?", # Add Custom Memory Queries
]

for goal in goals:
    run_agent(goal)