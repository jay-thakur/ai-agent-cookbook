# 9_react_with_tools.py
# Demonstrates how a ReAct agent interacts with a simple search & calculator tool.

import re
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class BasicEnvironment:
    def __init__(self):
        self.state = ""

    def get_state(self):
        return self.state

    def change_state(self, new_state):
        self.state = new_state

class ReActAgent:
    def __init__(self, environment):
        self.environment = environment

    def observe(self):
        return self.environment.get_state()

    def think(self, observation, goal):
        return [] # Base agent doesn't do any thinking by default

    def act(self, step, debug=True):
        if debug:
            print(f"Executing action: {step}")
        return "Action completed."

class CalculatorTool:
    """A simple calculator tool."""

    def calculate(self, expression):
        try:
            # Use a safer evaluator (replace with a suitable library in real applications)
            import ast
            result = ast.literal_eval(expression)
            return result
        except (SyntaxError, ValueError):
            return "Invalid calculation."
        
class SearchTool:
    def search(self, query):
        # Simulate a search (replace with a real search API in a real application)
        if "population of London" in query.lower():
            return "8.982 million"  # Example population
        else:
            return "Information not found."

class ReActAgentWithTools(ReActAgent): 
    def __init__(self, environment, tools, llm): # tools is a dictionary
        super().__init__(environment)
        self.tools = tools # Store tools in a dictionary
        self.llm = llm
        self.memory = []  # List to store tool results

    def think(self, observation, goal):
        """Uses the LLM to generate a plan."""
        prompt = f"""
        Tools Available: {list(self.tools.keys())}
        Goal: {goal}
        Observation: {observation}

        Plan:
        """
        
        """
        # Example 1: Search for capital
        Goal: What is the capital of France?
        Plan:
            - Use SearchTool: What is the capital of France?

        # Example 2: Calculate expression
        Goal: What is 10 + 5?
        Plan:
            - Use Calculator: 10 + 5

        # Example 3: Combine tools
        Goal: What is the population of Paris divided by 2?
        Plan:
        1. Use SearchTool: Population of Paris
        2. Use Calculator: [Result from SearchTool] / 2

        Your Plan:
        """
        llm_output = self.llm(prompt)
        plan = [step.strip() for step in llm_output.strip().split('\n') if step]
        return plan

    def act(self, step, debug=True):
        if debug:
            print(f"Executing step: {step}")
        for tool_name, tool in self.tools.items():
            if f"Use {tool_name}:" in step:
                query = step.split(f"Use {tool_name}: ")[1].strip()
                if tool_name == "Calculator" and "[Result from SearchTool]" in query:
                    if self.memory:
                        query = query.replace("[Result from SearchTool]", self.memory[-1])
                    else:
                        return "Error: No previous result in memory for calculation."
                result = tool.calculate(query) if hasattr(tool, 'calculate') else tool.search(query)
                print(f"{tool_name} Result: {result}")
                self.memory.append(result)  # Update memory with the latest result
                return result
        return super().act(step, debug)


if __name__ == "__main__":
    environment = BasicEnvironment()
    search_tool = SearchTool()
    calculator_tool = CalculatorTool()
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    # Create the agent with the tools
    tools = {
        "SearchTool": search_tool,
        "Calculator": calculator_tool
    }
    agent = ReActAgentWithTools(environment, tools, llm)

    goals = [
        "What is the population of London divided by 10?",
        "What is 20 * 3?",
        "What is the population of New York",
        "What is 10 / 0?",
        "What is (2 + 3) * 4?",
        "What is 2 ** 3",
        "What is 10 - 7?",
        "What is 2.5 * 4 ?",
        "What is 1 + (2 * 3)",
        "What is (10 + 5) / (2 + 1)",
    ]

    for goal in goals:
        print(f"\nGoal: {goal}")
        plan = agent.think(environment.get_state(), goal)
        print(f"Plan: {plan}")
        for step in plan:
            result = agent.act(step)
            print(f"Result: {result}")
            environment.change_state(result)
        print(f"Final Environment State: {environment.get_state()}")
        print("-" * 20)