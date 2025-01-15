# 4_react_with_llm_basic.py
# This script introduces a basic LLM into the ReAct cycle.

import random
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get the API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class BasicEnvironment:
    """
    Represents a simple environment with different states.

    Attributes:
        current_state (str): The current state of the environment.
    """
    def __init__(self, initial_state):
        self.current_state = initial_state

    def get_state(self):
        """Returns the current state of the environment."""
        return self.current_state

    def change_state(self, new_state):
        """Changes the state of the environment."""
        self.current_state = new_state

class ReActAgent:
    # ... (Agent class remains the same except for the act method)
    def __init__(self, environment, openai_api_key):
        self.environment = environment
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    def observe(self):
        return self.environment.get_state()

    def think(self, observation):
        prompt = f"""
        You are an agent in a simple environment. Your goal is to keep the room clean.
        The current state of the room is: {observation}

        Based on this observation, what action should you take?

        Action:
        """
        try:
            llm_output = self.llm(prompt)
            action = llm_output.strip()
        except Exception as e:
            print(f"Error during LLM call: {e}")
            action = "unknown state"
        return action

    def act(self, action, debug=False): # Added debug parameter
        """
        Performs an action and updates the environment if necessary. Now handles partial matches.

        Args:
            action (str): The action to perform (potentially from LLM).
            debug (bool, optional): If True, prints the raw LLM output. Defaults to False.

        Returns:
            str: The result of the action.
        """

        if debug:
            print(f"Raw LLM Output: {action}") # Print raw LLM output if debug is True

        if "clean" in action.lower(): # Partial matching
            self.environment.change_state("clean")
            return "You cleaned the room. It is now clean."
        elif "dust" in action.lower(): # Partial Matching
            self.environment.change_state("less messy")
            return "You dusted the room. It is now less messy, but still needs cleaning."
        elif "nothing" in action.lower() or "relax" in action.lower():  # Partial matching for "do nothing" and added "relax"
            return "You did nothing."
        elif "unknown" in action.lower():
            return "I don't know what to do in this state."

        else:
            return f"I don't know how to do '{action}'." # More informative message

if __name__ == "__main__":
    possible_states = ["messy", "clean", "dusty"]
    initial_state = random.choice(possible_states)
    room_environment = BasicEnvironment(initial_state)

    # Initialize the agent with your OpenAI API key
    agent = ReActAgent(room_environment, OPENAI_API_KEY)

    num_cycles = 3
    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---")

        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought (Now using the LLM)
        thought = agent.think(observation)
        print(f"Thought: {thought}")

        # Step 3: Action
        action_result = agent.act(thought, debug=True) # debug set to true so we can see the output
        print(f"Action Result: {action_result}")
        print(f"Cycle Complete: The room's current state is {room_environment.get_state()}")
        print("-" * 20)