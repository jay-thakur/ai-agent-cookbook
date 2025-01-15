# 6_react_with_llm_memory.py
# This script demonstrates using an LLM with memory in the ReAct cycle.

import random
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class BasicEnvironment:
    """
    Represents a simple environment with different states and a goal state.

    Attributes:
        current_state (str): The current state of the environment.
        goal_state (str): The desired state of the environment.
    """
    def __init__(self, initial_state, goal_state="clean"):  # Default goal is "clean"
        self.current_state = initial_state
        self.goal_state = goal_state

    def get_state(self):
        """Returns the current state of the environment."""
        return self.current_state

    def change_state(self, new_state):
        """Changes the state of the environment."""
        self.current_state = new_state

    def is_goal_state(self):
        """Checks if the current state matches the goal state."""
        return self.current_state == self.goal_state

class ReActAgent:
    """Base class for ReAct agents."""
    def __init__(self, environment, openai_api_key):
        self.environment = environment
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    def observe(self):
        return self.environment.get_state()

    def act(self, action, debug=False):
        if debug:
            print(f"Raw LLM Output: {action}")

        if "clean" in action.lower():
            self.environment.change_state("clean")
            return "You cleaned the room. It is now clean."
        elif "dust" in action.lower():
            self.environment.change_state("less messy")
            return "You dusted the room. It is now less messy, but still needs cleaning."
        elif "nothing" in action.lower() or "relax" in action.lower():
            return "You did nothing."
        elif "unknown" in action.lower():
            return "I don't know what to do in this state."
        else:
            return f"I don't know how to do '{action}'."

class ReActMemoryAgent(ReActAgent):
    """A ReAct agent that uses an LLM with memory."""

    def __init__(self, environment, openai_api_key):
        super().__init__(environment, openai_api_key)
        self.memory = []

    def think(self, observation, goal):
        """Uses the LLM with memory to make a decision (single action)."""
        memory_string = "\n".join(self.memory) if self.memory else "No memory available."
        prompt = f"""
        You are an agent in a simple environment. Your current goal is: {goal}
        The current state of the room is: {observation}
        Here is your memory of past observations, actions, and outcomes:
        {memory_string}

        Based on your memory and the current state, what is the BEST single action to take NOW to achieve your goal?

        Action:
        """
        try:
            llm_output = self.llm(prompt)
            thought = llm_output.strip()
            return thought
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return "unknown state"

    def act(self, action, debug=False):
        action_result = super().act(action, debug)
        self.memory.append(f"Observation: {self.environment.get_state()}, Action: {action}, Outcome: {action_result}")
        self.memory = self.memory[-5:]
        return action_result

if __name__ == "__main__":
    possible_states = ["messy", "clean", "dusty", "less messy"]
    initial_state = random.choice(possible_states)
    goal = "Make the room clean."
    room_environment = BasicEnvironment(initial_state, goal)
    agent = ReActMemoryAgent(room_environment, OPENAI_API_KEY)

    num_cycles = 5

    if not agent.memory: # Clarify empty memory at start
        print("Memory is initially empty.")

    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---")

        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought (Now using the LLM for memory-based decision making)
        thought = agent.think(observation, goal)

        valid_actions = ["clean the room", "dust the room", "do nothing", "unknown state"] # Action validation
        if thought not in valid_actions:
            thought = "unknown state"

        print(f"Thought: {thought}")

        # Step 3: Action
        if thought != "unknown state":
            action_result = agent.act(thought, debug=True)
            print(f"Action Result: {action_result}")
        else:
            print("No valid thought available or unknown state. Skipping action.")

        print(f"Cycle Complete: The room's current state is {room_environment.get_state()}")
        print(f"Memory after cycle {cycle + 1}: {agent.memory}")
        if room_environment.is_goal_state():
            print("Goal achieved!")
            break
        print("-" * 20)