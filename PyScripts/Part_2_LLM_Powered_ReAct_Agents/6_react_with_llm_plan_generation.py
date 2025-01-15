# 6_react_with_llm_plan_generation.py
# This script demonstrates using an LLM for planning within the ReAct cycle.

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


class ReActPlanGeneratingAgent(ReActAgent):
    """A ReAct agent that uses an LLM for plan generation."""

    def think(self, observation, goal):
        """Uses the LLM to generate a plan (sequence of actions)."""
        prompt = f"""
        You are an agent in a simple environment. Your current goal is: {goal}
        The current state of the room is: {observation}

        Create a plan (a sequence of actions) to achieve your goal. List the actions as numbered steps.

        Plan:
        """
        try:
            llm_output = self.llm(prompt)
            plan = [step.strip() for step in llm_output.strip().split('\n') if step]
            return plan
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return ["unknown state"]

if __name__ == "__main__":
    possible_states = ["messy", "clean", "dusty", "less messy"]
    initial_state = random.choice(possible_states)
    goal = "Make the room clean."
    room_environment = BasicEnvironment(initial_state, goal)
    agent = ReActPlanGeneratingAgent(room_environment, OPENAI_API_KEY)

    print(f"Initial State: {initial_state}")
    print(f"Goal: {goal}")

    num_cycles = 3
    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---")

        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought (Now using the LLM for planning)
        plan = agent.think(observation, goal)

        valid_actions = ["clean the room", "dust the room", "do nothing"]
        plan = [step for step in plan if any(valid_action in step.lower() for valid_action in valid_actions)]
        print(f"Filtered Plan: {plan} (only valid actions retained)") # Explain plan validation

        if not plan or "unknown state" in plan: # Handle unknown actions more informatively
            print("No valid plan was generated. This might happen due to unexpected LLM output or no suitable action.")
            continue

        print("Generated Plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

        # Step 3: Action (Execute the first action in the plan)
        action = plan[0]
        action_result = agent.act(action, debug=True)
        print(f"Action Result: {action_result}")

        print(f"Cycle {cycle + 1} Complete: State = {room_environment.get_state()}, Plan = {plan}")
        if room_environment.is_goal_state():
            print("Goal achieved!")
            break
        print("-" * 20)