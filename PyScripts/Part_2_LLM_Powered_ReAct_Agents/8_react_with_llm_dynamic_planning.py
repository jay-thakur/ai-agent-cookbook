# 8_react_with_llm_dynamic_planning.py
# This script demonstrates dynamic planning with an LLM within the ReAct cycle.

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

class ReActDynamicPlanningAgent(ReActAgent):
    """A ReAct agent that uses the LLM for dynamic planning."""

    def think(self, observation, goal, is_replanning=False):  # Add is_replanning parameter
        """Uses the LLM to generate/replan based on current observation."""
        prompt_prefix = "Replan" if is_replanning else "Create a plan" # Change prompt based on replanning

        prompt = f"""
        You are an agent in a simple environment. Your current goal is: {goal}
        You have observed: {observation}

        {prompt_prefix} (a sequence of actions) to achieve your goal. List the actions as numbered steps.

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
    agent = ReActDynamicPlanningAgent(room_environment, OPENAI_API_KEY)

    print(f"Initial State: {initial_state}")
    print(f"Goal: {goal}")

    num_cycles = 3
    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---")

        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought (Dynamic planning)
        plan = agent.think(observation, goal)

        valid_actions = ["clean the room", "dust the room", "do nothing"]
        plan = [step for step in plan if any(valid_action in step.lower() for valid_action in valid_actions)]
        print(f"Filtered Plan: {plan} (Retaining only valid actions)")

        if not plan or "unknown state" in plan:
            print("No valid plan generated. This could be due to unexpected LLM output or no suitable action. Skipping cycle.")
            continue

        print("Generated Plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

        # Step 3: Action (Dynamic execution with replanning)
        max_plan_length = 5
        plan = plan[:max_plan_length]
        for step_index, step in enumerate(plan):  # Iterate through the current plan, keeping track of the step index
            if room_environment.is_goal_state():  # Check if the goal has already been achieved
                print("Goal already achieved. Skipping remaining actions.")
                break  # Exit the loop if the goal is achieved

            action_result = agent.act(step, debug=True)  # Execute the current step in the plan
            print(f"Step {step_index + 1}: {action_result}")  # Print the result of the action
            print(f"Updated Environment State: {room_environment.get_state()}")  # Print the updated state of the environment

            # Replanning based on new observation after each step
            new_observation = agent.observe()  # Observe the environment *again* to get the updated state after the action
            replanned_plan = agent.think(new_observation, goal, is_replanning=True)  # Generate a *new* plan based on the updated observation

            valid_actions = ["clean the room", "dust the room", "do nothing"]
            replanned_plan = [step for step in replanned_plan if any(valid_action in step.lower() for valid_action in valid_actions)]

            if not replanned_plan:  # Handle Edge Case for Empty Plans
                print("Replanning resulted in no valid actions. Continuing with the current plan.")
                continue

            if replanned_plan and replanned_plan != plan[step_index + 1:]:  # Check if a valid new plan exists AND if it's different from the remaining part of the current plan
                plan = plan[:step_index + 1] + replanned_plan  # Update the plan
                print("\033[93mReplanning Triggered!\033[0m")
                print(f"Plan updated dynamically: {plan}")
            elif replanned_plan and replanned_plan[0] == plan[step_index]:  # Improved comparison for redundant steps
                print("Replanned action matches the current step. No need to re-execute.")
            elif replanned_plan:
                print("Replanning did not result in any changes. Continuing with the current plan.")

        if not room_environment.is_goal_state():
            print("This cycle's plan execution is complete, but the goal is not yet achieved.")

        print(f"Cycle {cycle + 1} Summary: Plan Executed = {plan[:step_index + 1] if step_index < len(plan) else plan}, Remaining Plan = {plan[step_index + 1:] if step_index < len(plan) else []}, State = {room_environment.get_state()}")
        if room_environment.is_goal_state():
            print("Goal achieved!")
            break
        print("-" * 20)

    if room_environment.is_goal_state():
        print("\nGoal achieved!")
    else:
        print("\nGoal not achieved. Final state:", room_environment.get_state())
