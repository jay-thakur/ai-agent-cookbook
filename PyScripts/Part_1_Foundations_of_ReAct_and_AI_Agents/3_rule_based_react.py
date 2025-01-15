# 3_rule_based_react.py
# This script demonstrates the ReAct cycle with more advanced rule-based reasoning.

import random

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
    """
    Represents a ReAct agent with more complex rule-based reasoning.

    Attributes:
        environment (BasicEnvironment): The agent's environment.
    """
    def __init__(self, environment):
        self.environment = environment

    def observe(self):
        """Observes the current state of the environment."""
        return self.environment.get_state()

    def think(self, observation):
        """
        Implements more complex rule-based thinking.

        Args:
            observation (str): The current observation (environment state).

        Returns:
            str: The action to perform.
        """
        if observation == "messy":
            return "clean the room"
        elif observation == "dusty":
            return "dust the room"
        elif observation == "less messy":
            return "clean the room"  # if less messy, clean the room
        elif observation == "clean":
            return "do nothing"
        else:
            return "unknown state"

    def act(self, action):
        """
        Performs an action and updates the environment if necessary.

        Args:
            action (str): The action to perform.

        Returns:
            str: The result of the action.
        """
        if action == "clean the room":
            self.environment.change_state("clean")
            return "You cleaned the room. It is now clean." # More verbose feedback
        elif action == "dust the room":
            self.environment.change_state("less messy")
            return "You dusted the room. It is now less messy, but still needs cleaning." # More verbose feedback
        elif action == "do nothing":
            return "You did nothing."
        elif action == "unknown state":
            return "I don't know what to do in this state."
        else:
            return "I don't know how to do that."

if __name__ == "__main__":
    possible_states = ["messy", "clean", "dusty", "less messy"] # Added "less messy" to possible states
    initial_state = random.choice(possible_states)
    room_environment = BasicEnvironment(initial_state)
    agent = ReActAgent(room_environment)

    num_cycles = 5
    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---")
        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought
        thought = agent.think(observation)
        print(f"Thought: {thought}")

        # Step 3: Action
        action = agent.act(thought)
        print(f"Action Result: {action}")
        print(f"Cycle Complete: The room's current state is {room_environment.get_state()}") # Cycle completion message
        print("-" * 20)