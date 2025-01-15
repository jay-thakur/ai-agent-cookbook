# 2_basic_environment.py
# This script demonstrates the ReAct cycle with a simple environment
# that has different states.

import random

class BasicEnvironment:
    """
    Represents a simple environment with different states.

    Attributes:
        current_state (str): The current state of the environment (e.g., "messy", "clean", "dusty").
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
    Represents a simple ReAct agent that interacts with the environment.

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
        Implements simple rule-based thinking based on the environment's state.

        Args:
            observation (str): The current observation (environment state).

        Returns:
            str: The action to perform.
        """
        if observation == "messy":
            return "clean the room"
        elif observation == "clean":
            return "do nothing"
        elif observation == "dusty":
            return "dust the room"
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
            return "You cleaned the room."
        elif action == "do nothing":
            return "You did nothing."
        elif action == "dust the room":
            self.environment.change_state("less messy")  # Dusting makes it less messy, not fully clean
            return "You dusted the room. It is now less messy."
        elif action == "unknown state":
            return "I don't know what to do in this state."
        else:
            return "I don't know how to do that."

if __name__ == "__main__":
    # Initialize with a random state
    possible_states = ["messy", "clean", "dusty"] # List of possible states
    initial_state = random.choice(possible_states) # Choosing random state
    room_environment = BasicEnvironment(initial_state)
    agent = ReActAgent(room_environment)

    # Run the ReAct cycle a few times
    num_cycles = 3 # Number of cycles to run
    for cycle in range(num_cycles):
        print(f"--- ReAct Cycle {cycle + 1} ---") # Cycle counter
        # Step 1: Observation
        observation = agent.observe()
        print(f"Observation: The room is {observation}")

        # Step 2: Thought
        thought = agent.think(observation)
        print(f"Thought: {thought}")

        # Step 3: Action
        action = agent.act(thought)
        print(f"Action Result: {action}")
        print("-" * 20)