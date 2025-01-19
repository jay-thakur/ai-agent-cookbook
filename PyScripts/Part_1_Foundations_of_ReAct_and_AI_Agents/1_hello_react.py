"""
This script demonstrates the basic ReAct (Reasoning and Acting) cycle
using a simple room cleaning example. It's designed as an introductory
example to illustrate the core concepts of observation, thought, and action.
"""

class HelloReActAgent:
    """
    A simple ReAct agent that simulates a basic cleaning task.
    """

    def observe(self):
        """
        Perceives the environment and returns an observation.

        Returns:
            str: A description of the current state of the environment.
        """
        return "The room is messy."

    def think(self, observation):
        """
        Processes the observation and returns a thought.

        Args:
            observation (str): The observation from the environment.

        Returns:
            str: A thought or plan based on the observation.
        """
        return "I need to clean the room. I should start by picking up the trash."

    def act(self, thought):
        """
        Performs an action based on the thought and returns a result.

        Args:
            thought (str): The thought or plan to act upon.

        Returns:
            str: The result of the action.
        """
        return "Picked up the trash."

agent = HelloReActAgent()

print("--- Welcome to the ReAct Cycle Demo! ---\n")

print("Step 1: The agent starts by observing its environment.")
observation = agent.observe()
print(f"Agent Observes: '{observation}'\n")

print("Step 2: The agent thinks about what to do based on the observation.")
thought = agent.think(observation)
print(f"Agent Thinks: '{thought}'\n")

print("Step 3: The agent takes action based on its thought process.")
action_result = agent.act(thought)
print(f"Agent Acts: '{action_result}'\n")

print("--- ReAct cycle complete! ---\n")