class HelloReActAgent:
    def __init__(self):
        pass

    def observe(self):
        """Perceives the environment and returns an observation."""
        return "Initial Observation"

    def think(self, observation):
        """Processes the observation and returns a thought."""
        return f"Thinking about: {observation}"

    def act(self, thought):
        """Performs an action based on the thought and returns a result."""
        return f"Acting on: {thought}"

agent = HelloReActAgent()

observation = agent.observe()
print(f"Observation: {observation}")

thought = agent.think(observation)
print(f"Thought: {thought}")

action_result = agent.act(thought)
print(f"Action Result: {action_result}")

print("ReAct cycle complete.")