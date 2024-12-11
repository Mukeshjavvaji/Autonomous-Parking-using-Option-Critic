from mlagents_envs.environment import UnityEnvironment
from replay import ReplayBuffer

env = UnityEnvironment(file_name=None, base_port=5004, worker_id=0)

env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

# Print observation and action specs
print("Observation space:", spec.observation_specs)
print("Action space:", spec.action_spec)

for episode in range(10):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    for agent_id in decision_steps.agent_id:
        observations = decision_steps[agent_id].obs
        reward = decision_steps[agent_id].reward
        print("obs", observations[1])
        print(f"Agent {agent_id} needs a decision. Reward: {reward}")
    for agent_id in decision_steps:
        action = spec.action_spec.random_action(len(decision_steps))
        print("action", action)
        env.set_actions(behavior_name, action)
    env.step()
env.close()
