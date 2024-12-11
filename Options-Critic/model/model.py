from OCAgent import OptionsCriticAgent
import torch
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

agent = OptionsCriticAgent(4, 2)
agent.load_state_dict(torch.load('/Users/mukeshjavvaji/Documents/Academics/RL and SDM/Project/AutonomousParking-main/Assets/MLAgents/Options-Critic/model/trial_model1.pt'))


# dummy = torch.randn(1, 58)
# output = agent.forward(dummy)
# print(output)

def get_action_tuple(action):
    continuous_actions = np.array([[action[0], action[1]]])  
    return ActionTuple(continuous=continuous_actions)

env = UnityEnvironment(file_name=None, base_port=5004, worker_id=0)
print("Connected to the environment")

env.reset()
print("Env reset done")

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
decision_steps, terminal_steps = env.get_steps(behavior_name)
state = decision_steps[0].obs[1]
for time_step in range(1500):
    if len(decision_steps.agent_id) > 0:
        action = agent.forward(torch.tensor(state))

        env.set_actions(behavior_name, get_action_tuple(action.detach().numpy()))
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        next_state = decision_steps[0].obs[1]
        state = next_state
