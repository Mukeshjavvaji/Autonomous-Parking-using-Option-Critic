from networks import PolicyOverOptions, SubPolicy, TerminationPolicy, QOmega, QU
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import random
from channel import OptionSideChannel
from collections import deque
np.bool = bool

# Options-Critic agent
class OptionsCriticAgent(nn.Module):
    def __init__(self, num_options, num_actions):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        self.options = ["Searching for the entrance", "Entering into the parking lot", "Searching for an empty parking spot", "Parking into the spot"]

        self.policy_over_options = PolicyOverOptions()
        self.poo_optimizer = Adam(self.policy_over_options.parameters(), lr=0.0000001)

        self.sub_policy = SubPolicy()
        self.sp_optimizer = Adam(self.sub_policy.parameters(), lr=0.0000001)

        self.termination_policy = TerminationPolicy()
        self.tp_optimizer = Adam(self.termination_policy.parameters(), lr=0.0000001)
        
        self.Qomega = QOmega()
        self.q_omega_optimizer = Adam(self.Qomega.parameters(), lr=0.0000001)

        self.Qu = QU()
        self.q_u_optimizer = Adam(self.Qu.parameters(), lr=0.0000001)

        self.current_option = None

    def get_one_hot_encoding(self, o, n):
        one_hot = np.zeros(n)
        one_hot[o] = 1
        return one_hot

    def forward(self, input):
        input = input.flatten()
        if self.current_option == None:
            with torch.no_grad():
                option_probs = self.policy_over_options(input[55:])
            option = np.random.choice([0,1,2,3], p=option_probs.detach().numpy())
            self.current_option = option
        else:
            one_hot_encoded_option = self.get_one_hot_encoding(self.current_option, self.num_options)
            with torch.no_grad():
                is_terminate = self.termination_policy(torch.tensor(np.concatenate((input[55:].detach().numpy(), one_hot_encoded_option))).to(torch.float32))
            if is_terminate > 0.5:
                if np.random.rand() > 0.1:
                    with torch.no_grad():
                        option_probs = self.policy_over_options(input[55:])
                    option = np.random.choice([0,1,2,3], p=option_probs.detach().numpy())
                else:
                    option = np.random.choice([0,1,2,3])
                self.current_option = option

        one_hot_encoded_option = self.get_one_hot_encoding(self.current_option, self.num_options)
        output = self.sub_policy(torch.tensor(np.concatenate((input[:55].detach().numpy(), one_hot_encoded_option))).to(torch.float32))
        return output
    
    def update_policy_over_options(self, state, option, q_omega, poo_reward):
        self.poo_optimizer.zero_grad()
        option_probs = self.policy_over_options(state)
        target = poo_reward + q_omega
        loss = -torch.log(option_probs[option]) * target
        loss.backward()
        self.poo_optimizer.step()
        return loss.item()
    
    def calculate_noise(self, predicted_action):
        # Gaussian noise with mean 0 and adjustable standard deviation
        noise_std = 0.2 
        return torch.normal(mean=0.0, std=noise_std, size=predicted_action.shape).to(predicted_action.device)

    def update_sub_policy(self, predicted_action, advantage, sp_reward):
        self.sp_optimizer.zero_grad()
        target = sp_reward + advantage
        noise = self.calculate_noise(predicted_action)
        noisy_action = predicted_action + noise
        noisy_action = torch.clamp(noisy_action, -1.0, 1.0)
        loss = torch.mean((noisy_action - target)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sub_policy.parameters(), max_norm=1.0)
        self.sp_optimizer.step()
        return loss.item()

    def update_termination_policy(self, state, option, q_omega, v_omega, t_reward):
        self.tp_optimizer.zero_grad()
        termination_prob = self.termination_policy(torch.tensor(np.concatenate((state, option))).to(torch.float32))
        target = t_reward + q_omega - v_omega
        loss = termination_prob * target
        loss.backward()
        self.tp_optimizer.step()
        return loss.item()

    def update_q_omega(self, state, option, target):
        self.q_omega_optimizer.zero_grad()
        prediction = self.Qomega(torch.tensor(np.concatenate((state, option))).to(torch.float32))
        target = torch.tensor(target).reshape(1)
        loss = nn.MSELoss()(prediction, target)
        loss.backward()
        self.q_omega_optimizer.step()
        return loss.item()

    def update_q_u(self, state, option, action, target):
        self.q_u_optimizer.zero_grad()
        prediction = self.Qu(torch.tensor(np.concatenate((state, option, action))).to(torch.float32))
        loss = nn.MSELoss()(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sub_policy.parameters(), max_norm=1.0)
        self.q_u_optimizer.step()
        return loss.item()
    
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, option, next_state, reward, poo_reward, iop_reward, termination_reward):
        self.buffer.append((state, action, option, next_state, reward, poo_reward, iop_reward, termination_reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, option, next_state, reward, poo_reward, iop_reward, termination_reward = zip(*batch)
        return torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor(option), torch.FloatTensor(next_state), torch.FloatTensor(reward), poo_reward, iop_reward, termination_reward

    def size(self):
        return len(self.buffer)


def get_action_tuple(action):
    continuous_actions = np.array([[action[0], action[1]]])  
    return ActionTuple(continuous=continuous_actions)

def back_propagation(agent, state, action, reward, next_state, option, poo_reward, iop_reward, termination_reward, time_step):
    one_hot_encoded_option = agent.get_one_hot_encoding(option, agent.num_options)
    with torch.no_grad():
        q_omega_value = agent.Qomega(torch.tensor(np.concatenate((state[55:].detach().numpy(), one_hot_encoded_option))).to(torch.float32))
        q_u_value = agent.Qu(torch.tensor(np.concatenate((state[:55].detach().numpy(), one_hot_encoded_option, action.detach().numpy()))).to(torch.float32))
        next_q_omega = agent.Qomega(torch.tensor(np.concatenate((next_state[55:].detach().numpy(), one_hot_encoded_option))).to(torch.float32))
        termination_prob = agent.termination_policy(torch.tensor(np.concatenate((next_state[55:].detach().numpy(), one_hot_encoded_option))).to(torch.float32))
        a_prime = agent.forward(torch.tensor(next_state))
        next_q_u = agent.Qu(torch.tensor(np.concatenate((next_state[:55].detach().numpy(), one_hot_encoded_option, a_prime.detach().numpy()))).to(torch.float32))

    v_omega = get_max_q_omega(next_state[-3:])
    target_q_u = reward + ((1 - termination_prob) * next_q_omega + termination_prob * v_omega)
    diff = target_q_u.item() - q_omega_value.item()
    q_omega_loss = agent.update_q_omega(state[-3:], one_hot_encoded_option, diff)
    u_target = reward + next_q_u
    error = u_target - q_u_value
    q_u_loss = agent.update_q_u(state[:55], one_hot_encoded_option, action.detach().numpy(), error)

    q_omega_losses.append(q_omega_loss)
    q_u_losses.append(q_u_loss)

    if time_step % update_frequency == 0:
        advantage = q_u_value.detach() - q_omega_value.detach()
        poo_loss = agent.update_policy_over_options(state[-3:], option, q_omega_value.detach(), poo_reward)
        sp_loss = agent.update_sub_policy(action, advantage, iop_reward)
        tf_loss = agent.update_termination_policy(next_state[-3:], one_hot_encoded_option, next_q_omega.detach(), v_omega.detach(), termination_reward)

        poo_losses.append(poo_loss)
        sp_losses.append(sp_loss)
        tf_losses.append(tf_loss)



if __name__ == '__main__':
    file = '/Users/mukeshjavvaji/Documents/Academics/RL and SDM/Project/AutonomousParking-main/parkingLot.app'
    option_channel = OptionSideChannel()
    env = UnityEnvironment(file_name=file, base_port=5004, worker_id=0, side_channels=[option_channel], no_graphics=True)
    print("Connected to the environment")

    env.reset()
    print("Env reset done")

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    gamma = 1
    update_frequency = 2
    batch_size = 512
    epsilon = 1.0
    decay = 0.9999
    

    q_omega_losses = []
    q_u_losses = []
    poo_losses = []
    sp_losses = []
    tf_losses = []
    all_rewards = []
    ep_poo_rewards = []
    ep_iop_rewards = []
    ep_termination_rewards = []

    agent = OptionsCriticAgent(4, 3)

    replay = ReplayBuffer(100000)

    def get_max_q_omega(state):
        maxi = -1000000
        for i in range(agent.num_options):
            with torch.no_grad():
                val = agent.Qomega(torch.tensor(np.concatenate((state, agent.get_one_hot_encoding(i, agent.num_options)))).to(torch.float32))
                if val > maxi:
                    maxi = val
        return maxi
    
    def save_loss_vals(l, file_name):
        with open(file_name, "a") as file:
            for value in l:
                file.write(str(value) + "\n")


    for episode in range(1000):
        print(episode)
        done = True
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        state = decision_steps[0].obs[1]
        ep_poo_reward = 0
        ep_iop_reward = 0
        ep_termination_reward = 0
        for time_step in range(1500):
            if len(decision_steps.agent_id) > 0:
                action = agent.forward(torch.tensor(state))
                option_channel.send_active_option(int(agent.current_option))
                numpy_actions = action.detach().numpy()

                env.set_actions(behavior_name, get_action_tuple(numpy_actions))
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                next_state = decision_steps[0].obs[1]
                reward = decision_steps[0].reward
                poo_reward = option_channel.poo_reward
                iop_reward = option_channel.iop_reward
                termination_reward = option_channel.termination_reward
                ep_poo_reward += poo_reward
                ep_iop_reward += iop_reward
                ep_termination_reward += termination_reward
                all_rewards.append(reward)
                replay.push(state, numpy_actions, agent.current_option, next_state, reward, poo_reward, iop_reward, termination_reward)
                if time_step % 100 == 0 and time_step != 0:
                    if replay.size() > batch_size:
                        states, actions, options, next_states, rewards, poo_rewards, iop_rewards, termination_rewards = replay.sample(batch_size)
                        for i in range(batch_size):
                            back_propagation(agent, states[i], actions[i], rewards[i].item(), next_states[i], int(options[i].item()), poo_rewards[i], iop_rewards[i], termination_rewards[i], 1)

                back_propagation(agent, torch.tensor(state), action, reward, torch.tensor(next_state), agent.current_option, poo_reward, iop_reward, termination_reward, time_step)
                state = next_state
        ep_poo_rewards.append(ep_poo_reward)
        ep_iop_rewards.append(ep_iop_reward)
        ep_termination_rewards.append(ep_termination_reward)
        env.reset()

        if len(all_rewards) > 5000:
            save_loss_vals(q_omega_losses, "q_omega.txt")
            q_omega_losses = []
            save_loss_vals(q_u_losses, "q_u.txt")
            q_u_losses = []
            save_loss_vals(poo_losses, "poo.txt")
            poo_losses = []
            save_loss_vals(sp_losses, "sp.txt")
            sp_losses = []
            save_loss_vals(tf_losses, "tf.txt")
            tf_losses = []
            save_loss_vals(all_rewards, "rewards.txt")
            all_rewards = []
            save_loss_vals(ep_poo_rewards, "poo_rewards.txt")
            ep_poo_rewards = []
            save_loss_vals(ep_iop_rewards, "iop_rewards.txt")
            ep_iop_rewards = []
            save_loss_vals(ep_termination_rewards, "termination_rewards.txt")
            ep_termination_rewards = []

        if episode % 10 == 0:
            torch.save(agent.state_dict(), "/Users/mukeshjavvaji/Documents/Academics/RL and SDM/Project/AutonomousParking-main/Assets/MLAgents/Options-Critic/model/model1.pt")
        if episode % 15 == 0:
            torch.save(agent.state_dict(), "/Users/mukeshjavvaji/Documents/Academics/RL and SDM/Project/AutonomousParking-main/Assets/MLAgents/Options-Critic/model/model2.pt")
    






    

    
    
        