import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.optim as optim # We need an optimizer to adjust weights
from tabulate import tabulate
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/PPO_Experiment_1')

class SimpleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 64) # Increased neurons slightly
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 4)
        self.relu = nn.Tanh() # Tanh often works better for PPO than ReLU

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # OUTPUT: Raw Logits (Not probabilities yet)
        return self.layer3(x)

class SimpleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)



actorNetwork = SimpleActor()
criticNetwork = SimpleCritic()
optimizerActor = optim.Adam(actorNetwork.parameters(), lr=0.003)
optimizerCritic = optim.Adam(criticNetwork.parameters(), lr=0.003)


loss_fn_critic = nn.MSELoss()

#env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5)
env = gym.make("LunarLander-v3", continuous=False)

BATCH_SIZE = 2048

def getBatchObs(env, actorNetwork=None, batch_size=2048):
    
    batch_obs = []
    batch_acts = []
    batch_logprobs = [] 
    batch_rews = []
    batch_dones = [] 
    observation, info = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    for _ in range(batch_size):
        #action = env.action_space.sample()
        with torch.no_grad():
            predicted_action_scores = actorNetwork(torch.from_numpy(observation))
             
            dist = Categorical(logits=predicted_action_scores)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        observationNew, reward, terminated, truncated, info = env.step(action.item())

        current_episode_reward += reward
        
        
        batch_obs.append(torch.from_numpy(observation).float())
        batch_acts.append(action)
        batch_logprobs.append(log_prob)
        batch_rews.append(reward)
        observation=observationNew
        
        done = terminated or truncated
        batch_dones.append(done)
        
        if done:
          episode_rewards.append(current_episode_reward)
          observation, info = env.reset()    
          current_episode_reward = 0
        
        

    env.close()
    return batch_obs, batch_acts, batch_logprobs, batch_rews, batch_dones, episode_rewards

def trace_graph(root, depth=0):
    indent = "  " * depth
    
    # 1. If the user passed a Tensor (like ActorLoss), grab its grad_fn
    if hasattr(root, 'grad_fn'):
        grad_fn = root.grad_fn
    else:
        grad_fn = root # It's already a grad_fn or None

    # 2. If grad_fn is None, we hit a Leaf Node (Weights or Input)
    if grad_fn is None:
        print(f"{indent}--> Reached Leaf Node (Weights or Inputs)")
        return

    # 3. Print the operation name
    print(f"{indent}Operation: {grad_fn.__class__.__name__}")

    # 4. Recurse into parents
    # grad_fn.next_functions returns a list of (parent_grad_fn, index)
    if hasattr(grad_fn, 'next_functions'):
        for parent, _ in grad_fn.next_functions:
            # We trace the first parent to see the main path
            # (Remove 'break' if you want to see the full massive tree)
            trace_graph(parent, depth+1)
            break

def calculate_returns(rewards, dones, discount_factor=0.99):
    returns = []
    R = 0
    
    # Iterate backwards
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d: # If 'done' is True (Game Over)
            R = 0 
        
        R = r + discount_factor * R
        returns.insert(0, R)
        
    return torch.tensor(returns, dtype=torch.float32)

def watch_agent(actor_model):
    print("--- WATCHING AGENT PLAY ---")
    test_env = gym.make("LunarLander-v3", continuous=False, render_mode="human")
    
    obs, _ = test_env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    
    actor_model.eval()
    
    while not (terminated or truncated):
        obs_tensor = torch.from_numpy(obs).float()
        
        with torch.no_grad():
            logits = actor_model(obs_tensor)
            
            
            action = torch.argmax(logits).item()
                        

        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward

    print(f"--- FINISHED WATCHING. Score: {total_reward:.2f} ---")
    test_env.close()
    
    # Switch model back to training mode
    actor_model.train()


global_step = 0
for _batches in range(500):

    batch_obs, batch_acts, batch_logprobs, batch_env_rews, batch_dones, episode_rewards = getBatchObs(env, actorNetwork, BATCH_SIZE)


    #tensorBatches
    tensor_batch_obs = torch.stack(batch_obs)
    tensor_batch_acts = torch.stack(batch_acts)
    tensor_batch_Oldlogprobs = torch.stack(batch_logprobs)
    tensor_batch_env_rews = torch.tensor(batch_env_rews, dtype=torch.float32)

    tensor_batch_env_discounted_rewards = calculate_returns(tensor_batch_env_rews, batch_dones,discount_factor=0.99)

    with torch.no_grad():
        tensor_batch_predicted_rewards = criticNetwork(tensor_batch_obs).squeeze() 
        advantages = tensor_batch_env_discounted_rewards - tensor_batch_predicted_rewards.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    avg_ep_reward = np.mean(episode_rewards) if episode_rewards else -1000

    for _epoch in range(10):

        tensor_batch_newAlllogprobs = actorNetwork(tensor_batch_obs)

        dist = Categorical(logits=tensor_batch_newAlllogprobs)
        tensor_batch_Newlogprobs = dist.log_prob(tensor_batch_acts)

        tensor_ratio = torch.exp(tensor_batch_Newlogprobs - tensor_batch_Oldlogprobs)
        tensor_clipped_ratio = torch.clamp(tensor_ratio, 0.8, 1.2)

        term1 = tensor_ratio*advantages
        term2 = tensor_clipped_ratio*advantages

        ActorLoss = -torch.mean(torch.min(term1, term2))

        tensor_batch_predicted_rewards_current = criticNetwork(tensor_batch_obs).squeeze() 
        criticLoss = loss_fn_critic(tensor_batch_predicted_rewards_current.float(), tensor_batch_env_discounted_rewards.float())


        print ( f"Batch {_batches+1}  Epoch {_epoch+1}: Actor Loss = {ActorLoss.item():.4f}, Critic Loss = {criticLoss.item():.4f}, Avg Episode Reward = {avg_ep_reward:.2f}" )
        
        optimizerCritic.zero_grad()
        criticLoss.backward()
        optimizerCritic.step()


        optimizerActor.zero_grad()
        ActorLoss.backward()
        optimizerActor.step()
   
   
    
        
    writer.add_scalar("Loss/Actor", ActorLoss.item(), global_step)
    writer.add_scalar("Loss/Critic", criticLoss.item(), global_step)
    writer.add_scalar("Reward/Average", avg_ep_reward, global_step)
        
    global_step += BATCH_SIZE
    if (_batches + 1) % 25 == 0:
        watch_agent(actorNetwork)

writer.close()

#print(actorNetwork.layer2.weight)
#print(actorNetwork.layer2.weight)

#print(advantages)