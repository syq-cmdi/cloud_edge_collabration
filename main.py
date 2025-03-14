import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import networkx as nx
import math
import time

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.95  # Discount factor
EPS_START = 0.1  # Starting exploration rate
EPS_END = 0.01  # Final exploration rate
EPS_DECAY = 200  # Decay rate for exploration
TARGET_UPDATE = 10  # How often to update target network
LEARNING_RATE = 0.01  # Learning rate for neural network
MEMORY_SIZE = 500  # Replay memory size
NUM_EPISODES = 400  # Number of training episodes
MAX_STEPS = 100  # Maximum steps per episode

# Define Experience namedtuple
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Zipf distribution function for content popularity
def zipf_distribution(num_contents, alpha=0.73):
    # alpha is the skewness parameter (default 0.73 as used in the paper)
    # Returns the probability distribution for content popularity
    ranks = np.arange(1, num_contents + 1)
    probs = (ranks + 1) ** (-alpha)
    return probs / np.sum(probs)

# Deep Q Network for each agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Edge Caching (EC) Server Agent
class ECServerAgent:
    def __init__(self, agent_id, is_leader, state_dim, action_dim, cache_capacity):
        self.agent_id = agent_id
        self.is_leader = is_leader
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cache_capacity = cache_capacity
        
        # Initialize policy network and target network
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Initialize cached content
        self.cached_content = set()
        
        # Initialize game matrix for Stackelberg equilibrium (for follower agents)
        if not is_leader:
            self.game_matrix = np.zeros((action_dim, action_dim))
            
    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, self.action_dim - 1)
    
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        # 转换为numpy数组再转为tensor以提高性能
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(device)
        
        # Compute Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        if self.is_leader:
            # For leader agent (Nash equilibrium)
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        else:
            # For follower agent (Stackelberg equilibrium)
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute target Q values
        target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
        
        # Compute loss
        loss = F.mse_loss(q_values.squeeze(), target_q_values)
        
        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_cache(self, content, action_type):
        # "all cache", "keep", "cache local requests", "cache neighbor requests", "pre-caching"
        if action_type == 0:  # "all cache"
            if len(self.cached_content) >= self.cache_capacity:
                # Remove least popular content
                self.cached_content.pop()
            self.cached_content.add(content)
        elif action_type == 1:  # "keep"
            pass  # Do nothing
        elif action_type == 2:  # "cache local requests"
            if content[0] == self.agent_id:  # If content was requested by users of this agent
                if len(self.cached_content) >= self.cache_capacity:
                    # Remove least popular content
                    self.cached_content.pop()
                self.cached_content.add(content)
        elif action_type == 3:  # "cache neighbor requests"
            if content[0] != self.agent_id:  # If content was requested by users of other agents
                if len(self.cached_content) >= self.cache_capacity:
                    # Remove least popular content
                    self.cached_content.pop()
                self.cached_content.add(content)
        elif action_type == 4:  # "pre-caching"
            # Pre-cache most popular content
            if len(self.cached_content) >= self.cache_capacity:
                # Remove least popular content
                self.cached_content.pop()
            self.cached_content.add(content)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Network Environment for Edge Caching System
class RailwayEdgeCachingEnvironment:
    def __init__(self, num_agents, num_contents, content_sizes, cache_capacities, zipf_param=0.73):
        self.num_agents = num_agents
        self.num_contents = num_contents
        self.content_sizes = content_sizes  # Size of each content in MB
        self.cache_capacities = cache_capacities  # Cache capacity of each agent in MB
        
        # Network topology setup
        self.graph = self._create_network_topology()
        
        # Content popularity from Zipf distribution
        self.content_popularity = zipf_distribution(num_contents, zipf_param)
        
        # Cloud server always has all content
        self.cloud_server = set(range(num_contents))
        
        # Communication costs
        self.setup_communication_costs()
        
        # Determine leader and follower agents using closeness centrality
        self.leader_id, self.follower_ids = self._determine_hierarchy()
        
        # State dimensions: [user requests, neighbor requests, cached content]
        self.state_dim = 3 * num_contents
        
        # Action space: "all cache", "keep", "cache local requests", "cache neighbor requests", "pre-caching"
        self.action_dim = 5
        
        # Initialize agent objects
        self.agents = {}
        for i in range(num_agents):
            is_leader = (i == self.leader_id)
            self.agents[i] = ECServerAgent(i, is_leader, self.state_dim, self.action_dim, self.cache_capacities[i])
    
    def _create_network_topology(self):
        # Create a railway-like topology
        # Simplifying as a line topology for railway stations
        G = nx.Graph()
        for i in range(self.num_agents):
            G.add_node(i)
        
        # Connect agents in a line (railway-like)
        for i in range(self.num_agents - 1):
            G.add_edge(i, i + 1, weight=1)
        
        # Add some cross-connections to make it more realistic
        if self.num_agents > 3:
            G.add_edge(0, 2, weight=2)  # Example additional connection
        
        return G
    
    def _determine_hierarchy(self):
        # Determine hierarchy using closeness centrality
        closeness = nx.closeness_centrality(self.graph)
        
        # Find the node with highest closeness centrality as the leader
        leader_id = max(closeness, key=closeness.get)
        follower_ids = [i for i in range(self.num_agents) if i != leader_id]
        
        return leader_id, follower_ids
    
    def setup_communication_costs(self):
        # Setup communication costs based on distance
        self.comm_costs = {}
        
        # Distance between nodes in the graph
        distances = dict(nx.all_pairs_shortest_path_length(self.graph))
        
        # Set communication costs proportional to distance
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    self.comm_costs[(i, j)] = 0
                else:
                    self.comm_costs[(i, j)] = 0.05 * distances[i][j]  # 0.05$/MB as in the paper
        
        # Cost to communicate with cloud server (higher than inter-agent)
        self.cloud_cost = 0.1  # 0.1$/MB
    
    def _get_content_request(self):
        # Generate a content request based on Zipf distribution
        content_id = np.random.choice(range(self.num_contents), p=self.content_popularity)
        agent_id = np.random.choice(range(self.num_agents))  # Random agent for the request
        return content_id, agent_id
    
    def _get_state(self, agent_id):
        # Create state representation for an agent
        # State consists of user requests, neighbor requests, and cached content
        
        # User requests - one-hot encoding for requested content
        user_requests = np.zeros(self.num_contents)
        for _ in range(5):  # Sample 5 requests
            content_id, req_agent_id = self._get_content_request()
            if req_agent_id == agent_id:  # Request from users of this agent
                user_requests[content_id] = 1
        
        # Neighbor requests - one-hot encoding for content requested by neighbors
        neighbor_requests = np.zeros(self.num_contents)
        for neighbor in self.graph.neighbors(agent_id):
            for _ in range(3):  # Sample 3 requests per neighbor
                content_id, req_agent_id = self._get_content_request()
                if req_agent_id == neighbor:  # Request from users of neighbor agent
                    neighbor_requests[content_id] = 1
        
        # Cached content - one-hot encoding for content already cached
        cached_content = np.zeros(self.num_contents)
        for content in self.agents[agent_id].cached_content:
            cached_content[content[0]] = 1
        
        return np.concatenate([user_requests, neighbor_requests, cached_content])
    
    def _compute_reward(self, agent_id, content_request, action):
        # Compute reward based on content delivery cost
        content_id, req_agent_id = content_request
        
        # Check if content is cached locally
        if content_request in self.agents[agent_id].cached_content:
            # Deliver from local cache - lowest cost
            delivery_cost = 0
        else:
            # Check if content is cached in neighboring agents
            found_in_neighbor = False
            for neighbor in self.graph.neighbors(agent_id):
                if content_request in self.agents[neighbor].cached_content:
                    # Deliver from neighbor cache
                    delivery_cost = self.comm_costs[(agent_id, neighbor)] * self.content_sizes[content_id]
                    found_in_neighbor = True
                    break
            
            if not found_in_neighbor:
                # Deliver from cloud server - highest cost
                delivery_cost = self.cloud_cost * self.content_sizes[content_id]
        
        # Reward is negative cost (we want to minimize cost)
        reward = -delivery_cost
        
        return reward, delivery_cost
    
    def step(self, agent_id, action):
        # Process one step for the given agent
        
        # Generate a content request
        content_request = self._get_content_request()
        
        # Compute reward based on content delivery cost
        reward, cost = self._compute_reward(agent_id, content_request, action)
        
        # Update agent's cache based on action
        self.agents[agent_id].update_cache(content_request, action)
        
        # Get next state
        next_state = self._get_state(agent_id)
        
        # Always False for now - episodes end after MAX_STEPS
        done = False
        
        return next_state, reward, done, {"cost": cost}
    
    def reset(self):
        # Reset cached content for all agents
        for agent in self.agents.values():
            agent.cached_content = set()
        
        # Return initial states for all agents
        states = {}
        for agent_id in range(self.num_agents):
            states[agent_id] = self._get_state(agent_id)
        
        return states

# Function to train the HG-MCEC system
def train_hg_mcec(env, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS):
    # Store metrics for evaluation
    leader_rewards = []
    follower_rewards = []
    total_costs = []
    hit_rates = []
    
    # For tracking training time
    start_time = time.time()
    
    # Training loop
    for episode in range(num_episodes):
        try:
            # Reset environment
            states = env.reset()
            
            # Epsilon for exploration (decaying)
            epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / EPS_DECAY)
            
            # Initialize metrics for this episode
            episode_rewards = {agent_id: 0 for agent_id in range(env.num_agents)}
            episode_costs = {agent_id: 0 for agent_id in range(env.num_agents)}
            episode_hits = {agent_id: 0 for agent_id in range(env.num_agents)}
            episode_requests = {agent_id: 0 for agent_id in range(env.num_agents)}
            
            # Process leader agent first, then follower agents
            for step in range(max_steps):
                # Leader agent's turn
                leader_id = env.leader_id
                leader_action = env.agents[leader_id].select_action(states[leader_id], epsilon)
                next_state, reward, done, info = env.step(leader_id, leader_action)
                
                # Store leader's experience
                env.agents[leader_id].memory.push(
                    states[leader_id], 
                    leader_action, 
                    reward, 
                    next_state, 
                    done
                )
                
                # Update leader's state
                states[leader_id] = next_state
                
                # Update metrics
                episode_rewards[leader_id] += reward
                episode_costs[leader_id] += info["cost"]
                episode_requests[leader_id] += 1
                if info["cost"] == 0:  # Hit if cost is 0
                    episode_hits[leader_id] += 1
                
                # Follower agents' turns
                for follower_id in env.follower_ids:
                    follower_action = env.agents[follower_id].select_action(states[follower_id], epsilon)
                    next_state, reward, done, info = env.step(follower_id, follower_action)
                    
                    # Store follower's experience
                    env.agents[follower_id].memory.push(
                        states[follower_id], 
                        follower_action, 
                        reward, 
                        next_state, 
                        done
                    )
                    
                    # Update follower's state
                    states[follower_id] = next_state
                    
                    # Update metrics
                    episode_rewards[follower_id] += reward
                    episode_costs[follower_id] += info["cost"]
                    episode_requests[follower_id] += 1
                    if info["cost"] == 0:  # Hit if cost is 0
                        episode_hits[follower_id] += 1
                
                # Optimize all agents
                for agent_id in range(env.num_agents):
                    env.agents[agent_id].optimize()
                
                if done:
                    break
            
            # Update target networks periodically
            if episode % TARGET_UPDATE == 0:
                for agent_id in range(env.num_agents):
                    env.agents[agent_id].update_target_network()
            
            # Calculate metrics for this episode
            leader_reward = episode_rewards[env.leader_id]
            follower_reward = sum(episode_rewards[f_id] for f_id in env.follower_ids) / len(env.follower_ids)
            total_cost = sum(episode_costs.values())
            
            # Calculate hit rates
            hit_rate = {}
            for agent_id in range(env.num_agents):
                if episode_requests[agent_id] > 0:
                    hit_rate[agent_id] = episode_hits[agent_id] / episode_requests[agent_id]
                else:
                    hit_rate[agent_id] = 0
            
            # Store metrics
            leader_rewards.append(leader_reward)
            follower_rewards.append(follower_reward)
            total_costs.append(total_cost)
            hit_rates.append(hit_rate)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode+1}/{num_episodes}, Leader Reward: {leader_reward:.2f}, "
                      f"Avg Follower Reward: {follower_reward:.2f}, Total Cost: {total_cost:.2f}, "
                      f"Leader Hit Rate: {hit_rate[env.leader_id]:.2f}, Time: {elapsed_time:.1f}s")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Returning current progress...")
            break
    
    print(f"Training completed in {time.time() - start_time:.1f} seconds")
    return leader_rewards, follower_rewards, total_costs, hit_rates

# Function to evaluate different caching strategies
def evaluate_caching_strategies(num_agents=3, num_contents=800, zipf_param=0.73, cache_capacity=100):
    # Create content sizes (1-3 MB as in the paper)
    content_sizes = np.random.uniform(1, 3, num_contents)
    
    # Set cache capacities (100 MB as default)
    cache_capacities = [cache_capacity] * num_agents
    
    # Create environment
    env = RailwayEdgeCachingEnvironment(num_agents, num_contents, content_sizes, cache_capacities, zipf_param)
    
    # Train HG-MCEC
    print("Training HG-MCEC...")
    leader_rewards, follower_rewards, total_costs, hit_rates = train_hg_mcec(env)
    
    # Evaluate other baseline strategies (simplified simulation)
    # In a full implementation, these would be separate classes with their own training loops
    
    # Plot results
    plot_results(leader_rewards, follower_rewards, total_costs, hit_rates, env.leader_id)
    
    return env

# Function to plot the training results
def plot_results(leader_rewards, follower_rewards, total_costs, hit_rates, leader_id):
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot leader rewards
    axes[0, 0].plot(leader_rewards)
    axes[0, 0].set_title('Leader Agent Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Plot follower rewards
    axes[0, 1].plot(follower_rewards)
    axes[0, 1].set_title('Average Follower Agent Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    
    # Plot total costs
    axes[1, 0].plot(total_costs)
    axes[1, 0].set_title('Total System Cost')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cost')
    
    # Plot hit rates
    leader_hit_rates = [hr[leader_id] for hr in hit_rates]
    follower_hit_rates = []
    for i in range(len(hit_rates[0])):
        if i != leader_id:
            follower_hit_rates.append([hr[i] for hr in hit_rates])
    
    axes[1, 1].plot(leader_hit_rates, label='Leader')
    for i, fhr in enumerate(follower_hit_rates):
        axes[1, 1].plot(fhr, label=f'Follower {i+1}')
    axes[1, 1].set_title('Hit Rates')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Hit Rate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('hg_mcec_results.png')
    plt.show()

# Function to test different parameters based on the paper's experiments
def experiment_zipf_parameters():
    zipf_params = [0.5, 0.73, 1.0, 1.5, 2.0]
    results = {}
    
    for param in zipf_params:
        print(f"Running experiment with Zipf parameter = {param}")
        env = RailwayEdgeCachingEnvironment(
            num_agents=3, 
            num_contents=800, 
            content_sizes=np.random.uniform(1, 3, 800),
            cache_capacities=[100, 100, 100],
            zipf_param=param
        )
        
        leader_rewards, follower_rewards, total_costs, hit_rates = train_hg_mcec(
            env, 
            num_episodes=100,  # Reduced for faster experimentation
            max_steps=50
        )
        
        # Store the final hit rates
        results[param] = {
            'leader_hit_rate': hit_rates[-1][env.leader_id],
            'follower_hit_rates': [hit_rates[-1][f_id] for f_id in env.follower_ids],
            'total_cost': total_costs[-1]
        }
    
    return results

def experiment_request_numbers():
    request_numbers = [5, 20, 40, 60, 80, 100]
    # This would modify how many requests are processed each step
    # Implementation would need to modify the step function
    
def experiment_cache_capacities():
    cache_capacities = [40, 60, 80, 100, 120]
    results = {}
    
    for capacity in cache_capacities:
        print(f"Running experiment with cache capacity = {capacity}")
        env = RailwayEdgeCachingEnvironment(
            num_agents=3, 
            num_contents=800, 
            content_sizes=np.random.uniform(1, 3, 800),
            cache_capacities=[capacity, capacity, capacity],
            zipf_param=0.73
        )
        
        leader_rewards, follower_rewards, total_costs, hit_rates = train_hg_mcec(
            env, 
            num_episodes=100,  # Reduced for faster experimentation
            max_steps=50
        )
        
        # Store the final hit rates and costs
        results[capacity] = {
            'leader_hit_rate': hit_rates[-1][env.leader_id],
            'follower_hit_rates': [hit_rates[-1][f_id] for f_id in env.follower_ids],
            'total_cost': total_costs[-1]
        }
    
    return results

# Main function to run the simulation
def main():
    # Run a basic simulation
    env = evaluate_caching_strategies()
    
    # Run experiments with different parameters
    # Uncomment to run specific experiments
    # zipf_results = experiment_zipf_parameters()
    # capacity_results = experiment_cache_capacities()
    
    return env

if __name__ == "__main__":
    main()
