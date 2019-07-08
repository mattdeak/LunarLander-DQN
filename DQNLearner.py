from utils import ReplayTable, EpsilonDecay
from tqdm import tqdm
import numpy as np

class DQNLearner:
    
    def __init__(self, environment, keras_net, replay_table, gamma=0.9, epsilon=EpsilonDecay()):
        self.environment = environment
        self.action_space = environment.action_space
        self.net = keras_net
        self.replay = replay_table
        self.gamma = gamma
        self.epsilon = epsilon
        self.search_method = self._epsilon_greedy
        
    def choose_action(self, observation):
        return self.search_method(observation)
    
    def get_best_action(self, observation):
        return np.argmax(self.net.predict(observation.reshape(1, -1)))
    
    def get_best_q(self, observation):
        return np.max(self.net.predict(observation.reshape(1, -1)))
    
    def update_replay(self, s, a, r, s_prime, done=False):
        
        if done:
            v_s_prime = 0
        else:
            v_s_prime = self.get_best_q(s_prime)
            
        # Store state in replay table
        self.replay.insert(s, a, r, s_prime, done)
        
    def train(self, batch_size=16):
        
        states, actions, rewards, next_states, done = self.replay.sample(batch_size)
        
        target = self.net.predict(states)
        
        next_qs = np.zeros(done.shape[0])
        next_qs[done != 1] = self.net.predict(next_states[done != 1, :]).max(axis=1)
        
        # Exctracing the q values at the actions we took
        
        # Adjust targets at the actions we took
        target[np.arange(len(states)), actions] = (
            rewards 
            + self.gamma*next_qs)
        
        self.net.fit(states, target, verbose=0)
        
    def run_episode(self, store_data=True, training=True, sample_threshold=500, max_episode_length=500, timeout_reward=-100):
        observation = self.environment.reset()
        done = False
        current_episode = 0
        rewards = []
        while not done and current_episode < sample_threshold:
            if training:
                action = self.choose_action(observation)
            else:
                action = self.get_best_action(observation)
                
            next_state, reward, done, _ = self.environment.step(action)
            rewards.append(reward)

            if current_episode > max_episode_length and store_data:
                self.update_replay(observation, action, timeout_reward, next_state, done=True)
            
            if store_data:
                self.update_replay(observation, action, reward, next_state, done=done)
            
            if training and len(self.replay) > sample_threshold:
                self.train()
                
            observation = next_state
            current_episode += 1

        return rewards

          
                
    def simulate(self, n_episodes, train=True, show_progress=True, max_episode_length=3000):
        rewards = []
        if show_progress:
            for i in tqdm(range(n_episodes)):
                episode_r = self.run_episode(training=train, max_episode_length=max_episode_length)
                rewards.append(episode_r)
        else:
            for i in range(n_episodes):
                episode_r = self.run_episode(training=train, max_episode_length=max_episode_length)
                rewards.append(episode_r)
        
        return rewards
            
    def _epsilon_greedy(self, observation):
        if np.random.rand() < self.epsilon.get():
            return self.action_space.sample()
        
        else:
            return self.get_best_action(observation)
