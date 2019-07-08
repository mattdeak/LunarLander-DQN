import numpy as np
class EpsilonDecay:
    
    def __init__(self, initial_val=1, decay_rate=0.001, min_val=0.05):
        self.val = initial_val
        self.decay_multiplier = (1 - decay_rate)
        self.min_val = min_val
        
    def get(self):
        initial = self.val
        self.val = max(self.val * self.decay_multiplier, self.min_val)
        return initial
        
class ReplayTable:
    
    def __init__(self, n_features, max_size=10000):
        self._state_table = np.zeros((max_size, n_features))
        self._action_table = np.zeros(max_size, dtype=np.int32)
        self._s_prime_table = np.zeros((max_size, n_features))
        self._reward_table = np.zeros(max_size)
        self._done_table = np.zeros(max_size)
        
        self._max_size = max_size
        self._size = 0
        self._current_index = 0
        
    def sample(self, n):
        ids = np.random.choice(np.arange(self._size), min(n, self._size), replace=False)
        
        return self._state_table[ids], self._action_table[ids], self._reward_table[ids],  self._s_prime_table[ids], self._done_table[ids]
        
    def insert(self, state, action, reward, s_prime, done):
        
        self._state_table[self._current_index] = state
        self._reward_table[self._current_index] = reward
        self._action_table[self._current_index] = action
        self._s_prime_table[self._current_index] = s_prime
        
        if done:
            self._done_table[self._current_index] = 1
        else:
            self._done_table[self._current_index] = 0
        
        # Increment the size
        if self._size < self._max_size:
            self._size += 1
        
        self._current_index = (self._current_index + 1) % self._max_size
        

    def __len__(self):
        return self._size