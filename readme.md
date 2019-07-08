## File and Class Descriptions
* `utils.py`: Contains code for the `ReplayTable` and `EpsilonDecay`. `ReplayTable` is a utility class for storing samples in a replay buffer.  `EpsilonDecay` is a utility class for a self-decaying epsilon value.
* `DQNLearner.py`: Contains code for the DQNLearner. The DQNLearner takes as arguments:
  * `environment` - An OpenAI Gym environment.
  * `keras_net` - A neural network implemented in Keras.
  * `replay_table` - A replay table as defined by the `ReplayTable` class in `utils.py`
  * `gamma` - A value for the discount rate between 0 and 1. (Optional, default=0.9)
  * `epsilon` - An `EpsilonDecay` instance. Defaults to the default `EpsilonDecay`.
 
Experiments were run in `DQN Data Collection.ipynb` and graphs were generated in `Graph Generation.ipynb`
