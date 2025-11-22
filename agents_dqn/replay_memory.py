"""
This module defines the ReplayMemory class used by the DQN agent.

Replay Memory is a key component that stores the agent's experiences,
allowing it to learn from a diverse batch of past events instead of just
the most recent ones, which helps stabilize training.
"""
import random
from collections import namedtuple, deque

# A namedtuple to define the structure for a single experience.
# This makes the code more readable, allowing access to fields by name
# (e.g., transition.state) instead of by index.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """
    A simple cyclic buffer that stores transitions observed by the agent.

    By sampling from this memory, we break the correlation between consecutive
    experiences, a crucial technique for stabilizing the DQN training process.
    """
    def __init__(self, capacity: int):
        """
        Initializes the memory buffer.

        Args:
            capacity: The maximum number of transitions to store. Once the
                      limit is reached, older transitions are automatically
                      discarded when new ones are added.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Saves a new transition to the memory.

        Args:
            *args: The components of the transition, expected in the order of
                   (state, action, next_state, reward, done).
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        """
        Selects a random batch of transitions from memory for training.

        Args:
            batch_size: The number of transitions to select.

        Returns:
            A list containing `batch_size` random transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current number of transitions stored in memory.

        This allows using the `len()` function directly on a ReplayMemory object.
        """
        return len(self.memory)