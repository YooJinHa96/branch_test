from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        if(self.num_experiences > batch_size):
            return random.sample(self.buffer, batch_size)
        else:
            return random.sample(self.buffer, self.num_experiences)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0