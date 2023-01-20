import numpy as np


class Qmodel:

    def __init__(self, num_states, num_actions, alpha, gamma):
        # Initialize the Q-table with small random values
        self.q_table = np.random.rand(num_states, num_actions) * 0.1  # States = rows, Actions = columns

        # Save possible actions
        self.actions = num_actions

        # Define the learning rate and discount factor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor / how great are rewards
        self.state = 0

    # Function for setting the state
    def set_state(self, state_values):
        self.state = int("".join(map(str, state_values)), 2)

    # Function for randomly taking an action based on the exploration probability
    def choose_action(self, epsilon):
        if np.random.rand() < epsilon:
            # Explore: choose a random action
            return np.random.randint(self.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.random.choice(np.where(self.q_table[self.state, :] == np.max(self.q_table[self.state, :]))[0])

    # Function for randomly taking an action based on the softmax approach
    def softmax_choose_action(self, tau):
        action_prob = np.exp(self.q_table[self.state, :] / tau) / np.sum(np.exp(self.q_table[self.state, :] / tau))
        return np.random.choice(range(self.actions), p=action_prob)

    # Function for updating the q_table
    def update_table(self, reward, new_state, action):
        new_state = int("".join(map(str, new_state)), 2)
        # Update the Q-value for the current state-action pair using the Q-learning rule
        self.q_table[self.state, action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[self.state, action])
