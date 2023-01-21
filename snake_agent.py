import numpy as np
from snake_class import SnakeGame
from deep_q_model import QLearning, QNetwork
from collections import deque
import torch
import torch.multiprocessing as mp
import random
import matplotlib.pyplot as plt

prev_distance = 2000
BATCH_SIZE = 5000

q_states = ["Danger straight", "Danger right", "Danger Left", "left", "right", "up", "down", "food left",
            "food right", "food up", "food down"]
actions = ["straight", "right turn", "left turn"]

class Agent:
    def __init__(self, alpha, gamma, batch_size):
        self.episode = 0  # Amount of lives
        self.epsilon = 0  # Chance for random inputs
        self.gamma = gamma
        self.memory = deque(maxlen=100_000)  # popleft()
        self.BATCH_SIZE = batch_size

        # Init the model and trainer
        self.model = QNetwork(len(q_states), 256, len(actions))
        self.trainer = QLearning(self.model, learning_rate=alpha, discount_factor=self.gamma)

        # Plotting stuff
        plt.ion()
        plt.show()
        # labels
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        # Legends
        plt.plot(0, 0, 'b-', label='score')
        plt.plot(0, 0, 'r-', label='mean score')
        # plot legend
        plt.legend()


    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.episode
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()  # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > self.BATCH_SIZE):
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def plot_data(self, scores, mean_scores):
        episodes = range(1, len(scores) + 1)
        # plot data
        plt.plot(episodes, scores, 'b-', label='score')
        plt.plot(episodes, mean_scores, 'r-', label='mean score')

        # show plot
        plt.draw()
        plt.pause(0.001)


# Function for taking actions in the snake game
def take_action(snake, action):
    global prev_distance

    snake.q_action(action)
    snake.iterate_game()  # input is wait time in hz

    # Get state from the game
    current_state, distance, event, score = snake.get_state_qmodel()

    # Check if the snake failed
    if event < 0:
        reward = -100  # failed
        return current_state, reward, True, score
    elif event > 0:
        reward = 100  # got an apple
    elif prev_distance == 2000:
        reward = 0  # Nothing happened
    elif distance > prev_distance:
        # print("bad")
        reward = -1
    else:
        # print("better")
        reward = 1

    prev_distance = distance
    return current_state, reward, False, score


def train_model(n_episodes):
    # Init values for tracking
    scores = []
    mean_scores = []
    record = 0
    # Initialise the agent
    agent = Agent(0.01, 0.9, n_episodes)
    # Start the snake game
    snake = SnakeGame("Training", 1000, 800)
    # Initialise the game
    snake.new_game()
    while True:
        # Get starting state
        state_old, _, _, _ = snake.get_state_qmodel()

        # Get the calculated action
        calc_action = agent.get_action(state_old)

        # Proceed with the game
        state_new, reward, completion, new_score = take_action(snake, calc_action)
        if not completion:
            score = new_score

        # Train the sucker
        agent.train_short_memory(state_old, calc_action, reward, state_new, completion)

        # remember
        agent.remember(state_old, calc_action, reward, state_new, completion)

        if completion:
            agent.episode += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save("DQN_model.pth")
            print("----------------------------------------------")
            print(f'Game:{agent.episode}, Score:{score}, Record:{record}')
            print("----------------------------------------------")

            scores.append(score)
            mean_score = np.mean(score)
            mean_scores.append(mean_score)
            agent.plot_data(scores, mean_scores)


# MAIN LOOP
if __name__ == "__main__":
    train_model(1000)