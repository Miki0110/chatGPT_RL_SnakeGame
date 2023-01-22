import numpy as np
from snake_class import SnakeGame
from deep_q_model import QLearning, QNetwork
from collections import deque
import torch
import random
import matplotlib.pyplot as plt
import time

prev_distance = 2000
BATCH_SIZE = 5000

q_states = ["Danger straight", "Danger right", "Danger Left", #"Danger Diagonal Right", "Danger Diagonal Left",
            "left", "right", "up", "down",
            "food left", "food right", "food up", "food down"]
actions = ["straight", "right turn", "left turn"]

class Agent:
    def __init__(self, alpha, gamma, model, PLOT_BOOL):
        self.episode = 0  # Amount of lives
        self.epsilon = 0.15  # Chance for random inputs
        self.gamma = gamma
        self.memory = deque(maxlen=100_000)  # popleft()
        self.BATCH_SIZE = 5000
        self.decay_rate = 0.01
        self.plot = PLOT_BOOL

        # Init the model and trainer
        self.model = model
        self.trainer = QLearning(self.model, learning_rate=alpha, discount_factor=self.gamma)

        if self.plot:
            # Plotting stuff
            plt.ion()
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
        if self.episode < 400:
            epsilon = self.epsilon*np.exp(-self.decay_rate*self.episode)
        else:
            epsilon = 0
        #print(epsilon)

        final_move = [0, 0, 0]
        if np.random.rand() < epsilon:
            move = np.random.randint(3)
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
        next_states = np.array(next_states)
        states = np.array(states)
        actions = np.array(actions)

        self.trainer.train_step(states, actions, np.array(rewards, dtype=int), next_states, np.array(dones, dtype=int))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def plot_data(self, scores, mean_scores):
        episodes = range(1, len(scores) + 1)
        # plot data
        plt.plot(episodes, scores, 'b-', label='score')
        plt.plot(episodes, mean_scores, 'r-', label='mean score')

        # show plot
        plt.draw()


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
    elif distance > prev_distance:
        reward = -1
    elif distance == prev_distance or prev_distance == 2000:
        reward = 0  # Nothing happened
    else:
        reward = 1

    prev_distance = distance
    return current_state, reward, False, score


def train_model(n_snakes):
    # Init values for tracking
    snakes = []
    agents = []
    scores = []
    mean_scores = []
    record = 0
    # Initiate the model
    model = QNetwork(len(q_states), 256, len(actions))
    # Initialise the first agent -> this is the one we plot through
    agents.append(Agent(0.01, 0.9, model, True))
    # Start the snake game
    snakes.append(SnakeGame("Training plotter", 1000, 800, True, view_distance=1))
    for n in range(n_snakes-1):
        agents.append(Agent(0.01, 0.9, model, False))
        snakes.append(SnakeGame("Training", 1000, 800, False, view_distance=1))

    # Initialise the game
    for snake in snakes:
        snake.new_game()

    while True:
        for snake, agent in zip(snakes, agents):
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
                snake.new_game()
                # check for debug
                agent.trainer.debug = snake.debug

                agent.episode += 1
                agent.train_long_memory()
                if score > record:
                    record = score
                    agent.model.save("DQN_model.pth")
                scores.append(score)
                mean_score = np.mean(scores)
                mean_scores.append(mean_score)

                if agent.plot:
                    print("----------------------------------------------")
                    print(f'Game:{len(scores)}, Score:{score}, Record:{record}')
                    print("----------------------------------------------")
                    agent.plot_data(scores, mean_scores)


# MAIN LOOP
if __name__ == "__main__":
    train_model(1)