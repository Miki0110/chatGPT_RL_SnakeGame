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
        self.model = QNetwork(len(q_states), 256, len(actions)).share_memory()
        self.trainer = QLearning(self.model, learning_rate=alpha, discount_factor=self.gamma)

        # Plotting stuff
        #
        # labels
        #plt.xlabel('Episodes')
        #plt.ylabel('Scores')
        # Legends
        #plt.plot(0, 0, 'b-', label='score')
        #plt.plot(0, 0, 'r-', label='mean score')
        # plot legend
        #plt.legend()
        self.scores = []
        self.mean_scores = []


    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.episode
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cpu()
            prediction = self.model(state0).cpu()  # prediction by model
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

    def plot_data(self):
        episodes = range(1, len(self.scores) + 1)
        # plot data
        plt.ion()
        plt.show()
        plt.plot(episodes, self.scores, 'b-', label='score')
        plt.plot(episodes, self.mean_scores, 'r-', label='mean score')

        # show plot
        plt.draw()
        plt.pause(0.001)

    def train_model(self, plot, name):
        # Init values for tracking

        record = 0

        # Start the snake game
        snake = SnakeGame(name, 1000, 800)
        # Initialise the game
        snake.new_game()
        while True:
            # Get starting state
            state_old, _, _, _ = snake.get_state_qmodel()

            # Get the calculated action
            calc_action = self.get_action(state_old)

            # Proceed with the game
            state_new, reward, completion, new_score = take_action(snake, calc_action)
            if not completion:
                score = new_score

            # Train the sucker
            self.train_short_memory(state_old, calc_action, reward, state_new, completion)

            # remember
            self.remember(state_old, calc_action, reward, state_new, completion)

            if completion:
                self.episode += 1
                self.train_long_memory()
                if score > record:
                    record = score
                    self.model.save("DQN_model.pth")
                print("----------------------------------------------")
                print(f'Game:{self.episode}, Score:{score}, Record:{record}')
                print("----------------------------------------------")
                if plot:
                    self.scores.append(int(score))
                    mean_score = np.mean(self.scores)
                    self.mean_scores.append(mean_score)
                    self.plot_data()


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


# MAIN LOOP
if __name__ == "__main__":
    # Initialise the agent
    agent = Agent(0.01, 0.9, 5000)

    num_processes = 4
    processes = []

    # Start the first process with plots
    p = mp.Process(target=agent.train_model, args=(True, "Trainer plotter",))
    p.start()
    processes.append(p)
    for rank in range(num_processes-1):
        p = mp.Process(target=agent.train_model, args=(False, "Trainer", ))
        p.start()
        processes.append(p)
    # Wait for them to finish
    for p in processes:
        p.join()
        #agent.train_model(1000)