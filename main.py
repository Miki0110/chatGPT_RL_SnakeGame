import numpy as np
from snake_class import SnakeGame
import pickle

q_states = ["Danger straight", "Danger right", "Danger Left", "left", "right", "up", "down", "food left", "food right", "food up", "food down"]
moves = ["left", "right", "up", "down"]
actions = ["straight", "right turn", "left turn"]
num_states = 2**len(q_states)
num_episodes = 1000  # Amount of lives
t_HZ = -1  # set to -1 if it should not wait at all
p_HZ = 30  # Speed it will play after being done training
prev_distance = 2000


# Function for randomly taking an action based on the exploration probability
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.randint(len(actions))
    else:
        # Exploit: choose the action with the highest Q-value
        return np.random.choice(np.where(q_table[state, :] == np.max(q_table[state, :]))[0])


# Function for randomly taking an action based on the softmax approach
def softmax_choose_action(state, q_table, tau):
    action_prob = np.exp(q_table[state, :] / tau) / np.sum(np.exp(q_table[state, :] / tau))
    return np.random.choice(range(len(actions)), p=action_prob)


# Function for taking actions in the snake game
def take_action(snake, action, speed):
    global prev_distance

    snake.q_action(action)
    snake.iterate_game(speed)  # input is wait time in hz

    current_state, distance, event = snake.get_state_qmodel()
    current_state = int("".join(map(str, current_state)), 2)

    # Check if the snake failed
    if event < 0:
        reward = -100  # failed
        #print("failed")
        return current_state, reward, True
    elif event > 0:
        reward = 100  # got an apple
    elif prev_distance == 2000:
        reward = 0  # Nothing happened
    elif distance > prev_distance:
        #print("bad")
        reward = -1
    else:
        #print("better")
        reward = 1

    prev_distance = distance
    return current_state, reward, False


def train_model(n_episodes, start_epsilon, alpha, gamma, start_table):
    q_table = start_table
    decay_rate = 0.001 # Bot recommended to start the decay rate at 0.001 and try making it smaller
    start_tau = 1
    # Loop through the game equal to the amount defined
    for episode in range(n_episodes):
        print(f"Episode = {episode+1}")
        # The more the snake trains the less random should it be
        #epsilon = start_epsilon*np.exp(-decay_rate*episode)

        tau = start_tau * np.exp(-decay_rate * episode)

        # Initialise the game
        snake = SnakeGame()
        snake.new_game()

        # Get starting state
        current_state, _, _ = snake.get_state_qmodel()
        current_state = int("".join(map(str, current_state)), 2)
        while True:
            # Choose an action according to some policy
            current_action = softmax_choose_action(current_state, q_table, tau)

            # Take the action and observe new state
            new_state, reward, completion = take_action(snake, current_action, t_HZ)

            # Update the Q-value for the current state-action pair using the Q-learning rule
            q_table[current_state, current_action] += alpha * (
                        reward + gamma * np.max(q_table[new_state, :]) - q_table[current_state, current_action])

            # Update the current state and check if the episode is finished
            current_state = new_state
            if completion:
                break
    # When it's done training return the trained table
    return q_table


# MAIN LOOP
if __name__ == "__main__":
    # Define the exploration probability (epsilon)
    start_epsilon = 0.1
    # Initialize the Q-table with small random values
    q_table = np.random.rand(num_states, len(actions)) * 0.1  # States = rows, Actions = columns

    # Define the learning rate and discount factor
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor / how great are rewards

    q_table = train_model(num_episodes, start_epsilon, alpha, gamma, q_table)

    # Last snake game without any randomness
    snake = SnakeGame("SNAKE")
    snake.new_game()

    # Get starting state
    current_state, _, _ = snake.get_state_qmodel()
    current_state = int("".join(map(str, current_state)), 2)
    while True:
        # Choose an action according to some policy
        current_action = choose_action(current_state, q_table, 0)

        # Take the action and observe new state
        new_state, _, completion = take_action(snake, current_action, p_HZ)

        # Update the current state and check if the episode is finished
        current_state = new_state
        if completion:
            break
    snake.end_game()

    # Lastly the trained model is saved
    with open('trained_model.pkl', 'wb') as f:
        # Pickle the array and write it to the file
        pickle.dump(q_table, f)
