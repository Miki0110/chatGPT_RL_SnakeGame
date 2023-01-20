import numpy as np
from snake_class import SnakeGame
from q_model import Qmodel
import pickle

q_states = ["Danger straight", "Danger right", "Danger Left", "left", "right", "up", "down", "food left", "food right", "food up", "food down"]
moves = ["left", "right", "up", "down"]
actions = ["straight", "right turn", "left turn"]

num_states = 2**len(q_states)
num_episodes = 1000  # Amount of lives
t_HZ = -1  # set to -1 if it should not wait at all
p_HZ = 30  # Speed it will play after being done training
prev_distance = 2000


# Function for taking actions in the snake game
def take_action(snake, action, speed):
    global prev_distance

    snake.q_action(action)
    snake.iterate_game(speed)  # input is wait time in hz

    # Get state from the game
    current_state, distance, event = snake.get_state_qmodel()

    # Check if the snake failed
    if event < 0:
        reward = -100  # failed
        # print("failed")
        return current_state, reward, True
    elif event > 0:
        reward = 100  # got an apple
    elif prev_distance == 2000:
        reward = 0  # Nothing happened
    elif distance > prev_distance:
        # print("bad")
        reward = 0
    else:
        # print("better")
        reward = 0

    prev_distance = distance
    return current_state, reward, False


def train_model(n_episodes, softmax, start_value, model):
    decay_rate = 0.001  # Bot recommended to start the decay rate at 0.001 and try making it smaller

    # Loop through the game equal to the amount defined
    for episode in range(n_episodes):
        print(f"Episode = {episode + 1}")
        # The more the snake trains the less random should it be
        if softmax:
            tau = start_value * np.exp(-decay_rate * episode)
        else:
            epsilon = start_value*np.exp(-decay_rate*episode)


        # Initialise the game
        snake = SnakeGame("Training")
        snake.new_game()

        # Get starting state
        state_values, _, _ = snake.get_state_qmodel()
        model.set_state(state_values)

        while True:
            # Choose an action according to some policy
            if softmax:
                predicted_action = model.softmax_choose_action(tau)
            else:
                predicted_action = model.choose_action(epsilon)

            # Take the action and observe new state
            state_values, reward, completion = take_action(snake, predicted_action, t_HZ)

            #Update the q_table
            model.update_table(reward, state_values, predicted_action)

            # Update the current state and check if the episode is finished
            model.set_state(state_values)
            if completion:
                break
    return


# MAIN LOOP
if __name__ == "__main__":
    # Define the exploration probability (epsilon)
    start_epsilon = 0.1
    start_tau = 1

    # Define the learning rate and discount factor
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor / how great are rewards

    # Start the model
    qmodel = Qmodel(num_states, len(actions), alpha, gamma)

    # softmax
    train_model(num_episodes, True, start_tau, qmodel)

    # greedy approach
    #train_model(num_episodes, False, start_epsilon, qmodel)

    # Last snake game without any randomness
    snake = SnakeGame("SNAKE")
    snake.new_game()

    # Get starting state
    state_values, _, _ = snake.get_state_qmodel()
    qmodel.set_state(state_values)
    while True:
        # Choose an action according to some policy
        current_action = qmodel.choose_action(0)  # Zero means it only does what the model predicts

        # Take the action and observe new state
        state_values, _, completion = take_action(snake, current_action, p_HZ)

        # Update the current state and check if the episode is finished
        qmodel.set_state(state_values)
        if completion:
            break
    snake.end_game()

    # Lastly the trained model is saved
    with open('trained_model.pkl', 'wb') as f:
        # Pickle the array and write it to the file
        pickle.dump(qmodel.q_table, f)