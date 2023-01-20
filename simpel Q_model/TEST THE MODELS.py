import multiprocessing as mp
import numpy as np
import pickle
from pathlib import Path
import os
from snake_class import SnakeGame


actions = ["straight", "right turn", "left turn"]
p_HZ = 20  # Speed it will play
prev_distance = 2000

# Function for randomly taking an action based on the exploration probability
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.randint(len(actions))
    else:
        # Exploit: choose the action with the highest Q-value
        return np.random.choice(np.where(q_table[state, :] == np.max(q_table[state, :]))[0])


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


def run_model(model_file):
    # Load the model from the pickle file
    with open(model_file, 'rb') as f:
        q_table = pickle.load(f)

    model_name = os.path.basename(model_file).split('.')[0]

    # Initialize pygame and run the game using the loaded model
    snake = SnakeGame(model_name)
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


if __name__ == '__main__':
    # Get the current script's location
    script_dir = Path(__file__).parent
    # The folder containing the pickle files is located in the same directory as the script.
    folder = script_dir / 'trained_models'
    print(folder)
    # Get a list of all pickle files in the folder
    model_files = [p for p in folder.glob('*.pkl') if p.is_file()]
    print(model_files)

    # Create a list of processes to run the models
    processes = [mp.Process(target=run_model, args=(model_file,)) for model_file in model_files]

    # Start the processes
    for p in processes:
        p.start()

    # Wait for the processes to complete
    for p in processes:
        p.join()