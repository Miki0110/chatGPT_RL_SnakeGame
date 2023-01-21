import pygame
import sys
import numpy as np
import time


class SnakeGame:

    def gen_apple(self):
        # Initialize the apple position
        board = np.zeros((int(self.width // 20), int(self.height // 20)))
        snake = self.snake_body // 20

        # Block out the part of the board the snake occupies
        for x, y in snake:
            board[int(x), int(y)] = 1
        # Find the indices of elements that are 0
        indices = np.where(board == 0)
        indices = np.array(indices).T

        # Select random indices from the list of 0-indices
        random_indices = np.random.choice(indices.shape[0], replace=False, size=1)

        # Get the random indices of elements that are 0
        self.apple = indices[random_indices][0] * 20

    def snake_collision(self):
        # Check if an apple has been reached
        if np.array_equal(self.current_pos, self.apple):
            self.gen_apple()
            self.reward = True
            return
        else:
            # Remove the last square from the snake's body
            self.snake_body = np.delete(self.snake_body, 0, 0)

        # Check for collision with the game window boundaries
        if self.current_pos[0] < 0 or self.current_pos[0] >= self.width\
                or self.current_pos[1] < 0 or self.current_pos[1] >= self.height:
            self.death = True
            #print(f'Model {self.name}:\n reached a length of {len(self.snake_body)}')
            self.new_game()

        # Check for collision with the snake's body
        if any(np.array_equal(sub, self.current_pos) for sub in self.snake_body[:-1]):
            self.death = True
            #print(f'Model {self.name}:\n reached a length of {len(self.snake_body)}')
            self.new_game()

    def update_screen(self):
        self.window.fill((0, 0, 0))

        # Draw the snake
        body_size = 18
        for square in self.snake_body:
            pygame.draw.rect(self.window, (0, 255, 0), (square[0]+1, square[1]+1, body_size, body_size))

        # Draw the eyes
        head = self.snake_body[-1]
        eye_size = 4
        eye_distance = 8
        pupil_size = 2

        eye1_pos = (head[0] + eye_distance, head[1] + eye_size) if self.direction == "up" else (
                    head[0] + eye_distance, head[1] + body_size - eye_size) if self.direction == "down" else (
                    head[0] + eye_size, head[1] + eye_distance) if self.direction == "left" else (
                    head[0] + body_size - eye_size, head[1] + eye_distance)

        eye2_pos = (head[0] + 2 * eye_distance, head[1] + eye_size) if self.direction == "up" else (
                    head[0] + 2 * eye_distance, head[1] + body_size - eye_size) if self.direction == "down" else (
                    head[0] + eye_size, head[1] + 2 * eye_distance) if self.direction == "left" else (
                    head[0] + body_size - eye_size, head[1] + 2 * eye_distance)

        pupil1_pos = (eye1_pos[0] + pupil_size, eye1_pos[1] + pupil_size)
        pupil2_pos = (eye2_pos[0] + pupil_size, eye2_pos[1] + pupil_size)

        pygame.draw.circle(self.window, (0, 0, 0), eye1_pos, eye_size)
        pygame.draw.circle(self.window, (0, 0, 0), eye2_pos, eye_size)
        pygame.draw.circle(self.window, (255, 255, 255), pupil1_pos, pupil_size)
        pygame.draw.circle(self.window, (255, 255, 255), pupil2_pos, pupil_size)

        # Draw the apple
        pygame.draw.circle(self.window, (255, 0, 0), (self.apple[0]+10, self.apple[1]+10), 10)

        # Update the display
        pygame.display.update()

    def __init__(self, model_name, width, height):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption(model_name)
        self.name = model_name
        self.update_rate = 50

        # Define the game window size
        self.width = width
        self.height = height
        self.square_size = 20

        # Initialize the game window
        self.window = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF, 32)

        # Initialize values
        self.direction = "left"
        self.actions = ["left", "right", "up", "down"]
        self.death = False
        self.reward = False
        self.debug = False
        self.display = False

    def new_game(self):
        # Define the snake's initial position and movement speed
        self.current_pos = np.array([self.width // 2, self.height // 2])

        # Initialize the snake's body as a list of squares
        self.snake_body = np.array([self.current_pos])

        # Set a starting apple
        self.gen_apple()

        # Initialize the direction the snake is moving
        self.direction = "left"

        # Initialize the game clock
        self.clock = int(time.perf_counter()*1000)

    def iterate_game(self):
        speed = self.square_size
        # Check for user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Increase or decrease update rate
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.update_rate > 0:
                    self.update_rate += -5
                if event.key == pygame.K_DOWN:
                    self.update_rate += 5
                if event.key == pygame.K_LEFT:
                    self.update_rate = 0
                if event.key == pygame.K_RIGHT:
                    self.update_rate = 50

                # Start the debugger
                if event.key == pygame.K_b:
                    if self.debug:
                        self.debug = False
                    else:
                        self.debug = True
                # Start new game incase the snake is stuck
                if event.key == pygame.K_ESCAPE:
                    self.new_game()
                # Turn off display
                if event.key == pygame.K_d:
                    if self.display:
                        self.display = False
                    else:
                        self.display = True
                print(f'update rate: {self.update_rate}')

        # To stop too many inputs at once
        # Update the snake's position
        if self.direction == "left":
            self.current_pos[0] -= speed
        if self.direction == "right":
            self.current_pos[0] += speed
        if self.direction == "up":
            self.current_pos[1] -= speed
        if self.direction == "down":
            self.current_pos[1] += speed

        self.snake_body = np.append(self.snake_body, [self.current_pos], axis=0)

        # Check for collision
        self.snake_collision()
        # update the screen
        if self.display:
            self.update_screen()

        # Wait for the timer to pass
        while True:
            if self.clock+self.update_rate < int(time.perf_counter()*1000):
                self.clock = int(time.perf_counter()*1000)
                break

    def q_action(self, action):
        # Incase that no action needs to be taken
        if action[0] == 1:
            return
        if self.direction == "left":
            if action[1] == 1:
                self.direction = "down"
            else:
                self.direction = "up"
        elif self.direction == "right":
            if action[1] == 1:
                self.direction = "up"
            else:
                self.direction = "down"
        elif self.direction == "up":
            if action[1] == 1:
                self.direction = "right"
            else:
                self.direction = "left"
        else:
            if action[1] == 1:
                self.direction = "left"
            else:
                self.direction = "right"

    # Function for returning grid position
    def get_position(self, coordinates):
        position = coordinates // 20
        return position[0]*5+(position[1]+1)

    # For returning the state of the game
    def get_state_qmodel(self):
        # Distance from apple
        distance = int(np.linalg.norm(self.apple - self.current_pos))
        # Current positions in the grid
        position = self.current_pos // 20
        apple_pos = self.apple // 20

        # states
        danger = self.collision_check(position)
        cur_direction = np.array([0, 0, 0, 0])

        # Find the current direction
        cur_direction[self.actions.index(self.direction)] = 1


        # Find the food direction
        food_direction = np.array([apple_pos[0] < position[0], apple_pos[0] > position[0],
                                   apple_pos[1] < position[1], apple_pos[1] > position[1]], dtype=int)


        if self.death:
            self.death = False
            return np.concatenate((danger, cur_direction, food_direction), axis=None, dtype=int), distance, -1, len(self.snake_body)
        elif self.reward:
            self.reward = False
            return np.concatenate((danger, cur_direction, food_direction), axis=None, dtype=int), distance, 1, len(self.snake_body)
        else:
            return np.concatenate((danger, cur_direction, food_direction), axis=None, dtype=int), distance, 0, len(self.snake_body)

    def collision_check(self, position):
        """[Danger straight", "Danger right", "Danger Left]"""
        danger = np.array([0, 0, 0])
        direction = self.direction
        # Check for walls by checking if the snake head is near an edge
        # Left or right side
        if int(position[0])+1 <= 4:
            # Find the distance
            distance = position[0]+1
            if direction == "left":
                danger[0] = distance  # Danger in front
            elif direction == "down":
                danger[1] = distance  # Danger to the right
            elif direction == "up":
                danger[2] = distance  # Danger to the left
        elif int(position[0]) >= int(self.width // 20) - 4:
            # Find the distance
            distance = int(self.width // 20) - position[0]
            if direction == "right":
                danger[0] = distance  # Danger in front
            elif direction == "up":
                danger[1] = distance  # Danger to the right
            elif direction == "down":
                danger[2] = distance  # Danger to the left

        # Above or below
        if int(position[1])+1 <= 4:
            # Find the distance
            distance = position[1]+1
            if direction == "up":
                danger[0] = distance  # Danger in front
            elif direction == "left":
                danger[1] = distance  # Danger to the right
            elif direction == "right":
                danger[2] = distance  # Danger to the left
        elif int(position[1]) >= int(self.height // 20) - 4:
            # Find the distance
            distance = int(self.height // 20) - position[1]
            if direction == "down":
                danger[0] = distance  # Danger in front
            elif direction == "right":
                danger[1] = distance  # Danger to the right
            elif direction == "left":
                danger[2] = distance  # Danger to the left

        # Check if the snake is near its own body
        snake_body = self.snake_body[:-2] // 20  # Convert into the grid format
        for body in snake_body:
            # Check to see if the body part is behind the snake
            if direction == "up" and body[1] > position[1] or direction == "down" and body[1] < position[1] \
            or direction == "left" and body[0] > position[0] or direction == "right" and body[0] < position[0]:
                continue

            # Check if the body parts are 1 unit away from the head
            vertical = (body[0] == position[0] and abs(position[1] - body[1]) <= 4)
            horizontal = (body[1] == position[1] and abs(position[0] - body[0]) <= 4)

            # Set distances
            if vertical:
                distance = abs(position[1] - body[1])
            elif horizontal:
                distance = abs(position[0] - body[0])

            # The direction of danger changes depending on the direction of the snake
            if self.direction == "up":
                if vertical:
                    danger[0] = distance
                elif horizontal:
                    if body[0] > position[0]:  # While moving up, to the right is +1 and left -1
                        danger[1] = distance
                    else:
                        danger[2] = distance

            elif self.direction == "down":
                if vertical:
                    danger[0] = distance
                elif horizontal:
                    if body[0] < position[0]:  # While moving down, to the right is -1 and left +1
                        danger[1] = distance
                    else:
                        danger[2] = distance

            elif self.direction == "left":  # While moving left, to the right is -1 and left +1
                if horizontal:
                    danger[0] = distance
                elif vertical:
                    if body[0] > position[0]:
                        danger[1] = distance
                    else:
                        danger[2] = distance

            elif self.direction == "right":  # While moving right, to the right is +1 and left -1
                if horizontal:
                    danger[0] = distance
                elif vertical:
                    if body[0] < position[0]:
                        danger[1] = distance
                    else:
                        danger[2] = distance
        # In case I want info
        if self.debug:
            dangers = ["straight", "right", "left"]
            for idx, val in enumerate(danger):
                if val != 0:
                    print(f"current direction: {direction}")
                    print(f"Danger at {dangers[idx]}: {val}")
        return danger

    def end_game(self):
        pygame.quit()