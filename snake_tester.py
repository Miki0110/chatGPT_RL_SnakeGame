from snake_class import SnakeGame
import pygame


# MAIN LOOP
if __name__ == "__main__":
    game = SnakeGame("Tester", 400, 400)
    game.new_game()
    game.debug = True
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(f"current pos:{game.current_pos // 20}")
                if event.key == pygame.K_LEFT:
                    game.direction = "left"
                if event.key == pygame.K_RIGHT:
                    game.direction = "right"
                if event.key == pygame.K_UP:
                    game.direction = "up"
                if event.key == pygame.K_DOWN:
                    game.direction = "down"
                clock.tick(10)
                game.iterate_game()
                game.collision_check(game.current_pos // 20)
