from snake_class import SnakeGame
import pygame


# MAIN LOOP
if __name__ == "__main__":
    game = SnakeGame("Tester", 400, 600)
    game.new_game()
    game.debug = True
    game.display = True
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(f"current pos:{game.current_pos // 20}")
                if event.key == pygame.K_LEFT:
                    game.q_action([0,0,1])
                if event.key == pygame.K_RIGHT:
                    game.q_action([0,1,0])
                if event.key == pygame.K_UP:
                    game.q_action([1,0,0])
                #if event.key == pygame.K_DOWN:
                #    game.direction = "down"
                clock.tick(10)
                game.iterate_game()
                state, _,_,_ = game.get_state_qmodel()
                print(f"Danger: {state[:5]}, Direction: {state[5:9]}, Food: {state[9:13]}")
                #game.collision_check(game.current_pos // 20)
