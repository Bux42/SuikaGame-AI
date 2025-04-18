import pygame
import pymunk
import pymunk.pygame_util
import random
import sys
import math
from enum import Enum, auto

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# Fruit class to categorize different fruits
class FruitType(Enum):
    CHERRY = 0
    STRAWBERRY = 1
    GRAPE = 2
    ORANGE = 3
    PERSIMMON = 4
    APPLE = 5
    PEAR = 6
    PEACH = 7
    PINEAPPLE = 8
    WATERMELON = 9

# Fruit properties: radius and color
FRUIT_PROPERTIES = {
    FruitType.CHERRY: {"radius": 20, "color": (185, 0, 0), "score": 1},
    FruitType.STRAWBERRY: {"radius": 30, "color": (255, 0, 0), "score": 3},
    FruitType.GRAPE: {"radius": 40, "color": (128, 0, 128), "score": 6},
    FruitType.ORANGE: {"radius": 50, "color": (255, 165, 0), "score": 10},
    FruitType.PERSIMMON: {"radius": 60, "color": (255, 69, 0), "score": 15},
    FruitType.APPLE: {"radius": 70, "color": (255, 0, 0), "score": 21},
    FruitType.PEAR: {"radius": 80, "color": (170, 255, 0), "score": 28},
    FruitType.PEACH: {"radius": 90, "color": (255, 218, 185), "score": 36},
    FruitType.PINEAPPLE: {"radius": 100, "color": (255, 255, 0), "score": 45},
    FruitType.WATERMELON: {"radius": 110, "color": (0, 128, 0), "score": 55},
}

class SuikaGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Suika Game Clone")
        self.clock = pygame.time.Clock()
        
        # Physics setup
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)  # Gravity in the y direction
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Game area
        self.box_width = 400
        self.box_height = 500
        self.box_left = (SCREEN_WIDTH - self.box_width) // 2
        self.box_right = self.box_left + self.box_width
        self.box_top = 200
        self.box_bottom = self.box_top + self.box_height
        
        # Create walls
        self.create_walls()
        
        # Game state
        self.fruits = []
        self.next_fruit = random.choice([FruitType.CHERRY, FruitType.STRAWBERRY, FruitType.GRAPE])
        self.fruit_x = SCREEN_WIDTH // 2
        self.game_over = False
        self.score = 0
        self.dropping = False
        self.drop_cooldown = 0
        self.next_fruit_after_drop = None
        
        # Setup collision handlers
        self.setup_collision_handlers()
        
        # Font for score
        self.font = pygame.font.Font(None, 36)

    def create_walls(self):
        # Create static walls for the box
        walls = [
            # Left wall
            [(self.box_left, self.box_top), (self.box_left, self.box_bottom)],
            # Right wall
            [(self.box_right, self.box_top), (self.box_right, self.box_bottom)],
            # Bottom wall
            [(self.box_left, self.box_bottom), (self.box_right, self.box_bottom)]
        ]
        
        for wall in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, wall[0], wall[1], 5)
            shape.elasticity = 0.3
            shape.friction = 0.8
            shape.collision_type = 1  # Wall collision type
            self.space.add(body, shape)
        
        # Add game over line (invisible)
        self.game_over_line = pymunk.Segment(
            pymunk.Body(body_type=pymunk.Body.STATIC),
            (self.box_left, self.box_top + 100),
            (self.box_right, self.box_top + 100),
            1
        )
        self.game_over_line.sensor = True
        self.game_over_line.collision_type = 3  # Game over line collision type
        self.space.add(self.game_over_line.body, self.game_over_line)

    def setup_collision_handlers(self):
        # Handler for fruit-to-fruit collisions
        handler = self.space.add_collision_handler(2, 2)  # Fruit to fruit
        handler.separate = self.handle_fruit_collision
        
        # Handler for fruit to game over line
        game_over_handler = self.space.add_collision_handler(2, 3)  # Fruit to game over line
        game_over_handler.begin = self.check_game_over

    def check_game_over(self, arbiter, space, data):
        # Check if fruit breaches the game over line and stays for too long
        for fruit in self.fruits:
            if fruit['shape'] in arbiter.shapes:
                body = fruit['shape'].body
                if body.position.y < self.box_top + 140 and body.velocity.length < 50:
                    # If a fruit is above the line and nearly stationary
                    if not self.dropping:  # Only check for game over when not dropping
                        self.game_over = True
        return True

    def handle_fruit_collision(self, arbiter, space, data):
        # Get the two colliding shapes
        shape_a, shape_b = arbiter.shapes
        
        # Find corresponding fruits
        fruit_a = next((f for f in self.fruits if f['shape'] == shape_a), None)
        fruit_b = next((f for f in self.fruits if f['shape'] == shape_b), None)
        
        if fruit_a and fruit_b and fruit_a['type'] == fruit_b['type']:
            # Check if next type exists (not the largest fruit)
            if fruit_a['type'].value < FruitType.WATERMELON.value:
                # Mark for deletion
                fruit_a['to_remove'] = True
                fruit_b['to_remove'] = True
                
                # Calculate merge position (average of both fruits)
                pos_x = (fruit_a['shape'].body.position.x + fruit_b['shape'].body.position.x) / 2
                pos_y = (fruit_a['shape'].body.position.y + fruit_b['shape'].body.position.y) / 2
                
                # Create new larger fruit
                new_type = FruitType(fruit_a['type'].value + 1)
                
                # Add score
                self.score += FRUIT_PROPERTIES[new_type]['score']
                
                # Schedule new fruit creation for next frame to avoid physics issues
                # Fix: Use lambda to properly pass the fruit_type parameter
                self.space.add_post_step_callback(
                    lambda space, key, x=pos_x, y=pos_y, t=new_type: self.create_merged_fruit(space, key, x, y, t),
                    None
                )
                
        return True

    def create_merged_fruit(self, space, key, pos_x, pos_y, fruit_type):
        # Remove marked fruits
        for fruit in list(self.fruits):
            if fruit.get('to_remove', False):
                self.space.remove(fruit['shape'], fruit['shape'].body)
                self.fruits.remove(fruit)
        
        # Create a new fruit at the merge position
        self.add_fruit(pos_x, pos_y, fruit_type)

    def add_fruit(self, x, y, fruit_type):
        properties = FRUIT_PROPERTIES[fruit_type]
        radius = properties["radius"]
        
        # Create body and shape
        mass = radius * radius * 0.01
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.3
        shape.friction = 0.8
        shape.collision_type = 2  # Fruit collision type
        
        self.space.add(body, shape)
        
        # Add to fruits list
        self.fruits.append({
            'type': fruit_type,
            'shape': shape,
            'radius': radius,
            'color': properties["color"]
        })

    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw the box
        pygame.draw.rect(self.screen, GRAY, (self.box_left, self.box_top, self.box_width, self.box_height), 5)
        
        # Draw game over line (red when game is over)
        if self.game_over:
            pygame.draw.line(
                self.screen, RED, 
                (self.box_left, self.box_top + 100), 
                (self.box_right, self.box_top + 100), 
                3
            )
        
        # Draw the next fruit above the box
        if not self.dropping and not self.game_over:
            next_props = FRUIT_PROPERTIES[self.next_fruit]
            pygame.draw.circle(
                self.screen, 
                next_props["color"], 
                (self.fruit_x, self.box_top - next_props["radius"] - 10), 
                next_props["radius"]
            )
        
        # Draw all fruits
        for fruit in self.fruits:
            pos = fruit['shape'].body.position
            pygame.draw.circle(
                self.screen,
                fruit['color'],
                (int(pos.x), int(pos.y)),
                fruit['radius']
            )
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (20, 20))
        
        # Draw game over text
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press R to restart", True, BLACK)
            self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 80, 100))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - 100, 140))
            
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                    
                elif event.key == pygame.K_SPACE and not self.dropping and not self.game_over:
                    # Drop the fruit
                    self.drop_fruit()
            
        # Move the next fruit with mouse
        if not self.dropping and not self.game_over:
            mouse_x, _ = pygame.mouse.get_pos()
            # Limit within box boundaries accounting for fruit radius
            radius = FRUIT_PROPERTIES[self.next_fruit]["radius"]
            self.fruit_x = max(self.box_left + radius, min(mouse_x, self.box_right - radius))
            
        # Check for mouse click to drop fruit
        if pygame.mouse.get_pressed()[0] and not self.dropping and not self.game_over and self.drop_cooldown <= 0:
            self.drop_fruit()

    def drop_fruit(self):
        self.dropping = True
        self.drop_cooldown = 30  # Cooldown frames
        
        # Add the fruit at the drop position
        self.add_fruit(
            self.fruit_x, 
            self.box_top - FRUIT_PROPERTIES[self.next_fruit]["radius"] - 5,
            self.next_fruit
        )
        
        # Prepare the next fruit type
        self.next_fruit_after_drop = random.choice([FruitType.CHERRY, FruitType.STRAWBERRY, FruitType.GRAPE])

    def update(self):
        if self.drop_cooldown > 0:
            self.drop_cooldown -= 1
        
        # If dropped fruit has settled (or after a time), allow dropping the next one
        if self.dropping:
            settled = True
            for fruit in self.fruits:
                if fruit['shape'].body.velocity.length > 20:  # Threshold for considering "settled"
                    settled = False
                    break
            
            if settled and self.drop_cooldown <= 0:
                self.dropping = False
                self.next_fruit = self.next_fruit_after_drop
        
        # Physics simulation step
        self.space.step(1/FPS)

    def reset_game(self):
        # Remove all fruits
        for fruit in self.fruits:
            self.space.remove(fruit['shape'], fruit['shape'].body)
        
        # Reset game state
        self.fruits = []
        self.next_fruit = random.choice([FruitType.CHERRY, FruitType.STRAWBERRY, FruitType.GRAPE])
        self.game_over = False
        self.score = 0
        self.dropping = False
        self.drop_cooldown = 0

    def run(self):
        while True:
            self.handle_events()
            if not self.game_over:
                self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    game = SuikaGame()
    game.run()