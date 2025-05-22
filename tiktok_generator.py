import sys
import os
import pygame
import pymunk
import pymunk.pygame_util
from moviepy import *  # Using your required import

# Constants
FPS = 60
DT = 1 / FPS
DURATION_SECONDS = 5
TOTAL_FRAMES = DURATION_SECONDS * FPS
OUTPUT_DIR = "frames"

BACKGROUND_COLOR = (55, 55, 55)

def setup_physics(space_width, space_height):
    space = pymunk.Space()
    space.gravity = (0, 900)

    # Add a ball
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 30))
    body.position = space_width // 2, 50
    shape = pymunk.Circle(body, 30)
    shape.elasticity = 0.95
    shape.collision_type = 1      # Ball
    space.add(body, shape)

    # Add floor
    segment = pymunk.Segment(space.static_body, (0, space_height - 50), (space_width, space_height - 50), 5)
    segment.friction = 0.9
    segment.elasticity = 0.95
    segment.collision_type = 2    # Floor
    space.add(segment)

    return space

def render_to_video():
    WIDTH, HEIGHT = 1080, 1920
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))  # Off-screen surface
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = setup_physics(WIDTH, HEIGHT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for frame_number in range(TOTAL_FRAMES):
        space.step(DT)
        screen.fill(BACKGROUND_COLOR)
        space.debug_draw(draw_options)
        pygame.image.save(screen, f"{OUTPUT_DIR}/frame_{frame_number:04d}.png")
        print(f"Saved frame {frame_number + 1}/{TOTAL_FRAMES}")

    clip = ImageSequenceClip(OUTPUT_DIR, fps=FPS)
    clip.write_videofile("output.mp4", codec="libx264")
    pygame.quit()

def interactive_mode():
    WIDTH, HEIGHT = 576, 1024
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pymunk Simulation")
    draw_options = pymunk.pygame_util.DrawOptions(window)

    space = setup_physics(WIDTH, HEIGHT)
    shape = space.shapes[0]  # The ball
    shape.collision_type = 1  # Ball
    
    clock = pygame.time.Clock()
    
    collision_effect = {"active": False, "timer": 0}

    def on_collision(arbiter, space, data):
        collision_effect["active"] = True
        collision_effect["timer"] = 5  # Show light effect for 5 frames
        return True

    handler = space.add_collision_handler(1, 2)
    handler.begin = on_collision

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        window.fill(BACKGROUND_COLOR)

        space.step(DT)
        
        if collision_effect["active"]:
            pos = int(shape.body.position.x), int(shape.body.position.y)
            print("Collision detected!", pos)
            
            pygame.draw.circle(window, (255, 255, 100), pos, 50, 0)  # yellow glow
            collision_effect["timer"] -= 1
            if collision_effect["timer"] <= 0:
                collision_effect["active"] = False


        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"

    if mode == "render":
        render_to_video()
    else:
        interactive_mode()
