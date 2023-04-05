import pygame
import sys
import time
import neat
import math

import numpy



WIDTH = 1500
HEIGHT = 750

CAR_SIZE_X = 15
CAR_SIZE_Y = 15

BORDER_COLOR = (255, 255, 255)




class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("E:\\Major_Project\\car_new.png").convert_alpha()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [846, 134]  # Starting Position
        # self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center

        self.radars = []  # Store radars
        self.drawing_radars = []  # Draw radars

        self.alive = True  # Check if Car is safe

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    def draw_radar(self, screen):
        # Drawing all the radars on the screen
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (255, 0, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def get_screenshot(self, game_map):
        # Another form of input. Image input to the neural network
        camera = pygame.Surface((240, 240))
        camera.set_colorkey((0, 255, 0))
        camera = pygame.transform.rotate(camera, self.angle)

        camera_rect = camera.get_rect(center=self.position)
        (x, y) = camera_rect.topleft
        (w, h) = camera_rect.size

        rect = pygame.Rect(x, y, w, h)
        sub = game_map.subsurface(rect)
        screenshot = pygame.Surface((w, h))
        screenshot.blit(sub, (0, 0))
        return screenshot

    def draw_camera(self,camera,screen):
      # Draw the camera and the car
        screen.blit(self.rotated_sprite, self.position)
        screen.blit(camera, (self.position[0], self.position[1]))
        screen.blit(camera, (self.position))
        view = self.get_screenshot(screen)
        screen.blit(view, (0, 0))

        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(x, y, 240, 240), 2)  # Draw Sprite
        self.draw_radar(screen)

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)



    def check_collision(self, game_map):
        # Crash the car when it touches the border
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash

            if game_map.get_at((int(point[0]), int(point[1]))) == (255, 255, 255):
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += 1  # self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = CAR_SIZE_X//2
        left_top = [self.center[0] + math.cos(math.radians((self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians((self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians((self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians((self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians((self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians((self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians((self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians((self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self, game_map):
        img = self.sprite
        map_rect = img.get_rect(center=(self.position[0], self.position[1]))
        view = game_map.subsurface(map_rect).copy()
        arr = numpy.array(view)
        arr_flat = arr.flatten()
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):

    nets = []
    cars = []

    # Initializing PyGame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # For All Genomes A New Neural Network Is Created
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("E:\\Major_Project\\new_track.png").convert()
    game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))
    # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Genearte Action
        for i, car in enumerate(cars):

            output = nets[i].activate(car.get_data(game_map))
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 5  # Left
            elif choice == 1:
                car.angle -= 5  # Right
            elif choice == 2:
                car.angle += 10
            elif choice == 3:
                car.angle -= 10
            elif choice == 4:
                car.angle += 15
            elif choice == 5:
                car.angle -= 15
            elif choice == 6:
                if car.speed - 2 >= 12:
                    car.speed -= 5  # Brake
            else:
                car.speed += 5  # Accelerate


        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()
                best_net = nets[i]

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()

        clock.tick(60)  # 60 FPS


if __name__ == "__main__":
    # Load Config

    config_path = 'E:\\Major_Project\\config.txt'
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    best = population.run(run_simulation, 200)
###################################################################################################
    # Taking the best genome car and testing it on a new track
    time.sleep(5)
    pygame.init()
    my_car = Car()
    my_car.position = [949, 150]
    game_map = pygame.image.load("E:\\Major_Project\\new_track.png").convert()
    hinton = neat.nn.FeedForwardNetwork.create(best, config)
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    while my_car.is_alive():
        output = hinton.activate(my_car.get_data(game_map))
        choice = output.index(max(output))
        if choice == 0:
            my_car.angle += 5
        elif choice == 1:
            my_car.angle -= 5
        elif choice == 2:
            my_car.angle += 10
        elif choice == 3:
            my_car.angle -= 10
        elif choice == 4:
            my_car.speed -= 5

        elif choice == 5:
            my_car.angle += 20
        elif choice == 6:
            my_car.angle -= 20
        else:
            my_car.speed += 5
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        my_car.update(game_map)
        screen.blit(game_map, (0, 0))
        my_car.draw(screen)
        pygame.display.flip()
        clock = pygame.time.Clock()
        clock.tick(20)

    print(f"The best neural net is {best}")


    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("E:\\Major_Project\\map2.png").convert()




