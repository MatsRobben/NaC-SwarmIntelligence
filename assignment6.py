import pygame
import random
import math
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TKAgg')

# Constants
WIDTH, HEIGHT = 800, 800
NUM_BOIDS = 50
MAX_SPEED = 3
BOID_SIZE = 5
BOID_LENGTH = 10
BOID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (220, 220, 220)
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20
ALIGNMENT_WEIGHT = 0.05
COHESION_WEIGHT = 0.0005
SEPARATION_WEIGHT = 0.05
TRUN_FACTOR = 0.3
FIGS_FOLDER = "figs"
MARGIN = {'left': 150, 'right': WIDTH-150, 'top':150, 'bottom': HEIGHT-150}
BOUNCE_OF_EDGES = False

# Constants for excersice 2
N = 20  # ABC population size
NB = 15  # Boids per simulation
MAX_ITER = 300
EPSILONS = [1.0, 0.5, 0.25, 0.1, 0.05]

class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

    def update(self, flock, cohesion, alignment, separation):
        dx, dy = self.compute_movement(flock, cohesion, alignment, separation)
        self.vx += dx
        self.vy += dy
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed <= 0:
            speed = 0.001
        scale = MAX_SPEED / speed
        self.vx *= scale
        self.vy *= scale
        
        self.x += self.vx
        self.y += self.vy
        self.wrap()

    def compute_movement(self, flock, cohesion, alignment, separation):
        avg_pos = [0, 0]
        avg_vel = [0, 0]
        sep = [0, 0]
        neighboring_boids = 0

        for boid in flock:
            if boid != self:
                distance = math.dist((self.x, self.y), (boid.x, boid.y))
                if distance < NEIGHBOR_RADIUS:
                    avg_pos[0] += boid.x
                    avg_pos[1] += boid.y
                    avg_vel[0] += boid.vx
                    avg_vel[1] += boid.vy
                    if distance < SEPARATION_RADIUS:
                        sep[0] += self.x - boid.x
                        sep[1] += self.y - boid.y
                    neighboring_boids += 1

        if neighboring_boids > 0:
            avg_pos[0] /= neighboring_boids
            avg_pos[1] /= neighboring_boids
            avg_vel[0] /= neighboring_boids
            avg_vel[1] /= neighboring_boids

        avg_pos[0] -= self.x
        avg_pos[1] -= self.y
        avg_vel[0] -= self.vx
        avg_vel[1] -= self.vy

        dx = alignment * avg_vel[0] + cohesion * avg_pos[0] + separation * sep[0]
        dy = alignment * avg_vel[1] + cohesion * avg_pos[1] + separation * sep[1]

        return dx, dy

    def wrap(self):
        if BOUNCE_OF_EDGES:
            if self.x < MARGIN['left']:
                self.vx = self.vx + TRUN_FACTOR
            if self.x > MARGIN['right']:
                self.vx = self.vx - TRUN_FACTOR
            if self.y > MARGIN['bottom']:
                self.vy = self.vy - TRUN_FACTOR
            if self.y < MARGIN['top']:
                self.vy = self.vy + TRUN_FACTOR
        else:
            if self.x < 0:
                self.x = WIDTH
            elif self.x > WIDTH:
                self.x = 0
            if self.y < 0:
                self.y = HEIGHT
            elif self.y > HEIGHT:
                self.y = 0
            
def draw_boids(screen, flock):
    for boid in flock:
        angle = math.atan2(boid.vy, boid.vx)
        p1 = (boid.x + BOID_LENGTH * math.cos(angle), boid.y + BOID_LENGTH * math.sin(angle))
        p2 = (boid.x + BOID_LENGTH * math.cos(angle - 2.5), boid.y + BOID_LENGTH * math.sin(angle - 2.5))
        p3 = (boid.x + BOID_LENGTH * math.cos(angle + 2.5), boid.y + BOID_LENGTH * math.sin(angle + 2.5))
        pygame.draw.polygon(screen, BOID_COLOR, (p1, p2, p3))

def create_population():
    population = []
    for _ in range(N):
        cohesion = random.uniform(0, 1)
        alignment = random.uniform(0, 1)
        separation = random.uniform(0, 1)
        population.append((cohesion, alignment, separation))
    return population


def pygame_sim():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()

    flock = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_BOIDS)]

    order_parameters = []
    nearest_neighbor_distances = []

    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for boid in flock:
            # boid.update(flock, COHESION_WEIGHT, ALIGNMENT_WEIGHT, SEPARATION_WEIGHT)
            # Insert values from approxemated posterior
            boid.update(flock, 0.7691528004956708, 0.7354003407767408, 0.7638551431684819)

        draw_boids(screen, flock)

        # Calculate order parameter
        avg_velocity = np.array([np.array([boid.vx, boid.vy]) / math.sqrt(boid.vx**2 + boid.vy**2) if math.sqrt(boid.vx**2 + boid.vy**2) > 0 else [0, 0] for boid in flock])
        
        order_parameter = np.linalg.norm(np.sum(avg_velocity, axis=0)) / NUM_BOIDS
        order_parameters.append(order_parameter)

        # Calculate nearest neighbor distances
        distances = []
        for boid in flock:
            distances.extend([math.dist((boid.x, boid.y), (other_boid.x, other_boid.y)) for other_boid in flock if boid != other_boid])
        nearest_neighbor_distances.append(np.mean(distances))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    return order_parameters, nearest_neighbor_distances

def simulate_boids(cohesion, alignment, separation, warmup=True):
    # Initialize boids
    flock = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NB)]

    # Run simulation
    order_parameters = []
    for _ in range(MAX_ITER):
        # Update boids
        for boid in flock:
            boid.update(flock, cohesion, alignment, separation)

        if not warmup:
            # Calculate order parameter
            avg_velocity = np.array([np.array([boid.vx, boid.vy]) / math.sqrt(boid.vx**2 + boid.vy**2) if math.sqrt(boid.vx**2 + boid.vy**2) > 0 else [0, 0] for boid in flock])
            order_parameter = np.linalg.norm(np.sum(avg_velocity, axis=0)) / NB
            order_parameters.append(order_parameter)

    if not warmup:
        # Final order parameter
        final_order_parameter = order_parameters[-1]
    else:
        for _ in range(100):
            for boid in flock:
                boid.update(flock, cohesion, alignment, separation)

            avg_velocity = np.array([np.array([boid.vx, boid.vy]) / math.sqrt(boid.vx**2 + boid.vy**2) if math.sqrt(boid.vx**2 + boid.vy**2) > 0 else [0, 0] for boid in flock])
            order_parameter = np.linalg.norm(np.sum(avg_velocity, axis=0)) / NB
            order_parameters.append(order_parameter)

        final_order_parameter = np.mean(order_parameters)

    return final_order_parameter

def run_abc():
    population = create_population()
    accepted_params = []

    for epsilon in EPSILONS:
        print(f"Running ABC with epsilon: {epsilon}")
        accepted = []
        for i in range(N):
            print(f"  Individual {i+1}/{N}")
            start_time = time.time()  # Start timing
            while True:
                candidate_params = population[i]
                final_order_parameter = simulate_boids(*candidate_params)
                if final_order_parameter >= 0.6 - epsilon:
                    end_time = time.time()  # End timing
                    time_taken = end_time - start_time
                    accepted.append(candidate_params)
                    print(f"    Accepted: {candidate_params}, order-param: {final_order_parameter}, time taken: {time_taken:.4f} seconds")
                    break
                else:
                    # Mutate parameters
                    mutation = [random.uniform(-epsilon, epsilon) for _ in range(3)]
                    mutated_params = [min(max(candidate_params[j] + mutation[j], 0), 1) for j in range(3)]
                    population[i] = mutated_params
        accepted_params.append(accepted)


    # Save accepted parameters
    os.makedirs("accepted_params", exist_ok=True)
    for i, accepted in enumerate(accepted_params):
        with open(f"accepted_params/accepted_params_{i}.txt", "w") as file:
            for params in accepted:
                file.write(f"{params[0]}, {params[1]}, {params[2]}\n")

def main(visualise=True):
    
    if visualise: 
        order_parameters, nearest_neighbor_distances = pygame_sim()

        # Save plots to figs folder
        os.makedirs(FIGS_FOLDER, exist_ok=True)

        # Plot order parameter over time
        plt.figure(figsize=(15, 10))
        plt.plot(order_parameters)
        plt.xlabel('Time')
        plt.ylabel('Order Parameter')
        plt.title('Order Parameter Over Time')
        plt.savefig('figs/order_parameter.png', format="png")
        plt.show()
        # plt.close()

        # Plot nearest neighbor distances over time
        plt.figure(figsize=(15, 10))
        plt.plot(nearest_neighbor_distances)
        plt.xlabel('Time')
        plt.ylabel('Nearest Neighbor Distance')
        plt.title('Nearest Neighbor Distances Over Time')
        plt.savefig('figs/nearest_neighbor_distances.png', format="png")
        plt.show()
        # plt.close()
    else:
        run_abc()

def plot_accepted_params(folder_path):
    all_accepted_params = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            accepted_params = []
            with open(os.path.join(folder_path, file_name), "r") as file:
                lines = file.readlines()
                params = [[float(val) for val in line.strip().split(",")] for line in lines]
                accepted_params.extend(params)
            all_accepted_params.append(accepted_params)

    plt.figure(figsize=(15, 10))

    for i, accepted_params in enumerate(all_accepted_params):
        param_1 = [params[0] for params in accepted_params]
        param_2 = [params[1] for params in accepted_params]
        param_3 = [params[2] for params in accepted_params]

        plt.subplot(3, len(all_accepted_params), i + 1)
        plt.hist(param_1, bins=30, edgecolor='black')
        plt.title(f'Parameter 1 (Epsilon={EPSILONS[i]})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.subplot(3, len(all_accepted_params), i + len(all_accepted_params) + 1)
        plt.hist(param_2, bins=30, edgecolor='black')
        plt.title(f'Parameter 2 (Epsilon={EPSILONS[i]})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.subplot(3, len(all_accepted_params), i + 2 * len(all_accepted_params) + 1)
        plt.hist(param_3, bins=30, edgecolor='black')
        plt.title(f'Parameter 3 (Epsilon={EPSILONS[i]})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')


    plt.tight_layout()
    plt.show()

def plot_accepted_params_stacked(folder_path, epsilons, save_path=None):
    all_accepted_params = []

    param_names = ['Cohesion', 'Alignment', 'Separation']

    for i, epsilon in enumerate(epsilons):
        accepted_params = []
        with open(os.path.join(folder_path, f"accepted_params_{i}.txt"), "r") as file:
            lines = file.readlines()
            params = [[float(val) for val in line.strip().split(",")] for line in lines]
            accepted_params.extend(params)
        all_accepted_params.append(accepted_params)

    plt.figure(figsize=(10, 2 * len(epsilons)))

    for i, accepted_params in enumerate(all_accepted_params):
        param_values = np.array(accepted_params)

        plt.subplot(len(epsilons), 1, i + 1)
        plt.hist(param_values[:, 0], bins=30, color='r', alpha=0.5, label=param_names[0], stacked=True)
        plt.hist(param_values[:, 1], bins=30, color='g', alpha=0.5, label=param_names[1], stacked=True)
        plt.hist(param_values[:, 2], bins=30, color='b', alpha=0.5, label=param_names[2], stacked=True)
        plt.title(f'Epsilon={epsilons[i]}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    # Turning on visualise runs the simulation in pygame, turing it off runs the ABC algorithm
    main(visualise=True)

    # Call the function to plot the accepted parameters
    # plot_accepted_params("accepted_params")
    # plot_accepted_params_stacked("accepted_params", EPSILONS, save_path="figs/accepted_params_no-borders_plot.png")
