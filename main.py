import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math
import threading
import time

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class AdvancedMazeGame:
    def __init__(self, width=20, height=20, cell_size=30):
        pygame.init()
        
        # Game settings
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 50  # Added space for stats
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Enhanced Maze Game with ML")
        
        # Colors with enhanced palette
        self.COLORS = {
            'wall': (40, 40, 40),
            'path': (220, 220, 220),
            'player': (0, 255, 100),
            'goal': (255, 50, 50),
            'solution': (0, 191, 255),
            'visited': (130, 130, 230, 100),
            'training': (255, 165, 0),
            'text': (255, 255, 255),
            'stats_bg': (30, 30, 30)
        }
        
        # ML components
        self.input_size = 6
        self.hidden_size = 64
        self.output_size = 4
        self.q_network = QNetwork(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        
        # Game state
        self.is_training = False
        self.debug_mode = False
        self.steps_taken = 0
        self.goals_reached = 0
        self.current_episode = 0
        self.total_reward = 0
        self.last_training_time = 0
        
        self.reset_game()
        
    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

    def reset_game(self):
        if self.debug_mode:
            print("Game reset")
        self.maze = self.generate_maze()
        self.player_pos = [1, 1]
        self.goal_pos = [self.height-2, self.width-2]
        self.visited_cells = set()
        self.solution_path = []
        self.steps_taken = 0
        return self.get_state()
        

    def generate_maze(self):
        # Enhanced maze generation with guaranteed path
        maze = np.ones((self.height, self.width), dtype=int)
        
        def carve_paths(x, y):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                new_x, new_y = x + dy, y + dx
                if (0 < new_x < self.width-1 and 
                    0 < new_y < self.height-1 and 
                    maze[new_y][new_x] == 1):
                    maze[new_y][new_x] = 0
                    maze[y + dy//2][x + dx//2] = 0
                    carve_paths(new_x, new_y)
        
        carve_paths(1, 1)
        
        # Ensure path to goal exists
        if maze[self.height-2][self.width-2] == 1:
            self._create_path_to_goal(maze)
        
        return maze
    

    def _create_path_to_goal(self, maze):
        y, x = self.height-2, self.width-2
        start_y, start_x = 1, 1
        
        # Create path from goal to start
        while y > start_y or x > start_x:
            maze[y][x] = 0
            # Decide whether to move horizontally or vertically
            if random.random() < 0.5 and x > start_x:
                x -= 1
            elif y > start_y:
                y -= 1
            else:
                x -= 1
        
        # Ensure start and goal cells are clear
        maze[1][1] = 0
        maze[self.height-2][self.width-2] = 0
        return maze
    
    def get_state(self):
        px, py = self.player_pos
        gx, gy = self.goal_pos
        
        wall_distances = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dist = 0
            while (0 <= px + dx*dist < self.width and 
                   0 <= py + dy*dist < self.height and 
                   self.maze[py + dy*dist][px + dx*dist] == 0):
                dist += 1
            wall_distances.append(dist / max(self.width, self.height))
        
        goal_distance = math.sqrt((gx - px)**2 + (gy - py)**2) / math.sqrt(self.width**2 + self.height**2)
        goal_angle = math.atan2(gy - py, gx - px) / math.pi
        
        state = wall_distances + [goal_distance, goal_angle]
        return torch.FloatTensor(state)

    def step(self, action):
        self.steps_taken += 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = directions[action]
        new_x = self.player_pos[1] + dx
        new_y = self.player_pos[0] + dy
        
        if (0 <= new_x < self.width and 
            0 <= new_y < self.height and 
            self.maze[new_y][new_x] == 0):
            self.player_pos = [new_y, new_x]
            self.visited_cells.add((new_y, new_x))
            
            if [new_y, new_x] == self.goal_pos:
                if self.debug_mode:
                    print(f"Goal reached in {self.steps_taken} steps!")
                self.goals_reached += 1
                return self.get_state(), 100, True
            
            goal_distance = math.sqrt((self.goal_pos[1] - new_x)**2 + 
                                    (self.goal_pos[0] - new_y)**2)
            reward = -0.1 - 0.01 * goal_distance
            
            return self.get_state(), reward, False
        
        return self.get_state(), -1, False

    def train(self, episodes=500):
        self.is_training = True
        start_time = time.time()
        
        for episode in range(episodes):
            self.current_episode = episode
            state = self.reset_game()
            episode_reward = 0
            done = False
            
            while not done:
                if random.random() < self.epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        action = self.q_network(state).argmax().item()
                
                next_state, reward, done = self.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                if len(self.memory) > 32:
                    self._train_batch()
            
            self.total_reward = episode_reward
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if self.debug_mode and episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        
        self.last_training_time = time.time() - start_time
        self.is_training = False
        print(f"Training complete! Time taken: {self.last_training_time:.1f}s")

    def _train_batch(self):
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_in_thread(self):
        if not self.is_training:
            threading.Thread(target=self.train, args=(500,), daemon=True).start()

    def draw(self):
        self.screen.fill(self.COLORS['stats_bg'])
        
        # Draw maze
        maze_surface = pygame.Surface((self.screen_width, self.screen_height - 50))
        maze_surface.fill((0, 0, 0))
        
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(maze_surface, self.COLORS['wall'], rect)
                else:
                    pygame.draw.rect(maze_surface, self.COLORS['path'], rect)
                    pygame.draw.rect(maze_surface, (100, 100, 100), rect, 1)
        
        # Draw visited cells and other elements
        self._draw_game_elements(maze_surface)
        
        # Draw stats
        self._draw_stats()
        
        # Combine surfaces
        self.screen.blit(maze_surface, (0, 0))
        pygame.display.flip()

    def _draw_game_elements(self, surface):
        # Draw visited cells
        for cell in self.visited_cells:
            y, x = cell
            cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(cell_surface, self.COLORS['visited'], 
                           (0, 0, self.cell_size, self.cell_size))
            surface.blit(cell_surface, (x * self.cell_size, y * self.cell_size))
        
        # Draw solution path
        for cell in self.solution_path:
            y, x = cell
            pygame.draw.rect(surface, self.COLORS['solution'],
                           (x * self.cell_size + self.cell_size//4,
                            y * self.cell_size + self.cell_size//4,
                            self.cell_size//2, self.cell_size//2))
        
        # Draw player and goal
        pygame.draw.circle(surface, self.COLORS['player'],
                         (self.player_pos[1] * self.cell_size + self.cell_size//2,
                          self.player_pos[0] * self.cell_size + self.cell_size//2),
                         self.cell_size//3)
        
        pygame.draw.rect(surface, self.COLORS['goal'],
                        (self.goal_pos[1] * self.cell_size + self.cell_size//4,
                         self.goal_pos[0] * self.cell_size + self.cell_size//4,
                         self.cell_size//2, self.cell_size//2))

    def _draw_stats(self):
        font = pygame.font.Font(None, 24)
        stats = [
            f"Steps: {self.steps_taken}",
            f"Goals: {self.goals_reached}",
            f"ε: {self.epsilon:.2f}"
        ]
        
        if self.is_training:
            stats.append(f"Episode: {self.current_episode}")
            stats.append(f"Reward: {self.total_reward:.1f}")
        
        # Draw training indicator
        if self.is_training:
            pygame.draw.rect(self.screen, self.COLORS['training'], 
                           (10, self.screen_height - 45, 100, 30))
            text = font.render("Training...", True, (0, 0, 0))
            self.screen.blit(text, (20, self.screen_height - 40))
        
        # Draw stats
        x_pos = 150
        for stat in stats:
            text = font.render(stat, True, self.COLORS['text'])
            self.screen.blit(text, (x_pos, self.screen_height - 40))
            x_pos += 150

    def play(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        state = self.get_state()
                        with torch.no_grad():
                            action = self.q_network(state).argmax().item()
                        self.step(action)
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_t:
                        print("Training ML agent...")
                        self.train_in_thread()
                    elif event.key == pygame.K_d:
                        self.toggle_debug()
            
            self.draw()
            clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    game = AdvancedMazeGame()
    print("Controls:")
    print("SPACE: Let ML agent make a move")
    print("R: Reset maze")
    print("T: Train ML agent")
    print("D: Toggle debug mode")
    game.play()
