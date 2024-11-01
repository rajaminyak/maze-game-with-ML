Advanced Maze Game with ML Pathfinding
An interactive maze game featuring machine learning-based pathfinding using Q-learning neural networks and PyGame visualization.
Features

Procedurally generated mazes
Neural network-based pathfinding agent
Real-time visualization with PyGame
Interactive controls for testing and training
Memory replay system for improved learning
Gradient-based reward system

Installation

Clone the repository
Install dependencies:

bashCopypip install -r requirements.txt
Usage
Run the game:
bashCopypython maze_game.py
Controls

SPACE: Let ML agent make a move
R: Reset maze
T: Train ML agent (500 episodes)
Close window to quit

Technical Details
Machine Learning Components

Q-learning neural network with 3 layers
State space: 8 dimensions (wall distances + goal position)
Action space: 4 directions
Epsilon-greedy exploration strategy
Experience replay buffer
Gradient-based rewards based on goal distance

Visualization

Real-time rendering with PyGame
Color-coded elements:

Walls: Dark gray
Paths: Light gray
Player: Green
Goal: Red
Visited cells: Light blue
Solution path: Cyan



Performance Tips

Train the agent (T key) before letting it navigate
Multiple training sessions improve performance
Watch for decreasing epsilon values during training
Higher exploration (epsilon) values help find new solutions
