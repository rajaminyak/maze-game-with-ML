# Maze Game with Machine Learning

An interactive maze game that uses reinforcement learning (Q-learning) to find optimal paths through procedurally generated mazes.

## Features

- 🎮 Interactive maze generation
- 🧠 Machine learning pathfinding
- 📊 Real-time statistics display
- 🎯 Goal-oriented learning
- 🔄 Continuous training capability
- 🐍 Built with Python and PyTorch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rajaminyak/maze-game-ml.git
cd maze-game-ml
```

2. Install required packages:
```bash
pip install pygame torch numpy
```

## Usage

Run the game:
```bash
python main.py
```

### Controls
- `SPACE`: Let ML agent make a move
- `R`: Reset/generate new maze
- `T`: Train ML agent (500 episodes)
- `D`: Toggle debug mode

## How It Works

### Machine Learning Components

1. State Space (What the AI "sees"):
   - Distances to walls in 4 directions
   - Distance to goal
   - Angle to goal

2. Action Space (What the AI can do):
   - Move Up
   - Move Down
   - Move Left
   - Move Right

3. Reward System:
   - +100 for reaching goal
   - -1 for hitting walls
   - -0.1 step cost
   - Distance-based penalties

### Neural Network Architecture

```python
QNetwork Structure:
Input Layer (6 neurons) 
    → Hidden Layer (64 neurons) 
    → Hidden Layer (64 neurons) 
    → Output Layer (4 neurons)
```

### Learning Process
- Uses epsilon-greedy strategy for exploration
- Experience replay for better learning
- Adaptive learning rate
- Continuous improvement through training

## Visual Elements

- 🟢 Green Circle: Player
- 🔴 Red Square: Goal
- ⬛ Dark Gray: Walls
- ⬜ Light Gray: Paths
- 🔷 Blue: Visited cells
- 🟡 Orange: Training indicator

## Performance Metrics

The game displays real-time statistics:
- Steps taken
- Goals reached
- Current epsilon value (exploration rate)
- Training episode (during training)
- Current reward

## Debug Mode

Debug mode (`D` key) provides additional information:
- Episode details
- Reward calculations
- Path analysis
- Training progress

## Project Structure

```
maze-game-ml/
│
├── main.py           # Main game file
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Technical Details

### Key Classes

1. `QNetwork`: Neural network for decision making
2. `AdvancedMazeGame`: Main game logic and ML integration

### Machine Learning Parameters

```python
learning_rate = 0.001
gamma = 0.99         # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01   # Minimum exploration rate
memory_size = 10000  # Experience replay buffer size
batch_size = 32      # Training batch size
```

## Development

To contribute:
1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## Troubleshooting

Common issues and solutions:

1. Game freezes during training:
   - Reduce maze size
   - Lower episode count
   - Check system resources

2. Agent not learning:
   - Increase training episodes
   - Adjust reward values
   - Check maze connectivity

## Future Improvements

- [ ] Add different maze generation algorithms
- [ ] Implement A* comparison
- [ ] Add visualization of neural network decisions
- [ ] Support for larger mazes
- [ ] Save/load trained models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch documentation
- Pygame community
- Reinforcement learning resources

## Authors

- Your Name
- Contributors welcome!
