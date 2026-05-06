# Bio-Inspired Game of Life

**Module:** COMP5400M Bio-Inspired Computing  
**Assessment:** Coursework 2  

## Overview
This project implements a neuroevolutionary framework using **Compositional Pattern Producing Networks (CPPNs)** to generate and evolve optimal initial seed states for Conway's Game of Life. The goal of the algorithm is to discover seed patterns that yield sustained survival, structural complexity, movement (center of mass displacement), and dynamic behaviors over time.

By utilizing spatial features (radial distance, border distance, trigonometric properties) as inputs to a neural network, the system outputs structured boolean starting grids rather than purely random noise. The fitness function evaluates the subsequent Game of Life simulation across hundreds of steps to score the network's "creativity."

---

## File Structure

- **`train.py`**  
  The main engine for the neuroevolutionary process. It manages a population of CPPNs (genomes), simulates the Game of Life using hardware-accelerated PyTorch convolutions, evaluates population fitness based on spatial and temporal metrics, and performs crossover/mutation to generate successive generations.
  
- **`evaluate.py`**  
  The visual evaluation and comparison tool. It loads a specified saved checkpoint and visually contrasts the trained agent's generated grid against a completely random untrained agent. It uses `pygame` to render the Game of Life steps in real-time.

- **`requirements.txt`**  
  The list of required Python dependencies needed to run the project.

- **`Runs/`**  
  A directory generated automatically during training. Each training session produces a timestamped `run_<timestamp>` folder containing:
  - `checkpoint.json`: Contains the highest performing genome, population data, and hyperparameters.
  - `fitness_log.json`: The training trajectory over generations (best, mean, worst fitness).

---

## Installation & Requirements

Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

### 1. Training the Model (`train.py`)
To begin an evolutionary training run, execute the `train.py` script. The script relies on an evolutionary algorithm to iteratively improve the generators.

**Basic Usage:**
```bash
python train.py
```

**Advanced Usage / Arguments:**
You can customize the evolutionary hyperparameters via command-line arguments:
- `--max_generations [INT]`: Maximum generations to evolve (default: 10000).
- `--population [INT]`: Number of genomes in the population (default: 36).
- `--save_every [INT]`: Frequency (in generations) to dump a checkpoint (default: 50).
- `--resume [PATH]`: Path to an existing `checkpoint.json` file to continue training.
- `--compile`: Enables PyTorch `torch.compile()` for potentially faster tensor operations (requires PyTorch 2.0+).

*Example:*
```bash
python train.py --population 50 --max_generations 5000 --compile
```

Outputs will be logged locally to the console, and checkpoints will be saved iteratively to the `Runs/run_<timestamp>/` directory.

### 2. Evaluating a Trained Model (`evaluate.py`)
To see the results of your training, use the evaluation script. It displays a Pygame window showing the Game of Life initialized by your evolved generator on the left, and a random generator on the right.

**Basic Usage:**
```bash
python evaluate.py --checkpoint "Runs/run_16/checkpoint.json"
```

**Controls inside the simulation:**
- **Spacebar**: Pause or Resume the simulation.
- **R**: Reset the grid back to step 0.

*(If you omit the `--checkpoint` argument, it will default to a pre-set checkpoint path. Ensure the provided path accurately targets the `.json` checkpoint you wish to evaluate).*

---

## Key Metrics Evaluated
The fitness function within `train.py` is multi-objective, applying rewards and penalties based on:
- **Longevity:** How many simulation steps the generated cells survive.
- **Movement (CoM Travel):** Total displacement of the cellular pattern's center of mass.
- **Dynamic Activity:** Sustained changes across Early, Mid, and Late stages of the simulation.
- **Compressibility / Novelty:** Penalties for patterns that quickly fall into simple, short repeating static loops (stasis/periodicity penalties).
