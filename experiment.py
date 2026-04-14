from game_of_life import advanceBoard, boardToString, Cell
import random


def string_to_board(pattern):
    board = set()
    lines = pattern.strip().splitlines()

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == "O":
                board.add(Cell(x, y))

    return board


def run_simulation(board, steps):
    history = [board]
    current = board

    for _ in range(steps):
        current = advanceBoard(current)
        history.append(current)

    return history


def live_counts(history):
    return [len(board) for board in history]


def lifespan(history):
    for i, board in enumerate(history):
        if len(board) == 0:
            return i
    return len(history) - 1


def stabilises(history):
    for i in range(1, len(history)):
        if history[i] == history[i - 1]:
            return True, i
    return False, None


def first_repeat(history):
    for i in range(len(history)):
        for j in range(i + 1, len(history)):
            if history[i] == history[j]:
                return True, i, j, j - i
    return False, None, None, None


def random_board(size=10, density=0.3):
    board = set()

    for x in range(size):
        for y in range(size):
            if random.random() < density:
                board.add(Cell(x, y))

    return board


def analyse_pattern(name, pattern, steps=10):
    board = string_to_board(pattern)
    analyse_board(name, board, steps)

def analyse_random(size=10, density=0.3, steps=10):
    board = random_board(size, density)
    analyse_board(f"Random {size}x{size} (density={density})", board, steps)

def analyse_board(name, board, steps=10):
    history = run_simulation(board, steps)

    counts = live_counts(history)
    life = lifespan(history)
    stable, stable_step = stabilises(history)
    repeats, first, second, period = first_repeat(history)

    print(f"\n=== {name} ===")
    print("Simulation Results")
    print("------------------")
    print(f"Lifespan: {life}")
    print(f"Stable: {stable}")
    print(f"Stable at step: {stable_step}")
    print(f"Repeats: {repeats}")
    print(f"First repeat pair: {first}, {second}")
    print(f"Period: {period}")
    print(f"Live cells over time: {counts}")

    print("\nInitial board:")
    print(boardToString(history[0]))

    print("\nFinal board:")
    print(boardToString(history[-1]))
    print("\n" + "=" * 30)

#SHAPES
block = """
OO
OO
"""

blinker = """
OOO
"""

glider = """
.O.
..O
OOO
"""
##

if __name__ == "__main__":
    analyse_pattern("Block", block, 10)
    analyse_pattern("Blinker", blinker, 10)
    analyse_pattern("Glider", glider, 10)
    analyse_random(10, 0.3, 10)