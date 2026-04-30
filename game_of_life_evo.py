import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

# Configuration matches training
GRID_W = 60
GRID_H = 60
CELL_SIZE = 10
EVAL_STEPS = 600
INPUTS = 9

# Colors for aesthetic
BLACK = (0, 0, 0)
NEON_GREEN = (57, 255, 20)
TEXT_COLOR = (255, 255, 255)
DARK_GRAY = (30, 30, 30)

# Window Layout
MARGIN = 40
HEADER_H = 80
GRID_PIXELS_W = GRID_W * CELL_SIZE
GRID_PIXELS_H = GRID_H * CELL_SIZE

WINDOW_W = (GRID_PIXELS_W * 2) + (MARGIN * 3)
WINDOW_H = GRID_PIXELS_H + HEADER_H + MARGIN

if torch.cuda.is_available(): DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available(): DEVICE = torch.device("mps")
else: DEVICE = torch.device("cpu")

@dataclass
class Genome:
    w1: np.ndarray; b1: np.ndarray
    w2: np.ndarray; b2: np.ndarray
    w3: np.ndarray; b3: np.ndarray

    @staticmethod
    def from_dict(data: dict) -> "Genome":
        return Genome(
            np.array(data["w1"], dtype=np.float32), np.array(data["b1"], dtype=np.float32),
            np.array(data["w2"], dtype=np.float32), np.array(data["b2"], dtype=np.float32),
            np.array(data["w3"], dtype=np.float32), np.array(data["b3"], dtype=np.float32),
        )

def get_random_genome() -> Genome:
    """Generates a completely untrained network."""
    rng = np.random.default_rng()
    return Genome(
        rng.normal(0.0, 0.8, (INPUTS, 16)).astype(np.float32),
        rng.normal(0.0, 0.2, (16,)).astype(np.float32),
        rng.normal(0.0, 0.8, (16, 16)).astype(np.float32),
        rng.normal(0.0, 0.2, (16,)).astype(np.float32),
        rng.normal(0.0, 0.8, (16, 1)).astype(np.float32),
        rng.normal(0.0, 0.2, (1,)).astype(np.float32),
    )

def get_features() -> torch.Tensor:
    ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]

    xn = (xs / max(1, GRID_W - 1)) * 2.0 - 1.0
    yn = (ys / max(1, GRID_H - 1)) * 2.0 - 1.0

    radial = np.sqrt(xn * xn + yn * yn)
    radial_centred = radial - radial.mean()
    radial_norm = radial_centred / (radial_centred.std() + 1e-8)

    border_dist = np.minimum.reduce([
        xs / max(1, GRID_W - 1),
        ys / max(1, GRID_H - 1),
        (GRID_W - 1 - xs) / max(1, GRID_W - 1),
        (GRID_H - 1 - ys) / max(1, GRID_H - 1)
    ])
    border_dist = border_dist * 2.0 - 1.0

    feats = np.stack([
        xn,
        yn,
        radial_norm,
        np.sin(3.0 * math.pi * xn),
        np.cos(3.0 * math.pi * yn),
        np.sin(3.0 * math.pi * yn),
        np.cos(3.0 * math.pi * xn),
        border_dist,
        np.ones_like(xn)
    ], axis=-1)

    feats = feats.reshape(-1, INPUTS).astype(np.float32)
    return torch.tensor(feats, device=DEVICE).unsqueeze(0).expand(2, -1, -1)

@torch.no_grad()
def evaluate_genomes(trained: Genome, random: Genome) -> torch.Tensor:
    """Runs the CPPN for both genomes to get the starting seed grids."""
    population = [trained, random]
    print("trained w1 shape:", trained.w1.shape)
    print("random w1 shape:", random.w1.shape)
    W1 = torch.tensor(np.stack([g.w1 for g in population]), device=DEVICE)
    B1 = torch.tensor(np.stack([g.b1 for g in population]), device=DEVICE).unsqueeze(1)
    W2 = torch.tensor(np.stack([g.w2 for g in population]), device=DEVICE)
    B2 = torch.tensor(np.stack([g.b2 for g in population]), device=DEVICE).unsqueeze(1)
    W3 = torch.tensor(np.stack([g.w3 for g in population]), device=DEVICE)
    B3 = torch.tensor(np.stack([g.b3 for g in population]), device=DEVICE).unsqueeze(1)

    feats_t = get_features()
    h1 = torch.tanh(torch.bmm(feats_t, W1) + B1)
    h2 = torch.tanh(torch.bmm(h1, W2) + B2)
    out = torch.bmm(h2, W3) + B3

    # Output boolean grids [2, 1, 60, 60], converted to float32 for conv2d
    return (out > 0.5).view(2, 1, GRID_H, GRID_W).to(torch.float32)


@torch.no_grad()
def step_gol(grids: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    grids_padded = torch.cat([grids[:, :, -1:, :], grids, grids[:, :, :1, :]], dim=2)
    grids_padded = torch.cat([grids_padded[:, :, :, -1:], grids_padded, grids_padded[:, :, :, :1]], dim=3)

    neighbors = torch.round(F.conv2d(grids_padded, kernel, padding=0))

    nxt_grids = (
        ((grids > 0.5) & ((neighbors == 2.0) | (neighbors == 3.0))) |
        ((grids < 0.5) & (neighbors == 3.0))
    )

    return nxt_grids.to(torch.float32)

def draw_grid(surface, np_grid, x_offset, y_offset):
    """Draws a single grid."""
    # Draw Background
    pygame.draw.rect(surface, DARK_GRAY, (x_offset-1, y_offset-1, GRID_PIXELS_W+2, GRID_PIXELS_H+2), 1)
    pygame.draw.rect(surface, BLACK, (x_offset, y_offset, GRID_PIXELS_W, GRID_PIXELS_H))
    
    # Draw Cells
    # np_grid shape is (H, W)
    for y in range(GRID_H):
        for x in range(GRID_W):
            if np_grid[y, x] > 0.5:
                rect = (x_offset + x * CELL_SIZE, y_offset + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, NEON_GREEN, rect)

def main():
    checkpoint_file = Path("Runs/run_16/checkpoint.json")
    if not checkpoint_file.exists():
        print(f"Error: {checkpoint_file} not found. Run the training script first!")
        sys.exit(1)

    print("Loading Trained Checkpoint...")
    data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    trained_genome = Genome.from_dict(data["best_genome"])
    trained_fitness = float(data.get("best_fitness", 0.0))
    gen = int(data.get("generation", 0))

    random_genome = get_random_genome()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Game of Life AI Comparison")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 32)

    # Pre-allocate Convolution Kernel
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device=DEVICE)

    # Initial seeds
    base_grids = evaluate_genomes(trained_genome, random_genome)
    current_grids = base_grids.clone()

    step = 0
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset manually
                    step = 0
                    current_grids = base_grids.clone()

        screen.fill(BLACK)

        # Draw UI text
        title = font_large.render(f"Evolved Generator (Gen {gen})   vs   Random Generator", True, TEXT_COLOR)
        screen.blit(title, (WINDOW_W//2 - title.get_width()//2, 15))

        lbl_trained = font_small.render(f"Trained Agent (Fitness: {trained_fitness:.2f})", True, NEON_GREEN)
        screen.blit(lbl_trained, (MARGIN, 60))

        lbl_random = font_small.render("Untrained Random Agent", True, TEXT_COLOR)
        screen.blit(lbl_random, (MARGIN * 2 + GRID_PIXELS_W, 60))
        
        lbl_step = font_small.render(f"Step: {step} / {EVAL_STEPS} (Space: Pause, R: Reset)", True, TEXT_COLOR)
        screen.blit(lbl_step, (WINDOW_W//2 - lbl_step.get_width()//2, WINDOW_H - 30))

        # Convert tensors to CPU numpy arrays for rendering
        np_grids = current_grids.squeeze(1).cpu().numpy()

        # Draw left grid (Trained)
        draw_grid(screen, np_grids[0], MARGIN, HEADER_H)
        
        # Draw right grid (Random)
        draw_grid(screen, np_grids[1], MARGIN * 2 + GRID_PIXELS_W, HEADER_H)

        pygame.display.flip()

        if not paused:
            # Advance simulation
            current_grids = step_gol(current_grids, kernel)
            step += 1

            # Auto-reset after EVAL_STEPS to view the seeds again
            if step > EVAL_STEPS:
                step = 0
                current_grids = base_grids.clone()

        # Run at 15 FPS so we can clearly see the shapes evolve
        clock.tick(15)

    pygame.quit()

if __name__ == "__main__":
    main()

