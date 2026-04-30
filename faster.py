import argparse
import json
import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import os

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch required: pip install torch") from exc

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

@dataclass
class Config:
    grid_w: int = 60
    grid_h: int = 60
    population_size: int = 36
    elite_fraction: float = 0.08
    min_elites: int = 2
    eval_steps: int = 600
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.4
    min_mutation: float = 0.02
    anneal_cycle: int = 50
    inputs: int = 9
    h1: int = 16
    h2: int = 16
    seed: int = 777
    save_prefix: str = "life_evo_checkpoint_change_simplefitness"
    alive_threshold: float = 0.5

    @property
    def default_checkpoint(self) -> Path:
        return Path(f"{self.save_prefix}.json")

@dataclass
class Genome:
    w1: np.ndarray; b1: np.ndarray
    w2: np.ndarray; b2: np.ndarray
    w3: np.ndarray; b3: np.ndarray

    def clone(self) -> "Genome":
        return Genome(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(), self.w3.copy(), self.b3.copy())

    def to_dict(self) -> dict:
        return {"w1": self.w1.tolist(), "b1": self.b1.tolist(), "w2": self.w2.tolist(), "b2": self.b2.tolist(), "w3": self.w3.tolist(), "b3": self.b3.tolist()}

    @staticmethod
    def from_dict(data: dict) -> "Genome":
        return Genome(
            np.array(data["w1"], dtype=np.float32), np.array(data["b1"], dtype=np.float32),
            np.array(data["w2"], dtype=np.float32), np.array(data["b2"], dtype=np.float32),
            np.array(data["w3"], dtype=np.float32), np.array(data["b3"], dtype=np.float32),
        )

class LifeNeuroEvolver:
    def __init__(self, config: Config, use_compile: bool = False) -> None:
        self.config = config
        self.population_size = max(2, config.population_size)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
            
        self.population = [self.random_genome() for _ in range(self.population_size)]
        self.best_genome = self.random_genome()
        self.best_fitness = -1e9
        self.generation = 0
        self.current_mutation_sigma = self.config.mutation_sigma
        
        self.last_scores = []; self.last_mean_fitness = float("nan"); self.last_worst_fitness = float("nan")
        self.best_curve = []; self.last_seed_heat = np.zeros((60, 60), dtype=np.uint16)
        self.last_final_heat = np.zeros((60, 60), dtype=np.uint16); self.best_seed = np.zeros((60, 60), dtype=np.uint8)
        self.best_final = np.zeros((60, 60), dtype=np.uint8)
        self.fitness_history = []
        self.last_components = {}

        self._preallocate_tensors()
        if use_compile and hasattr(torch, "compile"): self._step_simulation = torch.compile(self._step_simulation)

    def _preallocate_tensors(self) -> None:
        H, W = self.config.grid_h, self.config.grid_w
        ys, xs = np.mgrid[0:H, 0:W]
        xn = (xs / max(1, W - 1)) * 2.0 - 1.0; yn = (ys / max(1, H - 1)) * 2.0 - 1.0
        radial = np.sqrt(xn * xn + yn * yn)
        radial_centred = radial - radial.mean()
        radial_norm = radial_centred / (radial_centred.std() + 1e-8)
        border_dist = np.minimum.reduce([
            xs / max(1, W - 1),
            ys / max(1, H - 1),
            (W - 1 - xs) / max(1, W - 1),
            (H - 1 - ys) / max(1, H - 1)
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

        feats = feats.reshape(-1, 9).astype(np.float32)
        self.feats_t = torch.tensor(feats, dtype=torch.float32, device=DEVICE)\
            .unsqueeze(0).expand(self.population_size, -1, -1)
        self.kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device=DEVICE)
        self.hash_proj = torch.randint(1, 1000000, (1, 1, H, W), device=DEVICE, dtype=torch.int64)

    def random_genome(self) -> Genome:
        return Genome(
            self.rng.normal(0.0, 0.8, (self.config.inputs, self.config.h1)).astype(np.float32),
            self.rng.normal(0.0, 0.2, (self.config.h1,)).astype(np.float32),
            self.rng.normal(0.0, 0.8, (self.config.h1, self.config.h2)).astype(np.float32),
            self.rng.normal(0.0, 0.2, (self.config.h2,)).astype(np.float32),
            self.rng.normal(0.0, 0.8, (self.config.h2, 1)).astype(np.float32),
            self.rng.normal(0.0, 0.2, (1,)).astype(np.float32),
        )

    def crossover(self, a: Genome, b: Genome) -> Genome:
        mix = lambda x, y: np.where(self.rng.random(x.shape) < 0.5, x, y).astype(np.float32)
        return Genome(mix(a.w1, b.w1), mix(a.b1, b.b1), mix(a.w2, b.w2), mix(a.b2, b.b2), mix(a.w3, b.w3), mix(a.b3, b.b3))

    def mutate(self, genome: Genome, rate: float, sigma: float) -> Genome:
        child = genome.clone()
        for arr in [child.w1, child.b1, child.w2, child.b2, child.w3, child.b3]:
            mask = self.rng.random(arr.shape) < rate
            if np.any(mask): arr[mask] += self.rng.normal(0.0, sigma, size=int(mask.sum())).astype(np.float32)
        return child

    def _calc_complexities_batched(self, g_f: torch.Tensor, n_f: torch.Tensor) -> torch.Tensor:
        density = g_f.mean(dim=(1, 2, 3))
        gx = torch.abs(torch.diff(g_f, dim=3)).sum(dim=(1, 2, 3)); gy = torch.abs(torch.diff(g_f, dim=2)).sum(dim=(1, 2, 3))
        edge_density = (gx + gy) / (self.config.grid_w * self.config.grid_h)
        n_flat = torch.round(n_f).view(self.population_size, -1).long()
        probs = F.one_hot(n_flat, num_classes=9).sum(dim=1).float() / (self.config.grid_h * self.config.grid_w)
        safe_probs = torch.where(probs > 0, probs, torch.tensor(1e-9, device=DEVICE))
        entropy = -(probs * torch.log2(safe_probs)).sum(dim=1) / math.log2(9.0)
        return 0.50 * edge_density + 0.35 * entropy + 0.15 * (4.0 * density * (1.0 - density))
    
    def _rle_complexity_batched(self, grids: torch.Tensor) -> torch.Tensor:
        flat = grids.squeeze(1).view(self.population_size, -1)
        transitions = (flat[:, 1:] != flat[:, :-1]).float().sum(dim=1)
        return transitions / (self.config.grid_h * self.config.grid_w - 1)

    def _step_simulation(self, grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_hashes = (grids.long() * self.hash_proj).sum(dim=(1, 2, 3))
        grids_padded = torch.cat([grids[:, :, -1:, :], grids, grids[:, :, :1, :]], dim=2)
        grids_padded = torch.cat([grids_padded[:, :, :, -1:], grids_padded, grids_padded[:, :, :, :1]], dim=3)
        neighbors = torch.round(F.conv2d(grids_padded, self.kernel, padding=0))
        nxt_grids = (((grids > 0.5) & ((neighbors == 2.0) | (neighbors == 3.0))) | ((grids < 0.5) & (neighbors == 3.0))).to(torch.float32)
        return nxt_grids, step_hashes, (nxt_grids != grids).float().mean(dim=(1, 2, 3)), self._calc_complexities_batched(nxt_grids, neighbors)

    @torch.no_grad()
    def evaluate_population(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        H = self.config.grid_h
        W = self.config.grid_w

        ys = torch.arange(H, device=DEVICE).float().view(1, 1, H, 1)
        xs = torch.arange(W, device=DEVICE).float().view(1, 1, 1, W)

        com_x_history = []
        com_y_history = []
        W1 = torch.tensor(np.stack([g.w1 for g in self.population]), device=DEVICE)
        B1 = torch.tensor(np.stack([g.b1 for g in self.population]), device=DEVICE).unsqueeze(1)
        W2 = torch.tensor(np.stack([g.w2 for g in self.population]), device=DEVICE)
        B2 = torch.tensor(np.stack([g.b2 for g in self.population]), device=DEVICE).unsqueeze(1)
        W3 = torch.tensor(np.stack([g.w3 for g in self.population]), device=DEVICE)
        B3 = torch.tensor(np.stack([g.b3 for g in self.population]), device=DEVICE).unsqueeze(1)

        feats_flat = self.feats_t.reshape(self.population_size,self.config.grid_h * self.config.grid_w,self.config.inputs)

        raw_out = torch.bmm(torch.tanh(torch.bmm(torch.tanh(torch.bmm(feats_flat, W1) + B1),W2) + B2),W3) + B3


        grids = (raw_out > self.config.alive_threshold).view(self.population_size, 1, self.config.grid_h, self.config.grid_w).to(torch.float32)
        # Measure initial spatial spread of seed:
        seed_active = grids.squeeze(1)  # (pop, H, W)
        seed_y = torch.any(seed_active, dim=2).sum(dim=1).float() / self.config.grid_h
        seed_x = torch.any(seed_active, dim=1).sum(dim=1).float() / self.config.grid_w
        initial_spread = seed_y * seed_x  # fraction of rows × fraction of cols occupied


        
        seed_grids = grids.clone()
        initial_density = seed_grids.mean(dim=(1,2,3))
        density_seed_score = 1.0 - torch.abs(initial_density - 0.18) / 0.18
        density_seed_score = torch.clamp(density_seed_score, 0.0, 1.0)
        density_top = seed_grids[:, :, :H//2, :].mean()
        density_bottom = seed_grids[:, :, H//2:, :].mean()

        if self.generation % 50 == 0:
            print(f"Seed density top: {density_top:.3f}, bottom: {density_bottom:.3f}")

        all_hashes = torch.empty((self.population_size, self.config.eval_steps), device=DEVICE, dtype=torch.int64)
        curve_history = torch.empty((self.population_size, self.config.eval_steps), device=DEVICE, dtype=torch.float32)
        max_complexities = torch.zeros(self.population_size, device=DEVICE)
        final_complexities = torch.zeros(self.population_size, device=DEVICE)
        activity_sum = torch.zeros(self.population_size, device=DEVICE)
        prev_grid = None
        period2_matches = []
        alive_at_step = torch.zeros(self.population_size, self.config.eval_steps, device=DEVICE)
        activity_early = torch.zeros(self.population_size, device=DEVICE)
        activity_mid   = torch.zeros(self.population_size, device=DEVICE)
        activity_late  = torch.zeros(self.population_size, device=DEVICE)
        density_in_range = torch.zeros(self.population_size, device=DEVICE)
        interaction_events = torch.zeros(self.population_size, device=DEVICE)
        prev_local_activity = torch.zeros(self.population_size, 1, H, W, device=DEVICE)

        compressibility_sum = torch.zeros(self.population_size, device=DEVICE)
        
        

        third = self.config.eval_steps // 3

        late_zero_change = torch.zeros(self.population_size, device=DEVICE)

        prev_grids = grids.clone()
        

        for step in range(self.config.eval_steps):
            grids, step_hashes, activity, complexity = self._step_simulation(grids)
            currently_changing = (grids != prev_grids).float()
            interaction_mask = (currently_changing > 0.5) & (prev_local_activity < 0.1)
            interaction_events += interaction_mask.float().sum(dim=(1, 2, 3))
            prev_local_activity = 0.8 * prev_local_activity + 0.2 * currently_changing
            rle_complexity = self._rle_complexity_batched(grids)
            compressibility_sum += rle_complexity
            all_hashes[:, step] = step_hashes; curve_history[:, step] = complexity; activity_sum += activity
            max_complexities = torch.max(max_complexities, complexity); final_complexities = complexity
            mass = grids.sum(dim=(1, 2, 3)).clamp(min=1.0)
            com_y = (grids * ys).sum(dim=(1, 2, 3)) / mass
            com_x = (grids * xs).sum(dim=(1, 2, 3)) / mass
            alive_at_step[:, step] = (grids.sum(dim=(1,2,3)) > 0).float()

            com_x_history.append(com_x)
            com_y_history.append(com_y)
            prev_grid = grids.clone()

            if step >= 2 * third:
                consecutive_change = (grids != prev_grids).float().mean(dim=(1,2,3))
                late_zero_change += (consecutive_change < 0.005).float()

            prev_grids = grids.clone()

            if step < third:
                activity_early += activity
            elif step < 2 * third:
                activity_mid += activity
            else:
                activity_late += activity

            density = grids.mean(dim=(1,2,3))
            density_in_range += ((density > 0.05) & (density < 0.35)).float()

        density_viability = density_in_range / self.config.eval_steps
        compressibility_avg = compressibility_sum / self.config.eval_steps

        # Reward medium compressibility: not too simple, not random noise
        compressibility_score = 1.0 - torch.abs(compressibility_avg - 0.15) / 0.15
        compressibility_score = torch.clamp(compressibility_score, 0.0, 1.0)

        interaction_score = torch.clamp(interaction_events / 2000.0, 0.0, 1.0)

        mid_min_activity = curve_history[:, third:2*third].min(dim=1).values
        late_max_activity = curve_history[:, 2*third:].max(dim=1).values
        revival_score = torch.clamp(late_max_activity - mid_min_activity, 0.0, 0.5) / 0.5

        stasis_fraction = late_zero_change / max(1, (self.config.eval_steps - 2 * third))


        activity_early /= third
        activity_mid   /= third
        activity_late  /= (self.config.eval_steps - 2 * third)
        born_static_penalty = (activity_early < 0.005).float()

        active = grids.squeeze(1)  # shape: (pop, H, W)
        total = active.sum(dim=(1,2)).clamp(min=1.0)

        xs_grid = torch.linspace(-1, 1, self.config.grid_w, device=DEVICE).view(1, 1, self.config.grid_w)
        ys_grid = torch.linspace(-1, 1, self.config.grid_h, device=DEVICE).view(1, self.config.grid_h, 1)

        cx = (active * xs_grid).sum(dim=(1,2)) / total
        cy = (active * ys_grid).sum(dim=(1,2)) / total

        centring_reward = 1.0 - (cx.abs() + cy.abs()) / 2.0

        com_x_history = torch.stack(com_x_history, dim=1)  # (pop, steps)
        com_y_history = torch.stack(com_y_history, dim=1)  # (pop, steps)
        com_travel = com_x_history.var(dim=1) + com_y_history.var(dim=1)
        # Mask CoM by alive steps before computing variance:
        alive_mask = alive_at_step  # (pop, steps)
        com_x_masked = com_x_history * alive_mask
        com_y_masked = com_y_history * alive_mask

        # Only compute variance over alive steps to avoid dead-pattern inflation:
        n_alive = alive_mask.sum(dim=1).clamp(min=1)
        com_x_mean = com_x_masked.sum(dim=1) / n_alive
        com_y_mean = com_y_masked.sum(dim=1) / n_alive
        com_travel_raw = ((com_x_masked - com_x_mean.unsqueeze(1)) ** 2 * alive_mask).sum(dim=1) / n_alive \
                    + ((com_y_masked - com_y_mean.unsqueeze(1)) ** 2 * alive_mask).sum(dim=1) / n_alive
        com_travel = torch.clamp(com_travel_raw / 200.0, 0.0, 1.0)  # normalise: ~14-cell RMS travel = 1.0
        lifespan = alive_at_step.sum(dim=1) / self.config.eval_steps
        sustained_survival = lifespan * torch.clamp(activity_late * 10.0, 0.0, 1.0)

        
        period_penalty = torch.zeros(self.population_size, device=DEVICE)
        mid = self.config.eval_steps // 2
        for period in range(2, 31):
            if self.config.eval_steps > period * 2:
                h_now = all_hashes[:, period:]
                h_back = all_hashes[:, :-period]
                matches = (h_now == h_back).float().mean(dim=1)
                period_penalty = torch.max(period_penalty, matches)

        sorted_hashes, _ = torch.sort(all_hashes, dim=1)
        unique_counts = (sorted_hashes[:, 1:] != sorted_hashes[:, :-1]).sum(dim=1) + 1
        
        novelty = unique_counts.float() / max(1, self.config.eval_steps)
        motion = activity_sum / max(1, self.config.eval_steps)
        temporal_richness = curve_history.mean(dim=1)
        freeze_penalty = (motion < 0.05).float() #New
        #NEW: Spatial Spread 
        active = grids.squeeze(1)  # shape: (population, H, W)


        y_coords = torch.any(active, dim=2)  # rows with activity
        x_coords = torch.any(active, dim=1)  # columns with activity

        height = y_coords.sum(dim=1).float()
        width = x_coords.sum(dim=1).float()
        third = self.config.eval_steps // 3

        died_early = (alive_at_step[:, third] < 0.5).float()

        spread = (height * width) / (self.config.grid_h * self.config.grid_w)
        repeat_penalty = 1.0 - novelty

        born_static_penalty = (activity_early < 0.005).float()
        low_late_activity_penalty = (activity_late < 0.01).float()

        dynamic_gate = (
            lifespan
            * torch.clamp(activity_late * 20.0, 0.0, 1.0)
            * (1.0 - stasis_fraction)
            * (1.0 - period_penalty)
        )
        
        fitnesses = (
            4.0 * activity_late
            + 3.0 * activity_mid
            + 3.0 * com_travel
            + 2.0 * density_viability
            + 2.0 * density_seed_score
            + 1.5 * compressibility_score
            + 1.5 * revival_score

            - 6.0 * born_static_penalty
            - 5.0 * stasis_fraction
            - 5.0 * period_penalty
            - 2.0 * died_early
        )



    #fitnesses = (
 #   1.2 * max_complexities
  #  + 0.8 * final_complexities
   # + 1.5 * com_travel
    #+ 0.8 * spread
 #   + 0.8 * novelty
  #  - 2.0 * period2_penalty
   # + 3.0 * sustained_survival
   # - 2.0 * stasis_fraction
   # + 1.0 * centring_reward
   # - 1.5 * died_early
   # + 0.3 * activity_early
   # + 0.6 * activity_mid
   # + 2.5 * activity_late
   # + 1.5 * density_viability
   # )

        self.last_components = {
            "sustained_survival": sustained_survival.cpu().numpy(),
            "com_travel": com_travel.cpu().numpy(),
            "novelty": novelty.cpu().numpy(),
            "density_viability": density_viability.cpu().numpy(),
            "activity_mid": activity_mid.cpu().numpy(),
            "activity_late": activity_late.cpu().numpy(),
            "period_penalty": period_penalty.cpu().numpy(),
            "stasis_fraction": stasis_fraction.cpu().numpy(),
            "died_early": died_early.cpu().numpy(),
            "initial_spread": initial_spread.cpu().numpy(),
            "compressibility_score": compressibility_score.cpu().numpy(),
            "interaction_score": interaction_score.cpu().numpy(),
            "revival_score": revival_score.cpu().numpy(),
            "born_static_penalty": born_static_penalty.cpu().numpy(),
            "density_seed_score": density_seed_score.cpu().numpy(),
        }


        return fitnesses.cpu().numpy(), seed_grids.squeeze(1).cpu().numpy().astype(np.uint8), grids.squeeze(1).cpu().numpy().astype(np.uint8), curve_history.cpu().numpy()

    def evolve_one_generation(self) -> None:
        fitnesses, seed_grids, final_grids, curves = self.evaluate_population()
        scored = list(zip(fitnesses, self.population, seed_grids, final_grids, curves))
        scored.sort(key=lambda item: item[0], reverse=True)

        self.last_seed_heat = seed_grids.astype(np.uint16).sum(axis=0)
        self.last_final_heat = final_grids.astype(np.uint16).sum(axis=0)
        self.last_scores = [float(score) for score, *_ in scored]
        self.last_mean_fitness = float(np.mean(self.last_scores))
        self.last_worst_fitness = float(self.last_scores[-1])
        
        best_score, best_genome, best_seed, best_final, best_curve = scored[0]

        if best_score > self.best_fitness:
            self.best_fitness = float(best_score); self.best_genome = best_genome.clone(); self.best_seed = best_seed.copy(); self.best_final = best_final.copy(); self.best_curve = [float(x) for x in best_curve]
        elif not self.best_curve:
            self.best_curve = [float(x) for x in best_curve]; self.best_seed = best_seed.copy(); self.best_final = best_final.copy()

        progress = (self.generation % self.config.anneal_cycle) / self.config.anneal_cycle
        self.current_mutation_sigma = self.config.min_mutation + (self.config.mutation_sigma - self.config.min_mutation) * (0.5 * (1 + math.cos(math.pi * progress)))

        elites = [genome for _, genome, *_ in scored[: max(self.config.min_elites, int(round(self.population_size * self.config.elite_fraction)))]]
        new_population = [g.clone() for g in elites[:2]]

        num_random_immigrants = max(3, int(0.15 * self.population_size))

        while len(new_population) < self.population_size - num_random_immigrants:
            i1, i2 = self.rng.integers(0, len(elites), size=2)

            child = self.mutate(
                self.crossover(elites[int(i1)], elites[int(i2)]),
                rate=self.config.mutation_rate,
                sigma=self.current_mutation_sigma
            )

            new_population.append(child)

        while len(new_population) < self.population_size:
            new_population.append(self.random_genome())
        
        self.population = new_population
        self.generation += 1
        self.fitness_history.append({
        "generation": self.generation,
        "best": float(self.best_fitness),
        "mean": float(self.last_mean_fitness),
        "worst": float(self.last_worst_fitness)
     })

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "generation": self.generation,
            "population_size": self.population_size,
            "best_fitness": float(self.best_fitness),
            "best_genome": self.best_genome.to_dict(),
            "population": [genome.to_dict() for genome in self.population],
            "rng_state": self.rng.bit_generator.state,
            "best_curve": [float(x) for x in self.best_curve],
            "best_seed": self.best_seed.tolist(),
            "best_final": self.best_final.tolist(),
            "last_seed_heat": self.last_seed_heat.tolist(),
            "last_final_heat": self.last_final_heat.tolist(),
            "last_scores": [float(x) for x in self.last_scores],
            "last_mean_fitness": float(self.last_mean_fitness),
            "last_worst_fitness": float(self.last_worst_fitness),
        }, cls=NumpyEncoder), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.generation = int(data.get("generation", 0))
        self.population = [Genome.from_dict(item) for item in data["population"]]
        self.best_fitness = float(data["best_fitness"]); self.best_genome = Genome.from_dict(data["best_genome"])
        if "rng_state" in data: self.rng.bit_generator.state = data["rng_state"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--max_generations", type=int, default=10000)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def inspect_random_boards(engine, n=30, threshold=None):
    if threshold is None:
        threshold = engine.config.alive_threshold
    
    densities = []
    boards = []

    for _ in range(n):
        g = engine.random_genome()

        # convert genome → board (same as your forward pass)
        feats = engine.feats_t[0].reshape(-1, engine.config.inputs).unsqueeze(0)

        W1 = torch.tensor(g.w1, device=DEVICE).unsqueeze(0)
        B1 = torch.tensor(g.b1, device=DEVICE).unsqueeze(0).unsqueeze(1)
        W2 = torch.tensor(g.w2, device=DEVICE).unsqueeze(0)
        B2 = torch.tensor(g.b2, device=DEVICE).unsqueeze(0).unsqueeze(1)
        W3 = torch.tensor(g.w3, device=DEVICE).unsqueeze(0)
        B3 = torch.tensor(g.b3, device=DEVICE).unsqueeze(0).unsqueeze(1)

        out = torch.bmm(torch.tanh(torch.bmm(torch.tanh(torch.bmm(feats, W1) + B1),W2) + B2),W3) + B3

        grid = (out > threshold).view(engine.config.grid_h, engine.config.grid_w).cpu().numpy()

        density = grid.mean()
        densities.append(density)
        boards.append(grid)

    if len(densities) == 0:
        print("\n=== RAW OUTPUT STATS FOR FIRST RANDOM GENOME ===")
        print(f"min raw:  {out.min().item():.3f}")
        print(f"max raw:  {out.max().item():.3f}")
        print(f"mean raw: {out.mean().item():.3f}")

    # show 9 boards
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(boards[i], cmap="gray")
        ax.set_title(f"{densities[i]:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def main():
    args = parse_args()
    config = Config(population_size=args.population)

    run_name = f"run_{int(time.time())}"
    run_dir = Path("/Users/ziad/Desktop/Meng/Bio-inspired/CW2/MarkI/Runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {run_dir}")

    checkpoint_path = run_dir / "checkpoint.json"
    fitness_path = run_dir / "fitness_log.json"

    engine = LifeNeuroEvolver(config, use_compile=args.compile)
    inspect_random_boards(engine, n=30)

    if args.resume:
        engine.load_checkpoint(Path(args.resume))

    start_time = time.perf_counter()
    try:
        while engine.generation < args.max_generations:
            gen_start = time.perf_counter()
            engine.evolve_one_generation()

            print(
                f"Gen {engine.generation:05d} | "
                f"Best Fit: {engine.best_fitness:7.3f} | "
                f"Mean Fit: {engine.last_mean_fitness:7.3f} | "
                f"Sigma: {engine.current_mutation_sigma:.4f} | "
                f"Time: {time.perf_counter() - gen_start:.3f}s"
            )

            
            if engine.generation % 50 == 0:
                for name, vals in engine.last_components.items():
                    print(f"  {name}: min={vals.min():.3f} mean={vals.mean():.3f} max={vals.max():.3f}")

            if args.save_every > 0 and engine.generation % args.save_every == 0:
                engine.save_checkpoint(checkpoint_path)
                fitness_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fitness_path, "w") as f:
                    json.dump(engine.fitness_history, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")

    finally:
        print(f"\nFinished in {time.perf_counter() - start_time:.2f}s")
        engine.save_checkpoint(checkpoint_path)
        fitness_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fitness_path, "w") as f:
            json.dump(engine.fitness_history, f, indent=2)

if __name__ == "__main__": main()


