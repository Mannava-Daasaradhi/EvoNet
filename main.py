import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from copy import deepcopy

# --------------------------
# === Utilities & Setup ====
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --------------------------
# ==== Activations / RNA ===
# --------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation_by_name(name):
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "swish": Swish(),
        "leakyrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "gelu": nn.GELU()
    }.get(name, nn.ReLU())

class MixedActivation(nn.Module):
    def __init__(self, act_name_a, act_name_b, alpha=0.5):
        super().__init__()
        self.a = get_activation_by_name(act_name_a)
        self.b = get_activation_by_name(act_name_b)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, x):
        return self.alpha * self.a(x) + (1.0 - self.alpha) * self.b(x)

class RNA:
    ACTIVATIONS = ["ReLU", "Swish", "LeakyReLU", "Tanh", "ELU", "GELU"]
    activation_pairs = {}

    @staticmethod
    def select_new_activation(current_activation):
        if random.random() < 0.3 or not RNA.activation_pairs:
            return random.choice(RNA.ACTIVATIONS)
        candidates = [(pair, score) for pair, score in RNA.activation_pairs.items() if current_activation in pair]
        if not candidates:
            return random.choice(RNA.ACTIVATIONS)
        best_pair, _ = max(candidates, key=lambda x: x[1])
        return best_pair[1] if best_pair[0] == current_activation else best_pair[0]

    @staticmethod
    def update_activation_pair(act1, act2, reward):
        key = tuple(sorted([act1, act2]))
        RNA.activation_pairs[key] = 0.9 * RNA.activation_pairs.get(key, 0.0) + 0.1 * reward

# --------------------------
# ===== GlobalDNA ==========
# --------------------------
class GlobalDNA:
    def __init__(self):
        self.traits = {
            "structure": {
                "conv_channels": [1, 32, 64],
                "kernel_sizes": [3, 3],
                "pool": 2,
                "dropout": 0.25,
                "activation_mix": {"ReLU": 0.4, "Swish": 0.3, "LeakyReLU": 0.3},
                "residual": False
            },
            "learning": {"learning_rate": 0.01, "mutation_rate": 0.1, "temperature": 0.05}
        }
        self.previous_loss = None

    def metropolis_accept(self, old_loss, new_loss, temperature):
        if old_loss is None: return True
        delta = new_loss - old_loss
        if delta < 0: return True
        return np.random.rand() < np.exp(-delta / max(temperature, 1e-8))

    def normalize_activation_mix(self):
        mix = self.traits["structure"]["activation_mix"]
        total = sum(mix.values())
        if total <= 0:
            n = max(1, len(mix))
            for k in mix: mix[k] = 1.0 / n
            return
        for k in mix: mix[k] = mix[k] / total

    def mutate(self, current_loss):
        learning = self.traits["learning"]
        struct = self.traits["structure"]
        if self.previous_loss is None: self.previous_loss = current_loss
        if not self.metropolis_accept(self.previous_loss, current_loss, learning["temperature"]): return
        learning["temperature"] = max(1e-4, learning["temperature"] * 0.95)
        for act in list(struct["activation_mix"].keys()):
            if random.random() < learning["mutation_rate"]:
                struct["activation_mix"][act] = np.clip(struct["activation_mix"][act] + random.uniform(-0.05, 0.05), 0, 1)
        self.normalize_activation_mix()
        if random.random() < learning["mutation_rate"]:
            struct["dropout"] = float(np.clip(struct["dropout"] + random.uniform(-0.05, 0.05), 0.0, 0.9))
        if random.random() < learning["mutation_rate"]:
            ch = struct["conv_channels"]
            for i in range(1, len(ch)):
                if random.random() < 0.5:
                    ch[i] = max(8, int(ch[i] + np.random.randint(-8, 9)))
            struct["conv_channels"] = ch
        self.previous_loss = current_loss

# --------------------------
# ===== LocalDNA ===========
# --------------------------
class LocalDNA:
    def __init__(self, base_lr=0.01, base_mutation=0.1, act_fn=("ReLU", "Swish"), act_alpha=0.5, dropout=0.25):
        self.loss_history = deque(maxlen=20)
        self.learning_rate = float(base_lr)
        self.mutation_rate = float(base_mutation)
        self.activation_pair = act_fn
        self.activation_alpha = float(act_alpha)
        self.dropout = float(dropout)
        self.life = 30
        self.is_updating = True
        self.best_hyperparams = {"learning_rate": self.learning_rate, "dropout": self.dropout,
                                 "activation_alpha": self.activation_alpha, "mutation_rate": self.mutation_rate}

    def update_loss(self, loss):
        self.loss_history.append(loss)
        self.failure_function()

    def average_loss(self):
        return np.mean(self.loss_history) if self.loss_history else float('inf')

    def mutate(self):
        self.learning_rate = max(1e-6, self.learning_rate + np.random.normal(0, self.learning_rate * 0.1))
        self.mutation_rate = float(np.clip(self.mutation_rate + np.random.normal(0, 0.02), 0.0, 1.0))
        self.activation_alpha = float(np.clip(self.activation_alpha + np.random.normal(0, 0.05), 0.0, 1.0))
        if random.random() < 0.1: self.activation_pair = (
        random.choice(RNA.ACTIVATIONS), random.choice(RNA.ACTIVATIONS))
        if random.random() < 0.1: self.dropout = float(np.clip(self.dropout + np.random.normal(0, 0.02), 0.0, 0.9))
        self.life = 30
        self.is_updating = True

    def failure_function(self):
        if len(self.loss_history) < 4: return
        recent = list(self.loss_history)[-4:]
        if recent[-1] > recent[0] * 1.01:
            self.learning_rate *= 0.9
            self.mutation_rate = min(1.0, self.mutation_rate * 1.1 + 0.02)
            self.dropout = float(np.clip(self.dropout + 0.02, 0.0, 0.9))
            self.activation_alpha = float(0.9 * self.activation_alpha + 0.1 * 0.5)
        else:
            self.best_hyperparams["learning_rate"] = 0.95 * self.best_hyperparams["learning_rate"] + 0.05 * self.learning_rate
            self.best_hyperparams["dropout"] = 0.95 * self.best_hyperparams["dropout"] + 0.05 * self.dropout
            self.best_hyperparams["activation_alpha"] = 0.95 * self.best_hyperparams[
                "activation_alpha"] + 0.05 * self.activation_alpha
            self.best_hyperparams["mutation_rate"] = 0.95 * self.best_hyperparams["mutation_rate"] + 0.05 * self.mutation_rate

    def integrate_grid_search_result(self, best_result):
        step = 0.3
        self.learning_rate = float(
            (1 - step) * self.learning_rate + step * best_result.get("learning_rate", self.learning_rate))
        self.dropout = float((1 - step) * self.dropout + step * best_result.get("dropout", self.dropout))
        self.activation_alpha = float(
            (1 - step) * self.activation_alpha + step * best_result.get("activation_alpha", self.activation_alpha))
        self.mutation_rate = float(
            (1 - step) * self.mutation_rate + step * best_result.get("mutation_rate", self.mutation_rate))

# --------------------------
# ===== NeuralCell (CNN) ===
# --------------------------
class NeuralCell(nn.Module):
    _flat_cache = {}

    def __init__(self, global_dna: GlobalDNA, local_dna: LocalDNA):
        super().__init__()
        self.global_dna = global_dna
        self.local_dna = local_dna
        self.growth_rate = 1.0
        self.contribution_score = 0.0
        self.age = 0
        self.loss_value = None
        self.life = int(getattr(self.local_dna, "life", 30))
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.build_model()

    def _compute_flat_size(self, ch_list, pool):
        n_pools = max(0, len(ch_list) - 1)
        final_h = 28 // (pool ** n_pools)
        final_w = final_h
        final_c = ch_list[-1]
        return final_c * final_h * final_w

    def build_model(self):
        struct = self.global_dna.traits["structure"]
        ch = struct["conv_channels"]
        ks = struct["kernel_sizes"]
        pool = struct["pool"]
        dropout = self.local_dna.dropout
        layers = []
        in_ch = ch[0]
        for i in range(1, len(ch)):
            out_ch = ch[i]
            k = ks[i - 1] if i - 1 < len(ks) else 3
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2), nn.BatchNorm2d(out_ch),
                       MixedActivation(self.local_dna.activation_pair[0], self.local_dna.activation_pair[1],
                                       alpha=self.local_dna.activation_alpha),
                       nn.MaxPool2d(pool)]
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*layers).to(DEVICE)
        flat_key = (tuple(ch), pool)
        if flat_key in NeuralCell._flat_cache:
            flat = NeuralCell._flat_cache[flat_key]
        else:
            flat = self._compute_flat_size(ch, pool)
            NeuralCell._flat_cache[flat_key] = flat
        hidden = 128
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.BatchNorm1d(hidden),
                                        get_activation_by_name(self.local_dna.activation_pair[0]),
                                        nn.Dropout(dropout), nn.Linear(hidden, 10)).to(DEVICE)
        self.to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=self.local_dna.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def rebuild_model_with_updated_dna(self):
        self.build_model()

    def encode_dna_to_tensor(self, n_samples=32):
        alpha = float(self.local_dna.activation_alpha)
        lr = float(self.local_dna.learning_rate)
        drop = float(self.local_dna.dropout)
        z = torch.randn(n_samples, 1, 28, 28, device=DEVICE, dtype=torch.float32) * 0.1
        coords = torch.linspace(-1.0, 1.0, 28, device=DEVICE)
        xx, yy = torch.meshgrid(coords, coords, indexing='xy')
        rr = xx ** 2 + yy ** 2
        circle = torch.exp(-rr / (0.5 + 0.5 * (1.0 - alpha))).unsqueeze(0).unsqueeze(0)
        factor = (0.2 + 0.8 * alpha) * (lr * 5.0)
        z = z + circle * factor
        if drop > 0:
            mask = (torch.rand(n_samples, 1, 28, 28, device=DEVICE) > drop).float()
            z = z * mask
        labels = (torch.randint(0, 10, (n_samples,), device=DEVICE).long() + int(alpha * 9)) % 10
        return z, labels

    def train_step(self, x, y):
        self.train()
        for g in self.optimizer.param_groups: g['lr'] = self.local_dna.learning_rate
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if self.scaler is not None:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = self.forward(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_val = float(loss.detach().cpu().item())
        else:
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            loss_val = float(loss.detach().cpu().item())

        self.loss_value = loss_val
        self.local_dna.update_loss(loss_val)
        self.age += 1
        avg = self.local_dna.average_loss()
        if self.loss_value < avg:
            self.growth_rate *= 1.02
            self.contribution_score += (avg - self.loss_value)
            self.life = int(getattr(self.local_dna, "life", 30))
        else:
            self.growth_rate *= 0.995
            self.life = max(0, int(self.life) - 1)

        return self.loss_value

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def evaluate(self):
        return self.loss_value if self.loss_value is not None else float('inf')

# --------------------------
# ===== Data Manager =======
# --------------------------
class DataManager:
    def __init__(self, x, y):
        self.x_data = x.to(DEVICE)
        self.y_data = y.to(DEVICE)
        self.prev_size = len(x)

    def sample_batch(self, batch_size=64):
        if len(self.x_data) == 0: return None, None
        idx = np.random.choice(len(self.x_data), min(batch_size, len(self.x_data)), replace=False)
        return self.x_data[idx], self.y_data[idx]

# ==========================================================
# ===== START: TISSUE CLASS AND LOGIC (RE-INTEGRATED) ======
# ==========================================================
class Tissue:
    def __init__(self, cells):
        self.cells = cells
        self.aggregate_loss = np.mean([c.evaluate() for c in cells if c.evaluate() is not None])
        self.hyperparam_signature = self.compute_signature()

    def compute_signature(self):
        if not self.cells:
            return {"lr": 0.01, "drop": 0.25, "alpha": 0.5, "mut": 0.1} # Default signature
        lrs = [c.local_dna.learning_rate for c in self.cells]
        drops = [c.local_dna.dropout for c in self.cells]
        alphas = [c.local_dna.activation_alpha for c in self.cells]
        muts = [c.local_dna.mutation_rate for c in self.cells]
        return {
            "lr": float(np.mean(lrs)),
            "drop": float(np.mean(drops)),
            "alpha": float(np.mean(alphas)),
            "mut": float(np.mean(muts))
        }

    def spawn_offspring(self, global_dna, n_offspring=8):
        offspring = []
        if not self.cells: return offspring
        for _ in range(n_offspring):
            parent = random.choice(self.cells)
            child_dna = deepcopy(parent.local_dna)
            child_dna.mutate()
            # Nudge child's DNA toward the tissue's signature
            sig = self.hyperparam_signature
            child_dna.learning_rate = float(0.9 * child_dna.learning_rate + 0.1 * sig["lr"])
            child_dna.dropout = float(0.9 * child_dna.dropout + 0.1 * sig["drop"])
            child = NeuralCell(global_dna, child_dna)
            offspring.append(child)
        return offspring
# ========================================================
# ===== END: TISSUE CLASS AND LOGIC (RE-INTEGRATED) ======
# ========================================================

# --------------------------
# ===== Population logic ===
# --------------------------
MAX_POPULATION = 32
FAILURE_THRESHOLD = 1.2
EVOLVE_EVERY = 10
NUM_CHILDREN_PER_PARENT = 2
GRID_SEARCH_BUDGET_STEPS = 2

def choose_activation_from_mix(global_dna):
    mix = global_dna.traits["structure"]["activation_mix"]
    acts, probs = zip(*mix.items())
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
    return tuple(np.random.choice(acts, size=2, p=probs))

def apply_immunity_block(population, global_dna, avg_loss):
    for cell in population:
        loss = cell.evaluate()
        if loss is None: continue

        fail = loss > avg_loss * FAILURE_THRESHOLD
        stagnant = cell.age > 50 and cell.contribution_score < 1e-3

        if fail or stagnant:
            a1, a2 = cell.local_dna.activation_pair
            mix = global_dna.traits["structure"]["activation_mix"]
            for act in (a1, a2):
                if act in mix:
                    mix[act] = max(0.0, mix[act] - 0.02)
            global_dna.normalize_activation_mix()
            cell.local_dna.mutate()
            cell.rebuild_model_with_updated_dna()
            cell.life = cell.local_dna.life

        if loss < avg_loss * 0.9:
            a1, a2 = cell.local_dna.activation_pair
            mix = global_dna.traits["structure"]["activation_mix"]
            for act in (a1, a2):
                mix[act] = mix.get(act, 0.0) + 0.01
            global_dna.normalize_activation_mix()

def _evaluate_localdna_candidate(global_dna, local_dna, steps=2, batch_size=32):
    tmp = NeuralCell(global_dna, local_dna)
    losses = []
    for _ in range(steps):
        x, y = tmp.encode_dna_to_tensor(batch_size)
        losses.append(tmp.train_step(x, y))
    return np.mean(losses) if losses else float('inf')

def evolve_population(population, global_dna, data_manager=None):
    if not population: return

    scored = sorted(population, key=lambda c: c.evaluate() or float('inf'))
    num_parents = max(1, int(len(scored) * 0.3))
    parents = scored[:num_parents]
    children = []
    for parent in parents:
        for _ in range(NUM_CHILDREN_PER_PARENT):
            child_local = deepcopy(parent.local_dna)
            if random.random() < 0.3:
                child_local.activation_pair = choose_activation_from_mix(global_dna)
            child_local.mutate()
            score = _evaluate_localdna_candidate(global_dna, child_local, steps=GRID_SEARCH_BUDGET_STEPS)
            children.append((score, child_local))

    existing_losses = [c.evaluate() for c in population if c.evaluate() is not None]
    median_loss = np.median(existing_losses) if existing_losses else float('inf')
    children = [(s, l) for s, l in children if s < median_loss * 0.95]
    children.sort(key=lambda x: x[0])

    for score, localdna in children:
        if len(population) >= MAX_POPULATION: break
        new_cell = NeuralCell(global_dna, localdna)
        # Vaccinate child toward population mean
        step = 0.1
        mean_lr = np.mean([c.local_dna.learning_rate for c in population])
        mean_drop = np.mean([c.local_dna.dropout for c in population])
        new_cell.local_dna.learning_rate = (1 - step) * new_cell.local_dna.learning_rate + step * mean_lr
        new_cell.local_dna.dropout = (1 - step) * new_cell.local_dna.dropout + step * mean_drop
        population.append(new_cell)

def apply_vaccination(population, vaccination_rate=0.05):
    if not population: return
    mean_lr = np.mean([c.local_dna.learning_rate for c in population])
    mean_drop = np.mean([c.local_dna.dropout for c in population])
    mean_alpha = np.mean([c.local_dna.activation_alpha for c in population])
    mean_mut = np.mean([c.local_dna.mutation_rate for c in population])
    worst = sorted(population, key=lambda c: c.evaluate() or float('inf'), reverse=True)
    n_vacc = max(1, int(len(population) * vaccination_rate))
    for cell in worst[:n_vacc]:
        step = 0.15
        cell.local_dna.learning_rate = (1 - step) * cell.local_dna.learning_rate + step * mean_lr
        cell.local_dna.dropout = (1 - step) * cell.local_dna.dropout + step * mean_drop
        cell.local_dna.activation_alpha = (1 - step) * cell.local_dna.activation_alpha + step * mean_alpha
        cell.local_dna.mutation_rate = (1 - step) * cell.local_dna.mutation_rate + step * mean_mut
        cell.life = max(cell.life, 10)

def prune_population(population):
    while len(population) > MAX_POPULATION:
        population.sort(key=lambda c: (
            c.evaluate() or float('inf'),
            -c.contribution_score,
            -c.life
        ), reverse=True) # Worst is now at the end
        population.pop()

def decay_mutation_rates(population, progress, min_rate=0.01):
    decay_factor = 1.0 - (0.8 * progress)
    for cell in population:
        cell.local_dna.mutation_rate = max(min_rate, cell.local_dna.mutation_rate * decay_factor)

# --------------------------
# ===== Training loop ======
# --------------------------
def run_training_loop():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    
    global_dna = GlobalDNA()
    population = []
    
    # Initialize population
    for _ in range(6):
        local = LocalDNA(base_lr=global_dna.traits["learning"]["learning_rate"],
                         base_mutation=global_dna.traits["learning"]["mutation_rate"],
                         act_fn=(random.choice(RNA.ACTIVATIONS), random.choice(RNA.ACTIVATIONS)),
                         act_alpha=random.uniform(0.3, 0.8),
                         dropout=global_dna.traits["structure"]["dropout"])
        population.append(NeuralCell(global_dna, local))
        
    best_loss = float('inf')
    best_cell = None
    avg_loss_history = []
    start = time.time()
    
    # Initialize the tissue pool here
    tissue_pool = []
    max_cycles = 3
    
    for cycle in range(max_cycles):
        for batch_idx, (data, target) in enumerate(train_loader):
            dm = DataManager(data, target)
            for cell in list(population):
                bx, by = dm.sample_batch(64)
                if bx is None: bx, by = cell.encode_dna_to_tensor(64)
                loss = cell.train_step(bx, by)
                if loss < best_loss:
                    best_loss = loss
                    best_cell = deepcopy(cell)
            
            avg_loss = np.mean([c.evaluate() for c in population if c.evaluate() is not None])
            if avg_loss is not np.nan:
              avg_loss_history.append(avg_loss)

            if batch_idx % 20 == 0:
                print(f"[INFO] Cycle {cycle+1}, Batch {batch_idx}: Avg Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}, Pop: {len(population)}")
            
            global_dna.mutate(avg_loss)
            
            # --- Evolutionary and tissue operations ---
            if batch_idx > 0 and batch_idx % EVOLVE_EVERY == 0:
                apply_immunity_block(population, global_dna, avg_loss)
                progress = (batch_idx + cycle * len(train_loader)) / (max_cycles * len(train_loader))
                decay_mutation_rates(population, progress)
                evolve_population(population, global_dna, dm)
                apply_vaccination(population, vaccination_rate=0.05)
                prune_population(population)
            
            # --- Re-integrated Tissue Logic ---
            # Runs at a different frequency
            if batch_idx > 0 and batch_idx % 50 == 0:
                population.sort(key=lambda c: c.evaluate() or float('inf'))
                top_k = min(6, len(population))
                if top_k > 0:
                    tissue_cells = population[:top_k]
                    new_tissue = Tissue(tissue_cells)
                    tissue_pool.append(new_tissue)
                    # Spawn from the last two created tissues
                    for t in tissue_pool[-2:]:
                        new_offspring = t.spawn_offspring(global_dna, n_offspring=4)
                        population.extend(new_offspring)
                    prune_population(population) # Prune after adding offspring
            
            if len(population) >= MAX_POPULATION:
                break
        if len(population) >= MAX_POPULATION:
            break
            
    print(f"\n✅ Training finished in {time.time()-start:.2f}s")
    print(f"Best Loss Achieved: {best_loss:.4f}")
    if best_cell:
        print("\n--- Best Cell Hyperparameters ---")
        print(f"  Learning Rate: {best_cell.local_dna.learning_rate:.6f}")
        print(f"  Dropout: {best_cell.local_dna.dropout:.4f}")
        print(f"  Activation Pair: {best_cell.local_dna.activation_pair}")
        print(f"  Activation Alpha: {best_cell.local_dna.activation_alpha:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss_history)
    plt.title("Average Population Loss Over Training Steps")
    plt.xlabel("Training Step (Batch)")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_training_loop()