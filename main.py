import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import bisect
import matplotlib.pyplot as plt
import time

# --------------------------
# === Utilities & Setup ====
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------
# ==== Activations / RNA ===
# --------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# MixedActivation: blends two activations using a DNA-controlled blend coefficient alpha in [0,1]
class MixedActivation(nn.Module):
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self, act_name_a, act_name_b, alpha=0.5):
        super().__init__()
        self.a = get_activation_by_name(act_name_a)
        self.b = get_activation_by_name(act_name_b)
        # alpha stored as a learnable parameter in DNA; here it's buffer to be set by build_model
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        return self.alpha * a_out + (1.0 - self.alpha) * b_out

def get_activation_by_name(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "swish":
        return Swish()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()

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
        old_reward = RNA.activation_pairs.get(key, 0.0)
        RNA.activation_pairs[key] = 0.9 * old_reward + 0.1 * reward

# --------------------------
# ===== GlobalDNA ==========
# --------------------------
class GlobalDNA:
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self):
        # Structure: defines CNN channels, kernel sizes etc. These are high-level traits.
        self.traits = {
            "structure": {
                "conv_channels": [1, 32, 64],
                "kernel_sizes": [3, 3],
                "pool": 2,
                "dropout": 0.25,
                # activation mix holds probabilities for picking pairs
                "activation_mix": {
                    "ReLU": 0.4,
                    "Swish": 0.3,
                    "LeakyReLU": 0.3
                },
                "residual": False
            },
            "learning": {
                "learning_rate": 0.01,
                "mutation_rate": 0.1,
                "temperature": 0.05
            }
        }
        self.previous_loss = None

    def metropolis_accept(self, old_loss, new_loss, temperature):
        if old_loss is None:
            return True
        delta = new_loss - old_loss
        if delta < 0:
            return True
        prob = np.exp(-delta / max(temperature, 1e-8))
        return np.random.rand() < prob

    def normalize_activation_mix(self):
        mix = self.traits["structure"]["activation_mix"]
        total = sum(mix.values())
        for key in list(mix.keys()):
            mix[key] = mix[key] / total if total > 0 else 1.0 / len(mix)

    def mutate(self, current_loss):
        learning = self.traits["learning"]
        struct = self.traits["structure"]

        if self.previous_loss is None:
            self.previous_loss = current_loss

        if not self.metropolis_accept(self.previous_loss, current_loss, learning["temperature"]):
            return

        # Anneal temperature over time
        learning["temperature"] *= 0.95
        learning["temperature"] = max(learning["temperature"], 1e-4)

        # mutate activation_mix mildly
        for act in struct["activation_mix"]:
            if random.random() < learning["mutation_rate"]:
                struct["activation_mix"][act] = np.clip(struct["activation_mix"][act] + random.uniform(-0.05, 0.05), 0, 1)
        self.normalize_activation_mix()

        if random.random() < learning["mutation_rate"]:
            struct["dropout"] = np.clip(struct["dropout"] + random.uniform(-0.05, 0.05), 0.0, 0.9)

        # small chance to change conv channels slightly
        if random.random() < learning["mutation_rate"]:
            ch = struct["conv_channels"]
            # mutate intermediate channels
            for i in range(1, len(ch)):
                if random.random() < 0.5:
                    ch[i] = max(8, int(ch[i] + np.random.randint(-8, 9)))
            struct["conv_channels"] = ch

        self.previous_loss = current_loss

# --------------------------
# ===== LocalDNA ===========
# --------------------------
class LocalDNA:
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self, base_lr=0.01, base_mutation=0.1, act_fn=("ReLU", "Swish"), act_alpha=0.5, dropout=0.25):
        # loss history for failure detection
        self.loss_history = deque(maxlen=20)
        # hyperparameters
        self.learning_rate = base_lr
        self.mutation_rate = base_mutation
        self.activation_pair = act_fn  # tuple of two activations to blend
        self.activation_alpha = float(act_alpha)  # blend weight (alpha)
        self.dropout = dropout
        # life / state
        self.life = 30
        self.is_updating = True
        # tracking best found via local grid search
        self.best_hyperparams = {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "activation_alpha": self.activation_alpha,
            "mutation_rate": self.mutation_rate
        }

    def update_loss(self, loss):
        self.loss_history.append(loss)
        self.failure_function()

    def average_loss(self):
        return np.mean(self.loss_history) if self.loss_history else float('inf')

    def mutate(self):
        # small gaussian nudges to all hyperparams (failure function also can change them)
        self.learning_rate = max(1e-6, self.learning_rate + np.random.normal(0, self.learning_rate * 0.1))
        self.mutation_rate = float(np.clip(self.mutation_rate + np.random.normal(0, 0.02), 0.0, 1.0))
        self.activation_alpha = float(np.clip(self.activation_alpha + np.random.normal(0, 0.05), 0.0, 1.0))
        if random.random() < 0.1:
            # randomly pick new activation mixing pair
            a1 = random.choice(RNA.ACTIVATIONS)
            a2 = random.choice(RNA.ACTIVATIONS)
            self.activation_pair = (a1, a2)
        if random.random() < 0.1:
            self.dropout = float(np.clip(self.dropout + np.random.normal(0, 0.02), 0.0, 0.9))
        # reset life
        self.life = 30
        self.is_updating = True

    def failure_function(self):
        """
        If loss is increasing, nudge all hyperparameters slightly toward exploration or toward best found values.
        Also, if recent loss is improving, reinforce best_hyperparams via exponential moving average.
        This ties the grid-search short-run best to the local DNA's parameters.
        """
        if len(self.loss_history) < 4:
            return

        recent = list(self.loss_history)[-4:]
        # detect trend
        if recent[-1] > recent[0] * 1.01:  # increased by >1%
            # degrade: increase mutation, slightly decrease lr, increase dropout
            self.learning_rate *= 0.9
            self.mutation_rate = min(1.0, self.mutation_rate * 1.1 + 0.02)
            self.dropout = float(np.clip(self.dropout + 0.02, 0.0, 0.9))
            # shift activation_alpha toward 0.5 to encourage mixing
            self.activation_alpha = float(0.9 * self.activation_alpha + 0.1 * 0.5)
        else:
            # improving: slightly move best_hyperparams toward current
            self.best_hyperparams["learning_rate"] = 0.95 * self.best_hyperparams["learning_rate"] + 0.05 * self.learning_rate
            self.best_hyperparams["dropout"] = 0.95 * self.best_hyperparams["dropout"] + 0.05 * self.dropout
            self.best_hyperparams["activation_alpha"] = 0.95 * self.best_hyperparams["activation_alpha"] + 0.05 * self.activation_alpha
            self.best_hyperparams["mutation_rate"] = 0.95 * self.best_hyperparams["mutation_rate"] + 0.05 * self.mutation_rate

    def integrate_grid_search_result(self, best_result):
        """
        Move current hyperparameters slightly toward best_result (tradeoff speed vs accuracy).
        """
        lr_target = best_result.get("learning_rate", self.learning_rate)
        drop_target = best_result.get("dropout", self.dropout)
        alpha_target = best_result.get("activation_alpha", self.activation_alpha)
        mut_target = best_result.get("mutation_rate", self.mutation_rate)

        # step-size controls tradeoff between speed and accuracy; small step = fast adaptation
        step = 0.3
        self.learning_rate = float((1 - step) * self.learning_rate + step * lr_target)
        self.dropout = float((1 - step) * self.dropout + step * drop_target)
        self.activation_alpha = float((1 - step) * self.activation_alpha + step * alpha_target)
        self.mutation_rate = float((1 - step) * self.mutation_rate + step * mut_target)

# --------------------------
# ===== NeuralCell (CNN) ===
# --------------------------
class NeuralCell(nn.Module):
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self, global_dna: GlobalDNA, local_dna: LocalDNA):
        super().__init__()
        self.global_dna = global_dna
        self.local_dna = local_dna
        # growth and contribution tracking
        self.growth_rate = 1.0
        self.contribution_score = 0.0
        self.age = 0
        self.loss_value = None
        # build model based on DNA
        self.build_model()

    def build_model(self):
        struct = self.global_dna.traits["structure"]
        ch = struct["conv_channels"]
        ks = struct["kernel_sizes"]
        pool = struct["pool"]
        dropout = self.local_dna.dropout

        layers = []
        in_ch = ch[0]  # usually 1 for MNIST
        # conv layers
        for i in range(1, len(ch)):
            out_ch = ch[i]
            k = ks[i - 1] if i - 1 < len(ks) else 3
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            # use MixedActivation
            a1, a2 = self.local_dna.activation_pair
            mixed = MixedActivation(a1, a2, alpha=self.local_dna.activation_alpha)
            # attach alpha value buffer so it can be tuned when DNA changes (rebuild_model called)
            # mixed.alpha will be set by copying the local_dna.activation_alpha
            mixed.alpha = torch.tensor(self.local_dna.activation_alpha, dtype=torch.float32)
            layers.append(mixed)
            layers.append(nn.MaxPool2d(pool))
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        # compute flat dimension: for MNIST 28x28 — we will compute by a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, ch[0], 28, 28)
            feat = self.feature_extractor(dummy)
            flat = int(np.prod(feat.shape[1:]))

        # fully connected head
        hidden = 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.BatchNorm1d(hidden),
            get_activation_by_name(self.local_dna.activation_pair[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10)
        )

        # move modules to device
        self.to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=self.local_dna.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def rebuild_model_with_updated_dna(self):
        # Rebuild model (used when DNA changes: activation or dropout or alpha)
        # Save optimizer state? Simpler to re-create optimizer to match new lr
        self.build_model()

    def encode_dna_to_tensor(self, n_samples=32):
        """
        Create synthetic mini-batch from DNA traits.
        - use activation_alpha and conv channels to induce patterns in images.
        """
        # base random noise
        z = np.random.randn(n_samples, 1, 28, 28).astype(np.float32) * 0.1
        # stamp simple patterns from activation alpha and dropout into images
        alpha = self.local_dna.activation_alpha
        lr = self.local_dna.learning_rate
        drop = self.local_dna.dropout
        # create a circular blob whose intensity depends on alpha and lr
        xx, yy = np.meshgrid(np.linspace(-1,1,28), np.linspace(-1,1,28))
        # FIX: Corrected the formula for the circle. It should be xx**2 + yy**2.
        circle = np.exp(-1 * ((xx**2 + yy**2) / (0.5 + 0.5 * (1 - alpha))))
        circle = circle.astype(np.float32)
        for i in range(n_samples):
            z[i, 0] += circle * (0.2 + 0.8 * alpha) * (lr * 5)
            if random.random() < drop:
                z[i, 0] *= np.random.binomial(1, 0.9, size=(28,28)).astype(np.float32)
        # labels: random digits but biased by alpha
        labels = (np.abs((alpha * 9) + np.random.randint(0, 10, size=(n_samples,))) % 10).astype(np.int64)
        return torch.from_numpy(z).to(DEVICE), torch.from_numpy(labels).to(DEVICE)

    def train_step(self, x, y):
        self.train()
        # ensure optimizer lr matches DNA
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.local_dna.learning_rate

        self.optimizer.zero_grad()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        self.loss_value = float(loss.detach().cpu().item())
        self.local_dna.update_loss(self.loss_value)
        # update growth contribution metrics
        self.age += 1
        # small heuristic: if last loss improved vs avg, increment growth_rate
        avg = self.local_dna.average_loss()
        if self.loss_value < avg:
            self.growth_rate *= 1.02
            self.contribution_score += (avg - self.loss_value)
        else:
            self.growth_rate *= 0.995
        return self.loss_value

    def forward(self, x):
        # assume x shape = (B, 1, 28, 28)
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out

    def evaluate(self):
        return self.loss_value if self.loss_value is not None else float('inf')

# --------------------------
# ===== Data Manager =======
# --------------------------
class DataManager:
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self, x, y):
        # expects tensors shaped [N, 1, 28, 28], labels [N]
        self.x_data = x.to(DEVICE)
        self.y_data = y.to(DEVICE)
        self.prev_size = len(x)

    def add_data(self, new_x, new_y):
        self.x_data = torch.cat([self.x_data, new_x.to(DEVICE)], dim=0)
        self.y_data = torch.cat([self.y_data, new_y.to(DEVICE)], dim=0)

    def get_data(self):
        return self.x_data, self.y_data

    def sample_batch(self, batch_size=64):
        if len(self.x_data) == 0:
            return None, None
        idx = np.random.choice(len(self.x_data), min(batch_size, len(self.x_data)), replace=False)
        return self.x_data[idx], self.y_data[idx]

    def new_data_added(self):
        if len(self.x_data) > self.prev_size:
            self.prev_size = len(self.x_data)
            return True
        return False

# --------------------------
# ===== Population logic ===
# --------------------------
MAX_POPULATION = 100
FAILURE_THRESHOLD = 1.2
IMMUNITY_REGEN_COUNT = 2
GRID_SEARCH_TRIALS = 6  # small for speed; increase for thorough search
GRID_SEARCH_BUDGET_STEPS = 4  # small number of train steps per trial during CV

def apply_immunity_block(cells, global_dna, average_loss):
    regenerated = []
    for cell in list(cells):
        if cell.evaluate() is None:
            continue
        if cell.evaluate() > average_loss * FAILURE_THRESHOLD:
            for _ in range(IMMUNITY_REGEN_COUNT):
                if len(cells) + len(regenerated) >= MAX_POPULATION:
                    break
                new_dna = copy.deepcopy(cell.local_dna)
                new_dna.mutate()
                # change activation pair slightly
                a1 = RNA.select_new_activation(new_dna.activation_pair[0])
                a2 = RNA.select_new_activation(new_dna.activation_pair[1])
                new_dna.activation_pair = (a1, a2)
                child = NeuralCell(global_dna, new_dna)
                regenerated.append(child)
    cells.extend(regenerated)

def choose_activation_from_mix(activation_mix):
    acts = list(activation_mix.keys())
    weights = list(activation_mix.values())
    cumulative = np.cumsum(weights)
    r = random.uniform(0, cumulative[-1])
    idx = bisect.bisect_right(cumulative, r)
    return acts[min(idx, len(acts)-1)]

def lightweight_grid_search_cv(cell_template: NeuralCell, data_manager: DataManager, trials=GRID_SEARCH_TRIALS):
    """
    Very lightweight grid search:
    - Try random combinations (lr, dropout, activation_alpha) for a few trials.
    - Train for a few mini steps (GRID_SEARCH_BUDGET_STEPS) on small held-out batch to estimate loss.
    Returns best hyperparams dict.
    """
    best = None
    best_loss = float('inf')
    # sample a small validation set from data manager; if empty, use DNA-generated samples
    xval, yval = data_manager.sample_batch(256)
    if xval is None:
        # fallback: create synthetic dataset from template cell DNA
        xval, yval = cell_template.encode_dna_to_tensor(256)
    # create small train/val split
    split = int(0.8 * len(xval))
    train_x, val_x = xval[:split], xval[split:]
    train_y, val_y = yval[:split], yval[split:]

    # search ranges (keep small)
    lr_candidates = [max(1e-5, cell_template.local_dna.learning_rate * f) for f in [0.5, 1.0, 2.0]]
    drop_candidates = [max(0.0, cell_template.local_dna.dropout + d) for d in [-0.05, 0.0, 0.05]]
    alpha_candidates = [float(np.clip(cell_template.local_dna.activation_alpha + d, 0.0, 1.0)) for d in [-0.1, 0.0, 0.1]]

    combos = []
    for lr in lr_candidates:
        for dr in drop_candidates:
            for a in alpha_candidates:
                combos.append({"learning_rate": lr, "dropout": dr, "activation_alpha": a, "mutation_rate": cell_template.local_dna.mutation_rate})

    random.shuffle(combos)
    combos = combos[:trials]

    for combo in combos:
        # instantiate a quick ephemeral cell from template
        local = copy.deepcopy(cell_template.local_dna)
        local.learning_rate = combo["learning_rate"]
        local.dropout = combo["dropout"]
        local.activation_alpha = combo["activation_alpha"]
        local.mutation_rate = combo["mutation_rate"]
        tmp_cell = NeuralCell(cell_template.global_dna, local)
        # quick mini-training for a few small batches
        tmp_cell.train()
        bs = min(64, len(train_x))
        for step in range(GRID_SEARCH_BUDGET_STEPS):
            # sample mini-batch
            idx = np.random.choice(len(train_x), bs, replace=False)
            bx, by = train_x[idx], train_y[idx]
            tmp_cell.train_step(bx, by)
        # evaluate on val set
        tmp_cell.eval()
        with torch.no_grad():
            bx, by = val_x, val_y
            out = tmp_cell.forward(bx)
            loss = float(nn.CrossEntropyLoss()(out, by).cpu().item())
        if loss < best_loss:
            best_loss = loss
            best = combo
        # clean up
        del tmp_cell
    return best if best is not None else cell_template.local_dna.best_hyperparams

def evolve_population(cells, global_dna, data_manager):
    """
    Keep parents (sorted by evaluate) and produce more children than parents (offspring > parent).
    Use grid search results to nudge child hyperparams.
    """
    # sort ascending by loss (best first)
    cells.sort(key=lambda c: c.evaluate())
    parents = cells[:max(1, len(cells)//2)]
    new_cells = []
    mix = global_dna.traits["structure"]["activation_mix"]

    for parent in parents:
        parent_loss = parent.evaluate()
        # create multiple children per parent (offspring count > 1)
        num_children = 4  # bigger than parent batch => population growth
        for _ in range(num_children):
            # limit total population
            if len(cells) + len(new_cells) >= MAX_POPULATION:
                break
            child_dna = copy.deepcopy(parent.local_dna)
            old_act = child_dna.activation_pair
            # choose activation pair from mix: pick two possibly different activations
            a1 = choose_activation_from_mix(mix)
            a2 = choose_activation_from_mix(mix)
            child_dna.activation_pair = (a1, a2)
            # mutate baseline
            child_dna.mutate()
            # lightweight grid search cv to find local best -> integrate
            best_result = lightweight_grid_search_cv(parent, data_manager, trials=3)
            if best_result:
                child_dna.integrate_grid_search_result(best_result)
            # update RNA pair reward based on parent improvement
            reward = 1.0 if parent_loss < parent.local_dna.average_loss() else -0.5
            RNA.update_activation_pair(old_act[0], child_dna.activation_pair[0], reward)
            RNA.update_activation_pair(old_act[1], child_dna.activation_pair[1], reward)
            # create new child cell
            new_cell = NeuralCell(global_dna, child_dna)
            new_cells.append(new_cell)
    cells.extend(new_cells)

def prune_population(cells):
    # sort ascending by loss (best first)
    cells.sort(key=lambda c: c.evaluate())
    # remove low contributors: consider both loss and growth_rate/contribution
    while len(cells) > MAX_POPULATION // 2:
        # compute a score where lower is worse: combine loss (higher) and low growth
        scores = [c.evaluate() / (1e-6 + c.contribution_score + 1e-3) for c in cells]
        worst_idx = int(np.argmax(scores))
        removed = cells.pop(worst_idx)
        del removed

def apply_vaccination(cells, vaccination_rate=0.1):
    for cell in cells:
        if random.random() < vaccination_rate:
            cell.local_dna.mutation_rate = min(1.0, cell.local_dna.mutation_rate + 0.25)
            # randomise activation pair slightly
            a1 = RNA.select_new_activation(cell.local_dna.activation_pair[0])
            a2 = RNA.select_new_activation(cell.local_dna.activation_pair[1])
            cell.local_dna.activation_pair = (a1, a2)
            # rebuild model to apply new activations & alpha
            cell.rebuild_model_with_updated_dna()

# --------------------------
# ===== Tissue grouping ====
# --------------------------
class Tissue:
    # FIX: Corrected constructor from _init_ to __init__
    def __init__(self, cells):
        self.cells = cells  # list of NeuralCell
        self.aggregate_loss = np.mean([c.evaluate() for c in cells])
        self.hyperparam_signature = self.compute_signature()

    def compute_signature(self):
        # average local dna hyperparams to get a signature
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
        for _ in range(n_offspring):
            # choose a random cell from tissue as parent
            parent = random.choice(self.cells)
            child_dna = copy.deepcopy(parent.local_dna)
            child_dna.mutate()
            # small nudge toward tissue signature
            sig = self.hyperparam_signature
            child_dna.learning_rate = float(0.9 * child_dna.learning_rate + 0.1 * sig["lr"])
            child_dna.dropout = float(0.9 * child_dna.dropout + 0.1 * sig["drop"])
            child = NeuralCell(global_dna, child_dna)
            offspring.append(child)
        return offspring

# --------------------------
# ===== Training loop ======
# --------------------------
def run_training_loop():
    # transforms: keep image shape (1,28,28)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # quick small subset for speed in example; comment out next line to use entire dataset
    # train_dataset, _ = random_split(train_dataset, [20000, len(train_dataset)-20000])  # optional reduction

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    global_dna = GlobalDNA()
    population = []

    # initial population: create small diverse set
    for _ in range(6):
        local_dna = LocalDNA(base_lr=global_dna.traits["learning"]["learning_rate"],
                                 base_mutation=global_dna.traits["learning"]["mutation_rate"],
                                 act_fn=(random.choice(RNA.ACTIVATIONS), random.choice(RNA.ACTIVATIONS)),
                                 act_alpha=random.uniform(0.3, 0.8),
                                 dropout=global_dna.traits["structure"]["dropout"])
        cell = NeuralCell(global_dna, local_dna)
        population.append(cell)

    max_cycles = 6
    best_cell = None
    best_loss = float('inf')
    avg_loss_history = []
    tissue_pool = []

    start_time = time.time()
    for cycle in range(max_cycles):
        print(f"\n--- Training Cycle {cycle + 1} ---")
        for batch_idx, (data, target) in enumerate(train_loader):
            # data shape (B, 1, 28, 28)
            # quick DataManager for this batch
            data_manager = DataManager(data, target)

            # Each cell trains on this mini-batch (or on synthetic DNA if data empty)
            for cell in list(population):
                bx, by = data_manager.sample_batch(64)
                if bx is None:
                    # generate synthetic batch from cell's DNA
                    bx, by = cell.encode_dna_to_tensor(64)
                loss = cell.train_step(bx, by)
                # track best
                if loss < best_loss:
                    best_loss = loss
                    best_cell = copy.deepcopy(cell)

            # compute average loss across population
            avg_loss = np.mean([c.evaluate() for c in population if c.evaluate() is not None])
            avg_loss_history.append(avg_loss)

            # Periodic logs
            if batch_idx % 20 == 0:
                print(f"[INFO] Batch {batch_idx} Avg Loss: {avg_loss:.4f}, Best Loss So Far: {best_loss:.4f}, Population: {len(population)}")

            # adapt global DNA gently
            global_dna.mutate(avg_loss)

            # immunity block: regen bad cells
            apply_immunity_block(population, global_dna, avg_loss)

            # cluster top cells into tissues occasionally
            if batch_idx % 50 == 0:
                # choose top-k cells to form a tissue
                population.sort(key=lambda c: c.evaluate())
                top_k = min(6, len(population))
                tissue_cells = population[:top_k]
                # create tissue only if it improves over existing tissues
                new_tissue = Tissue(tissue_cells)
                tissue_pool.append(new_tissue)
                # spawn offspring from tissues to increase sampling
                for t in tissue_pool[-2:]:
                    new_off = t.spawn_offspring(global_dna, n_offspring=4)
                    population.extend(new_off)

            # evolutionary step: produce children > parents
            evolve_population(population, global_dna, data_manager)

            # vaccination block
            apply_vaccination(population, vaccination_rate=0.05)

            # prune low contributors and control population size
            prune_population(population)

            # stop if population hits max
            if len(population) >= MAX_POPULATION:
                print("[STOP] Maximum population reached.")
                break

        if len(population) >= MAX_POPULATION:
            break

    end_time = time.time()
    print(f"\n✅ Training finished. Best Loss Achieved: {best_loss:.4f} in {end_time - start_time:.1f}s")
    # show best cell hyperparams
    if best_cell is not None:
        print("Best cell hyperparams:")
        print(" LR:", best_cell.local_dna.learning_rate)
        print(" Dropout:", best_cell.local_dna.dropout)
        print(" Activation pair:", best_cell.local_dna.activation_pair)
        print(" Activation alpha:", best_cell.local_dna.activation_alpha)
        print(" Mutation rate:", best_cell.local_dna.mutation_rate)

    # === Plotting Average Loss ===
    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss_history, marker='o')
    plt.title("Average Population Loss Over Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# FIX: Corrected the main execution block to use double underscores.
if __name__ == "__main__":
    run_training_loop()