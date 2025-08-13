# ==============================================================================
# A Modular, Self-Healing, and Evolvable Neural Architecture
# Inspired by "A Modular, Self-Healing, and Evolvable Neural Architecture
# Inspired by Biological Systems" by Raghav et al. [cite: 1]
#
# This script implements the core concepts described in the presentation,
# including Modular Neural Cells, DNA-like trait encoding, Quantum Principles
# for behavior, and an Evolution-Driven Training pipeline.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import heapq

# --------------------------
# === Utilities & Setup ====
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ===================================================================
# ===== Checkpointing & Early Stopping Configuration ==============
# ===================================================================
CHECKPOINT_FILE = "neuro_evo_final_checkpoint.pth"
PATIENCE = 500  # Stop if no improvement after this many batches [cite: 25]

# --------------------------
# ==== Activations / RNA ===
# --------------------------
# DNA-like encoding for activation functions [cite: 24]
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

# --------------------------
# ===== GlobalDNA ==========
# --------------------------
# Manages shared traits and tracks global success rates 
class GlobalDNA:
    def __init__(self):
        # Using hash maps for efficient DNA encoding and trait lookup [cite: 80]
        self.traits = {
            "learning_rate": {"value": 0.01, "success_history": deque(maxlen=100)},
            "dropout": {"value": 0.25, "success_history": deque(maxlen=100)},
            "mutation_rate": 0.1,
            "temperature": 0.05
        }
        self.previous_loss = None

    def metropolis_accept(self, old_loss, new_loss, temperature):
        if old_loss is None: return True
        delta = new_loss - old_loss
        if delta < 0: return True
        return np.random.rand() < np.exp(-delta / max(temperature, 1e-8))

    def record_trait_success(self, trait_name, value, success_metric):
        """Records the success of a specific trait value."""
        if trait_name in self.traits:
            self.traits[trait_name]["success_history"].append((value, success_metric))

    def evolve_trait(self, trait_name):
        """Evolves a trait based on its historical success."""
        if trait_name not in self.traits or not self.traits[trait_name]["success_history"]:
            return
        
        history = self.traits[trait_name]["success_history"]
        # Simple weighted average for evolution
        total_weight = sum(item[1] for item in history)
        if total_weight > 0:
            weighted_sum = sum(item[0] * item[1] for item in history)
            new_value = weighted_sum / total_weight
            # Add small noise for exploration
            noise = np.random.normal(0, new_value * 0.05)
            self.traits[trait_name]["value"] = max(1e-6, new_value + noise)

    def mutate(self, current_loss):
        if self.previous_loss is None: self.previous_loss = current_loss
        if not self.metropolis_accept(self.previous_loss, current_loss, self.traits["temperature"]):
            return
        
        self.traits["temperature"] = max(1e-4, self.traits["temperature"] * 0.95)
        # Evolve traits based on recorded success rates 
        self.evolve_trait("learning_rate")
        self.evolve_trait("dropout")


# --------------------------
# ===== LocalDNA ===========
# --------------------------
# Encodes traits for a single Neural Cell module [cite: 32, 34]
class LocalDNA:
    def __init__(self, global_dna: GlobalDNA):
        # Using queues for temporal buffering of training data (loss history) [cite: 79]
        self.loss_history = deque(maxlen=20)
        self.learning_rate = float(global_dna.traits["learning_rate"]["value"])
        self.dropout = float(global_dna.traits["dropout"]["value"])
        self.activation_pair = ("ReLU", "Swish") # Could be evolved as well
        self.activation_alpha = float(np.random.uniform(0.3, 0.7))

    def average_loss(self):
        return np.mean(self.loss_history) if self.loss_history else float('inf')

    def mutate(self, mutation_rate):
        # Trait selection inspired by evolutionary biology [cite: 36]
        self.learning_rate = max(1e-6, self.learning_rate + np.random.normal(0, mutation_rate * 0.1))
        self.dropout = float(np.clip(self.dropout + np.random.normal(0, mutation_rate * 0.2), 0.0, 0.7))
        self.activation_alpha = float(np.clip(self.activation_alpha + np.random.normal(0, mutation_rate * 0.5), 0.0, 1.0))

# ===================================================================
# ===== QuantumState Class (for Superposition) ======================
# ===================================================================
# This class implements the Superposition principle [cite: 55]
class QuantumState:
    def __init__(self, base_dna: LocalDNA):
        self.states = {'base': base_dna}
        self.probabilities = {'base': 1.0}
        self.current_state_key = 'base'

    def collapse_state(self) -> LocalDNA:
        """Choose one state (behavior) to be active, based on probabilities."""
        keys, probs = list(self.probabilities.keys()), list(self.probabilities.values())
        self.current_state_key = random.choices(keys, weights=probs, k=1)[0]
        return self.states[self.current_state_key]

    def update_probabilities(self, reward: float):
        current_prob = self.probabilities[self.current_state_key]
        self.probabilities[self.current_state_key] = max(0.01, current_prob + reward * 0.1)
        self.normalize_probabilities()

    def normalize_probabilities(self):
        total_prob = sum(self.probabilities.values())
        if total_prob > 0:
            for key in self.probabilities:
                self.probabilities[key] /= total_prob

# ===================================================================
# ===== AttentionController (for Global Influence) ==================
# ===================================================================
# Implements global influence using an attention-like mechanism [cite: 69]
# This creates an implicit, fully-connected interaction graph between cells,
# fulfilling the "Graphs" data structure concept[cite: 77].
class AttentionController:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def calculate_attention_weights(self, population):
        if not population: return [], []
        # Evaluation score incorporates loss and divergence [cite: 45, 46]
        scores = [-cell.evaluate() for cell in population]
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        attention_weights = F.softmax(scores_tensor / self.temperature, dim=0)
        return population, attention_weights.cpu().numpy()

# --------------------------
# ===== NeuralCell =========
# --------------------------
# An autonomous learning unit, as described in the architecture [cite: 23, 32]
class NeuralCell(nn.Module):
    def __init__(self, global_dna: GlobalDNA):
        super().__init__()
        self.global_dna = global_dna
        self.local_dna = LocalDNA(self.global_dna)
        self.quantum_state = QuantumState(self.local_dna) # Each cell can maintain a mixed state [cite: 55]
        
        self.contribution_score = 0.0
        self.divergence_score = 0.0
        self.loss_value = None
        self.last_output_sample = None
        
        self.build_model()

    def build_model(self):
        # Rebuild model based on the current DNA state
        self.local_dna = self.quantum_state.collapse_state()
        layers = [
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            MixedActivation(self.local_dna.activation_pair[0], self.local_dna.activation_pair[1], alpha=self.local_dna.activation_alpha),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            MixedActivation(self.local_dna.activation_pair[0], self.local_dna.activation_pair[1], alpha=self.local_dna.activation_alpha),
            nn.MaxPool2d(2)
        ]
        self.feature_extractor = nn.Sequential(*layers).to(DEVICE)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.BatchNorm1d(128),
            get_activation_by_name(self.local_dna.activation_pair[0]),
            nn.Dropout(self.local_dna.dropout), nn.Linear(128, 10)
        ).to(DEVICE)
        self.to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=self.local_dna.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        # Store a sample of the output for divergence calculation [cite: 46]
        self.last_output_sample = F.softmax(out.detach(), dim=1)
        return out

    def train_step(self, x, y):
        self.train()
        # Each cell undergoes localized backpropagation [cite: 65]
        for g in self.optimizer.param_groups: g['lr'] = self.local_dna.learning_rate
        x, y = x.to(DEVICE), y.to(DEVICE)
        self.optimizer.zero_grad()
        out = self(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        
        prev_loss = self.loss_value if self.loss_value is not None else loss.item()
        self.loss_value = loss.item()
        self.local_dna.loss_history.append(self.loss_value)
        
        reward = prev_loss - self.loss_value
        self.quantum_state.update_probabilities(reward)
        self.contribution_score += max(0, reward)
        
        # Record trait performance for global evolution 
        self.global_dna.record_trait_success("learning_rate", self.local_dna.learning_rate, max(0, reward))
        self.global_dna.record_trait_success("dropout", self.local_dna.dropout, max(0, reward))
        
        return self.loss_value

    def evaluate(self):
        """Combined evaluation score for self-healing and evolution."""
        # Evaluation considers local performance and functional divergence [cite: 45, 46]
        base_score = self.local_dna.average_loss()
        # Higher divergence is rewarded to encourage exploration
        return base_score - (self.divergence_score * 0.1)

# --------------------------
# ===== Population Mgt =====
# --------------------------
def calculate_divergence(population):
    """Calculates functional divergence based on output drift[cite: 46]."""
    for i, cell_i in enumerate(population):
        total_kl_div = 0
        if cell_i.last_output_sample is None: continue
        
        for j, cell_j in enumerate(population):
            if i == j or cell_j.last_output_sample is None: continue
            # Use KL Divergence to measure difference in output distributions
            kl_div = F.kl_div(cell_i.last_output_sample.log(), cell_j.last_output_sample, reduction='batchmean')
            total_kl_div += kl_div.item()
        
        cell_i.divergence_score = total_kl_div / len(population) if len(population) > 1 else 0

def save_checkpoint(state, filename=CHECKPOINT_FILE):
    print(f"💾 Saving checkpoint to {filename}...")
    torch.save(state, filename)

def load_checkpoint(filename=CHECKPOINT_FILE):
    if os.path.exists(filename):
        print(f"✅ Found checkpoint! Resuming from {filename}...")
        return torch.load(filename)
    return None

# --- Constants ---
MAX_POPULATION = 16
EVOLVE_EVERY = 20

# --------------------------
# ===== Main Training Loop =====
# --------------------------
def run_training_loop():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    
    global_dna = GlobalDNA()
    attention_controller = AttentionController()
    population = []
    
    start_cycle = 0
    best_loss = float('inf')
    no_improvement_count = 0
    
    checkpoint = load_checkpoint()
    if checkpoint:
        global_dna.traits = checkpoint['global_dna_traits']
        dna_states = checkpoint['population_dna']
        for dna_state in dna_states:
            cell = NeuralCell(global_dna)
            # This is a simplified restore; a full restore would save/load weights
            cell.local_dna.__dict__.update(dna_state) 
            population.append(cell)
        start_cycle = checkpoint['cycle']
        best_loss = checkpoint['best_loss']
        no_improvement_count = checkpoint['no_improvement_count']
    else:
        for _ in range(MAX_POPULATION // 2): population.append(NeuralCell(global_dna))

    start_time = time.time()
    max_cycles = 1000 # Increased for a longer run
    
    for cycle in range(start_cycle, max_cycles):
        for batch_idx, (data, target) in enumerate(train_loader):
            
            for cell in population:
                if batch_idx % 50 == 0: cell.build_model() # Periodically collapse superposition
                loss = cell.train_step(data.to(DEVICE), target.to(DEVICE))
                if loss < best_loss:
                    best_loss = loss
                    no_improvement_count = 0

            no_improvement_count += 1
            if batch_idx % 20 == 0:
                avg_lr = np.mean([c.local_dna.learning_rate for c in population])
                print(f"[INFO] Cycle {cycle+1}, Batch {batch_idx}: Best Loss: {best_loss:.4f}, Pop: {len(population)}, Avg LR: {avg_lr:.6f}")
            
            if no_improvement_count > PATIENCE:
                print(f"🛑 Stopping early after {PATIENCE} batches with no improvement.")
                break

            # --- Evolution using Heap and Attention [cite: 67] ---
            if batch_idx > 0 and batch_idx % EVOLVE_EVERY == 0:
                # Update global knowledge based on recent performance
                global_dna.mutate(best_loss)
                calculate_divergence(population)

                # Prune with heap: Isolate and remove poorly performing cells [cite: 43, 78]
                if len(population) > MAX_POPULATION:
                    heap = [(c.evaluate(), id(c), c) for c in population]
                    heapq.heapify(heap)
                    num_to_prune = len(population) - MAX_POPULATION
                    for _ in range(num_to_prune):
                        heapq.heappop(heap)
                    population = [item[2] for item in heap]

                # Reproduce with attention: Reintroduce repaired/evolved cells [cite: 48, 69]
                parents, weights = attention_controller.calculate_attention_weights(population)
                num_children = MAX_POPULATION - len(population)
                if num_children > 0 and len(parents) > 0:
                    chosen_parents = random.choices(parents, weights=weights, k=num_children)
                    for parent in chosen_parents:
                        child = NeuralCell(global_dna)
                        # Child inherits and mutates parent's DNA
                        child.local_dna = copy.deepcopy(parent.local_dna)
                        child.local_dna.mutate(global_dna.traits["mutation_rate"])
                        population.append(child)

        # --- Save Checkpoint & Handle Early Stop ---
        state = {
            'cycle': cycle + 1,
            'global_dna_traits': global_dna.traits,
            'population_dna': [cell.local_dna.__dict__ for cell in population],
            'best_loss': best_loss,
            'no_improvement_count': no_improvement_count
        }
        save_checkpoint(state)
        if no_improvement_count > PATIENCE: break

    print(f"\n✅ Training finished in {time.time()-start_time:.2f}s")
    print(f"Final Global DNA Traits: {global_dna.traits}")

if __name__ == "__main__":
    run_training_loop()