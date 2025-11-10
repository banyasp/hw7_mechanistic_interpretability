"""
Mechanistic Interpretability: Greater-Than Comparison Task
===========================================================

This module implements a tiny transformer that learns to compare two numbers
and output which one is greater. We then peek inside to understand HOW it
makes this decision.

Task: Given [num1, num2], predict which is larger (0 = first, 1 = second, 2 = equal)

The goal is to discover:
- What attention patterns emerge?
- Do specific heads specialize in comparison?
- Can we identify the "greater-than circuit"?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time


# ============================================================================
# PART 1: DATA GENERATION
# ============================================================================

@dataclass
class ComparisonDataset:
    """Dataset for greater-than comparison task"""
    inputs: torch.Tensor  # Shape: (N, 2) - pairs of numbers
    labels: torch.Tensor  # Shape: (N,) - 0: first is greater, 1: second is greater, 2: equal

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def generate_comparison_data(
    num_samples: int = 2000,
    num_range: Tuple[int, int] = (0, 20),
    include_equal: bool = True,
    seed: int = 42
) -> ComparisonDataset:
    """
    Generate training data for the greater-than comparison task.

    Args:
        num_samples: Number of comparison pairs to generate
        num_range: Range of numbers to compare (min, max)
        include_equal: Whether to include cases where both numbers are equal
        seed: Random seed for reproducibility

    Returns:
        ComparisonDataset containing input pairs and labels
    """
    np.random.seed(seed)

    min_val, max_val = num_range
    num1 = np.random.randint(min_val, max_val + 1, size=num_samples)
    num2 = np.random.randint(min_val, max_val + 1, size=num_samples)

    # Create labels: 0 = first is greater, 1 = second is greater, 2 = equal
    labels = np.where(num1 > num2, 0, np.where(num1 < num2, 1, 2))

    if not include_equal:
        # Filter out equal cases
        mask = labels != 2
        num1 = num1[mask]
        num2 = num2[mask]
        labels = labels[mask]

    inputs = np.stack([num1, num2], axis=1)

    return ComparisonDataset(
        inputs=torch.from_numpy(inputs).long(),
        labels=torch.from_numpy(labels).long()
    )


def create_train_val_split(
    dataset: ComparisonDataset,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[ComparisonDataset, ComparisonDataset]:
    """Split dataset into train and validation sets"""
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    val_size = int(n * val_fraction)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = ComparisonDataset(
        inputs=dataset.inputs[train_indices],
        labels=dataset.labels[train_indices]
    )
    val_dataset = ComparisonDataset(
        inputs=dataset.inputs[val_indices],
        labels=dataset.labels[val_indices]
    )

    return train_dataset, val_dataset


# ============================================================================
# PART 2: TRANSFORMER MODEL ARCHITECTURE
# ============================================================================

class SimpleAttentionHead(nn.Module):
    """
    A single attention head that we can inspect.

    The key insight: attention learns to route information.
    For greater-than, we expect heads to learn to:
    1. Compare the two numbers
    2. Attend to the larger one
    3. Copy its information forward
    """

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head

        # Query, Key, Value projections
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Shape (batch, seq_len, d_model)
            return_attention: If True, also return attention weights

        Returns:
            output: Shape (batch, seq_len, d_head)
            attention_weights (optional): Shape (batch, seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, d_head)
        K = self.W_K(x)  # (batch, seq_len, d_head)
        V = self.W_V(x)  # (batch, seq_len, d_head)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        if return_attention:
            return output, attention_weights
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention that we can inspect head-by-head.

    Hypothesis: Different heads will specialize in different sub-tasks:
    - One head might attend to "which position has the larger number?"
    - Another might copy that number's value
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Create separate attention heads
        self.heads = nn.ModuleList([
            SimpleAttentionHead(d_model, self.d_head)
            for _ in range(n_heads)
        ])

        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Shape (batch, seq_len, d_model)

        Returns:
            output: Shape (batch, seq_len, d_model)
            attention_per_head (optional): List of attention weights per head
        """
        head_outputs = []
        attention_per_head = []

        for head in self.heads:
            if return_attention:
                out, attn = head(x, return_attention=True)
                head_outputs.append(out)
                attention_per_head.append(attn)
            else:
                head_outputs.append(head(x))

        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, d_model)

        # Final projection
        output = self.W_O(concat_output)

        if return_attention:
            return output, attention_per_head
        return output


class TransformerBlock(nn.Module):
    """
    A single transformer block (attention + residual)

    We're using attention-only (no MLP) to make interpretation easier!
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights = self.attention(x, return_attention=True)
            x = self.ln(x + attn_out)
            return x, attn_weights
        else:
            x = self.ln(x + self.attention(x))
            return x


class GreaterThanTransformer(nn.Module):
    """
    Tiny transformer for comparing two numbers.

    Architecture:
    1. Embedding layer (numbers → vectors)
    2. Positional encoding (so model knows "first" vs "second")
    3. 2 transformer layers
    4. Classification head

    Our goal: Understand what this model learns!
    """

    def __init__(
        self,
        vocab_size: int = 50,  # Max number we can represent
        d_model: int = 64,     # Model dimension
        n_heads: int = 4,      # Number of attention heads
        n_layers: int = 2,     # Number of transformer layers
        n_classes: int = 3     # 0: first greater, 1: second greater, 2: equal
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Token embeddings: convert numbers to vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings: distinguish "first" vs "second" number
        self.positional_embedding = nn.Embedding(2, d_model)  # Only 2 positions!

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # Classification head: map final hidden state to prediction
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_hidden_states: bool = False
    ):
        """
        Args:
            x: Shape (batch, 2) - two numbers to compare
            return_attention: If True, return attention weights from all layers
            return_hidden_states: If True, return hidden states from all layers

        Returns:
            logits: Shape (batch, n_classes) - predicted class logits
            attention_weights (optional): List of attention weights per layer
            hidden_states (optional): List of hidden states per layer
        """
        batch_size = x.shape[0]

        # Create embeddings
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)  # (batch, 2, d_model)
        pos_emb = self.positional_embedding(positions)  # (batch, 2, d_model)

        # Combine token and position embeddings
        h = token_emb + pos_emb  # (batch, 2, d_model)

        # Track attention and hidden states if requested
        all_attention = []
        all_hidden_states = [h] if return_hidden_states else []

        # Pass through transformer blocks
        for block in self.blocks:
            if return_attention:
                h, attn = block(h, return_attention=True)
                all_attention.append(attn)
            else:
                h = block(h)

            if return_hidden_states:
                all_hidden_states.append(h)

        # Final layer norm
        h = self.ln_final(h)

        # Pool: take mean over sequence (simple aggregation)
        h_pooled = h.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(h_pooled)  # (batch, n_classes)

        # Prepare return values
        outputs = [logits]
        if return_attention:
            outputs.append(all_attention)
        if return_hidden_states:
            outputs.append(all_hidden_states)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


# ============================================================================
# PART 3: TRAINING
# ============================================================================

def train_model(
    model: GreaterThanTransformer,
    train_dataset: ComparisonDataset,
    val_dataset: ComparisonDataset,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the transformer on the comparison task.

    Returns:
        history: Dictionary containing training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Mini-batch training
        n_batches = (len(train_dataset) + batch_size - 1) // batch_size
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(train_dataset))

            batch_inputs = train_dataset.inputs[start_idx:end_idx].to(device)
            batch_labels = train_dataset.labels[start_idx:end_idx].to(device)

            # Forward pass
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * len(batch_inputs)
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == batch_labels).sum().item()
            train_total += len(batch_inputs)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_inputs = val_dataset.inputs.to(device)
            val_labels = val_dataset.labels.to(device)

            logits = model(val_inputs)
            val_loss = criterion(logits, val_labels).item()

            predictions = torch.argmax(logits, dim=-1)
            val_acc = (predictions == val_labels).float().mean().item()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")

    return history


# ============================================================================
# PART 4: INTERPRETABILITY & ANALYSIS
# ============================================================================

def visualize_attention_patterns(
    model: GreaterThanTransformer,
    inputs: torch.Tensor,
    layer_idx: int = 0,
    figsize: Tuple[int, int] = (15, 4),
    device: str = 'cpu'
):
    """
    Visualize attention patterns for specific input examples.

    This is where the magic happens! We can see:
    - Does the model attend to the larger number?
    - Are attention patterns consistent across examples?
    - Do different heads specialize differently?
    """
    model.eval()
    model = model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        _, attention_weights, _ = model(
            inputs,
            return_attention=True,
            return_hidden_states=True
        )

    # Get attention for specified layer
    layer_attention = attention_weights[layer_idx]  # List of attention per head
    n_heads = len(layer_attention)
    n_examples = min(3, len(inputs))

    fig, axes = plt.subplots(n_examples, n_heads, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for ex_idx in range(n_examples):
        num1, num2 = inputs[ex_idx].cpu().numpy()
        for head_idx in range(n_heads):
            ax = axes[ex_idx, head_idx] if n_examples > 1 else axes[head_idx]

            # Get attention weights for this head and example
            attn = layer_attention[head_idx][ex_idx].cpu().numpy()

            # Plot heatmap
            sns.heatmap(
                attn,
                ax=ax,
                cmap='viridis',
                cbar=True,
                vmin=0,
                vmax=1,
                square=True,
                xticklabels=[f'{num1}', f'{num2}'],
                yticklabels=[f'{num1}', f'{num2}'],
                annot=True,
                fmt='.2f'
            )

            if ex_idx == 0:
                ax.set_title(f'Head {head_idx}')
            if head_idx == 0:
                label = ">" if num1 > num2 else ("<" if num1 < num2 else "=")
                ax.set_ylabel(f'{num1} {label} {num2}')

    plt.suptitle(f'Attention Patterns - Layer {layer_idx}', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def analyze_head_behavior(
    model: GreaterThanTransformer,
    dataset: ComparisonDataset,
    layer_idx: int = 0,
    head_idx: int = 0,
    device: str = 'cpu',
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Analyze what a specific attention head is doing.

    Key questions:
    1. Does this head attend more to the larger number?
    2. Does it attend more to the first or second position?
    3. Is there a consistent pattern?

    Returns:
        Statistics about the head's behavior
    """
    model.eval()
    model = model.to(device)

    sample_indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    inputs = dataset.inputs[sample_indices].to(device)
    labels = dataset.labels[sample_indices].cpu().numpy()

    with torch.no_grad():
        _, attention_weights, _ = model(
            inputs,
            return_attention=True,
            return_hidden_states=True
        )

    # Get attention for specified layer and head
    head_attention = attention_weights[layer_idx][head_idx]  # (batch, 2, 2)

    # Analyze patterns
    stats = {}

    # Average attention from position 0 to position 0 and 1
    attn_from_0 = head_attention[:, 0, :].cpu().numpy()  # (batch, 2)
    stats['avg_0_to_0'] = attn_from_0[:, 0].mean()
    stats['avg_0_to_1'] = attn_from_0[:, 1].mean()

    # Average attention from position 1 to position 0 and 1
    attn_from_1 = head_attention[:, 1, :].cpu().numpy()  # (batch, 2)
    stats['avg_1_to_0'] = attn_from_1[:, 0].mean()
    stats['avg_1_to_1'] = attn_from_1[:, 1].mean()

    # Does it attend to the larger number?
    nums = inputs.cpu().numpy()

    # For cases where first > second
    first_greater_mask = labels == 0
    if first_greater_mask.sum() > 0:
        # Average attention to position 0 (the larger one)
        stats['attn_to_larger_when_first_greater'] = attn_from_1[first_greater_mask, 0].mean()

    # For cases where second > first
    second_greater_mask = labels == 1
    if second_greater_mask.sum() > 0:
        # Average attention to position 1 (the larger one)
        stats['attn_to_larger_when_second_greater'] = attn_from_1[second_greater_mask, 1].mean()

    return stats


def ablate_head(
    model: GreaterThanTransformer,
    dataset: ComparisonDataset,
    layer_idx: int,
    head_idx: int,
    device: str = 'cpu'
) -> float:
    """
    Ablation study: What happens if we "remove" this head?

    This tells us how important the head is for the task.
    If accuracy drops a lot, the head is crucial!
    """
    model.eval()
    model = model.to(device)
    inputs = dataset.inputs.to(device)
    labels = dataset.labels.to(device)

    # Get the attention head to ablate
    target_head = model.blocks[layer_idx].attention.heads[head_idx]

    # Save original weights
    original_W_O = model.blocks[layer_idx].attention.W_O.weight.data.clone()

    # Zero out this head's contribution to output
    d_head = model.d_model // model.n_heads
    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head
    model.blocks[layer_idx].attention.W_O.weight.data[:, start_idx:end_idx] = 0

    # Evaluate
    with torch.no_grad():
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()

    # Restore original weights
    model.blocks[layer_idx].attention.W_O.weight.data = original_W_O

    return accuracy


def plot_training_curves(history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 4)):
    """Plot training loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def inspect_qk_circuit(
    model: GreaterThanTransformer,
    layer_idx: int = 0,
    head_idx: int = 0,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Inspect the Query-Key (QK) circuit of a specific head.

    The QK circuit determines "what to attend to".
    For greater-than, we might expect to see patterns related to number comparison.
    """
    head = model.blocks[layer_idx].attention.heads[head_idx]

    W_Q = head.W_Q.weight.detach().cpu().numpy()
    W_K = head.W_K.weight.detach().cpu().numpy()

    # Compute QK matrix: this determines attention patterns
    QK = W_Q.T @ W_K  # (d_head, d_head)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Query matrix
    sns.heatmap(W_Q, ax=axes[0], cmap='RdBu_r', center=0, cbar=True)
    axes[0].set_title(f'Query Matrix W_Q\nLayer {layer_idx}, Head {head_idx}')
    axes[0].set_xlabel('d_head')
    axes[0].set_ylabel('d_model')

    # Key matrix
    sns.heatmap(W_K, ax=axes[1], cmap='RdBu_r', center=0, cbar=True)
    axes[1].set_title(f'Key Matrix W_K\nLayer {layer_idx}, Head {head_idx}')
    axes[1].set_xlabel('d_head')
    axes[1].set_ylabel('d_model')

    # QK composition
    sns.heatmap(QK, ax=axes[2], cmap='RdBu_r', center=0, cbar=True)
    axes[2].set_title(f'QK Circuit (W_Q^T W_K)\nLayer {layer_idx}, Head {head_idx}')
    axes[2].set_xlabel('d_head')
    axes[2].set_ylabel('d_head')

    plt.tight_layout()
    return fig


def test_hypothesis(
    model: GreaterThanTransformer,
    test_cases: List[Tuple[int, int]],
    device: str = 'cpu'
) -> None:
    """
    Test the model on specific cases and show predictions with attention.

    Use this to test hypotheses like:
    - "The model attends to the larger number in head 0"
    - "Head 1 copies the value of the larger number"
    """
    model.eval()
    model = model.to(device)

    inputs = torch.tensor(test_cases, dtype=torch.long).to(device)

    with torch.no_grad():
        logits, attention_weights, hidden_states = model(
            inputs,
            return_attention=True,
            return_hidden_states=True
        )
        predictions = torch.argmax(logits, dim=-1)

    class_names = ['First Greater', 'Second Greater', 'Equal']

    print("=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)

    for i, (num1, num2) in enumerate(test_cases):
        pred_idx = predictions[i].item()
        pred_label = class_names[pred_idx]

        true_label = class_names[0 if num1 > num2 else (1 if num1 < num2 else 2)]

        print(f"\nTest case: [{num1}, {num2}]")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label} {'[CORRECT]' if pred_label == true_label else '[INCORRECT]'}")

        # Show attention from each head in layer 0
        print(f"  Attention patterns (Layer 0):")
        for head_idx in range(len(attention_weights[0])):
            attn = attention_weights[0][head_idx][i].cpu().numpy()
            print(f"    Head {head_idx}:")
            print(f"      From pos 0 ({num1}): {attn[0, 0]:.3f} to self, {attn[0, 1]:.3f} to pos 1")
            print(f"      From pos 1 ({num2}): {attn[1, 0]:.3f} to pos 0, {attn[1, 1]:.3f} to self")


# ============================================================================
# PART 5: EXAMPLE USAGE
# ============================================================================

def main():
    """
    Complete pipeline for the mechanistic interpretability assignment.

    This demonstrates:
    1. Data generation
    2. Model training
    3. Exploratory analysis
    4. Hypothesis formation and testing
    """

    print("=" * 60)
    print("MECHANISTIC INTERPRETABILITY: GREATER-THAN TASK")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # ========================================
    # STEP 1: Generate Data
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: GENERATING DATA")
    print("=" * 60)

    dataset = generate_comparison_data(
        num_samples=2000,
        num_range=(0, 20),
        include_equal=True
    )

    train_dataset, val_dataset = create_train_val_split(dataset, val_fraction=0.2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"\nExample inputs: {train_dataset.inputs[:5]}")
    print(f"Example labels: {train_dataset.labels[:5]} (0=first greater, 1=second greater, 2=equal)")

    # ========================================
    # STEP 2: Create and Train Model
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODEL")
    print("=" * 60)

    model = GreaterThanTransformer(
        vocab_size=25,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=3
    )

    print(f"\nModel architecture:")
    print(f"  - Embedding dim: 64")
    print(f"  - Layers: 2")
    print(f"  - Heads per layer: 4")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    history = train_model(
        model,
        train_dataset,
        val_dataset,
        n_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
        verbose=True
    )

    # ========================================
    # STEP 3: Visualize Training
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: VISUALIZING TRAINING")
    print("=" * 60)

    fig = plot_training_curves(history)
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training curves to 'training_curves.png'")
    plt.close()

    # ========================================
    # STEP 4: Explore Attention Patterns
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: EXPLORING ATTENTION PATTERNS")
    print("=" * 60)

    # Pick some interesting test cases
    test_inputs = torch.tensor([
        [15, 3],   # First much greater
        [2, 18],   # Second much greater
        [10, 10],  # Equal
        [12, 11],  # First slightly greater
        [7, 9],    # Second slightly greater
    ])

    # Visualize Layer 0
    print("\nVisualizing Layer 0 attention patterns...")
    fig = visualize_attention_patterns(model, test_inputs, layer_idx=0, device=device)
    plt.savefig('attention_layer0.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_layer0.png'")
    plt.close()

    # Visualize Layer 1
    print("Visualizing Layer 1 attention patterns...")
    fig = visualize_attention_patterns(model, test_inputs, layer_idx=1, device=device)
    plt.savefig('attention_layer1.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_layer1.png'")
    plt.close()

    # ========================================
    # STEP 5: Analyze Specific Heads
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: ANALYZING HEAD BEHAVIOR")
    print("=" * 60)

    for layer_idx in range(2):
        for head_idx in range(4):
            stats = analyze_head_behavior(
                model,
                val_dataset,
                layer_idx=layer_idx,
                head_idx=head_idx,
                device=device
            )

            print(f"\nLayer {layer_idx}, Head {head_idx}:")
            print(f"  Avg attention [0→0]: {stats['avg_0_to_0']:.3f}")
            print(f"  Avg attention [0→1]: {stats['avg_0_to_1']:.3f}")
            print(f"  Avg attention [1→0]: {stats['avg_1_to_0']:.3f}")
            print(f"  Avg attention [1→1]: {stats['avg_1_to_1']:.3f}")

            if 'attn_to_larger_when_first_greater' in stats:
                print(f"  When first > second, attention to larger: "
                      f"{stats['attn_to_larger_when_first_greater']:.3f}")
            if 'attn_to_larger_when_second_greater' in stats:
                print(f"  When second > first, attention to larger: "
                      f"{stats['attn_to_larger_when_second_greater']:.3f}")

    # ========================================
    # STEP 6: Ablation Studies
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 6: ABLATION STUDIES")
    print("=" * 60)

    # Baseline accuracy
    model.eval()
    with torch.no_grad():
        logits = model(val_dataset.inputs.to(device))
        predictions = torch.argmax(logits, dim=-1)
        baseline_acc = (predictions == val_dataset.labels.to(device)).float().mean().item()

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print("\nTesting importance of each head (accuracy after ablation):")

    for layer_idx in range(2):
        for head_idx in range(4):
            ablated_acc = ablate_head(
                model,
                val_dataset,
                layer_idx=layer_idx,
                head_idx=head_idx,
                device=device
            )
            drop = baseline_acc - ablated_acc
            print(f"  Layer {layer_idx}, Head {head_idx}: {ablated_acc:.4f} "
                  f"(drop: {drop:+.4f})")

    # ========================================
    # STEP 7: Inspect QK Circuits
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 7: INSPECTING QK CIRCUITS")
    print("=" * 60)

    # Inspect the most important head (based on ablation)
    print("\nVisualizing QK circuits for Layer 0, Head 0...")
    fig = inspect_qk_circuit(model, layer_idx=0, head_idx=0)
    plt.savefig('qk_circuit_l0h0.png', dpi=150, bbox_inches='tight')
    print("Saved to 'qk_circuit_l0h0.png'")
    plt.close()

    # ========================================
    # STEP 8: Test Specific Hypotheses
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 8: HYPOTHESIS TESTING")
    print("=" * 60)

    test_cases = [
        (20, 5),   # Large difference
        (5, 20),   # Large difference (reversed)
        (10, 11),  # Small difference
        (15, 15),  # Equal
        (0, 19),   # Edge case: min and max
    ]

    test_hypothesis(model, test_cases, device=device)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nNext steps for your assignment:")
    print("1. Review the attention patterns - what do you notice?")
    print("2. Which heads seem important? What are they doing?")
    print("3. Form a hypothesis about the 'greater-than circuit'")
    print("4. Use visualizations to support your explanation")
    print("5. Write up your findings as a narrative!")


if __name__ == "__main__":
    main()
