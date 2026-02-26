#!/usr/bin/env python3
"""
LeRobot Pretrained Model Evaluation
====================================

Download and evaluate a pretrained model with visualizations of predictions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# LeRobot imports
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

print("=" * 80)
print("LeRobot Pretrained Model Evaluation")
print("=" * 80)

# Configuration
OUTPUT_DIR = Path("./evaluation_output")
OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Choose a pretrained model from HuggingFace Hub
# Options:
#   - lerobot/diffusion_pusht (Diffusion policy for PushT)
#   - lerobot/act_aloha_sim_insertion_human (ACT policy for Aloha)
PRETRAINED_MODEL_ID = "lerobot/diffusion_pusht"

print(f"\n📥 Downloading pretrained model: {PRETRAINED_MODEL_ID}")
print("   (First run downloads from HuggingFace, then uses cache)")

# Load pretrained policy based on type
if "diffusion" in PRETRAINED_MODEL_ID:
    policy = DiffusionPolicy.from_pretrained(PRETRAINED_MODEL_ID)
elif "act" in PRETRAINED_MODEL_ID:
    policy = ACTPolicy.from_pretrained(PRETRAINED_MODEL_ID)
else:
    # Try diffusion as default
    policy = DiffusionPolicy.from_pretrained(PRETRAINED_MODEL_ID)

print(f"\n✅ Model loaded!")
print(f"   Policy type: {policy.config.type}")
print(f"   Device: {next(policy.parameters()).device}")
print(f"   Model size: {sum(p.numel() for p in policy.parameters()):,} parameters")

# Get model device
device = next(policy.parameters()).device

# Visual 1: Test model with various observation inputs
print(f"\n🤖 Creating visualization 1: Action predictions for different observations...")

# Create a grid of test observations
num_tests = 9
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f'Action Predictions from {PRETRAINED_MODEL_ID}', fontsize=16, fontweight='bold')

# For PushT: observation.image (3, 96, 96) and observation.state (2,)
# We'll vary the input and show predicted actions
policy.eval()
predicted_actions = []

for idx, ax in enumerate(axes.flat):
    # Create varied dummy observations
    # Vary the random seed for diversity
    torch.manual_seed(idx * 42)
    
    dummy_obs = {
        "observation.image": torch.randn(1, 3, 96, 96).to(device),
        "observation.state": torch.randn(1, 2).to(device),
    }
    
    # Get action prediction
    with torch.no_grad():
        action = policy.select_action(dummy_obs)
    
    action_np = action.squeeze().cpu().numpy()
    predicted_actions.append(action_np)
    
    # Visualize the observation image
    img = dummy_obs["observation.image"].squeeze().cpu()
    if img.shape[0] == 3:  # CHW to HWC
        img = img.permute(1, 2, 0)
    img_np = img.numpy()
    
    # Normalize for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    ax.imshow(img_np)
    ax.set_title(f'Test {idx+1}\nAction: [{action_np[0]:.3f}, {action_np[1]:.3f}]', 
                fontsize=10)
    ax.axis('off')

plt.tight_layout()
save_path = VIZ_DIR / "action_predictions.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved to: {save_path}")
plt.close()

# Visual 2: Action space exploration
print(f"\n📊 Creating visualization 2: Action space distribution...")

# Generate many predictions to understand action distribution
num_samples = 500
all_actions = []

for i in range(num_samples):
    torch.manual_seed(i)
    dummy_obs = {
        "observation.image": torch.randn(1, 3, 96, 96).to(device),
        "observation.state": torch.randn(1, 2).to(device),
    }
    
    with torch.no_grad():
        action = policy.select_action(dummy_obs)
    
    all_actions.append(action.squeeze().cpu().numpy())

all_actions = np.array(all_actions)
action_dim = all_actions.shape[1]

fig = plt.figure(figsize=(14, 6))
fig.suptitle(f'Action Space Analysis ({num_samples} predictions)', fontsize=14, fontweight='bold')

# 2D scatter plot for first two action dimensions
if action_dim >= 2:
    ax1 = plt.subplot(1, 2, 1)
    scatter = ax1.scatter(all_actions[:, 0], all_actions[:, 1], 
                         alpha=0.5, c=np.arange(num_samples), cmap='viridis', s=20)
    ax1.set_xlabel('Action Dimension 0')
    ax1.set_ylabel('Action Dimension 1')
    ax1.set_title('Action Space Distribution')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Sample Index')
    
    # Add mean and std ellipse
    mean = all_actions[:, :2].mean(axis=0)
    std = all_actions[:, :2].std(axis=0)
    ax1.plot(mean[0], mean[1], 'r*', markersize=20, label='Mean')
    circle = plt.Circle(mean, std.max(), color='red', fill=False, 
                       linestyle='--', linewidth=2, label='1-sigma')
    ax1.add_patch(circle)
    ax1.legend()

# Histograms for each dimension
ax2 = plt.subplot(1, 2, 2)
for dim in range(min(action_dim, 4)):  # Plot up to 4 dimensions
    ax2.hist(all_actions[:, dim], bins=50, alpha=0.5, 
            label=f'Dim {dim} (μ={all_actions[:, dim].mean():.3f})')
ax2.set_xlabel('Action Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Action Distribution per Dimension')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = VIZ_DIR / "action_space_analysis.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved to: {save_path}")
plt.close()

# Visual 3: Model architecture summary
print(f"\n🏗️  Creating visualization 3: Model architecture summary...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

# Collect model information
model_info = []
model_info.append(("Model Information", "=" * 60))
model_info.append(("Repository:", PRETRAINED_MODEL_ID))
model_info.append(("Policy Type:", policy.config.type))
model_info.append(("Device:", str(device)))
model_info.append(("", ""))
model_info.append(("Model Statistics", "=" * 60))

total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
model_info.append(("Total Parameters:", f"{total_params:,}"))
model_info.append(("Trainable Parameters:", f"{trainable_params:,}"))
model_info.append(("Model Size (MB):", f"{total_params * 4 / 1024 / 1024:.2f}"))
model_info.append(("", ""))

# Add layer information
model_info.append(("Layer Breakdown", "=" * 60))
param_counts = {}
for name, param in policy.named_parameters():
    layer_name = name.split('.')[0]
    if layer_name not in param_counts:
        param_counts[layer_name] = 0
    param_counts[layer_name] += param.numel()

for layer, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    model_info.append((f"  {layer}:", f"{count:,} params"))

model_info.append(("", ""))
model_info.append(("Action Statistics (from 500 samples)", "=" * 60))
for dim in range(action_dim):
    model_info.append((f"  Dimension {dim}:", 
                      f"μ={all_actions[:, dim].mean():.4f}, "
                      f"σ={all_actions[:, dim].std():.4f}, "
                      f"range=[{all_actions[:, dim].min():.4f}, {all_actions[:, dim].max():.4f}]"))

# Format as text
text_str = "\n".join([f"{k:<35} {v}" for k, v in model_info])

ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
save_path = VIZ_DIR / "model_summary.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved to: {save_path}")
plt.close()

# Create inference example
print(f"\n🎯 Creating visualization 4: Detailed inference example...")

fig = plt.figure(figsize=(16, 6))
fig.suptitle('Detailed Inference Example', fontsize=16, fontweight='bold')

# Test observation
torch.manual_seed(123)
test_obs = {
    "observation.image": torch.randn(1, 3, 96, 96).to(device),
    "observation.state": torch.randn(1, 2).to(device),
}

# Run inference
with torch.no_grad():
    test_action = policy.select_action(test_obs)

# Visualize observation image
ax1 = plt.subplot(1, 3, 1)
img = test_obs["observation.image"].squeeze().cpu()
if img.shape[0] == 3:
    img = img.permute(1, 2, 0)
img_np = img.numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
ax1.imshow(img_np)
ax1.set_title('Input: Observation Image\n(96×96 RGB)', fontweight='bold')
ax1.axis('off')

# Visualize state
ax2 = plt.subplot(1, 3, 2)
state = test_obs["observation.state"].squeeze().cpu().numpy()
ax2.bar(range(len(state)), state, color='steelblue', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('State Dimension')
ax2.set_ylabel('Value')
ax2.set_title(f'Input: State Vector\n({len(state)} dimensions)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Visualize action
ax3 = plt.subplot(1, 3, 3)
action_np = test_action.squeeze().cpu().numpy()
colors = ['green' if a >= 0 else 'red' for a in action_np]
ax3.bar(range(len(action_np)), action_np, color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Action Dimension')
ax3.set_ylabel('Value')
ax3.set_title(f'Output: Predicted Action\n({len(action_np)} dimensions)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(action_np):
    ax3.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
save_path = VIZ_DIR / "inference_example.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved to: {save_path}")
plt.close()

# Summary
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print(f"\n📁 Visualizations saved to: {VIZ_DIR.absolute()}")
print(f"   - action_predictions.png: Predictions for 9 test observations")
print(f"   - action_space_analysis.png: Distribution of 500 action predictions")
print(f"   - model_summary.png: Model architecture and statistics")
print(f"   - inference_example.png: Detailed single inference walkthrough")
print(f"\n📊 Model Info:")
print(f"   Repository: {PRETRAINED_MODEL_ID}")
print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
print(f"   Action dimensions: {action_dim}")
print(f"\n💡 To run in actual environment:")
print(f"   1. Install: pip install gym-pusht  (for PushT)")
print(f"   2. Code:")
print(f"      env = gym.make('gym_pusht/PushT-v0')")
print(f"      obs, _ = env.reset()")
print(f"      action = policy.select_action(obs)")
print(f"      obs, reward, done, _, _ = env.step(action)")
print("=" * 80)
