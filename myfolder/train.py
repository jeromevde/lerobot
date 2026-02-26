#!/usr/bin/env python3
"""
LeRobot Policy Training
=======================

Train a policy from scratch with real-time visualizations of training progress.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import draccus
from accelerate import Accelerator
from torch.utils.data import DataLoader
import numpy as np

# LeRobot imports - order matters for registration!
import lerobot.policies
from lerobot.datasets.factory import make_dataset
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_policy
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.configs.train import TrainPipelineConfig

print("=" * 80)
print("LeRobot Policy Training")
print("=" * 80)

# Configuration
OUTPUT_DIR = Path("./training_output")
OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Training configuration (modifiable)
DATASET = "lerobot/pusht"
ENV_TYPE = "pusht"
POLICY_TYPE = "diffusion"
STEPS = 500  # Increase this for real training (50k-100k)
BATCH_SIZE = 8
EVAL_FREQ = 100
LOG_FREQ = 50
SAVE_FREQ = 250

print("\n⚙️  Configuration:")
print(f"   Dataset: {DATASET}")
print(f"   Policy: {POLICY_TYPE}")
print(f"   Training steps: {STEPS}")
print(f"   Batch size: {BATCH_SIZE}")

# Build configuration for training
cli_args = [
    "--dataset.repo_id", DATASET,
    "--env.type", ENV_TYPE,
    "--policy.type", POLICY_TYPE,
    "--steps", str(STEPS),
    "--batch_size", str(BATCH_SIZE),
    "--seed", "42",
    "--output_dir", str(OUTPUT_DIR / "checkpoints"),
    "--eval_freq", str(EVAL_FREQ),
    "--log_freq", str(LOG_FREQ),
    "--save_freq", str(SAVE_FREQ),
]

print("\n🔧 Parsing configuration...")
cfg = draccus.parse(TrainPipelineConfig, args=cli_args)

# Setup accelerator
accelerator = Accelerator()
print(f"🖥️  Using device: {accelerator.device}")

# Load dataset
if accelerator.is_main_process:
    dataset = make_dataset(cfg)
    print(f"📦 Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

accelerator.wait_for_everyone()

# Create environment for evaluation
env = make_env(cfg.env, n_envs=1)
print(f"🎮 Environment: {cfg.env.type}")

# Create policy
policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
num_params = sum(p.numel() for p in policy.parameters())
print(f"🤖 Policy: {cfg.policy.type}")
print(f"   Parameters: {num_params:,}")

# Create optimizer and scheduler
optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)

# Wrap with accelerator
policy, optimizer = accelerator.prepare(policy, optimizer)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
)

# Training loop with visualization
print(f"\n🏋️  Starting training...")
print(f"   {'Step':<10} {'Loss':<12} {'LR':<12}")
print("   " + "-" * 34)

policy.train()
losses = []
steps = []
learning_rates = []

for step, batch in enumerate(dataloader):
    if step >= cfg.steps:
        break
    
    # Forward pass
    output_dict = policy.forward(batch)
    loss = output_dict["loss"]
    
    # Backward pass
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    loss_val = loss.item()
    losses.append(loss_val)
    steps.append(step)
    lr = optimizer.param_groups[0]['lr']
    learning_rates.append(lr)
    
    if (step + 1) % cfg.log_freq == 0:
        print(f"   {step+1:<10} {loss_val:<12.4f} {lr:<12.6f}")
    
    # Periodic visualization
    if (step + 1) % cfg.eval_freq == 0 or step == cfg.steps - 1:
        # Create training curve visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss curve
        ax1.plot(steps, losses, label='Training Loss', linewidth=2, alpha=0.7)
        # Add moving average
        if len(losses) > 10:
            window = min(50, len(losses) // 5)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], moving_avg, label=f'Moving Avg ({window} steps)', 
                    linewidth=2, color='red', alpha=0.8)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss (Current: {loss_val:.4f})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(steps, learning_rates, label='Learning Rate', linewidth=2, color='green', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = VIZ_DIR / f"training_progress_step_{step+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualization saved: {save_path.name}")

# Save final model
if accelerator.is_main_process:
    # Unwrap and save
    unwrapped_policy = accelerator.unwrap_model(policy)
    save_path = OUTPUT_DIR / "checkpoints" / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)
    unwrapped_policy.save_pretrained(save_path)
    print(f"\n💾 Final model saved to: {save_path}")

# Create final summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Training Summary - {POLICY_TYPE} on {DATASET}', fontsize=16, fontweight='bold')

# Loss curve with moving average
ax = axes[0, 0]
ax.plot(steps, losses, alpha=0.4, label='Loss')
if len(losses) > 10:
    window = min(50, len(losses) // 5)
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving Avg ({window})', color='red')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Loss distribution
ax = axes[0, 1]
ax.hist(losses, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
ax.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
ax.axvline(np.median(losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.4f}')
ax.set_xlabel('Loss Value')
ax.set_ylabel('Frequency')
ax.set_title('Loss Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Learning rate schedule
ax = axes[1, 0]
ax.plot(steps, learning_rates, linewidth=2, color='green')
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.grid(True, alpha=0.3)

# Training statistics text
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""
Training Statistics
{'='*40}

Configuration:
  • Policy: {POLICY_TYPE}
  • Dataset: {DATASET}
  • Total Steps: {len(steps)}
  • Batch Size: {BATCH_SIZE}
  • Parameters: {num_params:,}

Loss Statistics:
  • Final Loss: {losses[-1]:.4f}
  • Mean Loss: {np.mean(losses):.4f}
  • Min Loss: {np.min(losses):.4f}
  • Max Loss: {np.max(losses):.4f}
  • Std Loss: {np.std(losses):.4f}

Learning Rate:
  • Initial: {learning_rates[0]:.6f}
  • Final: {learning_rates[-1]:.6f}

Output:
  • Model: {OUTPUT_DIR / 'checkpoints' / 'final_model'}
  • Visualizations: {VIZ_DIR}
"""
ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
        verticalalignment='center', transform=ax.transAxes)

plt.tight_layout()
save_path = VIZ_DIR / "training_summary.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n📊 Training summary saved: {save_path}")
plt.close()

# Summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📁 Outputs:")
print(f"   Model checkpoint: {OUTPUT_DIR / 'checkpoints' / 'final_model'}")
print(f"   Visualizations: {VIZ_DIR}")
print(f"\n📊 Training Results:")
print(f"   Final loss: {losses[-1]:.4f}")
print(f"   Mean loss: {np.mean(losses):.4f}")
print(f"   Training steps: {len(steps)}")
print("\n🚀 Next step: Run evaluate_pretrained.py to test the trained model!")
print("=" * 80)
