#!/usr/bin/env python3
"""
LeRobot Dataset Download & Inspection
======================================

Downloads a dataset from HuggingFace Hub, explores its structure,
and creates visualizations to understand the data.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import LeRobot dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("=" * 80)
print("LeRobot Dataset Download & Inspection")
print("=" * 80)

# Configuration
REPO_ID = "lerobot/pusht"  # Change this to try other datasets
OUTPUT_DIR = Path("./dataset_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

# Download/load dataset
print(f"\n📦 Loading dataset: {REPO_ID}")
print("   (First run downloads from HuggingFace Hub, then uses cache)")

dataset = LeRobotDataset(REPO_ID)

print(f"\n✅ Dataset loaded!")
print(f"   Total episodes: {dataset.num_episodes}")
print(f"   Total frames: {dataset.num_frames}")
print(f"   FPS: {dataset.fps}")
print(f"   Cache location: {dataset.root}")

# Explore dataset structure
print(f"\n📊 Dataset Structure:")
sample_frame = dataset[0]
print(f"   Keys in each frame: {list(sample_frame.keys())}")

print(f"\n🔍 Sample Frame Details:")
for key, value in sample_frame.items():
    if isinstance(value, torch.Tensor):
        print(f"   {key:30s} shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"   {key:30s} type={type(value).__name__}, value={value}")

# Visual 1: Display sample images from different episodes
print(f"\n🖼️  Creating visualization 1: Sample images from different episodes...")

if 'observation.image' in sample_frame:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Sample Images from {REPO_ID}', fontsize=16, fontweight='bold')
    
    # Sample from different episodes
    episode_indices = np.linspace(0, dataset.num_episodes - 1, 8, dtype=int)
    
    for idx, (ax, ep_idx) in enumerate(zip(axes.flat, episode_indices)):
        # Get first frame of this episode
        # Find the frame index for this episode
        frame_idx = ep_idx * (dataset.num_frames // dataset.num_episodes)
        frame_idx = min(frame_idx, dataset.num_frames - 1)
        
        frame = dataset[frame_idx]
        img = frame['observation.image']
        
        # Convert from CHW to HWC and handle normalization
        if img.shape[0] == 3:  # CHW format
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.numpy()
        
        # Clip values to valid range
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.set_title(f'Episode {ep_idx}, Frame {frame_idx}')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "sample_images.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {save_path}")
    plt.close()

# Visual 2: Action distribution
print(f"\n📈 Creating visualization 2: Action distribution...")

# Sample actions from the dataset
num_samples = min(1000, dataset.num_frames)
sample_indices = np.linspace(0, dataset.num_frames - 1, num_samples, dtype=int)

actions = []
for idx in sample_indices:
    frame = dataset[int(idx)]
    if 'action' in frame:
        actions.append(frame['action'].numpy())

if actions:
    actions = np.array(actions)
    action_dim = actions.shape[1]
    
    fig, axes = plt.subplots(1, action_dim, figsize=(6 * action_dim, 4))
    if action_dim == 1:
        axes = [axes]
    
    fig.suptitle(f'Action Distribution (sampled {num_samples} frames)', fontsize=14, fontweight='bold')
    
    for dim in range(action_dim):
        axes[dim].hist(actions[:, dim], bins=50, alpha=0.7, edgecolor='black')
        axes[dim].set_xlabel(f'Action Dimension {dim}')
        axes[dim].set_ylabel('Frequency')
        axes[dim].set_title(f'Dim {dim}: μ={actions[:, dim].mean():.3f}, σ={actions[:, dim].std():.3f}')
        axes[dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "action_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {save_path}")
    plt.close()

# Visual 3: Episode length distribution
print(f"\n📊 Creating visualization 3: Episode statistics...")

# Calculate episode lengths by looking at frame indices
episode_lengths = []
current_ep = 0
ep_length = 0

for idx in range(min(dataset.num_frames, 10000)):  # Sample to avoid slowness
    frame = dataset[idx]
    ep_idx = frame['episode_index'].item()
    
    if ep_idx != current_ep:
        if ep_length > 0:
            episode_lengths.append(ep_length)
        current_ep = ep_idx
        ep_length = 1
    else:
        ep_length += 1

if ep_length > 0:
    episode_lengths.append(ep_length)

if episode_lengths:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Episode Statistics', fontsize=14, fontweight='bold')
    
    # Histogram
    ax1.hist(episode_lengths, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_xlabel('Episode Length (frames)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Episode Length Distribution\n(mean={np.mean(episode_lengths):.1f}, std={np.std(episode_lengths):.1f})')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(episode_lengths, vert=True)
    ax2.set_ylabel('Episode Length (frames)')
    ax2.set_title('Episode Length Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "episode_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {save_path}")
    plt.close()

# Summary
print("\n" + "=" * 80)
print("DATASET INSPECTION COMPLETE!")
print("=" * 80)
print(f"\n📁 Visualizations saved to: {OUTPUT_DIR.absolute()}")
print(f"   - sample_images.png: Images from different episodes")
print(f"   - action_distribution.png: Distribution of action values")
print(f"   - episode_statistics.png: Episode length analysis")
print(f"\n💾 Dataset info:")
print(f"   Repository: {REPO_ID}")
print(f"   Episodes: {dataset.num_episodes}")
print(f"   Frames: {dataset.num_frames}")
print(f"   Cached at: {dataset.root}")
print("\n🚀 Next step: Run train.py to train a policy on this data!")
print("=" * 80)
