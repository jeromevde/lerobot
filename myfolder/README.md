# LeRobot Learning Scripts

This folder contains 3 separate scripts that demonstrate the complete LeRobot workflow:

## 📁 Scripts

### 1. `download.py` - Dataset Download & Inspection
Downloads a dataset from HuggingFace Hub and creates visualizations to understand the data structure.

**What it does:**
- Downloads dataset from HuggingFace Hub (cached locally)
- Displays dataset statistics (episodes, frames, FPS)
- Shows data structure and tensor shapes

**Visualizations created:**
- `sample_images.png`: Sample images from different episodes
- `action_distribution.png`: Distribution of action values
- `episode_statistics.png`: Episode length analysis

**Usage:**
```bash
python download.py
```

**Output folder:** `./dataset_inspection/`

---

### 2. `train.py` - Train a Policy from Scratch
Trains a policy on the downloaded dataset with real-time training visualizations.

**What it does:**
- Trains a Diffusion policy on the PushT dataset
- Logs loss and learning rate during training
- Saves model checkpoints
- Creates training progress visualizations

**Visualizations created:**
- `training_progress_step_*.png`: Loss curves at intervals
- `training_summary.png`: Final training statistics and plots

**Configuration (edit in script):**
- `DATASET`: Dataset to use (default: "lerobot/pusht")
- `POLICY_TYPE`: Policy type (default: "diffusion")
- `STEPS`: Training steps (default: 500, increase to 50k+ for real training)
- `BATCH_SIZE`: Batch size (default: 8)

**Usage:**
```bash
python train.py
```

**Output folder:** `./training_output/`

⚠️ **Note:** Training on CPU is very slow! For real experiments, use a GPU. See `../CHEAP_GPU_COMPUTE.md` for affordable GPU options.

---

### 3. `evaluate_pretrained.py` - Evaluate Pretrained Models
Downloads and evaluates a pretrained model from HuggingFace with detailed visualizations.

**What it does:**
- Downloads a pretrained model from HuggingFace Hub
- Tests model inference on various observations
- Analyzes action space distribution
- Shows model architecture and statistics

**Visualizations created:**
- `action_predictions.png`: Predictions for 9 different test observations
- `action_space_analysis.png`: Distribution of 500 action predictions
- `model_summary.png`: Model architecture and layer breakdown
- `inference_example.png`: Detailed step-by-step inference walkthrough

**Configuration (edit in script):**
- `PRETRAINED_MODEL_ID`: Model to download (default: "lerobot/diffusion_pusht")
  - Other options: "lerobot/act_aloha_sim_insertion_human"

**Usage:**
```bash
python evaluate_pretrained.py
```

**Output folder:** `./evaluation_output/`

---

## 🚀 Recommended Workflow

Run the scripts in order to learn the full LeRobot pipeline:

```bash
# Step 1: Download and explore the dataset
python download.py

# Step 2: Train a policy from scratch (optional, requires GPU for speed)
python train.py

# Step 3: Evaluate a pretrained model
python evaluate_pretrained.py
```

---

## 📊 All Visualizations

After running all scripts, you'll have:

1. **Dataset Understanding** (from `download.py`)
   - What the robot observations look like
   - Distribution of actions in the dataset
   - Episode lengths and statistics

2. **Training Progress** (from `train.py`)
   - Real-time loss curves
   - Learning rate schedules
   - Training convergence analysis

3. **Model Performance** (from `evaluate_pretrained.py`)
   - Action predictions on test inputs
   - Action space exploration
   - Model architecture breakdown
   - Detailed inference examples

---

## 💰 Need GPU for Training?

Training neural networks is **much faster** on GPUs. See `../CHEAP_GPU_COMPUTE.md` for:
- RunPod: $0.24/hr (easiest)
- vast.ai: $0.12/hr (cheapest)
- Lambda Labs: $0.75/hr (most reliable)

All support SSH access and on-demand billing!

---

## 🛠️ Requirements

All scripts require the LeRobot package to be installed:

```bash
cd /Users/jerome/Documents/lerobot
pip install -e .
```

Additional dependencies (installed with LeRobot):
- torch
- matplotlib
- numpy
- accelerate
- draccus

---

## 📚 Learn More

- LeRobot Documentation: Check `../docs/` folder
- HuggingFace Models: https://huggingface.co/lerobot
- Original test.py: `../test.py` (combined version of all 3 scripts)
