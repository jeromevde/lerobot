# Cheapest Ways to Get GPU Compute for LeRobot Training

## 🎯 Quick Answer: SSH-Accessible On-Demand GPU Providers

All of these support **SSH access** and **on-demand hourly billing** (no long-term commitments):

---

## 1. **RunPod** ⭐ RECOMMENDED FOR BEGINNERS
- **Price**: $0.20-0.34/hour for RTX 3090 (24GB VRAM)
- **SSH**: ✅ Full SSH + JupyterLab
- **Setup**: 2 minutes
- **Website**: https://runpod.io

### Why RunPod?
- Easiest to use
- No credit card required for first $10
- Pay-as-you-go per minute
- Pre-built PyTorch/CUDA templates

### Setup Steps:
```bash
# 1. Create account at runpod.io
# 2. Click "Deploy" → "GPU Instances"
# 3. Choose "PyTorch" template
# 4. Select RTX 3090 or A4000 ($0.24/hr recommended)
# 5. Click "Deploy On-Demand"
# 6. Wait 30 seconds, then click "Connect" → "Start SSH over exposed TCP"
# 7. Copy the SSH command, e.g.:
ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_ed25519
```

Then install LeRobot:
```bash
cd /workspace  # Persistent storage
git clone https://github.com/huggingface/lerobot
cd lerobot
pip install -e .
python test.py  # Your script!
```

**Cost Example**: 10 hours of training @ $0.24/hr = **$2.40**

---

## 2. **vast.ai** 💰 CHEAPEST
- **Price**: $0.10-0.25/hour for RTX 3090
- **SSH**: ✅ Direct SSH
- **Setup**: 5 minutes
- **Website**: https://vast.ai

### Why vast.ai?
- Absolute cheapest (peer-to-peer GPU marketplace)
- Filter by reliability score
- Good for long training runs

### Gotchas:
- Machines can be unreliable (check host rating >98%)
- Need to manage your own Docker container
- Less beginner-friendly

### Setup:
```bash
# 1. Create account, add $10 credit
# 2. Search for instances with:
#    - GPU: RTX 3090
#    - Reliability: >98%
#    - DLPerf: >60
# 3. Click "Rent", select "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
# 4. SSH appears after instance starts:
ssh root@<host> -p <port>
```

**Cost Example**: 10 hours @ $0.12/hr = **$1.20**

---

## 3. **Lambda Labs** 🏢 MOST RELIABLE
- **Price**: $0.50-1.10/hour for A10 (24GB)
- **SSH**: ✅ Native SSH
- **Setup**: 3 minutes
- **Website**: https://lambdalabs.com/service/gpu-cloud

### Why Lambda?
- Enterprise-grade reliability
- Best networking (fast dataset downloads)
- PyTorch/CUDA pre-installed
- Popular with researchers

### Setup:
```bash
# 1. Sign up, add payment ($10 min)
# 2. Launch instance: A10 (24GB) - $0.75/hr
# 3. SSH key auto-configured
ssh ubuntu@<instance-ip>

# Lambda includes PyTorch already!
git clone https://github.com/huggingface/lerobot
cd lerobot && pip install -e .
```

**Cost Example**: 10 hours @ $0.75/hr = **$7.50**

---

## 4. **Google Colab Pro** 🚀 NO SSH BUT CONVENIENT
- **Price**: $10/month unlimited OR $0.10/compute unit (~$1-2/session)
- **SSH**: ❌ (but has VSCode integration)
- **Setup**: 0 minutes
- **Website**: https://colab.google

### Why Colab?
- Click and run (zero setup)
- 15-30GB GPUs (T4, A100)
- Great for quick experiments
- Can use VS Code with extension

### Workaround for SSH:
Use ngrok tunnel (advanced):
```python
# In Colab notebook:
!pip install colab-ssh
from colab_ssh import launch_ssh
launch_ssh('YOUR_NGROK_TOKEN', password='temp123')
# Then SSH via ngrok URL
```

**Cost Example**: $10/month flat = unlimited experiments

---

## 5. **Paperspace Gradient** 📊 MIDDLE GROUND
- **Price**: $0.45/hour for RTX 5000 (16GB)
- **SSH**: ✅ Via web terminal or SSH keys
- **Setup**: 3 minutes
- **Website**: https://paperspace.com/gradient

---

## 📊 Price Comparison Table

| Provider      | RTX 3090 (24GB) | A10 (24GB) | SSH | Reliability | Beginner-Friendly |
|---------------|-----------------|------------|-----|-------------|-------------------|
| **RunPod**    | $0.24/hr        | $0.49/hr   | ✅   | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐           |
| **vast.ai**   | $0.12/hr        | $0.30/hr   | ✅   | ⭐⭐⭐        | ⭐⭐⭐              |
| **Lambda**    | $1.10/hr        | $0.75/hr   | ✅   | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐             |
| **Colab Pro** | N/A (T4/A100)   | N/A        | 🔀  | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐           |
| **Paperspace**| N/A             | $0.76/hr   | ✅   | ⭐⭐⭐⭐      | ⭐⭐⭐⭐             |

---

## 🎓 For Students: FREE Options

### 1. **Google Colab Free Tier**
- **Price**: FREE
- **GPU**: T4 (16GB) for 12 hours/session
- **Limit**: ~20 hours/week
- **Perfect for**: Learning & small experiments

### 2. **Kaggle Notebooks**
- **Price**: FREE  
- **GPU**: P100 (16GB) or T4
- **Limit**: 30 hours/week
- **Website**: https://kaggle.com/code

### 3. **University HPC Clusters**
- **Price**: FREE if you're a student
- **GPU**: Often A100s or V100s
- Ask your CS department!

---

## 💡 My Recommendation for LeRobot

### Just Starting Out?
→ **RunPod + RTX 3090** ($0.24/hr)
- 10 hours = $2.40 for a full training run
- Easy SSH setup
- Stop/restart anytime

### Serious Training?
→ **Lambda Labs + A10** ($0.75/hr)
- Reliable for overnight runs
- Fast downloads from HuggingFace
- Better support

### On a Budget?
→ **vast.ai + RTX 3090** ($0.12/hr)
- 10 hours = $1.20
- Check host reliability >98%
- Save instances to favorites

---

## 🔧 Quick Start: RunPod Example

```bash
# 1. Sign up at runpod.io (30 seconds)
# 2. Add $5 credit (lasts ~20 hours on RTX 3090)
# 3. Deploy instance:
#    Template: "RunPod PyTorch"
#    GPU: RTX 3090 (24GB)
#    Disk: 50GB
# 4. Connect via SSH (shown in dashboard)

# In the SSH session:
cd /workspace
git clone https://github.com/huggingface/lerobot
cd lerobot
pip install -e .

# Run your script!
python test.py
# Set RUN_TRAINING=True in test.py for real training

# When done:
# Go to RunPod dashboard → "Stop" (charges stop immediately)
# /workspace is persistent, next session picks up where you left off
```

---

## 📉 Cost Estimates for LeRobot

Typical training times on RTX 3090 (24GB):

| Task                  | Steps   | Time    | RunPod Cost | vast.ai Cost |
|-----------------------|---------|---------|-------------|--------------|
| PushT (Diffusion)     | 100k    | 3 hours | $0.72       | $0.36        |
| Aloha ACT             | 50k     | 5 hours | $1.20       | $0.60        |
| Aloha ACT (full 100k) | 100k    | 10 hours| $2.40       | $1.20        |

**Bottom line**: You can do serious robotics training for **under $5**.

---

## 🚨 Tips to Save Money

1. **Use dataset caching**: Download datasets once to `/workspace` (persists across sessions)
2. **Stop instances when not training**: Charges stop immediately
3. **Use spot/interruptible instances**: 50-70% cheaper (risk of interruption)
4. **Monitor GPU utilization**: Make sure you're actually using the GPU!
   ```bash
   watch -n 1 nvidia-smi  # Check GPU usage
   ```
5. **Start small**: Test with `--steps 100` first, then scale up

---

## 🎯 TL;DR

**I want the easiest setup**: → RunPod ($0.24/hr)  
**I want the cheapest**: → vast.ai ($0.12/hr)  
**I want the most reliable**: → Lambda ($0.75/hr)  
**I want FREE**: → Google Colab Free or Kaggle  
**I'm a student**: → Ask your university for HPC access  

All of these are **on-demand** (pay per minute), **SSH-accessible** (except Colab), and require **minimal effort** to set up.

Happy training! 🤖
