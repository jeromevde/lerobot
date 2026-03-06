"""
run_pi05.py — eval Pi0.5 (pretrained) on LIBERO-Spatial via Modal.
Saves metrics to results/pi05.json and MP4 videos to results/pi05/videos/ automatically.

╔══════════════════════════════════════════════╗
║  Estimated cost & runtime (A10G @ $1.10/hr)  ║
║  Container + lerobot[libero] install  ~4 min  ║
║  Pi0.5 weights download (first run)   ~8 min  ║
║  Pi0.5 weights download (cached)      ~1 min  ║
║  Eval: 10 tasks × 3 episodes         ~20 min  ║
║  ─────────────────────────────────────────── ║
║  Total first run   ~32 min   ~$0.60          ║
║  Total cached run  ~25 min   ~$0.46          ║
╚══════════════════════════════════════════════╝

Setup (one-time):
    pip install modal matplotlib
    modal token new

Run:
    python run_pi05.py

Then visualize:
    python plot.py
"""
import json
from pathlib import Path
import modal

POLICY_NAME = "pi05"
POLICY_PATH = "lerobot/pi05_libero_finetuned"   # ready-to-use HF checkpoint
TASK        = "libero_spatial"
N_EPISODES  = 3
N_VIDEOS    = 2   # MuJoCo rollout videos saved per run (0 to disable)

RESULTS_DIR = Path("results")

image = (
    # Use official PyTorch image — torch/cuda already baked in, saves ~900MB download
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .run_commands("apt-get update -q && apt-get install -y -q git libgl1 libegl1 cmake")
    .pip_install("lerobot[libero]")
    .env({"MUJOCO_GL": "egl"})
)

# Persistent volume: survives between runs, stores videos + HF cache
volume = modal.Volume.from_name("lerobot-results", create_if_missing=True)

app = modal.App("libero-pi05", image=image)


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={
        "/results": volume,
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True),
    },
)
def run() -> dict:
    import subprocess, json, os
    from pathlib import Path

    out_dir = f"/results/eval/{POLICY_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "lerobot-eval",
        f"--policy.path={POLICY_PATH}",
        "--env.type=libero",
        f"--env.task={TASK}",
        f"--eval.n_episodes={N_EPISODES}",
        "--eval.batch_size=1",

        f"--output_dir={out_dir}",
    ]
    print("cmd:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])

    f = Path(out_dir) / "eval_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


if __name__ == "__main__":
    print(f"Running {POLICY_NAME} on Modal A10G...")
    with modal.enable_output():
        with app.run():
            results = run.remote()

    # Save results locally
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"{POLICY_NAME}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out_file}")
    print(f"Overall success: {results.get('overall', {}).get('pc_success', 'n/a')}%")

    # Auto-download MP4 videos from the Modal Volume
    videos_local = RESULTS_DIR / POLICY_NAME / "videos"
    videos_local.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading videos → {videos_local} ...")
    import subprocess
    subprocess.run(
        ["modal", "volume", "get", "lerobot-results",
         f"/eval/{POLICY_NAME}/videos", str(videos_local)],
        check=False,
    )
    print("Done. Run `python plot.py` to compare policies.")
