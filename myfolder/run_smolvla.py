"""
run_smolvla.py — eval X-VLA on LIBERO-Spatial via Modal.
Uses the official pretrained checkpoint lerobot/xvla-libero (~93% success).
Saves metrics to results/xvla.json and MP4 videos to results/xvla/videos/ automatically.

Note: SmolVLA and ACT have no pretrained LIBERO checkpoints, so this script
was repurposed to evaluate X-VLA, which does.

╔══════════════════════════════════════════════╗
║  Estimated cost & runtime (A10G @ $1.10/hr)  ║
║  Container + lerobot[xvla] install    ~5 min  ║
║  X-VLA weights (first run)            ~3 min  ║
║  X-VLA weights (cached)               ~1 min  ║
║  Eval: 10 tasks × 3 episodes         ~15 min  ║
║  ─────────────────────────────────────────── ║
║  Total first run   ~23 min   ~$0.42          ║
║  Total cached run  ~16 min   ~$0.29          ║
╚══════════════════════════════════════════════╝

Setup (one-time):
    pip install modal matplotlib
    modal token new

Run:
    python run_smolvla.py

Then visualize:
    python plot.py
"""
import json
from pathlib import Path
import modal

POLICY_NAME = "xvla"
POLICY_PATH = "lerobot/xvla-libero"   # pretrained, ~93% on LIBERO
TASK        = "libero_spatial"
N_EPISODES  = 3
N_VIDEOS    = 2

RESULTS_DIR = Path("results")

image = (
    # Use official PyTorch image — torch/cuda already baked in, saves ~900MB download
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .run_commands("apt-get update -q && apt-get install -y -q git libgl1 libegl1 cmake")
    .pip_install("lerobot[libero]")
    .pip_install("lerobot[xvla]")
    .env({"MUJOCO_GL": "egl"})
)

volume = modal.Volume.from_name("lerobot-results", create_if_missing=True)

app = modal.App("libero-xvla", image=image)


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
        "--env.control_mode=absolute",   # required for X-VLA
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
    print(f"Running X-VLA ({POLICY_PATH}) on Modal A10G...")
    with modal.enable_output():
        with app.run():
            results = run.remote()

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
