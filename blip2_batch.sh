#!/bin/bash
# ====================================================
# Grid5000 BLIP-2 batch job script (optimized for Sophia fast GPUs)
# ====================================================
#OAR -n blip2_job                        # Job name
#OAR -l host=1/gpu=1,walltime=48:00:00  # 1 node + 1 GPU, 48 hours (can increase to 72:00:00 if needed)
#OAR -p "cluster='uvb' AND gpu_mem >= 40000"  # Target uvb cluster (H100 94GB — fastest on Sophia)
#OAR -O blip2_%jobid%.out                # Stdout log
#OAR -E blip2_%jobid%.err                # Stderr log
#OAR --besteffort                        # Besteffort: opportunistic, long walltime allowed, preemptible (but your script resumes!)

# Setup environment
module load conda 2>/dev/null || echo "Conda not available; skipping"
conda create -n blip -y --quiet 2>/dev/null || echo "Env exists or creation skipped"
conda activate blip

# Install dependencies (idempotent)
pip install --quiet transformers==4.40.0 accelerate bitsandbytes==0.43.3 pillow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "===== ENV & GPU INFO ====="
which python
python --version
nvidia-smi
echo "=========================="

# Project directory
cd /home/ykebour/blip2

echo "===== Running inference ====="
python generate_captions.py   

echo "===== Job finished ====="