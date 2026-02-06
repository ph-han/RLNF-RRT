
# Server execution script for run-experiment workflow
# 1. Update code from git
# 2. Activate venv
# 3. Run training script

sshpass -p 'phan' ssh phan@166.104.224.180 << 'EOF'
cd ~/RLNF-RRT
echo "Pulling latest code..."
git pull origin main

echo "Activating venv..."
export VIRTUAL_ENV="/home/phan/RLNF-RRT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

echo "Checking python version and torch..."
python --version
python -c "import torch; print(f'Torch version: {torch.__version__}')"

echo "Starting training..."
# Kill any existing training processes (optional, but safer for automation)
pkill -f train_plannerflows.py || true

CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_plannerflows.py > train.log 2>&1 &
PID=$!
echo "Training started with PID: $PID"
EOF
