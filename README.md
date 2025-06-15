# SCARA Pick-and-Place with Reinforcement Learning

This project simulates a SCARA robot performing a pick-and-place task using PyBullet and trains it using PPO from Stable-Baselines3.

## 🚀 Features
- SCARA robot simulated in PyBullet
- Cube pick and place using inverse kinematics
- Reward-based PPO training (Stable-Baselines3)
- Replay, evaluation, and success metrics
- Headless and GUI-compatible

## 🧰 Requirements
```bash
pip install -r requirements.txt
```

## 🏁 How to Train
```bash
python scara_train_checkpointed.py
```

## 🎬 How to Evaluate Pick-and-Place
```bash
python scara_pick_place_eval.py
```

## 📁 Project Structure
```
scara-pickplace-rl/
├── scara_orange.urdf
├── scara_pick_env_final.py
├── scara_pick_env_place.py
├── scara_train_checkpointed.py
├── scara_pick_place_eval.py
└── scara_rl_logs/
    └── scara_model_step_*.zip
```

## 📜 License
MIT License