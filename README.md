# 404 Pilots Not Found — AI Grand Prix

Autonomous drone racing competition by Anduril Industries.
Team Rayburn AI.

## Setup
```bash
cd ai-grand-prix
.\venv\Scripts\activate
pip install -r requirements.txt
python scripts/01_hover_demo.py
```

## Structure
- `scripts/` — runnable demos and experiments
  - `01_hover_demo.py` — PID hover + waypoint following (pure numpy sim)
  - `02_rl_hover.py` — RL training with PPO to learn hover control
  - `03_gate_detection_synthetic.py` — synthetic gate detection data generator
- `models/` — trained model checkpoints
- `configs/` — environment and training configs
- `data/` — datasets (synthetic gates, etc.)

## Stack
- Python 3.12 + PyTorch (CUDA 12.1)
- **Flightmare** (UZH RPG) — primary RL training simulator
  - High-throughput rendering, Unity-based, designed for drone racing RL
  - https://github.com/uzh-rpg/flightmare
- **AirSim / Colosseum** — secondary simulator for realistic visual perception
  - Photorealistic Unreal Engine environments
  - https://github.com/CodexLabsLLC/Colosseum
- **Custom lightweight sim** — numpy-based quadrotor dynamics for rapid prototyping
  - Point-mass + drag model, Crazyflie 2.x parameters
  - Used in scripts/01-02 before graduating to full sims
- stable-baselines3 (RL training — PPO, SAC, etc.)
- OpenCV + YOLOv8 / Ultralytics (gate perception)
- NumPy, SciPy, CasADi (planning/control)
- TensorBoard (training visualization)

## Roadmap
1. ✅ Lightweight sim + PID hover demo
2. ✅ RL hover training (PPO)
3. ✅ Synthetic gate detection data pipeline
4. 🔲 Train YOLOv8 gate detector on synthetic data
5. 🔲 Integrate Flightmare for high-fidelity RL training
6. 🔲 AirSim perception pipeline (camera → gate detection → planning)
7. 🔲 Full racing stack: perception → planning → control
