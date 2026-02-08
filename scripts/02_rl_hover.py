"""
02_rl_hover.py — Train an RL agent (PPO) to hover a quadrotor.

This creates a custom Gymnasium environment wrapping the same
point-mass quadrotor dynamics from 01_hover_demo.py, then trains
a PPO agent with stable-baselines3 to learn hover control.

No pybullet dependency — just numpy + gymnasium + sb3.

Run: python scripts/02_rl_hover.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import yaml


# ============================================================
# Custom Gymnasium Environment
# ============================================================

class QuadrotorHoverEnv(gym.Env):
    """Gymnasium environment for quadrotor hover control.
    
    Observation: [x, y, z, vx, vy, vz, tx-x, ty-y, tz-z]  (9-dim)
        - Position and velocity of drone
        - Position error to target
        
    Action: [ax, ay, az]  (3-dim, continuous)
        - Acceleration commands (normalized to [-1, 1])
        
    Reward: Negative distance to target + alive bonus - energy penalty
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, target=None, max_steps=500, dt=1/48, 
                 mass=0.027, drag=0.01, gravity=9.81):
        super().__init__()
        
        self.dt = dt
        self.mass = mass
        self.drag = drag
        self.g = gravity
        self.max_steps = max_steps
        self.max_accel = 2.4 * self.g
        
        # Target hover position
        self.target = target if target is not None else np.array([0.0, 0.0, 1.0])
        
        # Observation: [x, y, z, vx, vy, vz, err_x, err_y, err_z]
        obs_high = np.array([5, 5, 5, 10, 10, 10, 5, 5, 5], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        # Action: normalized acceleration [-1, 1] -> scaled to max_accel
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        
        self.state = np.zeros(6)
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial position near origin, on ground
        self.state = np.zeros(6)
        self.state[0:3] = self.np_random.uniform(-0.3, 0.3, size=3)
        self.state[2] = abs(self.state[2])  # Start above ground
        self.state[3:6] = self.np_random.uniform(-0.1, 0.1, size=3)
        
        self.step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # Scale action to acceleration
        accel = np.array(action, dtype=np.float64) * self.max_accel
        
        pos = self.state[0:3].copy()
        vel = self.state[3:6].copy()
        
        # Dynamics
        drag_accel = -self.drag * vel / self.mass
        gravity_accel = np.array([0, 0, -self.g])
        total_accel = accel + gravity_accel + drag_accel
        
        new_vel = vel + total_accel * self.dt
        new_pos = pos + new_vel * self.dt
        
        # Ground constraint
        if new_pos[2] < 0:
            new_pos[2] = 0
            new_vel[2] = max(0, new_vel[2])
        
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        
        # Reward
        error = np.linalg.norm(new_pos - self.target)
        vel_mag = np.linalg.norm(new_vel)
        action_mag = np.linalg.norm(action)
        
        # Dense reward: close to target is good, low velocity is good, low effort is good
        reward = -error * 2.0             # Position error penalty
        reward -= vel_mag * 0.1           # Velocity penalty (prefer stillness at hover)
        reward -= action_mag * 0.01       # Energy penalty
        reward += 0.1                     # Alive bonus
        
        # Bonus for being very close
        if error < 0.1:
            reward += 1.0
        if error < 0.05:
            reward += 2.0
        
        # Termination conditions
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Terminate if too far away
        if error > 3.0:
            terminated = True
            reward -= 10.0
        
        return self._get_obs(), reward, terminated, truncated, {"error": error}
    
    def _get_obs(self):
        pos = self.state[0:3]
        vel = self.state[3:6]
        err = self.target - pos
        return np.concatenate([pos, vel, err]).astype(np.float32)


# ============================================================
# Training Callback for Reward Logging
# ============================================================

class RewardCallback(BaseCallback):
    """Logs episode rewards for plotting."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = {}
        
    def _on_step(self):
        # Track rewards per environment
        for i, done in enumerate(self.locals.get("dones", [])):
            if i not in self._current_rewards:
                self._current_rewards[i] = 0
            self._current_rewards[i] += self.locals["rewards"][i]
            
            if done:
                self.episode_rewards.append(self._current_rewards[i])
                self._current_rewards[i] = 0
        return True


# ============================================================
# Main Training
# ============================================================

def main():
    print("=" * 60)
    print("  404 Pilots Not Found — RL Hover Training")
    print("  PPO on Custom Quadrotor Env")
    print("=" * 60)
    
    # Load params
    config_path = Path(__file__).parent.parent / "configs" / "quadrotor_params.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)['quadrotor']
        mass = params['mass']
        drag = params['drag']['linear']['x']
        gravity = params['gravity']
        print(f"\n[*] Loaded Crazyflie 2.x params")
    else:
        mass, drag, gravity = 0.027, 0.01, 9.81
        print(f"\n[*] Using default params")
    
    # Create environment
    env = QuadrotorHoverEnv(
        target=np.array([0.0, 0.0, 1.0]),
        max_steps=500,
        mass=mass,
        drag=drag,
        gravity=gravity,
    )
    
    print(f"    Observation space: {env.observation_space.shape}")
    print(f"    Action space: {env.action_space.shape}")
    
    # Training parameters
    TOTAL_TIMESTEPS = 200_000  # Increase for better results
    
    print(f"\n[*] Training PPO for {TOTAL_TIMESTEPS:,} timesteps...")
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(Path(__file__).parent.parent / "models" / "tb_logs"),
    )
    
    # Train with callback
    callback = RewardCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    
    # Save model
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ppo_hover"
    model.save(str(model_path))
    print(f"\n[*] Model saved to: {model_path}")
    
    # ============================================================
    # Evaluate
    # ============================================================
    print(f"\n[*] Evaluating trained agent...")
    
    eval_episodes = 10
    eval_rewards = []
    eval_errors = []
    
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        total_reward = 0
        final_error = 0
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_error = info["error"]
            if terminated or truncated:
                break
        
        eval_rewards.append(total_reward)
        eval_errors.append(final_error)
    
    print(f"    Eval over {eval_episodes} episodes:")
    print(f"    Mean reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"    Mean final error: {np.mean(eval_errors):.4f} m")
    
    # ============================================================
    # Plot Training Rewards
    # ============================================================
    print(f"\n[*] Generating training plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PPO Hover Training Results", fontsize=14, fontweight='bold')
    
    # Episode rewards
    ax1 = axes[0]
    if len(callback.episode_rewards) > 0:
        rewards = callback.episode_rewards
        ax1.plot(rewards, alpha=0.3, color='blue', linewidth=0.5)
        # Rolling average
        window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
        if window > 1:
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), rolling, color='red', linewidth=2, label=f'Rolling avg ({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Evaluation trajectory
    ax2 = axes[1]
    obs, _ = env.reset(seed=42)
    traj_pos = [obs[0:3].copy()]
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        traj_pos.append(obs[0:3].copy())
        if terminated or truncated:
            break
    traj_pos = np.array(traj_pos)
    
    ax2.plot(range(len(traj_pos)), traj_pos[:, 0], 'r-', label='X', linewidth=1)
    ax2.plot(range(len(traj_pos)), traj_pos[:, 1], 'g-', label='Y', linewidth=1)
    ax2.plot(range(len(traj_pos)), traj_pos[:, 2], 'b-', label='Z', linewidth=1)
    ax2.axhline(y=1.0, color='b', linestyle='--', alpha=0.4, label='Z target (1.0m)')
    ax2.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Evaluation Trajectory (Best Policy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = model_dir / "rl_hover_training.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved plot to: {plot_path}")
    plt.close()
    
    print(f"\n[*] Done! TensorBoard logs at: {model_dir / 'tb_logs'}")
    print(f"    Run: tensorboard --logdir {model_dir / 'tb_logs'}")


if __name__ == "__main__":
    main()
