"""
01_hover_demo.py — First flight: hover a drone with pure numpy simulation.

This demonstrates:
1. Simple 3D quadrotor dynamics (point-mass + drag + thrust)
2. PID-controlled hover and waypoint following
3. Matplotlib visualization (no external physics engine needed)

Uses Crazyflie 2.x parameters from configs/quadrotor_params.yaml.

Run: python scripts/01_hover_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


# ============================================================
# Quadrotor Dynamics (Point-Mass with Drag)
# ============================================================

class QuadrotorSim:
    """Simple 3D quadrotor simulation using point-mass dynamics.
    
    State: [x, y, z, vx, vy, vz]  (position + velocity)
    Action: [ax, ay, az]  (acceleration commands, simplified from thrust/attitude)
    
    This abstracts away full attitude dynamics for rapid prototyping.
    The PID controller outputs desired accelerations, and we apply them
    with drag and gravity.
    """
    
    def __init__(self, mass=0.027, drag_coeff=0.01, gravity=9.81, dt=1/48):
        self.mass = mass
        self.drag = drag_coeff
        self.g = gravity
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # Max acceleration (based on Crazyflie thrust-to-weight ~2.4:1)
        self.max_accel = 2.4 * self.g  # ~23.5 m/s²
        
    def reset(self, position=None):
        """Reset drone to initial position."""
        self.state = np.zeros(6)
        if position is not None:
            self.state[0:3] = position
        return self.state.copy()
    
    def step(self, accel_cmd):
        """Step simulation forward by dt.
        
        Args:
            accel_cmd: [ax, ay, az] desired acceleration (m/s²)
            
        Returns:
            state: updated state [x, y, z, vx, vy, vz]
        """
        # Clip acceleration command
        accel = np.clip(accel_cmd, -self.max_accel, self.max_accel)
        
        pos = self.state[0:3]
        vel = self.state[3:6]
        
        # Forces: command + gravity + drag
        drag_accel = -self.drag * vel / self.mass
        gravity_accel = np.array([0, 0, -self.g])
        
        total_accel = accel + gravity_accel + drag_accel
        
        # Semi-implicit Euler integration
        new_vel = vel + total_accel * self.dt
        new_pos = pos + new_vel * self.dt
        
        # Ground constraint
        if new_pos[2] < 0:
            new_pos[2] = 0
            new_vel[2] = max(0, new_vel[2])
        
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        
        return self.state.copy()


# ============================================================
# PID Controller
# ============================================================

class PIDController:
    """3D PID position controller for quadrotor."""
    
    def __init__(self, kp=None, ki=None, kd=None, gravity=9.81):
        self.kp = kp if kp is not None else np.array([2.0, 2.0, 2.0])
        self.ki = ki if ki is not None else np.array([0.0, 0.0, 0.1])
        self.kd = kd if kd is not None else np.array([1.5, 1.5, 1.5])
        self.g = gravity
        
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.max_integral = 5.0  # Anti-windup
        
    def compute(self, current_pos, current_vel, target_pos, dt):
        """Compute acceleration command to reach target position.
        
        Returns:
            accel_cmd: [ax, ay, az] acceleration command (m/s²)
        """
        error = target_pos - current_pos
        
        # P term
        p_term = self.kp * error
        
        # I term (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self.integral
        
        # D term (using velocity for smoother control)
        d_term = -self.kd * current_vel
        
        # Total acceleration + gravity compensation
        accel = p_term + i_term + d_term
        accel[2] += self.g  # Compensate for gravity
        
        return accel
    
    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)


# ============================================================
# Main Demo
# ============================================================

def load_params():
    """Load quadrotor parameters from YAML config."""
    config_path = Path(__file__).parent.parent / "configs" / "quadrotor_params.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        return params['quadrotor']
    return None


def main():
    print("=" * 60)
    print("  404 Pilots Not Found — Hover Demo")
    print("  AI Grand Prix Prep (Lightweight Sim)")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    if params:
        mass = params['mass']
        drag = params['drag']['linear']['x']
        gravity = params['gravity']
        kp = np.array(params['pid']['position']['kp'])
        ki = np.array(params['pid']['position']['ki'])
        kd = np.array(params['pid']['position']['kd'])
        print(f"\n[*] Loaded Crazyflie 2.x params from config")
    else:
        mass, drag, gravity = 0.027, 0.01, 9.81
        kp = np.array([2.0, 2.0, 2.0])
        ki = np.array([0.0, 0.0, 0.1])
        kd = np.array([1.5, 1.5, 1.5])
        print(f"\n[*] Using default parameters")
    
    # Simulation setup
    DT = 1.0 / 48  # Control frequency: 48 Hz
    DURATION = 20.0  # seconds
    
    sim = QuadrotorSim(mass=mass, drag_coeff=drag, gravity=gravity, dt=DT)
    pid = PIDController(kp=kp, ki=ki, kd=kd, gravity=gravity)
    
    # Initial position
    init_pos = np.array([0.0, 0.0, 0.0])
    state = sim.reset(init_pos)
    
    print(f"    Mass: {mass} kg")
    print(f"    Control freq: {1/DT:.0f} Hz")
    print(f"    Duration: {DURATION}s")
    
    # Waypoints (same spirit as original demo)
    waypoints = [
        np.array([0.0, 0.0, 1.0]),   # take off and hover
        np.array([1.0, 0.0, 1.0]),   # move forward
        np.array([1.0, 1.0, 1.5]),   # move right + up
        np.array([0.0, 1.0, 1.0]),   # move left
        np.array([0.0, 0.0, 1.0]),   # return to start
    ]
    
    print(f"\n[*] Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"    {i}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")
    
    # Flight loop
    print(f"\n[*] Starting flight...")
    total_steps = int(DURATION / DT)
    wp_idx = 0
    wp_threshold = 0.15  # meters
    wp_hold_time = 1.0   # seconds to hold at waypoint before advancing
    wp_hold_counter = 0.0
    
    # Data recording
    times = []
    positions = []
    velocities = []
    targets = []
    wp_events = []
    
    for step in range(total_steps):
        t = step * DT
        pos = state[0:3]
        vel = state[3:6]
        target = waypoints[wp_idx]
        
        # Check waypoint reached
        dist = np.linalg.norm(pos - target)
        if dist < wp_threshold:
            wp_hold_counter += DT
            if wp_hold_counter >= wp_hold_time and wp_idx < len(waypoints) - 1:
                wp_events.append((t, wp_idx))
                wp_idx += 1
                wp_hold_counter = 0.0
                pid.reset()  # Reset integral on waypoint switch
                target = waypoints[wp_idx]
                print(f"    [t={t:5.1f}s] Reached waypoint {wp_idx-1}! -> Next: {target}")
        else:
            wp_hold_counter = 0.0
        
        # PID control
        accel_cmd = pid.compute(pos, vel, target, DT)
        
        # Step simulation
        state = sim.step(accel_cmd)
        
        # Record
        times.append(t)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        targets.append(target.copy())
        
        # Print status every 2 seconds
        if step % (int(2.0 / DT)) == 0:
            print(f"    [t={t:5.1f}s] pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) "
                  f"dist={dist:.3f} wp={wp_idx}/{len(waypoints)-1}")
    
    pos = state[0:3]
    print(f"\n[*] Flight complete!")
    print(f"    Final position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"    Waypoints reached: {wp_idx}/{len(waypoints)-1}")
    
    # Convert to arrays
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    targets = np.array(targets)
    
    # ============================================================
    # Visualization
    # ============================================================
    print(f"\n[*] Generating plots...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("404 Pilots Not Found — Hover & Waypoint Demo", fontsize=14, fontweight='bold')
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1, alpha=0.7, label='Trajectory')
    for i, wp in enumerate(waypoints):
        ax1.scatter(*wp, c='red', s=100, marker='^', zorder=5)
        ax1.text(wp[0]+0.05, wp[1]+0.05, wp[2]+0.05, f'WP{i}', fontsize=8)
    ax1.scatter(*positions[0], c='green', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(*positions[-1], c='orange', s=100, marker='s', zorder=5, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Flight Path')
    ax1.legend(fontsize=8)
    
    # Position vs time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, positions[:, 0], 'r-', label='X', linewidth=1)
    ax2.plot(times, positions[:, 1], 'g-', label='Y', linewidth=1)
    ax2.plot(times, positions[:, 2], 'b-', label='Z', linewidth=1)
    ax2.plot(times, targets[:, 0], 'r--', alpha=0.4, linewidth=1)
    ax2.plot(times, targets[:, 1], 'g--', alpha=0.4, linewidth=1)
    ax2.plot(times, targets[:, 2], 'b--', alpha=0.4, linewidth=1)
    for t_event, _ in wp_events:
        ax2.axvline(x=t_event, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time (dashed = target)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity vs time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, velocities[:, 0], 'r-', label='Vx', linewidth=1)
    ax3.plot(times, velocities[:, 1], 'g-', label='Vy', linewidth=1)
    ax3.plot(times, velocities[:, 2], 'b-', label='Vz', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Tracking error
    ax4 = fig.add_subplot(2, 2, 4)
    error = np.linalg.norm(positions - targets, axis=1)
    ax4.plot(times, error, 'k-', linewidth=1)
    ax4.axhline(y=wp_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({wp_threshold}m)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Tracking Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "hover_demo_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved plot to: {output_path}")
    
    plt.close()
    print("\n[*] Done. Welcome to drone racing!")


if __name__ == "__main__":
    main()
