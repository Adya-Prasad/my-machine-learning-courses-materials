# mountain_car.py
# Exact Python translation of the official CS229 mountain_car.m
import numpy as np

class MountainCar:
    def __init__(self, mode=0):
        # mode 0: normal, mode 1: noisy (not used in pset)
        self.mode = mode
        self.gravity = 0.0025
        self.force = 0.001
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0

    def reset(self):
        # Start at bottom of valley
        self.position = -0.5 + np.random.rand() * 0.1
        self.velocity = 0.0
        return self.position, self.velocity

    def step(self, action):
        # action: -1 (left), 0 (none), +1 (right)
        assert action in [-1, 0, 1], "Action must be -1, 0, or 1"

        # Add noise if mode == 1 (not used)
        if self.mode == 1:
            action += np.random.randn() * 0.1

        # Physics update
        self.velocity += action * self.force - np.cos(3 * self.position) * self.gravity
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)

        # If car hits left wall, zero velocity
        if self.position <= self.min_position:
            self.velocity = 0.0

        # Reward and done
        done = bool(self.position >= self.goal_position)
        reward = -1.0

        return self.position, self.velocity, reward, done

    # ------------------------------------------------------------
    # Beautiful visualization function (exactly like the MATLAB one)
    # ------------------------------------------------------------
    def draw_mountain_car_q(self, Q, pos_bins, vel_bins, filename=None):

        import matplotlib.pyplot as plt

        # Use the provided bin coordinates so the mesh matches Q's shape.
        # `pos_bins` and `vel_bins` are 1-D arrays with length equal to
        # the number of discretization bins (same as Q's first two dims).
        Pos, Vel = np.meshgrid(pos_bins, vel_bins)  # Vel first → (N_pos,N_vel)

        # Best action — NO TRANSPOSE!
        best_action = np.argmax(Q, axis=2)   # shape (30,30) → perfect match

        plt.figure(figsize=(10, 6))
        plt.contourf(Pos, Vel, best_action, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=['#ffaaaa', '#aaffaa', '#aaaaff'], alpha=0.7)
        plt.colorbar(ticks=[0, 1, 2], label='Best Action (-1=left, 0=none, +1=right)')

        # Zero velocity line
        plt.plot([-1.2, 0.6], [0, 0], 'k--', linewidth=1)
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Learned Q-Learning Policy in Phase Space')
        plt.xlim(-1.2, 0.6)
        plt.ylim(-0.07, 0.07)

        # Draw hill shape (scaled)
        x = np.linspace(-1.2, 0.6, 500)
        height = np.sin(3 * x) * 0.045 + 0.055
        plt.plot(x, height, 'k-', linewidth=3)

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()