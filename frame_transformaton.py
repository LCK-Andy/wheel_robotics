import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------
# Utility Functions
# ----------------------------
def rot_x(theta_deg):
    """Rotation about X-axis (degrees)."""
    theta = np.radians(theta_deg)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rot_y(theta_deg):
    """Rotation about Y-axis (degrees)."""
    theta = np.radians(theta_deg)
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rot_z(theta_deg):
    """Rotation about Z-axis (degrees)."""
    theta = np.radians(theta_deg)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def homogeneous_transform(R, p):
    """Build a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def plot_frame_3d(ax, T, label, colors=("r", "g", "b")):
    """Plot a 3D coordinate frame from a homogeneous transform."""
    origin = T[:3, 3]
    R = T[:3, :3]
    for i, c in enumerate(colors):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            R[0, i],
            R[1, i],
            R[2, i],
            color=c,
            length=1,
            normalize=True,
        )
    ax.text(origin[0], origin[1], origin[2], label, fontsize=12, fontweight="bold")


# ----------------------------
# Define Transformations
# ----------------------------

# Frame A (base)
R_A = np.eye(3)
p_A = np.zeros(3)
T_A = homogeneous_transform(R_A, p_A)

# Frame B relative to A
R_AB = rot_x(90) @ rot_z(180)
p_AB = np.array([0, 4, 2])
T_AB = homogeneous_transform(R_AB, p_AB)

# Frame C relative to B
R_BC = rot_y(0)
p_BC = np.array([3, 0, 0])
T_BC = homogeneous_transform(R_BC, p_BC)

# ----------------------------
# Visualization
# ----------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot all frames in A's coordinates
plot_frame_3d(ax, T_A, "A")
plot_frame_3d(ax, T_AB, "B")
plot_frame_3d(ax, T_BC, "C")

# Format plot
ax.set_xlim([-1, 5])
ax.set_ylim([-1, 5])
ax.set_zlim([-1, 5])
ax.set_box_aspect([1, 1, 1])
ax.set_title("3D Reference Frame Transformations (Homogeneous Form)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.grid(True)
plt.show()
