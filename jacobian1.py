import numpy as np
import matplotlib.pyplot as plt

# Define the manipulator's link lengths
l1, l2, l3 = 1.0, 1.0, 1.0  # Lengths of each link

# Define obstacles as circles with center and radius
obstacles = [
    {"center": np.array([1.0, 1.0]), "radius": 0.3},
    {"center": np.array([-1.5, -0.5]), "radius": 0.3},
    {"center": np.array([0.5, -1.5]), "radius": 0.3},
]

def forward_kinematics(theta1, theta2, theta3):
    """Compute the (x, y) position of the end-effector based on joint angles."""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y])

def jacobian(theta1, theta2, theta3):
    """Compute the Jacobian matrix for the manipulator."""
    J = np.array([
        [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3),
         -l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3),
         -l3*np.sin(theta1+theta2+theta3)],
        [ l1*np.cos(theta1) + l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3),
          l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3),
          l3*np.cos(theta1+theta2+theta3)]
    ])
    return J

def inverse_kinematics_step(theta, target, alpha=0.1):
    """Perform one step of inverse kinematics to move towards the target."""
    current_position = forward_kinematics(*theta)
    error = target - current_position
    
    J = jacobian(*theta)
    dtheta = np.dot(np.linalg.pinv(J), error)  # Use pseudoinverse to compute joint changes
    theta += alpha * dtheta  # Update joint angles
    
    return theta

def plot_robot(theta):
    """Plot the manipulator in its current configuration."""
    # Calculate the positions of the joints
    joint1 = np.array([0, 0])
    joint2 = l1 * np.array([np.cos(theta[0]), np.sin(theta[0])])
    joint3 = joint2 + l2 * np.array([np.cos(theta[0] + theta[1]), np.sin(theta[0] + theta[1])])
    end_effector = joint3 + l3 * np.array([np.cos(theta[0] + theta[1] + theta[2]), np.sin(theta[0] + theta[1] + theta[2])])
    
    # Plot the arm
    plt.plot([joint1[0], joint2[0], joint3[0], end_effector[0]], 
             [joint1[1], joint2[1], joint3[1], end_effector[1]], 'ro-')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True)

# Plot obstacles
def plot_obstacles():
    for obs in obstacles:
        circle = plt.Circle(obs["center"], obs["radius"], color='gray', fill=True)
        plt.gca().add_artist(circle)

# Trajectory generation: Move step-by-step along the path
def follow_trajectory(theta_start, target, steps=100):
    """Move from the initial configuration to the target in small steps."""
    theta = np.copy(theta_start)
    for _ in range(steps):
        theta = inverse_kinematics_step(theta, target, alpha=0.05)
        
        # Clear and re-plot the manipulator in the new configuration
        plt.cla()
        plot_obstacles()  # Draw obstacles
        plot_robot(theta)
        plt.pause(0.05)  # Brief pause to visualize the motion
    return theta

# Store the current robot configuration globally
theta_current = np.array([0.0, 0.0, 0.0])  # Start from the initial configuration
clicked_points = []  # Store clicked points

# Callback for clicking event
def on_click(event):
    global theta_current  # Use the current configuration for the next target
    if event.inaxes:
        target = np.array([event.xdata, event.ydata])
        print(f"Clicked target: {target}")
        clicked_points.append(target)  # Add to clicked points for visual feedback
        
        # Perform stepwise movement towards the target from the current configuration
        theta_current = follow_trajectory(theta_current, target)

        # Final plot showing the robot at the final position and clicked points
        plt.cla()
        plot_obstacles()
        plot_robot(theta_current)
        # Plot clicked points as green dots
        for point in clicked_points:
            plt.plot(point[0], point[1], 'go')  
        plt.draw()

# Set up plot and click event
fig, ax = plt.subplots()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.grid(True)

# Draw obstacles initially
plot_obstacles()

# Connect the click event to the callback function
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
