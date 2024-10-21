import numpy as np
import matplotlib.pyplot as plt

# Define the manipulator's link lengths and mass properties
l1, l2, l3 = 1.0, 1.0, 1.0  # Lengths of each link (meters)
m1, m2, m3 = 1.0, 1.0, 1.0  # Mass of each link (kilograms)
g = 9.81  # Acceleration due to gravity (m/s²)

# Initial state variables
theta = np.array([0.0, 0.0, 0.0])  # Joint angles (radians)

# Time settings
dt = 0.01  # Time step (seconds)
sim_time = 2.0  # Total simulation time for each move (seconds)
num_steps = int(sim_time / dt)

# List to store clicked points
clicked_points = []

def forward_kinematics(theta):
    """Compute the (x, y) position of the end-effector based on joint angles."""
    x = (l1 * np.cos(theta[0]) +
         l2 * np.cos(theta[0] + theta[1]) +
         l3 * np.cos(theta[0] + theta[1] + theta[2]))
    y = (l1 * np.sin(theta[0]) +
         l2 * np.sin(theta[0] + theta[1]) +
         l3 * np.sin(theta[0] + theta[1] + theta[2]))
    return np.array([x, y])

def compute_gravitational_torques(theta):
    """Compute the gravitational torques at each joint."""
    tau = np.zeros(3)
    
    tau[0] = -m1 * g * (l1 / 2) * np.cos(theta[0]) - \
             m2 * g * (l1 * np.cos(theta[0]) + (l2 / 2) * np.cos(theta[0] + theta[1])) - \
             m3 * g * (l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1]) + (l3 / 2) * np.cos(theta[0] + theta[1] + theta[2]))
    
    tau[1] = -m2 * g * (l2 / 2) * np.cos(theta[0] + theta[1]) - \
             m3 * g * (l2 * np.cos(theta[0] + theta[1]) + (l3 / 2) * np.cos(theta[0] + theta[1] + theta[2]))
    
    tau[2] = -m3 * g * (l3 / 2) * np.cos(theta[0] + theta[1] + theta[2])
    
    return tau

def inverse_kinematics(target):
    """Compute the joint angles using a simple iterative inverse kinematics."""
    global theta
    for _ in range(num_steps):
        # Compute the end-effector position
        current_position = forward_kinematics(theta)
        
        # Compute the error to the target
        error = target - current_position
        
        # Use a simple proportional controller to adjust the joint angles
        k_p = 0.1  # Proportional gain
        # Calculate the Jacobian
        J = jacobian(theta)
        
        # Update the joint angles
        dtheta = np.dot(np.linalg.pinv(J), error)  # Compute the change in joint angles
        theta += k_p * dtheta  # Update the joint angles

        # Wrap theta values to allow for continuous rotation
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # Wrap angles to [-π, π]

        # Update the plot
        plot_robot()
        plt.pause(dt)

def jacobian(theta):
    """Compute the Jacobian matrix for the manipulator."""
    J = np.array([
        [-l1 * np.sin(theta[0]) - l2 * np.sin(theta[0] + theta[1]) - l3 * np.sin(theta[0] + theta[1] + theta[2]),
         -l2 * np.sin(theta[0] + theta[1]) - l3 * np.sin(theta[0] + theta[1] + theta[2]),
         -l3 * np.sin(theta[0] + theta[1] + theta[2])],
        [ l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1]) + l3 * np.cos(theta[0] + theta[1] + theta[2]),
          l2 * np.cos(theta[0] + theta[1]) + l3 * np.cos(theta[0] + theta[1] + theta[2]),
          l3 * np.cos(theta[0] + theta[1] + theta[2])]
    ])
    return J

def plot_robot(link_color='cyan'):
    """Plot the manipulator in its current configuration."""
    plt.cla()  # Clear the current axes
    
    # Calculate the positions of the joints and end-effector
    joint1 = np.array([0, 0])
    joint2 = l1 * np.array([np.cos(theta[0]), np.sin(theta[0])])
    joint3 = joint2 + l2 * np.array([np.cos(theta[0] + theta[1]), np.sin(theta[0] + theta[1])])
    end_effector = joint3 + l3 * np.array([np.cos(theta[0] + theta[1] + theta[2]), np.sin(theta[0] + theta[1] + theta[2])])
    
    # Plot the arm with specified color
    plt.plot([joint1[0], joint2[0], joint3[0], end_effector[0]], 
             [joint1[1], joint2[1], joint3[1], end_effector[1]], link_color, linewidth=4)  # Cyan links
    
    # Plot the joints as black circles
    plt.plot(joint1[0], joint1[1], 'ko', markersize=8)  # Joint 1 (Black dot)
    plt.plot(joint2[0], joint2[1], 'ko', markersize=8)  # Joint 2 (Black dot)
    plt.plot(joint3[0], joint3[1], 'ko', markersize=8)  # Joint 3 (Black dot)
    plt.plot(end_effector[0], end_effector[1], 'ko', markersize=8)  # End effector (Black dot)
    
    # Display torques on the plot
    torques = compute_gravitational_torques(theta)
    torque_text = f"Torque 1: {torques[0]:.2f} Nm\nTorque 2: {torques[1]:.2f} Nm\nTorque 3: {torques[2]:.2f} Nm"
    
    # Convert angles to degrees
    angles_degrees = np.degrees(theta)
    angles_text = f"Angle 1: {angles_degrees[0]:.2f}°\nAngle 2: {angles_degrees[1]:.2f}°\nAngle 3: {angles_degrees[2]:.2f}°"
    
    # Position for torque and angle text
    plt.text(-2.5, -2.5, torque_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8), ha='left')
    plt.text(2.7, -2.5, angles_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8), ha='right')  # Moved to the right

    # Plot the clicked points as green dots and dashed lines between them
    for i, point in enumerate(clicked_points):
        plt.plot(point[0], point[1], 'go', markersize=8)  # Green dot for each clicked point
        if i > 0:
            # Draw dashed lines between consecutive clicked points
            plt.plot([clicked_points[i-1][0], point[0]], [clicked_points[i-1][1], point[1]], 'k--')

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True)
    plt.title('3-DOF Robot Manipulator')

# Set up plot and click event
fig, ax = plt.subplots()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.grid(True)

# Draw the robot in the initial configuration
plot_robot()

# Callback for clicking event
def on_click(event):
    if event.inaxes:
        target = np.array([event.xdata, event.ydata])
        print(f"Clicked target: {target}")
        
        # Store clicked points
        clicked_points.append(target)
        
        inverse_kinematics(target)  # Move to the clicked point
        
        # Plot the robot with the updated clicked points
        plot_robot(link_color='cyan')  

# Connect the click event to the callback function
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
