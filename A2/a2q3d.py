import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the temperature function T(x,y)
def T(x, y):
    return x**2 * y

# Define the gradient components
def dT_dx(x, y):
    return 2*x*y

def dT_dy(x, y):
    return x**2

# Create meshgrid for plotting
x = np.linspace(-2, 0, 50)
y = np.linspace(0, 2, 50)
X, Y = np.meshgrid(x, y)
Z = T(X, Y)

# Calculate gradient at point (-1, 3/2)
point_x, point_y = -1, 3/2
grad_x = dT_dx(point_x, point_y)
grad_y = dT_dy(point_x, point_y)
gradient = np.array([grad_x, grad_y])

# Direction vector
direction = np.array([-1, -1/2])
# Normalize direction vector
direction = direction / np.linalg.norm(direction)

# Calculate directional derivative
directional_derivative = np.dot(gradient, direction)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surface = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

# Plot gradient vector at the point
point_z = T(point_x, point_y)
ax.quiver(point_x, point_y, point_z, 
          grad_x/2, grad_y/2, 0,  # Scaled down for better visualization
          color='red', label='Gradient')

# Plot direction vector at the point
ax.quiver(point_x, point_y, point_z,
          direction[0]/2, direction[1]/2, 0,  # Scaled down for better visualization
          color='blue', label='Direction')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
ax.set_title('Temperature Surface with Gradient and Direction Vector')

# Add colorbar
fig.colorbar(surface, ax=ax, label='Temperature')

# Add legend
ax.legend()

# Print the results
print(f"Gradient at point (-1, 3/2): ({grad_x}, {grad_y})")
print(f"Directional derivative in direction [-1, -1/2]: {directional_derivative}")

plt.show()