import numpy as np
import matplotlib.pyplot as plt
import math

# Solution using Triangular elements
# Defines the nodes and elements for Triangular elements
def generate_triangular_mesh(a, b, n_x, n_y, ax):
    nodes = []
    elements = []

    # Create nodes
    x_vals1 = np.linspace(0, ax, math.floor(1/2 * (n_x+1)), endpoint=False)
    x_vals2 = np.linspace(ax, a, math.ceil(1/2 * (n_x+1)), endpoint=True)
    x_vals = np.hstack([x_vals1, x_vals2])
    y_vals = np.linspace(0, b, n_y+1)
    nodes = np.array([[x, y] for y in y_vals for x in x_vals])

    # Create triangular elements
    for j in range(n_y):
        for i in range(n_x):
            n1 = j * (n_x + 1) + i
            n2 = n1 + 1
            n3 = n1 + (n_x + 1)
            n4 = n3 + 1

            # Two triangles per rectangular element
            elements.append([n1, n2, n3])  # First triangle

            elements.append([n2, n4, n3])  # Second triangle
    
    return nodes, elements


# Creates local stiffeness matrix for Triangular elements
def compute_stiffness_matrix(vertices,k):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Compute the area of the triangle
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    # Compute shape function gradients
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)

    # Compute the stiffness matrix
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = (b[i] * b[j] + c[i] * c[j]) * area
    K = k*K
    return K




# Creates glabal stiffeness matrix for triangular elements
def assemble_global_stiffness(nodes, elements, k1, k2, ax):
    n_nodes = len(nodes)
    global_K = np.zeros((n_nodes, n_nodes))

    for element in elements:
        # Get the vertices of the current element
        vertices = nodes[element]
        x_centroid = np.mean([nodes[i][0] for i in element])
        k = k1 if x_centroid < ax else k2
        # Compute the local stiffness matrix
        local_K = compute_stiffness_matrix(vertices, k)
        # Add the local stiffness matrix to the global matrix
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                global_K[ni, nj] += local_K[i, j] 
    return global_K



# Solves the equation for Triangular elements
def Triangular_finite_elements(nodes, elements, k1, k2, ax, boundary_conditions, boundary_nodes):
    global_K = assemble_global_stiffness(nodes, elements, k1, k2, ax)

    F = np.zeros(len(nodes))

    for boundary, condition in boundary_conditions.items():
        type = condition['type']
        b_elements = boundary_nodes[boundary]
        value = condition['values']
        if type == "dirichlet":
            Dirichlet_cond(b_elements, value[0], global_K, F)
        if type == "neumann":
            Neumann_cond(nodes, b_elements, value[0], F)

    T = np.linalg.solve(global_K,F)
    return global_K, F, T


# Solution using Rectangular elements

# Define the global node coordinates for Rectangular elements
def generate_rectangular_nodes(a, ax, b, n_x, n_y):
    x_vals1 = np.linspace(0, ax, math.floor(1/2 * (n_x+1)), endpoint=False)
    x_vals2 = np.linspace(ax, a, math.ceil(1/2 * (n_x+1)), endpoint=True)
    x_vals = np.hstack([x_vals1, x_vals2])
    y_vals = np.linspace(0, b, n_y+1)
    nodes = np.array([[x, y] for y in y_vals for x in x_vals])
    
    elements = []
    for j in range(n_y):
        for i in range(n_x):
            n1 = j * (n_x + 1) + i
            n2 = n1 + 1
            n3 = n1 + (n_x + 1)
            n4 = n3 + 1
            elements.append([n1, n2, n4, n3])
    
    return np.array(nodes), np.array(elements)


# Create a dictionary with boundary nodes
def get_boundary_nodes(nodes):
    boundary_nodes = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": []
    }

    for i in range(len(nodes)):
        x, y = nodes[i]
        node_index = i

        # Left boundary (x = 0)
        if np.isclose(x, 0):
            boundary_nodes["left"].append(node_index)

        # Right boundary (x = max x)
        elif np.isclose(x, nodes[-1, 0]):  # Last x value
            boundary_nodes["right"].append(node_index)

        # Bottom boundary (y = 0)
        if np.isclose(y, 0):
            boundary_nodes["bottom"].append(node_index)

        # Top boundary (y = max y)
        elif np.isclose(y, nodes[-1, 1]):  # Last y value
            boundary_nodes["top"].append(node_index)

    return boundary_nodes


# Compute local stiffeness matrix for rectangular elements
def compute_local_stiffness_matrix(hx, hy, k):
    K_e = np.zeros((4, 4))
    K11 = 1 / (3 * hx * hy) * (hx ** 2 + hy ** 2)
    K12 = 1 / (6 * hx * hy) * (hx ** 2 - 2 * hy ** 2)
    K13 = -1 / (6 * hx * hy) * (hx ** 2 + hy ** 2)
    K14 = -1 / (6 * hx * hy) * (2 * hx ** 2 - hy ** 2)
    K21 = K12
    K22 = K11
    K23 = K14
    K24 = K13
    K31 = K13
    K32 = K23
    K33 = K11
    K34 = K12
    K41 = K14
    K42 = K24
    K43 = K34
    K44 = K11
    K_e = np.array([[K11, K12, K13, K14], [K21, K22, K23, K24], [K31, K32, K33, K34], [K41, K42, K43, K44]]) * k
    return K_e

    

# Solves the equation for Rectangular elements
def Rectangular_finite_elements(nodes, elements, k1, k2, ax, boundary_conditions, boundary_nodes):
    n_nodes = len(nodes)
    K_global = np.zeros((n_nodes, n_nodes))

    for element in elements:
        # Determine which material to use based on element centroid
        x_centroid = np.mean([nodes[i][0] for i in element])
        k = k1 if x_centroid < ax else k2
        hx = nodes[element[1]][0] - nodes[element[0]][0] 
        hy = nodes[element[2]][1] - nodes[element[1]][1] 

        # Compute local stiffness matrix
        K_e = compute_local_stiffness_matrix(hx, hy, k)

        # Assemble into global stiffness matrix
        for i in range(4):
            for j in range(4):
                K_global[element[i], element[j]] += K_e[i, j]
    
    F = np.zeros(len(nodes))

    for boundary, condition in boundary_conditions.items():
        type = condition['type']
        b_elements = boundary_nodes[boundary]
        value = condition['values']
        if type == "dirichlet":
            Dirichlet_cond(b_elements, value[0], K_global, F)
        if type == "neumann":
            Neumann_cond(nodes, b_elements, value[0], F)

    T = np.linalg.solve(K_global,F)

    return K_global , F, T


# Boundary conditions

# Dirichlet boundary conditions
def Dirichlet_cond(nodes, value, global_K, F):
    for i in (nodes):
        global_K[i] = 0
        global_K[i][i] = 1
        F[i] = value


# Neumann boundary conditions
def Neumann_cond(nodes, boundary_elements, flux, F):
    for i in range(len(boundary_elements) - 1):
        n1 = boundary_elements[i]
        n2 = boundary_elements[i + 1]

        # Coordinates of the nodes
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        # Length of the boundary edge
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Compute contributions to the load vector (linear shape functions)
        F[n1] += flux * L / 2
        F[n2] += flux * L / 2


# Reading configuration file
def read_config(file_path):
    # Initialize data structure
    config_data = {
        "element_type": None,
        "a": 1,
        "b": 1,
        "hx": 0.2,
        "nx": 10,
        "ny": 10,
        "k1": 0.5,
        "k2": 1,
        "boundary_conditions": {},
    }
    
    # Parse the configuration file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:  # Skip comments and empty lines
            continue
        
        if ":" in line:  # Parse key-value pairs
            key, value = map(str.strip, line.split(":", 1))
            if key == "element_type":
                config_data["element_type"] = value
            elif key in ["a", "b", "hx"]:
                config_data[key] = float(value)
            elif key in ["nx", "ny"]:
                config_data[key] = int(value) - 1
            elif key in ["k1", "k2"]:
                config_data[key] = float(value)
            elif key in ["right", "left", "top", "bottom"]:
                condition_type, *values = value.split()
                config_data["boundary_conditions"][key] = {
                    "type": condition_type,
                    "values": list(map(float, values))
                }
    
    return config_data


# Presenting solution

# Printing the solution
def Print_solution(global_K, F, T):
    print("Global stiffness matrix K:")
    print(global_K)
    print("\n")
    print("Global F vector")
    print(F)
    print("\n")
    print("Temperature vector")
    print(T)


# Ploting solution for Triangular elements
def Plot_triangular_2D(T, nodes, elements, a, b):
    # Create a grid for visualization
    n_points = 200  # Number of points per dimension in the grid
    x_global = np.linspace(0, a, n_points)
    y_global = np.linspace(0, b, n_points)
    x_grid, y_grid = np.meshgrid(x_global, y_global)

    # Initialize the solution array
    T_global = np.zeros_like(x_grid)
    
    # Initialize weights array for accumulation
    weights = np.zeros_like(T_global)

    # Loop through each triangular element
    for element in elements:
        # Get the nodal coordinates and solution values for the current element
        element_nodes = nodes[element]
        element_T_values = T[element]

        # Get the coordinates of the triangle vertices
        x1, y1 = element_nodes[0]
        x2, y2 = element_nodes[1]
        x3, y3 = element_nodes[2]

        # Compute the area of the triangle
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Loop through the grid points
        for i in range(n_points):
            for j in range(n_points):
                xp, yp = x_grid[i, j], y_grid[i, j]

                # Check if the point lies inside the triangle using barycentric coordinates
                alpha = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / (2 * area)
                beta = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / (2 * area)
                gamma = 1 - alpha - beta

                if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
                    # Point is inside the triangle; interpolate the solution
                    T_point = alpha * element_T_values[0] + beta * element_T_values[1] + gamma * element_T_values[2]
                    
                    # Accumulate contributions to the grid point
                    T_global[i, j] += T_point
                    
                    # Accumulate weights to normalize later
                    weights[i, j] += alpha + beta + gamma

    # Normalize the solution by the accumulated weights
    # T_global /= weights
    T_global /= np.where(weights > 0, weights, 1)
    T_global[:,-1] = T_global[:,-2]
    T_global[0,:] = T_global[1,:]

    # Plot the solution as a 2D color map
    plt.figure()
    plt.contourf(x_global, y_global, T_global, levels=100, cmap='viridis')
    plt.colorbar(label="T [K]")
    plt.title("2D Solution Map")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.scatter(nodes[:, 0], nodes[:, 1], color="red", s=1, label="Nodes")  # Mark the nodes
    plt.legend()
    plt.grid(False)
    plt.savefig("triangular_color_map.jpg")
    plt.show()

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x_grid, y_grid, T_global, cmap='coolwarm')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Temperature [K]")

    # Plot settings
    ax.set_title("3D Temperature Field")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Temperature [K]")
    ax.view_init(elev=28, azim=-72)  # Set view angle
    plt.savefig("triangular_3D.jpg")
    plt.show()

    # Plotting the solution along a specific line in the 2D domain
    y_index = 100
    temperature_line = T_global[y_index, :]  # Extract row corresponding to y

    # Plot temperature over x
    plt.plot(x_global, temperature_line)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title(f"Temperature at y={b/2} m")
    plt.grid()
    plt.savefig("triangular_2D.jpg")
    plt.show()
    


# Plotting Rectangle

# Define the bilinear shape functions
def shape_functions(xi, eta):
    N1 = (1 - xi) * (1 - eta) / 4
    N2 = (1 + xi) * (1 - eta) / 4
    N3 = (1 + xi) * (1 + eta) / 4
    N4 = (1 - xi) * (1 + eta) / 4
    return np.array([N1, N2, N3, N4])

def xi_f(xp, hx, x):
    xi = 2 * (xp - x)/hx - 1
    return xi

def eta_f(yp, hy, y):
    eta = 2 * (yp - y)/hy - 1
    return eta

# Ploting solution for Rectangular elements
def Plot_rectangular_2D(T, nodes, elements, a, b):

    # Create a grid for visualization
    n_points = 200  # Number of points per dimension in the grid
    x_global = np.linspace(0, a, n_points)
    y_global = np.linspace(0, b, n_points)
    x_grid, y_grid = np.meshgrid(x_global, y_global)

    # Initialize the solution array
    T_global = np.zeros_like(x_grid)

    # Initialize weights array for accumulation
    weights = np.zeros_like(T_global)

    # Loop through each element and compute the solution
    for element in elements:
        # Get the nodal coordinates and solution values for the current element
        element_nodes = nodes[element]
        element_T_values = T[element]
        
        # Bottom-left corner and dimensions of the element
        x0, y0 = element_nodes[0]  # Bottom-left corner of the element
        hx = element_nodes[1][0] - x0  # Element width in x
        hy = element_nodes[3][1] - y0  # Element height in y
        
        # Loop through the grid points
        for i in range(n_points):
            for j in range(n_points):
                xp, yp = x_grid[i, j], y_grid[i, j]
                
                # Check if the point lies within the current element bounds
                if x0 <= xp <= x0 + hx and y0 <= yp <= y0 + hy:
                    # Map to reference coordinates
                    xi_pt = xi_f(xp, hx, x0)
                    eta_pt = eta_f(yp, hy, y0)
                    
                    # Evaluate shape functions
                    N = shape_functions(xi_pt, eta_pt)
                    
                    # Interpolate the solution and accumulate contributions
                    T_global[i, j] += np.dot(N, element_T_values)
                    
                    # Accumulate weights to normalize later
                    weights[i, j] += np.sum(N)

    # Normalize the solution by the accumulated weights
    T_global /= np.where(weights > 0, weights, 1)

    # Plot the solution as a 2D color map
    plt.figure()
    plt.contourf(x_global, y_global, T_global, levels=100, cmap='viridis')
    plt.colorbar(label="T [K]")
    plt.title("2D Solution Map")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.scatter(nodes[:, 0], nodes[:, 1], color="red", s = 1, label="Nodes")  # Mark the nodes
    plt.legend()
    plt.grid(False)
    plt.savefig("rectangular_color_map.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    surf = ax.plot_surface(x_grid, y_grid, T_global, cmap='coolwarm')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Temperature [K]")

    # Plot settings
    ax.set_title("3D Temperature Field")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Temperature [K]")
    ax.view_init(elev=28, azim=-72)  # Set view angle
    plt.savefig("rectangular_3D.jpg")
    plt.show()  

    # Plotting the solution along a specific line in the 2D domain
    y_index = 100
    temperature_line = T_global[y_index, :]  # Extract row corresponding to y

    # Plot temperature over x
    plt.plot(x_global, temperature_line)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title(f"Temperature at y={b/2} m")
    plt.grid()
    plt.savefig("rectangular_2D.jpg")
    plt.show()  


# Saving the solution to the file
def save_output_to_file(filename, global_K, F, T):
    with open(filename, 'w') as f:
        f.write(f"Element Type: {config['element_type']}\n")
        f.write(f"Physical Dimensions: {config['a']} x {config['b']}\n")
        f.write(f"Grid Division: {config["hx"]}\n" )
        f.write(f"Number of nodes (nx x ny, all): {config["nx"] + 1} x {config["ny"] + 1} , {(config["nx"] + 1) * (config["ny"] + 1)}\n")
        f.write(f"Material Properties (k1, k2): {config['k1']}, {config['k2']}\n\n") 

        for boundary, condition in boundary_conditions.items():
            f.write(f"Boundary: {boundary}, Type: {condition['type']}, Values: {condition['values']}")
            f.write("\n")

        f.write("\n")
        f.write("Global stiffness matrix K:\n")
        f.write(np.array2string(global_K, precision=6, separator=',', suppress_small=True))
        f.write("\n\n")

        f.write("Global force vector F:\n")
        f.write(np.array2string(F, precision=6, separator=',', suppress_small=True))
        f.write("\n\n")

        f.write("Solution vector T:\n")
        f.write(np.array2string(T, precision=6, separator=',', suppress_small=True))
        f.write("\n\n")
        print("Output saved to the files")


def write_results_to_file(nodes, temperatures):
    data = np.column_stack((nodes[:,0], nodes[:,1], temperatures))
    np.savetxt('temperature_data.txt', data, fmt='%.2f', delimiter='\t', header='x [m]\ty [m]\tT [K]')



if __name__ == "__main__":

    # Reading the configuration file
    config_path = "config.txt"
    config = read_config(config_path)

    print("Element Type:", config["element_type"])
    print("Physical Dimensions:", config["a"], config["b"])
    print("Grid Division (hx):", config["hx"])
    print("Number of nodes (nx, ny, all):", config["nx"] + 1, config["ny"] + 1, (config["nx"] + 1) * (config["ny"] + 1))
    print("Material properties (k1, k2):", config["k1"], config["k2"])    

    element_type = config["element_type"]
    a, b = config["a"], config["b"]
    ax = config["hx"]
    n_x, n_y = config["nx"], config["ny"]
    k1, k2 = config["k1"], config["k2"]
    boundary_conditions = config["boundary_conditions"]

    for boundary, condition in boundary_conditions.items():
        print(f"Boundary: {boundary}, Type: {condition['type']}, Values: {condition['values']}")

    # Running the programm with considerance of the element type
    if element_type == "triangular":
        nodes, elements = generate_triangular_mesh(a, b, n_x, n_y, ax)
        boundary = get_boundary_nodes(nodes)
        global_K, F, T = Triangular_finite_elements(nodes, elements, k1, k2, ax, boundary_conditions, boundary)
        Print_solution(global_K, F, T)
        Plot_triangular_2D(T, nodes, elements, a, b)

    if config["element_type"] == "rectangular":
        nodes, elements = generate_rectangular_nodes(a, ax, b, n_x, n_y)
        boundary = get_boundary_nodes(nodes)
        global_K, F, T = Rectangular_finite_elements(nodes, elements, k1, k2, ax, boundary_conditions, boundary)
        Print_solution(global_K, F, T)
        Plot_rectangular_2D(T, nodes, elements, a, b)

    # Saving the output
    save_output_to_file("output.txt", global_K, F, T)
    write_results_to_file(nodes, T)
