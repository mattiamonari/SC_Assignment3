import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

__all__ = ['solve_steady_state_diffusion', 'plot_solution']

def solve_steady_state_diffusion(radius=2.0, Nx=50, Ny=50, source_pos=(0.6, 1.2)):
    """
    Solves the steady-state diffusion equation in a circular domain using finite differences.
    
    Parameters:
    radius (float): Radius of the circular domain.
    Nx, Ny (int): Grid resolution.
    source_pos (tuple): Coordinates of the source (x, y).
    
    Returns:
    X, Y, C (ndarray): Grid coordinates and computed concentration field.
    """
    x = np.linspace(-radius, radius, Nx)
    y = np.linspace(-radius, radius, Ny)
    dx = x[1] - x[0]  # Grid spacing
    
    # Create grid
    X, Y = np.meshgrid(x, y)
    inside_circle = X**2 + Y**2 <= radius**2  # Boolean mask for inside points
    num_points = np.count_nonzero(inside_circle)
    
    # Mapping 2D (i,j) to 1D index
    index_map = -np.ones((Nx, Ny), dtype=int)  # Initialize with -1 to mark outside points
    index_map[inside_circle] = np.arange(num_points)  # Assign indices only to valid grid points
    
    # Construct sparse matrix M and vector b
    M = sp.lil_matrix((num_points, num_points))
    b = np.zeros(num_points)
    
    # Fill M using the finite difference scheme
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if inside_circle[i, j]:
                k = index_map[i, j]
                neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                
                M[k, k] = 4 / dx**2  # Corrected sign to ensure positive values
                for ni, nj in neighbors:
                    if inside_circle[ni, nj]:
                        M[k, index_map[ni, nj]] = -1 / dx**2  # Adjusted sign for consistency
                
                # Set source term at given position with stricter check
                if np.isclose(X[i, j], source_pos[0], atol=dx/2) and np.isclose(Y[i, j], source_pos[1], atol=dx/2):
                    b[k] = 1
    
    # Ensure all matrix entries are correctly populated before converting
    assert M.shape == (num_points, num_points), "Matrix M dimensions are incorrect."
    assert b.shape == (num_points,), "Vector b dimensions are incorrect."
    
    # Convert to CSR format for efficiency
    M = M.tocsr()
    
    # Solve Mc = b using sparse direct solver
    c = spla.spsolve(M, b)
    
    # Reconstruct full domain solution
    C = np.zeros((Nx, Ny))
    C[inside_circle] = c
    
    return X, Y, C

def plot_solution(X, Y, C, radius):
    """
    Plots the steady-state diffusion solution.
    
    Parameters:
    X, Y (ndarray): Grid coordinates.
    C (ndarray): Computed concentration field.
    radius (float): Radius of the circular domain.
    """
    plt.figure(figsize=(6,6))
    plt.contourf(X, Y, C, levels=50, cmap='viridis')
    plt.colorbar(label='Steady-state Concentration')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Steady-state Diffusion Solution")
    
    # Overlay boundary for verification
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    plt.plot(circle_x, circle_y, 'r--', label='Domain Boundary')
    plt.legend()
    plt.show()
