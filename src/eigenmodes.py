import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
import os 
import time
from src.plots import visualize_eigenmodes    
import tqdm

__all__ = ['create_system_matrix', 'eigenmode_analysis', 'compare_solvers', 'study_size_dependence']

def create_system_matrix(n, shape='square', L=1.0):
    """
    Create the discretized Laplacian matrix for different shapes.
    
    Parameters:
    n : int
        Number of internal grid points along one dimension
    shape : str
        Shape of the membrane ('square', 'rectangle', or 'circle')
    L : float
        Characteristic length (side length for square, side length for rectangle, diameter for circle)
    
    Returns:
    M : sparse matrix
        The Laplacian matrix
    points : list
        List of points in the domain
    h : float
        Grid spacing
    """
    if shape == 'square':
        # For a square with side length L
        h = L / (n + 1)  # Grid spacing
        N = n * n  # Total number of interior points
        
        # Create matrix in LIL format (efficient for constructing sparse matrices)
        M = lil_matrix((N, N))
        
        # Store coordinates of grid points
        points = []
        
        
        # Fill the matrix
        for i in range(n):
            for j in range(n):
                idx = i * n + j  # 1D index for the current point
                
                points.append([(j+1)*h, (i+1)*h])  # Store point coordinates
                
                # Diagonal element (center point)
                M[idx, idx] = -4
                
                # Connect to adjacent points if they're within the grid
                if i > 0:  # Connect to point below
                    M[idx, (i-1)*n + j] = 1
                if i < n-1:  # Connect to point above
                    M[idx, (i+1)*n + j] = 1
                if j > 0:  # Connect to point to the left
                    M[idx, i*n + (j-1)] = 1
                if j < n-1:  # Connect to point to the right
                    M[idx, i*n + (j+1)] = 1
        
    elif shape == 'rectangle':
        # For a rectangle with sides L and 2L
        h = L / (n + 1)  # Grid spacing for shorter side
        nx = 2 * n  # Number of points along the longer side (2L)
        ny = n     # Number of points along the shorter side (L)
        N = nx * ny  # Total number of interior points
        
        # Create matrix in LIL format
        M = lil_matrix((N, N))
        
        # Store coordinates of grid points
        points = []
        
        # Fill the matrix
        for i in range(ny):
            for j in range(nx):
                idx = i * nx + j  # 1D index for the current point
                
                points.append([(j+1)*h, (i+1)*h])  # Store point coordinates
                
                # Diagonal element (center point)
                M[idx, idx] = -4
                
                # Connect to adjacent points if they're within the grid
                if i > 0:  # Connect to point below
                    M[idx, (i-1)*nx + j] = 1
                if i < ny-1:  # Connect to point above
                    M[idx, (i+1)*nx + j] = 1
                if j > 0:  # Connect to point to the left
                    M[idx, i*nx + (j-1)] = 1
                if j < nx-1:  # Connect to point to the right
                    M[idx, i*nx + (j+1)] = 1
        
    elif shape == 'circle':
        # For a circle with diameter L
        radius = L / 2
        h = L / (n + 1)  # Grid spacing
        
        # We'll use a square grid and mark points inside the circle
        grid_size = n + 2  # Include boundary
        center = (grid_size - 1) / 2  # Center of the circle
        
        # Find points inside the circle and identify boundary points
        points = []
        indices = {}  # Maps (i,j) to the index in our matrix
        boundary_points = set()  # Set to store boundary points
        count = 0
        
        # First, identify interior and boundary points
        for i in range(grid_size):
            for j in range(grid_size):
                # Distance from center
                dx = (j - center) * h
                dy = (i - center) * h
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < radius - 0.5*h:  # If strictly inside the circle (interior)
                    points.append([j*h, i*h])
                    indices[(i, j)] = count
                    count += 1
                elif dist < radius + 0.5*h:  # If near the boundary
                    boundary_points.add((i, j))
        
        N = len(points)  # Number of points inside the circle
        M = lil_matrix((N, N))
        
        # Fill the matrix - only for interior points
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in indices:
                    idx = indices[(i, j)]
                    
                    # Start with the diagonal element
                    neighbors = 0
                    boundary_neighbors = 0
                    
                    # Check all four neighbors
                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if (ni, nj) in indices:  # If neighbor is interior point
                            M[idx, indices[(ni, nj)]] = 1
                            neighbors += 1
                        elif (ni, nj) in boundary_points:  # If neighbor is on boundary
                            # For fixed boundary conditions, boundary value is 0
                            # So we don't add any term to the RHS, just count it
                            boundary_neighbors += 1
                    
                    # Set diagonal element based on number of neighbors
                    # Each interior neighbor contributes 1, each boundary neighbor contributes 0 (fixed BC)
                    M[idx, idx] = -(neighbors + boundary_neighbors)

                    
    
    # Scale the matrix by 1/h^2
    M = M / (h * h)
    
    return M.tocsr(), points, h  # Convert to CSR format for efficient operations


def compare_solvers(M, num_modes=6, directory=None):
    """
    Compare the speed of different eigensolvers.
    """
    # Using eig()
    start_time = time.time()
    M_dense = M.toarray()
    _, _ = np.linalg.eig(M_dense)
    eig_time = time.time() - start_time
    
    # Using eigh() (for symmetric matrices)
    start_time = time.time()
    _, _ = np.linalg.eigh(M_dense)
    eigh_time = time.time() - start_time
    
    # Using sparse eigs()
    start_time = time.time()
    _, _ = eigs(M, k=num_modes, which='SM')
    eigs_time = time.time() - start_time

    if not os.path.exists(directory + 'solvers.txt'):
        open(directory + 'solvers.txt', 'w').close()
    
    with open(directory + 'solvers.txt', 'a') as f:
        f.write(f'Matrix size: {M.shape}\n')
        f.write(f'np.linalg.eig(): {eig_time}\n')
        f.write(f'np.linalg.eigh(): {eigh_time}\n')
        f.write(f'scipy.sparse.linalg.eigs(): {eigs_time}\n')

    print(f'np.linalg.eig(): {eig_time}')
    print(f'np.linalg.eigh(): {eigh_time}')
    print(f'scipy.sparse.linalg.eigs(): {eigs_time}')

    
def study_size_dependence(shape, n_values, L_values, num_modes=5, saveFig=False, directory=None):
    """
    Study how eigenfrequencies depend on size L and discretization n.
    """ 

    # Results for different L values (fixed n)
    fixed_n = n_values[-1]  # Use the largest n for accuracy
    L_results = []
    
    for L in tqdm.tqdm(L_values, desc="Running shape size dependence"):
        M, _, _ = create_system_matrix(fixed_n, shape, L)
        eigenvalues, _ = eigs(M, k=num_modes, which='SM')
        eigenvalues = np.sort(eigenvalues.real)
        frequencies = np.sqrt(-eigenvalues)
        L_results.append(frequencies)
    
    # Results for different n values (fixed L)
    fixed_L = 1.0
    n_results = []
        
    # Plot results
    fig = plt.figure(figsize=(12, 6))
    
    L_results = np.array(L_results)
    for i in range(num_modes):
        plt.plot(L_values, L_results[:, i], 'o-', label=f'Mode {i+1}')
    
    plt.xlabel('Size L')
    plt.ylabel('Frequency')
    plt.title(f'Eigenfrequencies vs. Size L ({shape})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if saveFig and directory is not None:
        plt.savefig(directory + f'{shape}_size_dependence.pdf')
        plt.close()
    plt.show()
    
    return L_results, n_results

def compute_eigenmodes(M, num_modes=6):
    eigenvalues, eigenvectors = eigs(M, k=num_modes, which='SM')
    
    # Sort the eigenvalues and eigenvectors based on who is closer to zero
    idxs = np.argsort(np.abs(eigenvalues.real))
    eigenvalues = eigenvalues.real[idxs]
    eigenvectors = eigenvectors[:, idxs]
   
    frequencies = np.sqrt(-eigenvalues)
    
    return frequencies, eigenvectors

def eigenmode_analysis(M, points, shape, L, n, h, num_modes=6, mode_to_animate=None, saveFig=False, directory=None):
    frequencies, eigenvectors = compute_eigenmodes(M, num_modes)

    if mode_to_animate is not None:
        ani = visualize_eigenmodes(
            frequencies, eigenvectors, points, shape, L, n, h, 
            animate=True, mode_idx=mode_to_animate, saveFig=saveFig, directory=directory
        )
        return frequencies, eigenvectors, ani
    else:
        visualize_eigenmodes(
            frequencies, eigenvectors, points, shape, L, n, h, 
            animate=False, saveFig=saveFig, directory=directory
        )
        return frequencies, eigenvectors, None

# Main execution
if __name__ == "__main__":
    # Parameters
    n = 100  # Number of grid points (try 20, 40, 60 for different resolutions)
    L = 1.0  # Characteristic length
    saveFig = True
    shapes = ['square', 'rectangle', 'circle']


    