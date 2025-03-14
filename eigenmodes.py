import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import time

def create_laplacian_matrix(n, shape='square', L=1.0):
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
        
        # Find points inside the circle
        points = []
        indices = {}  # Maps (i,j) to the index in our matrix
        count = 0
        
        for i in range(1, grid_size-1):  # Skip boundary
            for j in range(1, grid_size-1):  # Skip boundary
                # Distance from center
                dx = (j - center) * h
                dy = (i - center) * h
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < radius:  # If inside the circle
                    points.append([j*h, i*h])
                    indices[(i, j)] = count
                    count += 1
        
        N = len(points)  # Number of points inside the circle
        M = lil_matrix((N, N))
        
        # Fill the matrix
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if (i, j) in indices:
                    idx = indices[(i, j)]
                    
                    # Start with the diagonal element
                    neighbors = 0
                    
                    # Check all four neighbors
                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if (ni, nj) in indices:  # If neighbor is inside circle
                            M[idx, indices[(ni, nj)]] = 1
                            neighbors += 1
                        # If neighbor is outside (boundary), it contributes 0
                    
                    # Set diagonal element based on number of neighbors
                    M[idx, idx] = -neighbors
    
    # Scale the matrix by 1/h^2
    M = M / (h * h)
    
    return M.tocsr(), points, h  # Convert to CSR format for efficient operations

def solve_eigenmodes(M, points, shape, L, n, num_modes=6):
    """
    Solve the eigenvalue problem and plot the eigenmodes.
    
    Parameters:
    M : sparse matrix
        The Laplacian matrix
    points : list
        List of points in the domain
    shape : str
        Shape of the membrane
    L : float
        Characteristic length
    n : int
        Number of grid points along one dimension
    num_modes : int
        Number of eigenmodes to compute
    """
    # Solve the eigenvalue problem using scipy.sparse.linalg.eigs
    # We want the smallest eigenvalues (largest negative values)
    eigenvalues, eigenvectors = eigs(M, k=num_modes, which='SM')
    
    # Sort the eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues.real[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The frequencies are related to the eigenvalues
    frequencies = np.sqrt(-eigenvalues)
    
    # Plot the eigenmodes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Prepare grid for plotting
    if shape == 'square':
        h = L / (n + 1)
        x_grid = np.linspace(h, L-h, n)
        y_grid = np.linspace(h, L-h, n)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((n, n))
        for mode in range(min(num_modes, len(axes))):
            for idx, (x, y) in enumerate(points):
                i = int(round(y/h)) - 1
                j = int(round(x/h)) - 1
                Z[i, j] = eigenvectors[idx, mode].real
            
            ax = axes[mode]
            im = ax.contourf(X, Y, Z, cmap='coolwarm')
            ax.set_title(f'Mode {mode+1}, Frequency: {frequencies[mode]:.4f}')
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax)
            
    elif shape == 'rectangle':
        h = L / (n + 1)
        nx = 2 * n
        ny = n
        x_grid = np.linspace(h, 2*L-h, nx)
        y_grid = np.linspace(h, L-h, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((ny, nx))
        
        for mode in range(min(num_modes, len(axes))):
            for idx, (x, y) in enumerate(points):
                i = int(round(y/h)) - 1
                j = int(round(x/h)) - 1
                Z[i, j] = eigenvectors[idx, mode].real
            
            ax = axes[mode]
            im = ax.contourf(X, Y, Z, cmap='coolwarm')
            ax.set_title(f'Mode {mode+1}, Frequency: {frequencies[mode]:.4f}')
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax)
            
    elif shape == 'circle':
        # For circle, we'll use a scatter plot since points are irregular
        for mode in range(min(num_modes, len(axes))):
            ax = axes[mode]
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            scatter = ax.scatter(x_vals, y_vals, c=eigenvectors[:, mode].real, 
                                cmap='coolwarm', s=30)
            ax.set_title(f'Mode {mode+1}, Frequency: {frequencies[mode]:.4f}')
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect('equal')
            fig.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return frequencies, eigenvectors

def compare_solvers(M, num_modes=6):
    """
    Compare the speed of different eigensolvers.
    """
    # Using eig()
    start_time = time.time()
    M_dense = M.toarray()
    eigenvalues_dense, eigenvectors_dense = np.linalg.eig(M_dense)
    eig_time = time.time() - start_time
    
    # Using eigh() (for symmetric matrices)
    start_time = time.time()
    eigenvalues_eigh, eigenvectors_eigh = np.linalg.eigh(M_dense)
    eigh_time = time.time() - start_time
    
    # Using sparse eigs()
    start_time = time.time()
    eigenvalues_sparse, eigenvectors_sparse = eigs(M, k=num_modes, which='SM')
    eigs_time = time.time() - start_time
    
    print(f"np.linalg.eig() time: {eig_time:.4f} seconds")
    print(f"np.linalg.eigh() time: {eigh_time:.4f} seconds")
    print(f"scipy.sparse.linalg.eigs() time: {eigs_time:.4f} seconds")
    
    return {
        'eig': eig_time,
        'eigh': eigh_time,
        'eigs': eigs_time
    }

# Study how eigenfrequencies depend on size L
def study_size_dependence(shape, n_values, L_values):
    """
    Study how eigenfrequencies depend on size L and discretization n.
    """
    num_modes = 5  # Number of lowest modes to track
    
    # Results for different L values (fixed n)
    fixed_n = n_values[-1]  # Use the largest n for accuracy
    L_results = []
    
    for L in L_values:
        M, points, h = create_laplacian_matrix(fixed_n, shape, L)
        eigenvalues, _ = eigs(M, k=num_modes, which='SM')
        eigenvalues = np.sort(eigenvalues.real)
        frequencies = np.sqrt(-eigenvalues)
        L_results.append(frequencies)
    
    # Results for different n values (fixed L)
    fixed_L = 1.0
    n_results = []
    
    for n in n_values:
        M, points, h = create_laplacian_matrix(n, shape, fixed_L)
        eigenvalues, _ = eigs(M, k=num_modes, which='SM')
        eigenvalues = np.sort(eigenvalues.real)
        frequencies = np.sqrt(-eigenvalues)
        n_results.append(frequencies)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot frequency vs L
    ax = axes[0]
    L_results = np.array(L_results)
    for i in range(num_modes):
        ax.plot(L_values, L_results[:, i], 'o-', label=f'Mode {i+1}')
    
    ax.set_xlabel('Size L')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Eigenfrequencies vs. Size L ({shape})')
    ax.legend()
    ax.grid(True)
    
    # Plot frequency vs n
    ax = axes[1]
    n_results = np.array(n_results)
    for i in range(num_modes):
        ax.plot(n_values, n_results[:, i], 'o-', label=f'Mode {i+1}')
    
    ax.set_xlabel('Grid size n')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Eigenfrequencies vs. Grid Size ({shape})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return L_results, n_results

from matplotlib.animation import FuncAnimation

def animate_eigenmode(shape, L, n, mode=0):
    """
    Create an animation of a specific eigenmode.
    
    Parameters:
    shape : str
        Shape of the membrane
    L : float
        Characteristic length
    n : int
        Number of grid points
    mode : int
        Index of the eigenmode to animate
    """
    # Create the Laplacian matrix
    M, points, h = create_laplacian_matrix(n, shape, L)
    
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigs(M, k=mode+1, which='SM')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Get the selected mode
    eigenvalue = eigenvalues[mode]
    eigenmode = eigenvectors[:, mode]
    
    # Calculate the frequency
    frequency = np.sqrt(-eigenvalue)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Prepare grid for plotting based on shape
    if shape == 'square':
        # For square
        x_grid = np.linspace(h, L-h, n)
        y_grid = np.linspace(h, L-h, n)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((n, n))
        
        for idx, (x, y) in enumerate(points):
            i = int(round(y/h)) - 1
            j = int(round(x/h)) - 1
            Z[i, j] = eigenmode[idx]
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, cmap='coolwarm', levels=20)
        fig.colorbar(contour, ax=ax)
        
    elif shape == 'rectangle':
        # For rectangle
        nx = 2 * n
        ny = n
        x_grid = np.linspace(h, 2*L-h, nx)
        y_grid = np.linspace(h, L-h, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((ny, nx))
        
        for idx, (x, y) in enumerate(points):
            i = int(round(y/h)) - 1
            j = int(round(x/h)) - 1
            Z[i, j] = eigenmode[idx]
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, cmap='coolwarm', levels=20)
        fig.colorbar(contour, ax=ax)
        
    elif shape == 'circle':
        # For circle, use a scatter plot
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        scatter = ax.scatter(x_vals, y_vals, c=eigenmode, cmap='coolwarm', s=30)
        fig.colorbar(scatter, ax=ax)
        
        # Create a circle boundary
        circle = plt.Circle((L/2, L/2), L/2, fill=False, color='black')
        ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.set_title(f'Mode {mode+1}, Frequency: {frequency:.4f}')
    
    # Function to update the plot for each animation frame
    def update(frame):
        time = frame / 20  # Scale to get smoother animation
        # u(x, y, t) = v(x, y) * cos(omega * t)
        amplitude = np.cos(frequency * time)
        
        if shape == 'square' or shape == 'rectangle':
            ax.clear()
            if shape == 'square':
                Z_t = Z * amplitude
                contour = ax.contourf(X, Y, Z_t, cmap='coolwarm', levels=20)
            else:
                Z_t = Z * amplitude
                contour = ax.contourf(X, Y, Z_t, cmap='coolwarm', levels=20)
            ax.set_aspect('equal')
            ax.set_title(f'Mode {mode+1}, Frequency: {frequency:.4f}, Time: {time:.2f}')
            
        elif shape == 'circle':
            ax.clear()
            # Create a circle boundary
            circle = plt.Circle((L/2, L/2), L/2, fill=False, color='black')
            ax.add_patch(circle)
            
            colors = eigenmode * amplitude
            scatter = ax.scatter(x_vals, y_vals, c=colors, cmap='coolwarm', s=30, vmin=-np.max(np.abs(eigenmode)), vmax=np.max(np.abs(eigenmode)))
            ax.set_aspect('equal')
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_title(f'Mode {mode+1}, Frequency: {frequency:.4f}, Time: {time:.2f}')
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, interval=50)
    plt.close()  # Prevent display of the static plot
    
    return ani

# Main execution
if __name__ == "__main__":
    # Parameters
    n = 20  # Number of grid points (try 20, 40, 60 for different resolutions)
    L = 1.0  # Characteristic length
    
    print("Part A: Matrix formulation")
    # For visualization purposes, use a small example
    small_n = 4
    M_small, points_small, h_small = create_laplacian_matrix(small_n, 'square', L)
    print("Laplacian matrix for a 4x4 grid:")
    print(M_small.toarray())
    
    print("\nPart B: Solving for different shapes")
    shapes = ['square', 'rectangle', 'circle']
    
    for shape in shapes:
        print(f"\nSolving for {shape} with L={L}")
        M, points, h = create_laplacian_matrix(n, shape, L)
        frequencies, eigenvectors = solve_eigenmodes(M, points, shape, L, n)
    
    print("\nPart C: Comparing solver speeds")
    # Using a smaller grid for dense methods to be feasible
    small_n = 15
    for shape in shapes:
        print(f"\nSolver comparison for {shape}:")
        M, _, _ = create_laplacian_matrix(small_n, shape, L)
        compare_solvers(M)
    
    print("\nPart D: Size dependence study")
    n_values = [10, 20, 30, 40]
    L_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for shape in shapes:
        print(f"\nSize dependence for {shape}:")
        L_results, n_results = study_size_dependence(shape, n_values, L_values)
    
    print("\nPart E: Animating eigenmodes")
    # Animate the first few eigenmodes for the square
    for mode in range(3):
        print(f"Animating mode {mode+1} for square")
        ani = animate_eigenmode('square', L, n, mode)
        ani.save(f'square_mode_{mode+1}.gif', writer='pillow', fps=20)