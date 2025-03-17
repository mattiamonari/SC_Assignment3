import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# FPS * TIME (s)
ANIMATION_FRAMES = 30 * 5

def prepare_eigenmode_grid(shape, points, eigenvector, h, L, n):
    """
    Prepare grid data for visualizing an eigenmode.
    
    Parameters:
    shape : str
        Shape of the membrane ('square', 'rectangle', or 'circle')
    points : list
        List of points in the domain
    eigenvector : numpy array
        The eigenmode to visualize
    h : float
        Grid spacing
    L : float
        Characteristic length
    n : int
        Number of grid points along one dimension
    
    Returns:
    X, Y : numpy arrays
        Meshgrid coordinates
    Z : numpy array
        Values of eigenmode at grid points
    mask : numpy array or None
        Mask for points outside the domain (only for circle)
    """
    if shape == 'square':
        x_grid = np.linspace(h, L-h, n)
        y_grid = np.linspace(h, L-h, n)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((n, n))
        
        for idx, (x, y) in enumerate(points):
            i = int(round(y/h)) - 1
            j = int(round(x/h)) - 1
            Z[i, j] = eigenvector[idx]
        
        mask = None
            
    elif shape == 'rectangle':
        nx = 2 * n
        ny = n
        x_grid = np.linspace(h, 2*L-h, nx)
        y_grid = np.linspace(h, L-h, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((ny, nx))
        
        for idx, (x, y) in enumerate(points):
            i = int(round(y/h)) - 1
            j = int(round(x/h)) - 1
            Z[i, j] = eigenvector[idx]
        
        mask = None
            
    elif shape == 'circle':
        # For circle, we create a square grid and mask points outside the circle
        grid_size = n + 2  # Include boundary
        h_grid = L / (grid_size - 1)
        x_grid = np.linspace(0, L, grid_size)
        y_grid = np.linspace(0, L, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros((grid_size, grid_size))
        
        # Create a mask for points outside the circle
        center = L / 2
        radius = L / 2
        mask = np.sqrt((X - center)**2 + (Y - center)**2) > radius
        
        # Map eigenmode values to grid points
        for idx, (x, y) in enumerate(points):
            i = int(round(y/h_grid))
            j = int(round(x/h_grid))
            Z[i, j] = eigenvector[idx]
    
    return X, Y, Z, mask

def visualize_eigenmodes(frequencies, eigenvectors, points, shape, L, n, h, animate=False, mode_idx=0, saveFig=False, directory=None):
    num_modes = len(frequencies)
    
    if animate:
        # Animation of a single mode in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        frequency = frequencies[mode_idx]
        eigenmode = eigenvectors[:, mode_idx].real
        
        # Normalize eigenmode for better visualization
        eigenmode = eigenmode / np.max(np.abs(eigenmode))
        
        X, Y, Z, mask = prepare_eigenmode_grid(shape, points, eigenmode, h, L, n)
        
        # Initial surface plot
        if shape in ['square', 'rectangle']:
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        else:  # circle
            # Mask points outside the circle
            Z_masked = np.ma.array(Z, mask=mask)
            surf = ax.plot_surface(X, Y, Z_masked, cmap='coolwarm', edgecolor='none')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set plot limits and labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Amplitude')
        ax.set_title(f'Mode {mode_idx+1}, Frequency: {frequency:.4f}')
        
        # Function to update the plot for each animation frame
        def update(frame):
            ax.clear()
            time = frame / 20  # Scale for smoother animation
            amplitude = np.cos(frequency * time)
            
            if shape in ['square', 'rectangle']:
                Z_t = Z * amplitude
                surf = ax.plot_surface(X, Y, Z_t, cmap='coolwarm', edgecolor='none')
            else:  # circle
                Z_t = Z * amplitude
                Z_masked = np.ma.array(Z_t, mask=mask)
                surf = ax.plot_surface(X, Y, Z_masked, cmap='coolwarm', edgecolor='none')
            
            # Set consistent view limits
            max_val = np.max(np.abs(Z))
            ax.set_zlim(-max_val, max_val)
            
            # Add labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Amplitude')
            ax.set_title(f'Mode {mode_idx+1}, Frequency: {frequency:.4f}, Time: {time:.2f}')
            
            # Maintain a consistent view angle
            ax.view_init(elev=30, azim=frame % 360)
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, interval=50)
        plt.close()  # Prevent display of static plot
        return ani
        
    else:
        # Static 3D plots for multiple modes
        num_rows = 2
        num_cols = int(np.ceil(num_modes / 2))
        fig = plt.figure(figsize=(14.25, 7.5))
        
        for mode in range(min(num_modes, num_rows * num_cols)):
            eigenmode = eigenvectors[:, mode].real
            frequency = frequencies[mode]
            
            # Normalize eigenmode for better visualization
            eigenmode = eigenmode / np.max(np.abs(eigenmode))
            
            X, Y, Z, mask = prepare_eigenmode_grid(shape, points, eigenmode, h, L, n)
            
            # Create 3D subplot
            ax = fig.add_subplot(num_rows, num_cols, mode + 1, projection='3d')
            
            if shape in ['square', 'rectangle']:
                surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            else:  # circle
                # Mask points outside the circle
                Z_masked = np.ma.array(Z, mask=mask)
                surf = ax.plot_surface(X, Y, Z_masked, cmap='coolwarm', edgecolor='none')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Set consistent view limits
            max_val = np.max(np.abs(Z))
            ax.set_zlim(-max_val, max_val)
            
            # Add labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Amplitude')
            ax.set_title(f'Mode {mode+1}, Frequency: {frequency:.4f}')
            
            # Set a nice viewing angle
            ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        if saveFig and directory is not None:
            plt.savefig(directory + f'multiple_modes_{shape}.pdf')
            plt.close()
        else:
            plt.show()
        return None
    
def plot_solvers_timings(saveFig=False, directory=None):
    # Read the file and plot a candle chart with confidence intervals for each method
    with open(directory + 'solvers.txt', 'r') as f:
        lines = f.readlines()

    times = {}
    for line in lines:
        if 'Matrix size' in line:
            continue
        method, timing = line.strip().split(': ')
        if method not in times:
            times[method] = []
        times[method].append(float(timing))

    # Convert to NumPy arrays
    methods = list(times.keys())
    mean_times = np.array([np.mean(times[method]) for method in methods])
    std_times = np.array([np.std(times[method]) for method in methods])

    # Sort by mean time for better readability
    sorted_indices = np.argsort(mean_times)
    methods = [methods[i] for i in sorted_indices]
    mean_times = mean_times[sorted_indices]
    std_times = std_times[sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))  # Color gradient

    bars = ax.bar(methods, mean_times, yerr=std_times, capsize=5, color=colors, alpha=0.85)

    # Annotate bars with mean times
    for bar, mean in zip(bars, mean_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.2f}s",
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Comparison of Eigenvalue Solvers')
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    if saveFig and directory is not None:
        plt.savefig(directory + 'solvers.pdf')
        plt.close()
    else:
        plt.show()  

