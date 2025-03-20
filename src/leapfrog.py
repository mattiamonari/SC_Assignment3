import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def leapfrog_step(x, v, t, m, f, dt):
    """
    Leapfrog algorithm step, Velocity Varlet form
    This method has the advantage of matching the velocity and space points in time
    instead of staggering them
    """
    # Half-step for velocity
    v_half = v + 0.5 * dt * f(t, x) / m
    
    # Full step for position
    x_new = x + v_half * dt
    
    # Half-step to complete velocity update
    v_new = v_half + 0.5 * dt * f(t + dt, x_new) / m
    return x_new, v_new

def simulate_leapfrog(x0, v0, m, f, dt, t_end):
    """Simulate using leapfrog method"""
    steps = int(t_end / dt) + 1
    t = np.linspace(0, t_end, steps)
    x = np.zeros(steps)
    v = np.zeros(steps)
    
    # Initial conditions
    x[0] = x0
    # Initialize velocity at t = -dt/2
    v[0] = v0
    
    # Simulate
    for i in range(1, steps):
        x[i], v[i] = leapfrog_step(x[i-1], v[i-1], t[i-1], m, f, dt)
    
    return t, x, v

def rk45_simulate(x0, v0, m, k, t_end, dt, forced=False):
    """Simulate using RK45 method"""
    # Define the ODE system
    def f(t, y):
        if forced:
            return [y[1], (-k * y[0] + np.sin(t))/m]
        # y[0] is position, y[1] is velocity
        return [y[1], -k/m * y[0]]
    
    # Solve using RK45
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(f, [0, t_end], [x0, v0], method='RK45', t_eval=t_eval)
    
    return sol.t, sol.y[0], sol.y[1]

def analytical_solution(t, x0, v0, k, m):
    """Analytical solution for harmonic oscillator"""
    omega = np.sqrt(k/m)
    x = x0 * np.cos(omega * t) + v0/omega * np.sin(omega * t)
    v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    return x, v

def analytical_solution_forced(t, x0, v0, k, m):
    """
    Analytical solution for forced harmonic oscillator with f = -k*x + sin(t)
    
    Parameters:
    t : array_like
        Time points to evaluate the solution at
    x0, v0 : float
        Initial position and velocity
    k, m : float
        Spring constant and mass
    
    Returns:
    x, v : tuple of arrays
        Position and velocity at each time point
    """
    omega = np.sqrt(k/m)
    
    # Check for resonance case (k = m)
    if abs(k - m) < 1e-10:  # Numerical comparison to avoid floating point issues
        x = x0 * np.cos(t) + v0 * np.sin(t) + (t * np.sin(t)) / (2 * m)
        v = -x0 * np.sin(t) + v0 * np.cos(t) + (np.cos(t) - t * np.sin(t)) / (2 * m)
    else:
        x = x0 * np.cos(omega * t) + (v0/omega) * np.sin(omega * t) + np.sin(t) / (k - m)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t) + np.cos(t) / (k - m)
    
    return x, v

def compute_energy(x, v, m, k):
    """Compute total energy: kinetic + potential"""
    kinetic = 0.5 * m * v**2
    potential = 0.5 * k * x**2
    return kinetic + potential

def create_force_function(k, forced):
    if forced:
        return lambda t, x: -k * x + np.sin(t)
    return lambda t, x: -k * x

def run_leapfrog_experiment(m, k_values, x0, v0, t_end, dt_values, forced=False, savefig=False, directory=None):
    plt.figure(figsize=(10, 6))
    # Plot position for different dt values
    for k_idx, k in enumerate(k_values):
        analytical_error = []
        plt.subplot(1, 2, 1)
        
        # Create a force function for this k value
        f = create_force_function(k, forced)
        
        for dt_idx, dt in enumerate(dt_values):
            t, x, v = simulate_leapfrog(x0, v0, m, f, dt, t_end)
            
            if dt_idx == len(dt_values) - 1:
                plt.plot(t, x, label=f'K = {k}')
            else:
                plt.plot(t, x)
            
            # Compute analytical solution
            t_analytical = np.linspace(0, t_end, int(t_end/dt + 1))
            if forced:
                x_analytical, v_analytical = analytical_solution_forced(t_analytical, x0, v0, k, m)
            else:
                x_analytical, v_analytical = analytical_solution(t_analytical, x0, v0, k, m)
            
            # t_rk, x_rk, v_rk = rk45_simulate(x0, v0, m, k, t_end, dt, forced)
            # if dt_idx == len(dt_values) - 1:
            #     plt.plot(t_rk, x_rk, label=f'K = {k}')
            # else:
            #     plt.plot(t_rk, x_rk)

            # Compute error
            error = np.abs(np.interp(t_analytical, t, x) - x_analytical)
            analytical_error.append(np.max(error))
        
        if k_idx == 0:
            plt.plot(t_analytical, x_analytical, 'k--', label='Analytical')
        else:
            plt.plot(t_analytical, x_analytical, 'k--')
        
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Position vs Time for Different dt Values')
        
        # Plot error convergence
        plt.subplot(1, 2, 2)
        plt.loglog(dt_values, analytical_error, 'o-', label=f'k={k}')
        plt.xlabel('dt')
        plt.ylabel('Maximum Error')
        plt.title('Error Convergence with dt')
        plt.grid(True)

    # Add a single legend for the first subplot at the end
    plt.subplot(1, 2, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')

    # Add a legend for the second subplot
    plt.subplot(1, 2, 2)
    plt.legend(loc='best', fontsize='small')

    plt.tight_layout()
    if savefig:
        plt.savefig(f"{directory}/leapfrog_experiment_forced={forced}.pdf")
    plt.show()

def compare_with_rk(m, k_values, x0, v0, t_end, dt, forced=False, savefig=False, directory=None):
    for k in k_values:
        plt.figure(figsize=(10, 6))
        # Compare energy conservation: Leapfrog vs RK45
        plt.subplot(1, 2, 1)
        
        # Create a force function for this k value
        f = create_force_function(k, forced)

        # Leapfrog
        t_lf, x_lf, v_lf = simulate_leapfrog(x0, v0, m, f, dt, t_end)
        energy_lf = compute_energy(x_lf, v_lf, m, k)

        # RK45
        t_rk, x_rk, v_rk = rk45_simulate(x0, v0, m, k, t_end, dt, forced)
        energy_rk = compute_energy(x_rk, v_rk, m, k)

        plt.plot(t_lf, energy_lf, label='Leapfrog')
        plt.plot(t_rk, energy_rk, label='RK45')
        plt.xlabel('Time')
        plt.ylabel('Total Energy')
        plt.title('Energy Conservation: Leapfrog vs RK45')
        plt.legend()

        # Plot trajectories in phase space
        plt.subplot(1, 2, 2)
        plt.plot(x_lf, v_lf, label='Leapfrog')
        plt.plot(x_rk, v_rk, label='RK45')
        
        # Compute analytical solution
        t_analytical = np.linspace(0, t_end, int(t_end/dt + 1))
        if forced:
            x_analytical, v_analytical = analytical_solution_forced(t_analytical, x0, v0, k, m)
        else:
            x_analytical, v_analytical = analytical_solution(t_analytical, x0, v0, k, m)

        plt.plot(x_analytical, v_analytical, 'k--', label='Analytical')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Phase Space Trajectory')
        plt.axis('equal')
        plt.legend()

        plt.tight_layout()
        if savefig:
            plt.savefig(f"{directory}/rk_comparison_forced={forced}.pdf")
        plt.show()

if __name__ == "__main__":
    """
    TODO: 
        In general: 
        1. Separate points (I) and (J) clearly
        2. Clean the code
        3. Check what happens with high/low dt
        IDEA: - show energy conservation and phase space plots together for different dt
                to show how dt affects energy conservation and solution accuracy
              - show error convergence for different k values
        
        I)
        1. Better show the accuracy of the method and how the initial velocity affects it

        J)
        1. Try different oscillation frequencies to show what happens when the oscillation matches the natural one
        2. Show phase space plot of (v, x) for various frequencies
    """ 
    # Parameters
    m = 1.0
    x0 = 1.0
    v0 = 0.0
    t_end = 5.0
    
    # Test different dt and k values
    k_values = [0.5, 2.0, 5.0]
    dt_values = [0.01]
    
    # For standard harmonic oscillator
    run_leapfrog_experiment(m, k_values, x0, v0, t_end, dt_values, forced=False)
    
    # For forced harmonic oscillator
    run_leapfrog_experiment(m, k_values, x0, v0, t_end, dt_values, forced=True)
    
    # Compare with RK45
    dt = 0.01
    t_end = 50
    compare_with_rk(m, k_values, x0, v0, t_end, dt, forced=False)

    # Test with external forcing
    compare_with_rk(m, k_values, x0, v0, t_end, dt, forced=True)
    