import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def leapfrog_step(x, v, t, m, f, dt):
    """Leapfrog algorithm step"""
    # Full step in space using current velocity
    x_new = x + v * dt
    # Full step for velocity using new position
    v_new = v + f(t, x_new) * dt / m
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
    v[0] = v0 - f(0, x0) * 0.5 * dt / m
    
    # Simulate
    for i in range(1, steps):
        x[i], v[i] = leapfrog_step(x[i-1], v[i-1], t[i-1], m, f, dt)
    
    return t, x, v

def rk45_simulate(x0, v0, m, k, t_end, dt):
    """Simulate using RK45 method"""
    # Define the ODE system
    def f(t, y):
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

def compute_energy(x, v, m, k):
    """Compute total energy: kinetic + potential"""
    kinetic = 0.5 * m * v**2
    potential = 0.5 * k * x**2
    return kinetic + potential

if __name__ == "__main__":
    """
    TODO: 
        In general: 
        1. Separate points (I) and (J) clearly
        2. Clean the code
        
        I)
        1. Better show the accuracy of the method and how the initial velocity affects it
        2. Better comparison with RK45: show energy loss over time of the latter for various dt
        3. Try different k values and show results
        4. Fix analytical solution in phase space plot

        J)
        1. Try different oscillation frequencies to show what happens when the oscillation matches the natural one
        2. Show phase space plot of (v, x) for various frequencies
    """ 
    # Parameters
    m = 1.0
    k = 1  # f(x) = -kx
    x0 = 1.0
    v0 = 0.0
    t_end = 50.0  # Longer simulation to see energy conservation
    f = lambda t, x:np.sin(0.5*t) - k * x

    # Test different dt values
    dt_values = [1, 0.1, 0.05, 0.01]
    analytical_error = []

    plt.figure(figsize=(15, 10))

    # Plot position for different dt values
    plt.subplot(2, 2, 1)
    for dt in dt_values:
        t, x, v = simulate_leapfrog(x0, v0, m, f, dt, t_end)
        plt.plot(t, x, label=f'dt = {dt}')
        
        # Compute analytical solution
        t_analytical = np.linspace(0, t_end, int(t_end/dt + 1))  
        x_analytical, _ = analytical_solution(t_analytical, x0, v0, k, m)
        
        # Compute error (interpolate to match time points)
        error = np.abs(np.interp(t_analytical, t, x) - x_analytical)
        analytical_error.append(np.max(error))

    # RK45
    t_rk, x_rk, v_rk = rk45_simulate(x0, v0, m, k, t_end, dt)
    plt.plot(t_rk, x_rk, label='RK45')

    plt.plot(t_analytical, x_analytical, 'k--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time for Different dt Values')
    plt.legend()

    # Plot error convergence
    plt.subplot(2, 2, 2)
    plt.loglog(dt_values, analytical_error, 'o-')
    plt.xlabel('dt')
    plt.ylabel('Maximum Error')
    plt.title('Error Convergence with dt')
    plt.grid(True)

    # Compare energy conservation: Leapfrog vs RK45
    plt.subplot(2, 2, 3)
    t_end = 100
    dt = 0.0001  # Use a relatively large dt to show differences
    f = lambda t, x:  - k * x
    # Leapfrog
    t_lf, x_lf, v_lf = simulate_leapfrog(x0, v0, m, f, dt, t_end)
    # Adjust v_lf to be at the same time as x_lf (instead of half-step behind)
    v_lf_adjusted = v_lf + 0.5 * dt * (-k * x_lf) / m
    energy_lf = compute_energy(x_lf, v_lf_adjusted, m, k)

    # RK45
    t_rk, x_rk, v_rk = rk45_simulate(x0, v0, m, k, t_end, dt)
    energy_rk = compute_energy(x_rk, v_rk, m, k)

    plt.plot(t_lf, energy_lf, label='Leapfrog')
    plt.plot(t_rk, energy_rk, label='RK45')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Energy Conservation: Leapfrog vs RK45')
    plt.legend()

    # Plot trajectories in phase space
    plt.subplot(2, 2, 4)
    plt.plot(x_lf, v_lf, label='Leapfrog')
    plt.plot(x_rk, v_rk, label='RK45')
    # Analytical solution for phase space is a circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.sqrt(x0**2)  # Radius from initial conditions
    plt.plot(r*np.cos(theta), r*np.sin(theta), 'k--', label='Analytical')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Space Trajectory')
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()