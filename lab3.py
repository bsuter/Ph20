import numpy as np
import matplotlib.pyplot as plt
import sys

x0 = 0
v0 = 5

def explicit(x0, v0, N, h):
    time = np.arange(0, N, h)
    distance = np.empty(len(time))
    velocity = np.empty(len(time))

    distance[0] = x0
    velocity[0] = v0

    for i in range(1, len(time)):
        distance[i] = dx_dt(distance[i - 1], velocity[i - 1], h)
        velocity[i] = dv_dt(distance[i - 1], velocity[i - 1], h)
    return distance, velocity, time

def EulerSpringExplicit(x0, v0, N, h, plot):
    distance, velocity, time = explicit(x0, v0, N, h)
    if plot == 'dist':
        plotPosition(distance, time)
    if plot == 'vel'
        plotVelocity(velocity, time)

def dv_dt(x, v, h):
    return v - h * x

def dx_dt(x, v, h):
    return x + h * v

def plotPosition(distance, time):
    plt.plot(time, distance, '-', label='Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Explicit Euler Method for Position')
    plt.show()

def plotVelocity(velocity, time):
    plt.plot(time, velocity, '-', label='Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Explicit Euler Method for Velocity')
    plt.show()

def globalError(x0, v0, N, h):
    distance, velocity, time = explicit(x0, v0, N, h)
    vel_error = np.empty(len(time))
    dist_error = np.empty(len(time))

    vel_error[0] = 0
    dist_error[0] = 0

    for i in range(1, len(time)):
        vel_error[i] = np.abs(v0 * np.cos(time[i]) - velocity[i])
        dist_error[i] = np.abs(v0 * np.sin(time[i]) - distance[i])

    plt.plot(time, vel_error, '-', label='Velocity')
    plt.plot(time, dist_error, '-', label='Position')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Explicit Euler Error for Velocity and Position')
    plt.legend()
    plt.show()


def truncation(x0, v0, N, h0):
    h = np.array([h0, h0/2, h0/4, h0/8, h0/16])
    max_error = np.empty(len(h))

    for n in range(len(h)):
        time = explicit(x0, v0, N, h[n])
        dist_error[0] = 0

        for i in range(1, len(time)):
            dist_error[i] = np.abs(v0 * np.sin(time[i]) - distance[i])

        max_error[n] = dist_error[-1]

    plt.plot(h, max_error)
    plt.xlabel('h in terms of h0 = ' + str(h0))
    plt.ylabel('Maximum Error of Position')
    plt.title('Explicit Euler Error Vs h')
    plt.show()


def energy(x0, v0, N, h):
        distance, velocity, time = explicit(x0, v0, N, h)
        energy = np.empty(len(time))

        energy[0] = v0**2 + x0**2

        for i in range(1, len(time)):
            energy[i] = velocity[i]**2 + distance[i]**2

        plt.plot(time, energy)
        plt.xlabel('Energy')
        plt.ylabel('Time')
        plt.title('Explicit Euler Energy')
        plt.show()

def xi1(xi, vi, h):
    return (vi / h) * (1 - 1 / (h**2 + 1)) + (xi / (h**2 + 1))

def vi1(xi, vi, h):
    return (vi - h * xi)/(h**2 + 1)

def implicit(x0, v0, N, h):
    time = np.arange(0, N, h)
    distance = np.empty(len(time))
    velocity = np.empty(len(time))

    distance[0] = x0
    velocity[0] = v0

    for i in range(1, len(time)):
        distance[i] = xi1(distance[i - 1], velocity[i - 1], h)
        velocity[i] = vi1(distance[i - 1], velocity[i - 1], h)
    return distance, velocity, time


def EulerSpringImplicit(x0, v0, N, h, plot):
    distance, velocity, time = explicit(x0, v0, N, h)
    energy = np.empty(len(time))
    dist_error = np.empty(len(time))

    energy[0] = v0**2 + x0**2
    dist_error[0] = 0

    for i in range(1, len(time)):
        energy[i] = velocity[i]**2 + distance[i]**2
        dist_error[i] = np.abs(v0 * np.sin(time[i]) - distance[i])

    distance_i, velocity_i, time = implicit(x0, v0, N, h)
    energy_i = np.empty(len(time))
    dist_error_i = np.empty(len(time))

    energy_i[0] = v0**2 + x0**2
    dist_error_i[0] = 0

    for i in range(1, len(time)):
        energy_i[i] = velocity_i[i]**2 + distance_i[i]**2
        dist_error_i[i] = np.abs(v0 * np.sin(time[i]) - distance_i[i])

    if plot == 'error':
        plt.plot(time, dist_error, label='Explicit')
        plt.plot(time, dist_error_i, label='Implicit')
        plt.xlabel('Time')
        plt.ylabel('Amount of Error')
        plt.title('Global Error Comparison')
        plt.show()

    if plot == 'energy':
        plt.plot(time, energy, label='Explicit')
        plt.plot(time, energy_i, label='Implicit')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Comparison')
        plt.show()



def phaseSpace(x0, v0, N, h , plot):
    distance, velocity, time = explicit(x0, v0, N, h)
    distance_i, velocity_i, time = implicit(x0, v0, N, h)
    trueVel = np.empty(len(time))
    trueDist = np.empty(len(time))

    trueVel[0] = v0
    trueDist[0] = x0

    for i in range(1, len(time)):
        trueVel[i] = v0 * np.cos(time[i])
        trueDist[i] = v0 * np.sin(time[i])

    if plot == 'dist':
        plt.plot(distance, velocity, label='Explict')
        plt.plot(trueDist, trueVel, label='True')
        plt.xlabel('Distance')
        plt.ylabel('Velocity')
        plt.title('Explicit Euler Method - Phase Space')
        plt.legend()
        plt.show()

    if plot == 'vel':
        plt.plot(distance_i, velocity_i, label='Implicit')
        plt.plot(trueDist, trueVel, label='True')
        plt.xlabel('Distance')
        plt.ylabel('Velocity')
        plt.title('Implicit Euler Method - Phase Space')
        plt.legend()
        plt.show()

def symp_x(xi, vi, h):
    return xi + h * vi

def symp_v(xi, vi, h):
    return vi * (1 - h**2) - h * xi


def symplectic(x0, v0, N, h):
    time = np.arange(0, N, h)
    distance = np.empty(len(time))
    velocity = np.empty(len(time))

    distance_s[0] = x0
    velocity_s[0] = v0

    for i in range(1, len(time)):
        distance[i] = symp_x(distance[i - 1], velocity[i - 1], h)
        velocity[i] = symp_v(distance[i - 1], velocity[i - 1], h)

def EulerSpringSymplectic(x0, v0, N, h):
    distance, velocity, time = symplectic(x0, v0, N, h)

    plt.plot(distance, velocity)
    plt.xlabel('Distance')
    plt.ylabel('Velocity')
    plt.title('Symplectic Euler Method - Phase Space')
    plt.show()

def sympEnergy(x0, v0, N, h):
    distance_s, velocity_s, time = symplectic(x0, v0, N, h)
    energy_s = np.empty(len(time))

    energy_s[0] = v0**2 + x0**2

    for i in range(1, len(time)):
        energy_s[i] = velocity_s[i]**2 + distance_s[i]**2

    plt.plot(time, energy_s)
    plt.xlabel('Energy')
    plt.ylabel('Time')
    plt.title('Symplectic Euler Energy')
    plt.show()

def main(sys.argv):
    x0 = 0
    v0 = 5
    N = 10
    h = 0.01
    if sys.argv[0] == 'explicit':
        EulerSpringExplicit(x0, v0, N, h, sys.argv[1])
    elif sys.argv[0] == 'error':
        globalError(x0, v0, N, h)
    elif sys.argv[0] == 'truncation':
        truncation(x0, v0, N, h0)
    elif sys.argv[0] == 'energy':
        energy(x0, v0, N, h)
    elif sys.argv[0] == 'implicit':
        EulerSpringImplicit(x0, v0, N, h, sys.argv[1])
    elif sys.argv[0] == 'phaseSpace':
        phaseSpace(x0, v0, N, h , sys.argv[1])
    elif sys.argv[0] == 'symplectic':
        EulerSpringSymplectic(x0, v0, N, h)
    else sys.argv[0] == 'sympEnergy':
        sympEnergy(x0, v0, N, h)
        
if __name__ == '__main__':
    main()




# ksdk
