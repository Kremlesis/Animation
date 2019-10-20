import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
right_side_length = 5
down_side_length = 5

n = right_side_length*down_side_length
# Number of particles
m = np.ones(n)  # Particle masses
x = np.zeros((n, 2))  # Particle positions (x and y for ith particle in x[i,0], x[i,1])
v = np.zeros((n, 2))  # Particle velocities
f = np.zeros((n, 2))  # Force accumulator
dt = 0.02  # Time step
x_limit = 10  # limit of x axis
y_limit = 10  # limit of y axis
g = np.array([0, -9.8])  # Acceleration due to gravity

# Initialize
kd = 200
spring_length = 1
y_pos = down_side_length+1
x_pos = 1
for i in range(n):
    rem = i%right_side_length
    if rem ==0:
        y_pos-=1
        x_pos=1
    x[i,:] = np.array([x_pos,y_pos])
    x_pos+=1

# x[0, :] = np.array([1, 2])
# x[1, :] = np.array([2, 2])
# x[2, :] = np.array([1, 1])
# x[3, :] = np.array([2, 1])
for i in range(n):
    v[i, :] = np.array([0,0])
    f[i, :] = np.array([0, 0])


def bound_check(index, pos_array, vel_array):
    x = pos_array[index][0]
    y = pos_array[index][1]
    if x < 0.5:
        x = 0.5
        vel_array[index][0] *= -1
    if y < 0.5:
        y = 0.5
        vel_array[index][1] *= -1
    if x > x_limit - 0.5:
        x = x_limit - 0.5
        vel_array[index][0] *= -1
    if y > y_limit - 0.5:
        y = y_limit - 0.5
        vel_array[index][1] *= -1
    pos_array[index][0] = x
    pos_array[index][1] = y


def change_force(index1, index2, pos_array, force_array, spring_constant, rest_length):
    bw_len = ((pos_array[index1][0] - pos_array[index2][0]) ** 2) + ((pos_array[index1][1] - pos_array[index2][1]) ** 2)
    bw_len = bw_len ** (1 / 2)
    dum1 = spring_constant * (bw_len / rest_length - 1) * ((pos_array[index2][0] - pos_array[index1][0]) / bw_len)
    dum2 = spring_constant * (bw_len / rest_length - 1) * ((pos_array[index2][1] - pos_array[index1][1]) / bw_len)
    force_array[index1][0] += dum1
    force_array[index1][1] += dum2

#def check_boundary(index,right_side_length,down_side_length):


def spring_force(index1, pos_array, force_array, spring_constant, rest_length):
    index2_1 = index1 - right_side_length
    index2_2 = index1 + 1
    index2_3 = index1 + right_side_length
    index2_4 = index1 - 1
    if (index2_1 >= 0):
        change_force(index1, index2_1, pos_array, force_array, spring_constant, rest_length)
    if (index1%right_side_length +1 < right_side_length):
        change_force(index1, index2_2, pos_array, force_array, spring_constant, rest_length)
    if (index2_3 < down_side_length*right_side_length):
        change_force(index1, index2_3, pos_array, force_array, spring_constant, rest_length)
    if (index1%right_side_length-1 >= 0):
        change_force(index1, index2_4, pos_array, force_array, spring_constant, rest_length)
    index2_5 = index1 - right_side_length + 1
    index2_6 = index1 + right_side_length + 1
    index2_7 = index1 + right_side_length - 1
    index2_8 = index1 - right_side_length - 1

    if (index1//right_side_length > 0 and index1%right_side_length +1 < right_side_length):
        change_force(index1,index2_5,pos_array,force_array,spring_constant,rest_length)
    if (index1//right_side_length < down_side_length-1 and index1%right_side_length +1 < right_side_length):
        change_force(index1,index2_6,pos_array,force_array,spring_constant,rest_length)
    if (index1//right_side_length < down_side_length-1 and index1%right_side_length-1 >= 0):
        change_force(index1,index2_7,pos_array,force_array,spring_constant,rest_length)
    if (index1//right_side_length >0 and index1%right_side_length-1 >= 0):
        change_force(index1,index2_8,pos_array,force_array,spring_constant,rest_length)


# Time stepping (this is actually "semi-implicit Euler")
def step():
    # Accumulate forces on each particle
    f.fill(0)
    for i in range(n):
        spring_force(i, x, f, kd, spring_length)
        f[i,:] += m[i]*g
    # Update velocity of each particle
    for i in range(n):
        v[i, :] += f[i, :] / m[i] * dt
    # Update position of each particle
    for i in range(n):
        x[i, :] += v[i, :] * dt
        bound_check(i, x, v)


# Drawing code
fig, ax = plt.subplots()
points, = ax.plot(x[:, 0], x[:, 1], 'o')


def init():
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_aspect('equal')
    return points,


def animate(frame):
    step()
    points.set_data(x[:, 0], x[:, 1])
    if frame is frames - 1:
        plt.close()
    return points,


totalTime = 10
frames = int(totalTime / dt)
anim = FuncAnimation(fig, animate, frames=range(frames), init_func=init, interval=dt * 1000)
plt.show()

"""
mport matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

n = 4      # Number of particles
m = np.ones(n)      # Particle masses
x = np.zeros((n,2))  # Particle positions (x and y for ith particle in x[i,0], x[i,1])
v = np.zeros((n,2))  # Particle velocities
f = np.zeros((n,2))  # Force accumulator
dt = 0.005           # Time step
frc = np.zeros((n,2))  # Force accumulator

g = np.array([0,0])  # Acceleration due to gravity


# Class for particle

# class particle:
#     def __init__(self, mass, position, velocity, force, number):
#         self.mass = mass
#         self.position = position
#         self.velocity = velocity
#         self.force = force
#         self.number = number
#
# #Creating an array for particles
#
#
# my_objects = []
#
# my_objects.append(particle(0.1, 0.5, 0, 0, 0))
# my_objects.append(particle(0.1, 0, 0, 0, 0))

# Initialize


m[:] = 0.1


# for i in range(n):
#     x[i, :] = np.array([random.randint(0,100)/100, random.randint(0, 100)/100])
#     v[i, :] = np.array([random.randint(1, 5), random.randint(1, 5)])
x[0, :] = np.array([0.5, 0.5])
x[1, :] = np.array([0.6, 0.5])
x[2, :] = np.array([0.6, 0.6])
x[3, :] = np.array([0.5, 0.6])
v[1, :] = np.array([0.0,0.0])

# Time stepping (this is actually "semi-implicit Euler")


def step():
    # Accumulate forces on each particle
    f.fill(0)
    for i in range(n):
        f[i,:] += m[i]*g
    frc = np.zeros((n, 2))
    len = ((x[0][0] - x[1][0])*(x[0][0] - x[1][0]) + (x[0][1] - x[1][1])*(x[0][1] - x[1][1]))
    len = len**(1/2)
    kd = 5
    kl = 1
    frc[0][0] = kl*(len/0.2 - 1)*((x[1][0] - x[0][0]) / len)
    frc[0][1] = kl * (len/0.2 - 1)*((x[1][1] - x[0][1]) / len)
    frc[1][0] = -frc[0][0]
    frc[1][1] = -frc[0][1]
    len = ((x[0][0] - x[2][0]) * (x[0][0] - x[2][0]) + (x[0][1] - x[2][1]) * (x[0][1] - x[2][1]))
    len = len ** (1 / 2)
    frc[0][0] += kl * (len / 0.2 - 1) * ((x[2][0] - x[0][0]) / len)
    frc[0][1] += kl * (len / 0.2 - 1) * ((x[2][1] - x[0][1]) / len)
    frc[2][0] += -(kl * (len / 0.2 - 1) * ((x[2][0] - x[0][0]) / len))
    frc[2][1] += -(kl * (len / 0.2 - 1) * ((x[2][1] - x[0][1]) / len))
    len = ((x[0][0] - x[3][0]) * (x[0][0] - x[3][0]) + (x[0][1] - x[3][1]) * (x[0][1] - x[3][1]))
    len = len ** (1 / 2)
    frc[0][0] += kl * (len / 0.2 - 1) * ((x[3][0] - x[0][0]) / len)
    frc[0][1] += kl * (len / 0.2 - 1) * ((x[3][1] - x[0][1]) / len)
    frc[3][0] += -(kl * (len / 0.2 - 1) * ((x[3][0] - x[0][0]) / len))
    frc[3][1] += -(kl * (len / 0.2 - 1) * ((x[3][1] - x[0][1]) / len))
    len = ((x[1][0] - x[2][0]) * (x[1][0] - x[2][0]) + (x[1][1] - x[2][1]) * (x[1][1] - x[2][1]))
    len = len ** (1 / 2)
    frc[1][0] += kl * (len / 0.2 - 1) * ((x[2][0] - x[1][0]) / len)
    frc[1][1] += kl * (len / 0.2 - 1) * ((x[2][1] - x[1][1]) / len)
    frc[2][0] += -(kl * (len / 0.2 - 1) * ((x[2][0] - x[1][0]) / len))
    frc[2][1] += -(kl * (len / 0.2 - 1) * ((x[2][1] - x[1][1]) / len))
    len = ((x[1][0] - x[3][0]) * (x[1][0] - x[3][0]) + (x[1][1] - x[3][1]) * (x[1][1] - x[3][1]))
    len = len ** (1 / 2)
    frc[1][0] += kl * (len / 0.2 - 1) * ((x[3][0] - x[1][0]) / len)
    frc[1][1] += kl * (len / 0.2 - 1) * ((x[3][1] - x[1][1]) / len)
    frc[3][0] += -(kl * (len / 0.2 - 1) * ((x[3][0] - x[1][0]) / len))
    frc[3][1] += -(kl * (len / 0.2 - 1) * ((x[3][1] - x[1][1]) / len))
    len = ((x[2][0] - x[3][0]) * (x[2][0] - x[3][0]) + (x[2][1] - x[3][1]) * (x[2][1] - x[3][1]))
    len = len ** (1 / 2)
    frc[2][0] += kl * (len / 0.2 - 1) * ((x[3][0] - x[2][0]) / len)
    frc[2][1] += kl * (len / 0.2 - 1) * ((x[3][1] - x[2][1]) / len)
    frc[3][0] += -(kl * (len / 0.2 - 1) * ((x[3][0] - x[2][0]) / len))
    frc[3][1] += -(kl * (len / 0.2 - 1) * ((x[3][1] - x[2][1]) / len))
    for i in range(n):
        f[i,0] += frc[i][0]
        f[i,1] += frc[i][1]

    for i in range(n):
        if x[i][0] >= 1 or x[i][0] <= 0:
            v[i][0] = - 0.8*v[i][0]

    for i in range(n):
        if x[i][1] >= 1 or x[i][1] <= 0:
            v[i][1] = - 0.8*v[i][1]

    for i in range(n):
        v[i,:] += f[i,:]/m[i] * dt
    # Update position of each particle
    for i in range(n):
        x[i,:] += v[i,:] * dt

# Drawing code


fig, ax = plt.subplots()
points, = ax.plot(x[:,0], x[:,1], 'o')


def init():
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal')
    return points,


def animate(frame):
    step()
    points.set_data(x[:,0], x[:,1])
    if frame is frames-1:
        plt.close()
    return points,


totalTime = 2
frames = int(totalTime/dt)
anim = FuncAnimation(fig, animate, frames=range(frames), init_func=init, interval=dt*1000)
plt.show()
"""
