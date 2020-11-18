import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


animate = False


data = np.loadtxt('coords.dat', skiprows=1)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
ax.set_ylim((0, 5))

def update(frame):
    ax.scatter(data[frame, 1], data[frame, 0], alpha=0.25, s=3, edgecolors='none', c='k')
    if frame+1 == len(data):
        plt.cla()
        ax.set_ylim((0, 5))


if animate:
    anim = animation.FuncAnimation(fig, update, len(data), interval=1, blit=False)
else:
    ax.scatter(data[:, 1], data[:, 0], alpha=0.25, s=3, edgecolors='none', c='k')


plt.show()
