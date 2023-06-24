
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

k = 500 # number of generations
p = 100 # number of individuals
t = 0.01 # unit time, e.g., 1 second

x = np.arange(1, 11)

y1 = factorial(x) * t # brute-force solver
y2 = x * p * k * t # GA solver

y1 = y1 / (3600)
y2 = y2 / (3600)

fontsize = 15
linewidth = 2

fig, ax = plt.subplots()

ax.plot(x, y1, linewidth=linewidth, label='Brute-force Solver')
ax.plot(x, y2, linewidth=linewidth, label='GA Solver')

plt.xlabel('The number of tasks', fontsize=fontsize)
plt.ylabel('Time complexity (unit time)', fontsize=fontsize)
plt.xticks( fontsize=fontsize)
plt.yticks( fontsize=fontsize)
plt.grid()

legend = plt.legend(bbox_to_anchor=(-0.07, 0.96, 1.1,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

plt.show()
fig.set_size_inches(5, 3)
plt.subplots_adjust(
    left=0.15,
    bottom=0.2,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("complexity.pdf")
