
import numpy as np
import matplotlib.pyplot as plt

dataset = 'gtsrb'  # gtsrb or gsc

if dataset == 'gtsrb':
    values = [99.47, 99.59, 99.63] # gtsrb
else:
    values = [82.48, 81.69, 79.75] # GSC
x = np.arange(3)


fontsize = 15
linewidth = 2
width = 0.15

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

width_ctr = 1.6

if dataset == 'gtsrb':
    ax.bar(x , values, width*width_ctr,label='GTSRB', color='#9b9ca0', zorder = 2)
else:
    ax.bar(x, values, width * width_ctr, label='GSC', color='#935859', zorder = 2)




ax.margins(x=0.01)
ax.set_xticklabels([10,15,20])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( [15,30,45,60,75,90], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
# legend = plt.legend(bbox_to_anchor=(0.15, 0.92, 1.15,1), loc=3, shadow=False,mode='expand',ncol=1,fontsize='x-large',frameon=False)

plt.xlabel('Number of tasks', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(3, 1.9)
plt.subplots_adjust(
    left=0.26,
    bottom=0.28,
    right=0.96,
    top=0.93,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("task_number_{}.pdf".format(dataset))









