import numpy as np
import matplotlib.pyplot as plt

### show number above the bar
### for float
def autolabel_float(line, x, type, shift=None):

    if shift == None:
        shift = [0]*len(line)

    for index, height in enumerate(line):
        x_coordinate = x[index]
        y_coordinate = height

        if type == 'ne':
            if index == 0:
                x_coordinate += 0.07
                y_coordinate -= 0.0005
            if index == 1:
                x_coordinate += 0.04
            if index == 2:
                y_coordinate += 0.0001
            if index == 3:
                y_coordinate -= 0.001
            if index == len(x) - 1:
                x_coordinate -= 0.02
                y_coordinate -= 0.001




        if type == 'auc':
            if index == 0:
                x_coordinate += 0.08
                y_coordinate -= 0.00035
            if index == 1:
                x_coordinate += 0.04
                y_coordinate -= 0.0009
            if index == 2:
                y_coordinate -= 0.001

            if index == len(x) - 1:
                x_coordinate -= 0.02

        ax.annotate('{}'.format(height), # '{:.2f}'.format(height).lstrip('0'),
                    xy=(x_coordinate, y_coordinate),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-1,zorder=3)




ne = [0.82127, 0.81824, 0.81731, 0.81638, 0.81573]
auc = [0.81395, 0.81607, 0.81692, 0.81741, 0.81752]
x = [0.2, 0.4, 0.6, 0.8 ,1.0]

fontsize = 15
markersize = 10
linewidth = 3
width = 0.35

fig, ax = plt.subplots()

ax.plot(x, ne, '--', marker='*', markersize=markersize, linewidth=linewidth, label='Eval NE')
ax.plot(x, auc, marker='>', markersize=markersize, linewidth=linewidth, label='Eval AUC')

plt.xticks(x, fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.ylabel('Eval NE / AUC', fontsize=fontsize)
plt.xlabel('Downsampling Ratio', fontsize=fontsize)

autolabel_float(ne, x, type='ne')
autolabel_float(auc, x, type='auc')

# bbox_to_anchor = (x0, y0, width, height)
plt.legend(bbox_to_anchor=(0.1, 1.0, 0.8, 0.9), loc=3, shadow=False, mode='expand', ncol=2, fontsize='x-large',frameon=False)


fig.set_size_inches(8, 4)
plt.subplots_adjust(
    left=0.14,
    bottom=0.16,
    right=0.97,
    top=0.87,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("downsampling.pdf")