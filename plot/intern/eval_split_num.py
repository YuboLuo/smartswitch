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
            if index == 7:
                y_coordinate -=0.16

        # if type == 'ne':
        #     if index == 0:
        #         x_coordinate += 0.07
        #         y_coordinate -= 0.0005
        #     if index == 1:
        #         x_coordinate += 0.04
        #     if index == 2:
        #         y_coordinate += 0.0001
        #     if index == 3:
        #         y_coordinate -= 0.001
        #     if index == len(x) - 1:
        #         x_coordinate -= 0.02
        #         y_coordinate -= 0.001




        # if type == 'auc':
        #     if index == 0:
        #         x_coordinate += 0.08
        #         y_coordinate -= 0.00035
        #     if index == 1:
        #         x_coordinate += 0.04
        #         y_coordinate -= 0.0009
        #     if index == 2:
        #         y_coordinate -= 0.001
        #
        #     if index == len(x) - 1:
        #         x_coordinate -= 0.02

        ax.annotate('{:.2f}'.format(height), # '{:.2f}'.format(height).lstrip('0'),
                    xy=(x_coordinate, y_coordinate),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-2,zorder=3)




ne = [1.06, 1.07, 1.41, 1.50, 1.65, 1.52, 1.64, 1.84, 1.74, 1.60, 1.75]
auc = [0.7, 0.67, 0.86, 0.92, 0.92, 0.86, 0.97, 1.03, 0.99, 0.98, 1.02]
x = list(range(2,13))

fontsize = 15
markersize = 10
linewidth = 3
width = 0.35

fig, ax = plt.subplots()

ax.plot(x, ne, '--', marker='*', markersize=markersize, linewidth=linewidth, label='Eval NE Drop')
ax.plot(x, auc, marker='>', markersize=markersize, linewidth=linewidth, label='Eval AUC Gain')

plt.xticks(x, fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.ylabel('Eval NE Drop/ AUC Gain (%)', fontsize=fontsize)
plt.xlabel('Number of embedding split', fontsize=fontsize)

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
fig.savefig("eval_split_num.pdf")