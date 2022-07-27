import numpy as np
import matplotlib.pyplot as plt

### show number above the bar
### for float
def autolabel_float(rects,ax,shift=None):

    if shift == None:
        shift = [2] * len(rects)

    """Attach a text label above each bar in *rects*, displaying its height."""
    for index,rect in enumerate(rects):
        height = rect.get_height()
        print(height)
        ax.annotate('{}%'.format(height), # '{:.2f}'.format(height).lstrip('0'),
                    xy=(rect.get_x() + rect.get_width() / 2, height + shift[index] - 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-1,zorder=3)



ratio = [23.77, 16.91, 11.01, 6.07, 2.32]  # unit %, removal ratio w.r.t total training set
x = list(range(5))

fontsize = 15
markersize = 10
linewidth = 3
width = 0.5

fig, ax = plt.subplots()

rect = ax.bar(x, ratio, width=width, label='Eval NE')

plt.xticks( fontsize=fontsize)
ax.set_xticklabels(['','p0.5', 'p0.6', 'p0.7', 'p0.8', 'p0.9'])

plt.yticks(fontsize=fontsize)
plt.ylim([0,30])

plt.ylabel('Removed hard negatives\n(%)', fontsize=fontsize)
plt.xlabel('Removal percentile threshold', fontsize=fontsize)

autolabel_float(rect, ax)

# bbox_to_anchor = (x0, y0, width, height)
# plt.legend(bbox_to_anchor=(0.1, 1.0, 0.8, 0.9), loc=3, shadow=False, mode='expand', ncol=2, fontsize='x-large',frameon=False)


fig.set_size_inches(7, 3)
plt.subplots_adjust(
    left=0.18,
    bottom=0.19,
    right=0.97,
    top=0.87,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("remove_hard_negatives.pdf")