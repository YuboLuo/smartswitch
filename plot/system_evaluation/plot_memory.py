
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##########################################################################################################################
### show number above the bar
### for int
def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index,rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{}KB'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height - 15),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-2,zorder=3)

##########################################################################################################################


# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('memory')

values = df.values[1:2,1:]
x = np.arange(values.shape[1])
print(values)

fontsize = 13
linewidth = 2
width = 0.3

fig, ax = plt.subplots()


rects = ax.bar(x, values[0,:], width, color='#629fca')

autolabel(rects, ax)

ax.set_xticklabels(['Vanilla','Antler','NWS','NWV','YONO'])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( fontsize=fontsize)

ax.set_ylim(0,750)


# plt.xlabel('Methods', fontsize=fontsize)
plt.ylabel('Memory usage (KB)',fontsize=fontsize)

fig.set_size_inches(4, 2.5)
plt.subplots_adjust(
    left=0.18,
    bottom=0.18,
    right=0.962,
    top=0.907,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("memory.pdf")




