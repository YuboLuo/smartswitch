
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "empirical.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('empirical')

datasets = df.values[0,1:5]
values = df.values[1:6,1:5]

x = np.arange(values.shape[1])
print(values)

fontsize = 14
linewidth = 2
width = 0.16
width_ctr = 0.95

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)


rects11 = ax.bar(x - 1.5 * width, values[0,:], width * width_ctr, label='Original',color='#5d89a8', zorder = 2)
rects12 = ax.bar(x - 0.5 * width, values[1,:], width * width_ctr, label='Do Nothing',color='#64666a', zorder = 2)
rects13 = ax.bar(x + 0.5 * width, values[2,:], width * width_ctr, label='Data Imputation',color='#935859', zorder = 2)
rects14 = ax.bar(x + 1.5 * width, values[3,:], width * width_ctr, label='Data Augmentation',color='#9b9ca0', zorder = 2)

ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( range(0,101,20), fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.15, 0.92, 1.15, 1), loc=3, shadow=False,mode='expand',ncol=2,fontsize='x-large',frameon=False)

plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(5, 2)
plt.subplots_adjust(
    left=0.15,
    bottom=0.15,
    right=0.98,
    top=0.69,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("empirical.pdf")




