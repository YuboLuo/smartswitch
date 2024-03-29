
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('accuracy')

datasets = df.values[0,1:10]
values = df.values[1:7,1:10]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.15

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

## because we use data read from NWS/NWV/YONO paper and then use ratio coversion
## it might be possible there are some data that have a value larger than 1 after coversion
## we have to make the maximun value no bigger than 1

for i in range(len(values)):
    for j in range(len(values[0])):
        if values[i][j] >= 100:
            values[i][j] = 100

width_ctr = 0.8

rects11 = ax.bar(x - 2.5 * width, values[2,:], width*width_ctr, label='YONO',color='#d4d4cb', zorder = 2) #d4d4cb
rects12 = ax.bar(x - 1.5 * width, values[1,:], width*width_ctr, label='NWS',color='#9b9ca0', zorder = 2)
rects13 = ax.bar(x - 0.5 * width, values[0,:], width*width_ctr, label='NWV',color='#935859', zorder = 2)
rects14 = ax.bar(x + 0.5 * width, values[3,:], width*width_ctr, label='Vanilla',color='#64666a', zorder = 2)
rects16 = ax.bar(x + 1.5 * width, values[4,:], width*width_ctr, label='Antler',color='#5d89a8', zorder = 2)



ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( [15,30,45,60,75,90,100],fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.1, 0.92, 1.1,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

# plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(8, 2.6)
plt.subplots_adjust(
    left=0.099,
    bottom=0.16,
    right=0.962,
    top=0.85,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("accuracy.pdf")









