
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

fontsize = 13
linewidth = 2
width = 0.13

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

## because we use data read from NWS/NWV/YONO paper and then use ratio coversion
## it might be possible there are some data that have a value larger than 1 after coversion
## we have to make the maximun value no bigger than 1

for i in range(len(values)):
    for j in range(len(values[0])):
        if values[i][j] >= 100:
            values[i][j] = 100



rects11 = ax.bar(x - 2.5 * width, values[2,:], width*0.7, label='YONO',color='#1F77B4', edgecolor='#353337', zorder = 2)
rects12 = ax.bar(x - 1.5 * width, values[1,:], width*0.7, label='NWS',color='#ffd1a9', edgecolor='#353337', zorder = 2)
rects13 = ax.bar(x - 0.5 * width, values[0,:], width*0.7, label='NWV',color='#629fca', edgecolor='#353337', zorder = 2)
rects14 = ax.bar(x + 0.5 * width, values[3,:], width*0.7, label='Vanilla',color='#ffa352', edgecolor='#353337', zorder = 2)
rects15 = ax.bar(x + 1.5 * width, values[4,:], width*0.7, label='MTL',color='#3F5A8A', edgecolor='#353337', zorder = 2)
rects16 = ax.bar(x + 2.5 * width, values[5,:], width*0.7, label='SS',color='#8c0000', edgecolor='#353337', zorder = 2)

ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)

# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(0, 0.98, 1.,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='large')

plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(8, 2.6)
plt.subplots_adjust(
    left=0.082,
    bottom=0.2,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("accuracy.pdf")




