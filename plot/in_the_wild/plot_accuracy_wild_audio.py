
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "comparison_wild.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('accuracy')

# datasets = df.values[0,1:6]
values = df.values[1:4,1:6]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.28

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

## because we use data read from NWS/NWV/YONO paper and then use ratio coversion
## it might be possible there are some data that have a value larger than 1 after coversion
## we have to make the maximun value no bigger than 1

for i in range(len(values)):
    for j in range(len(values[0])):
        if values[i][j] >= 100:
            values[i][j] = 100

width_ctr = 1

rects11 = ax.bar(x - 0.6 * width, values[0,:]*100, width*width_ctr, label='Vanilla',color='#9b9ca0', zorder = 2)
# rects12 = ax.bar(x              , values[1,:]*100, width*width_ctr, label='MTL',color='#ffd1a9', edgecolor='#353337', zorder = 2)
rects13 = ax.bar(x + 0.6 * width, values[2,:]*100, width*width_ctr, label='Antler',color='#935859', zorder = 2)

# ax.margins(x=0.01)
ax.set_xticklabels(['Presence\ndetection', 'Command\ndetection', 'Person\nidentification', 'Emotion\nclassification', 'Distance\nclassification'])
plt.xticks( range(len(x)),fontsize=fontsize-2, rotation=90)
plt.yticks( range(10,101,15), fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.2, 0.92, 1.2, 1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

# plt.xlabel('Tasks', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)




fig.set_size_inches(3.5, 3)
plt.subplots_adjust(
    left=0.24,
    bottom=0.45,
    right=0.95,
    top=0.85,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("accuracy_wild_audio.pdf")









