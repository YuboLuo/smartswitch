
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('knowledge_distillation')

datasets = df.values[0,2:5]
values = df.values[1:4,2:5]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.15

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

width_ctr = 0.8

rects13 = ax.bar(x - 1 * width, values[0,:], width*width_ctr, label='Teacher',color='#d4d4cb', zorder = 2)
rects12 = ax.bar(x              , values[1,:], width*width_ctr, label='Student',color='#9b9ca0', zorder = 2)
rects11 = ax.bar(x + 1 * width, values[2,:], width*width_ctr, label='Vanilla',color='#935859', zorder = 2)




# rects11 = ax.bar(x - 2.5 * width, values[2,:], width*width_ctr, label='YONO',color='#d4d4cb', zorder = 2) #d4d4cb
# rects12 = ax.bar(x - 1.5 * width, values[1,:], width*width_ctr, label='NWS',color='#9b9ca0', zorder = 2)
# rects13 = ax.bar(x - 0.5 * width, values[0,:], width*width_ctr, label='NWV',color='#935859', zorder = 2)
# rects14 = ax.bar(x + 0.5 * width, values[3,:], width*width_ctr, label='Vanilla',color='#64666a', zorder = 2)
# # rects15 = ax.bar(x + 1.5 * width, values[4,:], width*width_ctr, label='MTL',color='#3F5A8A', zorder = 2)
# rects16 = ax.bar(x + 1.5 * width, values[5,:], width*width_ctr, label='Antler',color='#5d89a8', zorder = 2)


ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.15, 0.92, 1.15,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

# plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(5, 2.2)
plt.subplots_adjust(
    left=0.16,
    bottom=0.16,
    right=0.992,
    top=0.85,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("accuracy.pdf")









