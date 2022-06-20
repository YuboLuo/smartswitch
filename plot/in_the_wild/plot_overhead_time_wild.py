
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# To open Workbook
file = "comparison_wild.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)

df = xls.parse('overhead_audio')
values_audio = np.transpose(df.values[1:4,1:2])[0]

df = xls.parse('overhead_image')
values_image = np.transpose(df.values[1:4,1:2])[0]

x = np.arange(values_image.shape[0])
# print(values)

fontsize = 15
linewidth = 2
width = 0.14

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

# values[4,:] = values[6:]  # uncomment this for best MTL result

width_ctr = 1


rects11 = ax.bar(x - 0.75 * width, values_audio, width*width_ctr, label='Audio',color='#1F77B4', edgecolor='#353337', zorder = 2)
rects12 = ax.bar(x + 0.75 * width, values_image, width*width_ctr, label='Image',color='#ffd1a9', edgecolor='#353337', zorder = 2)
# rects13 = ax.bar(x - 0.5 * width, values[0,:], width*width_ctr, label='NWV',color='#629fca', edgecolor='#353337', zorder = 2)
# rects14 = ax.bar(x + 0.5 * width, values[3,:], width*width_ctr, label='Vanilla',color='#ffa352', edgecolor='#353337', zorder = 2)
# rects15 = ax.bar(x + 1.5 * width, values[4,:], width*width_ctr, label='MTL',color='#3F5A8A', edgecolor='#353337', zorder = 2) # #bbd6e8
# rects16 = ax.bar(x + 2.5 * width, values[5,:], width*width_ctr, label='SS',color='#8c0000', edgecolor='#353337', zorder = 2) # #bbd6e8


# ax.margins(x=0.01)
ax.set_xticklabels(['Vanilla', 'MTL', 'Antler'])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)


plt.yticks([0,2,4,6,8,10], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
# legend = plt.legend(bbox_to_anchor=(0, 0.98, 1.,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='large')
legend = plt.legend(bbox_to_anchor=(0, 0.96, 1.0,1), loc=3, shadow=False,mode='expand',ncol=2,fontsize='x-large',frameon=False)


# plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Time overhead (s)',fontsize=fontsize)


fig.set_size_inches(4, 2.6)
plt.subplots_adjust(
    left=0.175,
    bottom=0.13,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("overhead_time_wild.pdf")




