
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


board = 'msp'
# board = 'pico'

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
sheet_name = 'timeoverhead_' + board
df = xls.parse(sheet_name)

datasets = df.values[0,1:10]
values = df.values[1:8,1:10]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.14

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

# values[4,:] = values[6:]  # uncomment this for best MTL result

width_ctr = 0.8


rects11 = ax.bar(x - 2.5 * width, values[2,:], width*width_ctr, label='YONO',color='#1F77B4', edgecolor='#353337', zorder = 2)
rects12 = ax.bar(x - 1.5 * width, values[1,:], width*width_ctr, label='NWS',color='#ffd1a9', edgecolor='#353337', zorder = 2)
rects13 = ax.bar(x - 0.5 * width, values[0,:], width*width_ctr, label='NWV',color='#629fca', edgecolor='#353337', zorder = 2)
rects14 = ax.bar(x + 0.5 * width, values[3,:], width*width_ctr, label='Vanilla',color='#ffa352', edgecolor='#353337', zorder = 2)
rects15 = ax.bar(x + 1.5 * width, values[4,:], width*width_ctr, label='MTL',color='#3F5A8A', edgecolor='#353337', zorder = 2) # #bbd6e8
rects16 = ax.bar(x + 2.5 * width, values[5,:], width*width_ctr, label='Antler',color='#8c0000', edgecolor='#353337', zorder = 2) # #bbd6e8


ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)

if board == 'msp':
    plt.yticks([0,20,40,60,80,100,120], fontsize=fontsize)
else:
    plt.yticks([0,3,6,9,12,15,18], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.09, 0.96, 1.1,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)


# plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Time overhead (s)',fontsize=fontsize)


fig.set_size_inches(8, 2.6)
plt.subplots_adjust(
    left=0.098,
    bottom=0.16,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("inferencetime_"+ board + ".pdf")




