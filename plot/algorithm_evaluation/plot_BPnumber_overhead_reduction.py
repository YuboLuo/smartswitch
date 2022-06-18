
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "algorithm_evaluation.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('BPnumber')

datasets = df.values[0,1:10]
values = df.values[1:4,1:10]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.17

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)


rects11 = ax.bar(x - 1.5 * width, values[0,:] / values[0,:], width * 1.2, label='#BP=3',color='#1F77B4', edgecolor='#353337', zorder = 2)
rects12 = ax.bar(x              , values[1,:] / values[0,:], width * 1.2, label='#BP=5',color='#8c0000', edgecolor='#353337', zorder = 2)
rects13 = ax.bar(x + 1.5 * width, values[2,:] / values[0,:], width * 1.2, label='#BP=7',color='#3F5A8A', edgecolor='#353337', zorder = 2)

ax.margins(x=0.01)
ax.set_xticklabels(datasets)

plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( [0,0.2,0.4,0.6,0.8,1],fontsize=fontsize)

# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.09, 0.92, 1.09,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)


plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Normalized \noverhead reduction',fontsize=fontsize)



fig.set_size_inches(8, 2.6)
plt.subplots_adjust(
    left=0.138,
    bottom=0.22,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("BPnumber_overhead_reduction.pdf")




