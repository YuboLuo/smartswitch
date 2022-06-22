
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "algorithm_evaluation.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('BPlocation')

datasets = df.values[0,1:9]
values = df.values[8:11,1:9]
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.17

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

# values = (values - np.min(values))/(np.max(values) - np.min(values))   # min-max normalization

rects11 = ax.bar(x - 1.5 * width, values[0,:], width * 1.2, label='#Min-budget',color='#9b9ca0',  zorder = 2)
rects12 = ax.bar(x              , values[1,:], width * 1.2, label='#Tradeoff-budget',color='#935859', zorder = 2)
rects13 = ax.bar(x + 1.5 * width, values[2,:], width * 1.2, label='#Max-budget',color='#64666a', zorder = 2)


ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( fontsize=fontsize)

# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.09, 0.96, 1.09,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)


plt.ylabel('Cost (s)',fontsize=fontsize)



fig.set_size_inches(7, 2.5)
plt.subplots_adjust(
    left=0.10,
    bottom=0.15,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("BPlocation_cost.pdf")




