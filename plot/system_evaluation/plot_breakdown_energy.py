
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('breakdown')

values1 = df.values[15:19,1:3]
values2 = df.values[22:26,1:3]

values1 = np.delete(values1,2,0)  # delete MTL baseline
values2 = np.delete(values2,2,0)

x = np.arange(values1.shape[0])

fontsize = 15
linewidth = 2
width = 0.35

width_ctr = 0.8

fig, ax = plt.subplots()

rects11 = ax.bar(x - 0.5 * width, values1[:,0], width*width_ctr, color='#935859', zorder = 2)
rects12 = ax.bar(x + 0.5 * width, values2[:,0], width*width_ctr, color='#9b9ca0', zorder = 2)


rects21 = ax.bar(x - 0.5 * width, values1[:,1], width*width_ctr,hatch='///', label='Weight-reloading-MSP',color='#64666a', zorder = 2, bottom = values1[:,0])
rects22 = ax.bar(x + 0.5 * width, values2[:,1], width*width_ctr,hatch='///', label='Weight-reloading-RP',color='#d4d4cb', zorder = 2, bottom = values2[:,0])

ax.grid(axis='y', linestyle=':', zorder = 0)

ax.set_xticklabels(['Vanilla','NWS','Antler'])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( range(0,600,100), fontsize=fontsize)

legend = plt.legend(bbox_to_anchor=(-0.4, 0.9, 1.40, 1), loc=3, shadow=False,mode='expand',ncol=1,fontsize='xx-large',frameon=False)


plt.ylabel('Energy\nConsumption (mJ)',fontsize=fontsize)

fig.set_size_inches(4, 2.8)
plt.subplots_adjust(
    left=0.27,
    bottom=0.13,
    right=0.97,
    top=0.76,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("breakdown_energy.pdf")




