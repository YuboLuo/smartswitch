
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# To open Workbook
file = "comparison_wild.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)

df = xls.parse('overhead_audio')
values_audio = np.transpose(df.values[1:4,2:3])[0]

df = xls.parse('overhead_image')
values_image = np.transpose(df.values[1:4,2:3])[0]

values_audio = np.delete(values_audio, 1, 0) # delete MTL baseline
values_image = np.delete(values_image, 1, 0)

x = np.arange(values_image.shape[0])
# print(values)

fontsize = 15
linewidth = 2
width = 0.2
width_ctr = 1

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)



rects11 = ax.bar(x - 0.6 * width, values_audio, width*width_ctr, label='Audio',color='#1F77B4', edgecolor='#353337', zorder = 2)
rects12 = ax.bar(x + 0.6 * width, values_image, width*width_ctr, label='Image',color='#ffd1a9', edgecolor='#353337', zorder = 2)


# ax.margins(x=0.01)
ax.set_xticklabels(['Vanilla', 'Antler'])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)


plt.yticks([0,20,40,60,80,100,120], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.2, 0.96, 1.2, 1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

plt.ylabel('Energy overhead (mJ)',fontsize=fontsize)


fig.set_size_inches(3.5, 2.6)
plt.subplots_adjust(
    left=0.23,
    bottom=0.13,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("overhead_energy_wild.pdf")




