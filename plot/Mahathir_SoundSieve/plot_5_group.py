
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook


file = 'harvester'  # 'harvester' or 'precision'
c_value = 15

filename = file + ".xlsx"

xls = pd.ExcelFile(filename)
print(xls.sheet_names)
df = xls.parse('C'+str(c_value))

datasets = df.values[0,1:6]
values = df.values[1:3,1:6]
print(values)
x = np.arange(datasets.shape[0])

fontsize = 33 # 24
linewidth = 2
width = 0.33
width_ctr = 0.9

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)


rects11 = ax.bar(x - 0.5 * width, values[0,:], width * width_ctr, label='SoundSieve',color='#64666a', zorder = 2) # 5d89a8
rects12 = ax.bar(x + 0.5 * width, values[1,:], width * width_ctr, label='Vanilla',color='#935859', zorder = 2) # 64666a
#rects13 = ax.bar(x + 0.5 * width, values[2,:], width * width_ctr, label='4',color='#935859', zorder = 2)
#rects14 = ax.bar(x + 1.5 * width, values[3,:], width * width_ctr, label='ICS1',color='#9b9ca0', zorder = 2)

ax.margins(x=0.01)
print(datasets)
ax.set_xticklabels(["Urban8k", "ESC\n indoor", "ESC\n human", "ESC\n animal", "Voice"])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)

yticks = ax.yaxis.get_major_ticks()  # hide 0 from y-label
yticks[0].label1.set_visible(False)
plt.yticks( fontsize=fontsize)

print(values)
# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.05, 0.92, 1.08, 1), loc=3, shadow=False,mode='expand',ncol=6,fontsize=fontsize,frameon=False)

plt.ylabel('Accuracy (%)',fontsize=fontsize)
#plt.xlabel('Length (s)',fontsize=fontsize)


fig.set_size_inches(9, 5.5)
plt.subplots_adjust(
    left=0.15,
    bottom=0.2,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig(file + "_c_{}.pdf".format(c_value))





