
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# To open Workbook
file = "comparison_wild.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)

df = xls.parse('overhead_audio')
values_audio = np.transpose(df.values[1:4,2:3])[0]
values_TSPCC = df.values[1:2,7:9][0]

df = xls.parse('overhead_image')
values_image = np.transpose(df.values[1:4,2:3])[0]
values_TSPPC = df.values[1:2,7:9][0]

values_audio = np.delete(values_audio, 1, 0) # delete MTL baseline
values_image = np.delete(values_image, 1, 0)


fontsize = 15
linewidth = 2
width = 0.32
width_ctr = 1

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

# we only have one data point for TSPPC and one data point for TSPCC
# we form the two values into one variable
array_TSPPC = [0, 0, values_TSPPC[1], 0]   # for energy
array_TSPCC = [0, 0, 0, values_TSPCC[1]]   # for energy

# expand to 4 numbers
values_audio = np.concatenate((values_audio, np.asarray(([0, 0]))))
values_image = np.concatenate((values_image, np.asarray(([0, 0]))))

x = np.arange(4)

rects11 = ax.bar(x - 0.6 * width, values_audio, width*width_ctr, label='Audio',color='#9b9ca0', zorder = 2)
rects12 = ax.bar(x + 0.6 * width, values_image, width*width_ctr, label='Image',color='#935859', zorder = 2)

rects13 = ax.bar(x, array_TSPPC, width*width_ctr, color='#935859', zorder = 2)
rects14 = ax.bar(x, array_TSPCC, width*width_ctr, color='#9b9ca0', zorder = 2)


# ax.margins(x=0.01)
ax.set_xticklabels(['Vanilla', 'Antler', 'Antler\nTSPPC', 'Antler\nTSPCC'])
plt.xticks( range(len(x)),fontsize=fontsize-2, rotation=0)


plt.yticks([0,20,40,60,80,100,120], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.2, 0.96, 1.2, 1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)

plt.ylabel('Energy overhead (mJ)',fontsize=fontsize)


fig.set_size_inches(3.5, 2.6)
plt.subplots_adjust(
    left=0.23,
    bottom=0.23,
    right=0.96,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("overhead_energy_wild.pdf")




