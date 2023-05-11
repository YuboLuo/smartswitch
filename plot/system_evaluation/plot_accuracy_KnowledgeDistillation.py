
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('knowledge_distillation')

datasets = df.values[0,2:5]
values = df.values[1:4,2:5]  # cifar10
# values = df.values[4:7,2:5]  # mnist
x = np.arange(values.shape[1])
print(values)

fontsize = 15
linewidth = 2
width = 0.15

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

width_ctr = 0.8

rects13 = ax.bar(x - 1 * width, values[0,:], width*width_ctr, label='Teacher',color='#d4d4cb', zorder = 2)
rects12 = ax.bar(x              , values[1,:], width*width_ctr, label='Stu-Distilled',color='#9b9ca0', zorder = 2)
rects11 = ax.bar(x + 1 * width, values[2,:], width*width_ctr, label='Stu-Labeled',color='#935859', zorder = 2)


ax.margins(x=0.01)
ax.set_xticklabels(datasets)
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
plt.yticks( [15,30,45,60,75,90], fontsize=fontsize)


# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.15, 0.92, 1.15,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='large',frameon=False)

# plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Accuracy (%)',fontsize=fontsize)



fig.set_size_inches(5, 2.2)
plt.subplots_adjust(
    left=0.16,
    bottom=0.16,
    right=0.972,
    top=0.85,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("accuracy_kd.pdf")









