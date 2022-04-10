
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "layerwise_overhead.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('time')

values = df.values[1:7,1:]
x = np.arange(values.shape[1])
print(values)

fontsize = 13
linewidth = 2
width = 0.13

fig, ax = plt.subplots()


rects11 = ax.bar(x - 2.5 * width, values[0,:], width, label='L1-Conv',color='#1F77B4')
rects14 = ax.bar(x + 0.5 * width, values[3,:], width, label='L4-FC',color='#ffd1a9')
rects12 = ax.bar(x - 1.5 * width, values[1,:], width, label='L2-Conv',color='#629fca')
rects15 = ax.bar(x + 1.5 * width, values[4,:], width, label='L5-FC',color='#ffa352')
rects13 = ax.bar(x - 0.5 * width, values[2,:], width, label='L3-Conv',color='#bbd6e8')
rects16 = ax.bar(x + 2.5 * width, values[5,:], width, label='L6-FC',color='#f47200')

ax.set_xticklabels(['MNIST','CIFAR-10','SVHN','GTSBR'])
plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)

# bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(0, 0.98, 1.,1), loc=3, shadow=False,mode='expand',ncol=3,fontsize='large')

plt.xlabel('Datasets', fontsize=fontsize)
plt.ylabel('Ratio of inference time',fontsize=fontsize)

fig.set_size_inches(4.8, 3.2)
plt.subplots_adjust(
    left=0.142,
    bottom=0.2,
    right=0.962,
    top=0.768,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("layer-wise_inference_ratio.pdf")




