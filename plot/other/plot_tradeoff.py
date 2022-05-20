import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "tradeoff.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('Sheet1')

values = df.values[:28,:2]
print(values)

score = values[:,0]
overhead = values[:,1]

fontsize = 13
linewidth = 2

fig, ax = plt.subplots()



x = np.linspace(0, 1, len(score))
ax.plot(x, score, 'r', label='Similarity score', linewidth=linewidth)
ax.plot(x, overhead, 'b', label='Overhead reduction', linewidth=linewidth)


# # plot the intersection point and a vertical line segment
point = (0.197, 0.498)  # the coordinates of the intersection point
ax.plot([point[0], point[0]], [0, point[1]], 'k', linewidth=linewidth, linestyle='dotted')  # plot a vertical line

plt.ylim([0, 1])
plt.xlim([0, 1])
ax.legend(loc='right', fontsize=fontsize - 2)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel('Task similarity score\nand Overhead reduction', fontsize=fontsize)
plt.xlabel('Model Size Budget', fontsize=fontsize)
# plt.title('Normalized Results', fontsize=fontsize)

# bbox_to_anchor = (x0, y0, width, height)
plt.legend(bbox_to_anchor=(-0.25, 1.0, 1.26, 0.9), loc=3, shadow=False, mode='expand', ncol=2, fontsize='large')

fig.set_size_inches(4.5, 3.5)
plt.subplots_adjust(
    left=0.193,
    bottom=0.14,
    right=0.951,
    top=0.877,
    wspace=1,
    hspace=0.5,
)

fig.show()
fig.savefig("algo1_tradeoff.pdf")




