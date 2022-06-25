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

similarity_score = values[:,0]
overhead_reduction = values[:,1]

# later, we decided to use (variety score, overhead) instead of (similarity score, overhead_reduction)
variety_score = 1 - similarity_score
overhead = 1 - overhead_reduction

fontsize = 16
linewidth = 2

fig, ax = plt.subplots()



x = np.linspace(0, 1, len(variety_score))
ax.plot(x, variety_score, 'k--', label='Variety score', linewidth=linewidth)
ax.plot(x, overhead, 'b', label='Execution cost', linewidth=linewidth)


# # plot the intersection point and a vertical line segment
# point = (0.197, 0.498)  # the coordinates of the intersection point
# ax.plot([point[0], point[0]], [0, point[1]], 'k', linewidth=linewidth, linestyle='dotted')  # plot a vertical line



plt.ylim([0, 1])
plt.xlim([0, 1])
ax.legend(loc='right', fontsize=fontsize - 2)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel('Normalized task variety score\nand execution cost', fontsize=fontsize)
plt.xlabel('Model Size Budget', fontsize=fontsize)
# plt.title('Normalized Results', fontsize=fontsize)

# bbox_to_anchor = (x0, y0, width, height)
plt.legend(bbox_to_anchor=(-0.2, 1.0, 1.26, 0.9), loc=3, shadow=False, mode='expand', ncol=2, fontsize='x-large',frameon=False)

fig.set_size_inches(5, 4)
plt.subplots_adjust(
    left=0.22,
    bottom=0.17,
    right=0.951,
    top=0.877,
    wspace=1,
    hspace=0.5,
)

fig.show()
fig.savefig("algo1_tradeoff.pdf")






