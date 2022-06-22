
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "algorithm_evaluation.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('BPlocation')

datasets = df.values[0,1:9]
score = df.values[1:4,1:9]
cost = df.values[8:11,1:9]
overhead = df.values[14:17,1:9]

# x = np.arange(values.shape[1])
# print(values)

fontsize = 15
linewidth = 2
width = 0.17

fig, ax = plt.subplots()


score = (score - np.min(score))/(np.max(score) - np.min(score))
cost = (cost - np.min(cost))/(np.max(cost) - np.min(cost))


# ax.grid(axis='y', linestyle=':', zorder = 0)


# rects11 = ax.bar(x - 1.5 * width, values[0,:] / values[0,:], width * 1.2, label='#BP=3',color='#5d89a8', zorder = 2)
# rects12 = ax.bar(x              , values[1,:] / values[0,:], width * 1.2, label='#BP=5',color='#64666a', zorder = 2)
# rects13 = ax.bar(x + 1.5 * width, values[2,:] / values[0,:], width * 1.2, label='#BP=7',color='#935859', zorder = 2)

# rects11 = ax.bar(x - 1.5 * width, values[0,:] / values[0,:], width * 1.2, label='#BP=3',color='#5d89a8', edgecolor='#353337', zorder = 2)
# rects12 = ax.bar(x              , values[1,:] / values[0,:], width * 1.2, label='#BP=5',color='#64666a', edgecolor='#353337', zorder = 2)
# rects13 = ax.bar(x + 1.5 * width, values[2,:] / values[0,:], width * 1.2, label='#BP=7',color='#935859', edgecolor='#353337', zorder = 2)

# markers = [".","o","v",">","<","1","s","p"]
# colors = ["#5d89a8","#64666a","#935859"]

markers = ["o","x","v"]
colors = ['#f61f1f','#ffda69','#b27e99','#7fa4ab','#76a163','#22ccff','#ff5aa6','#0e3560']

for row in range(score.shape[0]):
    for col in range(score.shape[1]):

        if row == 0:
            plt.scatter(score[row][col], cost[row][col], marker=markers[row], c=colors[col], label=datasets[col])
        else:
            plt.scatter(score[row][col], cost[row][col], marker=markers[row], c=colors[col])



# plt.scatter(score[0,:][0], cost[0,:][0], marker="<")
# plt.scatter(score[1,:], cost[1,:], marker="s")
# plt.scatter(score[2,:], cost[2,:], marker="o")



# ax.margins(x=0.01)
# ax.set_xticklabels(datasets)
# plt.xticks( range(len(x)),fontsize=fontsize, rotation=0)
# plt.yticks( [0,0.2,0.4,0.6,0.8,1],fontsize=fontsize)
#
#
# # bbox_to_anchor = (x0, y0, width, height)
legend = plt.legend(bbox_to_anchor=(-0.09, 0.99, 1.09,1), loc=4, shadow=False,mode='expand',ncol=4,fontsize='x-large',frameon=False)
#
plt.xlabel('Normalized variety score', fontsize=fontsize)
plt.ylabel('Normalized cost',fontsize=fontsize)



fig.set_size_inches(6, 6)
plt.subplots_adjust(
    left=0.14,
    bottom=0.22,
    right=0.992,
    top=0.848,
    wspace=0.2,
    hspace=0.2,
)
fig.show()
fig.savefig("BPnumber_variety_score.pdf")




