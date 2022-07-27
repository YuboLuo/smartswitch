import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

##########################################################################################################################
### show number above the bar
### for int
def autolabel(rects,ax,shift):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index,rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + shift[index] - 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-1,zorder=3)

### show number above the bar
### for float
def autolabel_float(rects,ax,shift=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index,rect in enumerate(rects):
        height = rect.get_height()
        print(height)
        ax.annotate('{:.3}'.format(height), # '{:.2f}'.format(height).lstrip('0'),
                    xy=(rect.get_x() + rect.get_width() / 2, height - shift),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-3,zorder=3)

##########################################################################################################################
### program starts here

# df = pd.read_excel('data_performance_entrystate.xlsx', sheet_name='eventtype2')
# a = df.values   # all data in this excel page
# b = a[:12,2:12]  # catches
# b = (np.around(b.astype(float))).astype(int)  # convert from dtype=object to int by np.around
#
# c = a[21:33,2:12]  # catches
# c = c.astype(float)

x = np.arange(1,5)

NE1 = [1.28, 1.52, 1.75, 1.75]
NE2 = [1.3, 1.49, 1.81, 1.92]
NE3 = [2.25, 2.285, 2.285, 2.285]

AUC1 = [0.84, 0.96, 1.05, 1.07]
AUC2 = [0.83, 0.93, 1.05, 1.11]
AUC3 = [1.3, 1.3, 1.3, 1.3]

# NE_barcolors = ['#9b9ca0', '#935859', '#64666a']
NE_barcolors = ['#bbd6e8', '#629fca', '#1F77B4']
AUC_barcolors = ['#ffd1a9', '#ffa352', '#f47200']

width = 0.25  # the width of the bars
width_ratio = 0.8
linewitdth = 3
markersize = 10
fontsize = 15


fig, axs = plt.subplots(2)

#############################################################
### for NE
ax = axs[0]
plt.sca(ax)

### NE
rects11 = ax.bar(x - width, NE1, width*width_ratio, label='4-day training ds',color=NE_barcolors[0])
rects12 = ax.bar(x        , NE2, width*width_ratio, label='7-day training ds',color=NE_barcolors[1])
rects13 = ax.bar(x + width, NE3, width*width_ratio, label='Upperbound',color=NE_barcolors[2])

ax.set_xticklabels(['2','4','6','8'])
plt.xticks( range(1,5),fontsize=fontsize, rotation=0)
ax.set_ylim(0,3)
plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=fontsize)

# ax.get_xaxis().set_ticks([]) # hidexticks
plt.xlabel('Number of embedding split', fontsize=fontsize)
plt.ylabel('NE Drop (%)',fontsize=fontsize)







### put legend above the figure
legend = plt.legend(bbox_to_anchor=(0, 0.9, 1., 1), loc=3, shadow=False,mode='expand',ncol=3,fontsize='x-large',frameon=False)


## show value above the bar
autolabel_float(rects11, ax)
autolabel_float(rects12, ax)
autolabel_float(rects13, ax)



#############################################################
### for enery efficiency
ax = axs[1]
plt.sca(ax)



### for energy efficiency, we only have binary results
rects21 = ax.bar(x - width, AUC1, width*width_ratio, label='4-day training ds',color=AUC_barcolors[0])
rects22 = ax.bar(x        , AUC2, width*width_ratio, label='7-day training ds',color=AUC_barcolors[1])
rects23 = ax.bar(x + width, AUC3, width*width_ratio, label='Upperbound',color=AUC_barcolors[2])


ax.set_xticklabels(['2','4','6','8'])
plt.xticks( range(1,5),fontsize=fontsize, rotation=0)
ax.set_ylim(0,1.8)
plt.yticks(fontsize=fontsize)


plt.xlabel('Number of embedding split', fontsize=fontsize)
plt.ylabel('AUC Gain(%)', fontsize=fontsize)
# plt.title('(b) ', fontsize=fontsize, y=1.08)


## show value above the bar
autolabel_float(rects21, ax, 0.3)
autolabel_float(rects22, ax)
autolabel_float(rects23, ax)


### put legend above the figure
legend = plt.legend(bbox_to_anchor=(0, 0.9, 1., 1), loc=3, shadow=False,mode='expand',ncol=3,fontsize='x-large',frameon=False)



#############################################################
fig.set_size_inches(9, 5)
plt.subplots_adjust(
top=0.94,
bottom=0.115,
left=0.1,
right=0.963,
hspace=0.752,
wspace=0.2
)
plt.show()

fig.savefig('eval_e2e.pdf', format='pdf')

