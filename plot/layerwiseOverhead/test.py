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
def autolabel2(rects,ax,shift):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index,rect in enumerate(rects):
        height = rect.get_height()
        print(height)
        ax.annotate('{}'.format(height), # '{:.2f}'.format(height).lstrip('0'),
                    xy=(rect.get_x() + rect.get_width() / 2, height + shift[index] - 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize-1,zorder=3)

##########################################################################################################################
### program starts here

df = pd.read_excel('data_performance_entrystate.xlsx', sheet_name='eventtype3')
a = df.values   # all data in this excel page
b = a[:12,2:12]  # catches
b = (np.around(b.astype(float))).astype(int)  # convert from dtype=object to int by np.around

c = a[21:33,2:12]  # catches
c = c.astype(float)

x = np.arange(1,11)






width = 0.25  # the width of the bars
linewitdth = 3
markersize = 10
fontsize = 15


# fig, ax = plt.subplots()

fig, axs = plt.subplots(2)

#############################################################
### for total catches
ax = axs[0]
plt.sca(ax)

### binary
rects11 = ax.bar(x - width, b[0,:], width, label='SD=60',color='#bbd6e8', yerr= c[0,:])
rects12 = ax.bar(x        , b[4,:], width, label='SD=30',color='#629fca', yerr= c[4,:])
rects13 = ax.bar(x + width, b[8,:], width, label='SD=20',color='#1F77B4', yerr= c[8,:])
Fixed = (np.around(a[12,12])).astype(int) # average value
GT = (np.around(a[15,12])).astype(int) # average value
rects14 = ax.axhline(y = Fixed, linestyle='-.',    linewidth=2, color='r', label='Fixed',zorder=0)
rects15 = ax.axhline(y = GT, linestyle=':',   linewidth=2, color='g', label='GroudTruth',zorder=0)
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
plt.xticks( range(1,11),fontsize=fontsize, rotation=0)
ax.set_ylim(0,70)
plt.yticks(range(0,69,20),fontsize=fontsize)

# ax.get_xaxis().set_ticks([]) # hidexticks
plt.xlabel('Entry state energy level', fontsize=fontsize)
plt.ylabel('Total Catches\n(count)',fontsize=fontsize)







### put legend above the figure
legend = plt.legend(bbox_to_anchor=(0, 0.96, 1.,1), loc=3, shadow=False,mode='expand',ncol=5,fontsize='x-large')


### show value above the bar
autolabel(rects11,ax,c[0,:])
autolabel(rects12,ax,c[4,:])
autolabel(rects13,ax,c[8,:])
plt.annotate('%d' % Fixed, xy=(1, Fixed), xytext=(3, -5), xycoords=('axes fraction', 'data'), textcoords='offset points',fontsize=fontsize)
plt.annotate('%d' % GT, xy=(1, GT), xytext=(3, -5), xycoords=('axes fraction', 'data'), textcoords='offset points',fontsize=fontsize)



#############################################################
### for enery efficiency
ax = axs[1]
plt.sca(ax)

### for energy efficiency, we only have binary results
rects21 = ax.bar(x - width, np.around(b[3,:]).astype(int), width, label='SD=60',color='#ffd1a9', yerr= c[3,:])
rects22 = ax.bar(x        , np.around(b[7,:]).astype(int), width, label='SD=40',color='#ffa352', yerr= c[7,:])
rects23 = ax.bar(x + width, np.around(b[11,:]).astype(int), width, label='SD=20',color='#f47200', yerr= c[11,:])
Fixed = np.around(a[12,12])/a[14,12]
GT = np.around(a[15,12])/a[17,12]




rects24 = ax.axhline(y = (np.around(Fixed*100)).astype(int), linestyle='-.',    linewidth=2, color='r', label='Fixed',zorder=0)
rects25 = ax.axhline(y = (np.around(GT*100)).astype(int), linestyle=':',   linewidth=2, color='g', label='GroudTruth',zorder=0)


ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
plt.xticks( range(1,11),fontsize=fontsize, rotation=0)
ax.set_ylim(0,90)
plt.yticks(range(0,90,20),fontsize=fontsize)


plt.xlabel('Entry state energy level', fontsize=fontsize)
plt.ylabel('Energy Efficiency\n(%)', fontsize=fontsize)
# plt.title('(b) ', fontsize=fontsize, y=1.08)

### show value labels
autolabel2(rects21,ax,c[3,:])
autolabel2(rects22,ax,c[7,:])
autolabel2(rects23,ax,c[11,:])
plt.annotate('{}'.format((np.around(GT*100)).astype(int)), xy=(1, Fixed), xytext=(3, 0), xycoords=('axes fraction', 'data'), textcoords='offset points',fontsize=fontsize)

### put legend above the figure
legend = plt.legend(bbox_to_anchor=(0, 0.96, 1.,1), loc=3, shadow=False,mode='expand',ncol=5,fontsize='x-large')



#############################################################
fig.set_size_inches(15, 6)
plt.subplots_adjust(
top=0.927,
bottom=0.101,
left=0.067,
right=0.963,
hspace=0.752,
wspace=0.2
)
plt.show()

fig.savefig('simu_PerformanceEntryState.pdf', format='pdf')
