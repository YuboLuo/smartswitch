
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# colors = ["#90AFC5", "#336B87", "#2a3132", "#763626"]
# colors = ["#EBF5FB","#D6EAF8","#AED6F1","#85C1E9","#5DADE2","#5DADE2","#3498DB","#2E86C1","#2874A6"]
# colors = ["#EBF5FB","#D6EAF8","#AED6F1","#85C1E9","#5DADE2"]
# colors = ["#EBF5FB","#EBF5FB","#D6EAF8","#D6EAF8","#AED6F1","#AED6F1","#85C1E9","#5DADE2"]
# colors = ["#EBF5FB","#EBF5FB","#EBF5FB","#EBF5FB","#EBF5FB","#AED6F1","#85C1E9","#5DADE2"]
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

########
# best match: net1-2, net2-3, net3-3
files = [{'net': 1, 'rsm': 2}, {'net': 2, 'rsm': 3}, {'net': 3, 'rsm': 3}]

# files = [{'net': 1, 'rsm': 1}, {'net': 1, 'rsm': 2}, {'net': 1, 'rsm': 3}]
arrays = []
vmin, vmax = 1, -1
for i in range(3):
    n_net, n_rsm = files[i]['net'], files[i]['rsm']
    rsm = np.load('RSM/mnist_net{}_rsm{}.npy'.format(n_net, n_rsm))
    array = np.mean(rsm, axis=0)
    array = array[:5,:5]
    arrays.append(array)

    vmin = min(vmin, array.min())
    vmax = max(vmax, array.max())

print('vmax = {}\nvmin = {}'.format(vmax, vmin))

fontsize = 18
num = len(arrays)

fig, axs = plt.subplots(ncols=num + 1, gridspec_kw=dict(width_ratios=[1,1,1,0.1]))
for i in range(num):

    plt.axes(axs[i])
    array = arrays[i]

    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    sn.heatmap(df_cm, vmin=0, vmax=1, cbar=False, cmap='crest')

    if i == 0:
        plt.ylabel('Tasks', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    else:
        axs[i].axes.get_yaxis().set_visible(False)

    plt.xlabel('Tasks', fontsize=fontsize)
    plt.title('Net{}'.format(i+1), fontsize=fontsize)
    plt.xticks( fontsize=fontsize)

# plot the color bar in the last axs
plt.axes(axs[-1])
fig.colorbar(axs[0].collections[0], cax=axs[-1])
plt.yticks(fontsize=fontsize)



fig.set_size_inches(9, 2.9)
plt.subplots_adjust(
    left=0.075,
    bottom=0.22,
    right=0.92,
    top=0.88,
    wspace=0.20,
    hspace=0.2,
)
fig.show()
fig.savefig("ConfusionMatrix.pdf")

