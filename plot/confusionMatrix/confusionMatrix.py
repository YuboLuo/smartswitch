
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib

# colors = ["#90AFC5", "#336B87", "#2a3132", "#763626"]
# colors = ["#EBF5FB","#D6EAF8","#AED6F1","#85C1E9","#5DADE2","#5DADE2","#3498DB","#2E86C1","#2874A6"]
# colors = ["#EBF5FB","#D6EAF8","#AED6F1","#85C1E9","#5DADE2"]
# colors = ["#EBF5FB","#EBF5FB","#D6EAF8","#D6EAF8","#AED6F1","#AED6F1","#85C1E9","#5DADE2"]
# colors = ["#EBF5FB","#EBF5FB","#EBF5FB","#EBF5FB","#EBF5FB","#AED6F1","#85C1E9","#5DADE2"]

colors = ["#ceeefb","#D6EAF8","#D6EAF8","#0eaeee","#0979a6","#05455f","#000000"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


def quantization(array):
    # quantize array values
    levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for low, high in zip(levels, levels[1:]):

        row, col = array.shape
        for i in range(row):
            for j in range(col):

                if array[i][j] < 0:
                    array[i][j] = 0
                elif array[i][j] > 1:
                    array[i][j] = 1

                if low <= array[i][j] <= high:
                    if array[i][j] <= low + (high - low) / 2:
                        array[i][j] = low
                    else:
                        array[i][j] = high



########
dataset = 'cifar10'
# dataset = 'mnist'

if dataset == 'cifar10':
    # best match - cifar10: net1-6, net2-3, net3-7
    files = [{'net': 1, 'rsm': 1}, {'net': 2, 'rsm': 1}, {'net': 3, 'rsm': 1}]
else:
    # best match - mnist: net1-8, net2-8, net3-1
    files = [{'net': 1, 'rsm': 6}, {'net': 2, 'rsm': 8}, {'net': 3, 'rsm': 7}]


arrays = []
vmin, vmax = 1, -1
for i in range(3):
    n_net, n_rsm = files[i]['net'], files[i]['rsm']
    rsm = np.load('RSM_joint/epoch10/{}_net{}_rsm{}.npy'.format(dataset, n_net, n_rsm))
    array = np.mean(rsm, axis=0)
    array = array[:5,:5]
    arrays.append(array)

    vmin = min(vmin, array.min())
    vmax = max(vmax, array.max())

    # quantize array
    # quantization(array)

print('vmax = {}\nvmin = {}'.format(vmax, vmin))

fontsize = 18
num = len(arrays)
Nets = ['6Layer', '6Layer', '8Layer']

fig, axs = plt.subplots(ncols=num + 1, gridspec_kw=dict(width_ratios=[1,1,1,0.1]))
for i in range(num):

    plt.axes(axs[i])
    array = arrays[i]

    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    sn.heatmap(df_cm, vmin=0, vmax=1, cbar=False, cmap='crest') # cmap = cmap

    if i == 0:
        plt.ylabel('Tasks', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    else:
        axs[i].axes.get_yaxis().set_visible(False)

    plt.xlabel('Tasks', fontsize=fontsize)
    plt.title('Net{}_{}'.format(i+1, Nets[i]), fontsize=fontsize)
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
fig.savefig("confusionMatrix_{}.pdf".format(dataset))

