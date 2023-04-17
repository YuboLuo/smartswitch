
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
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

colors = ["#ffffff","#D6EAF8","#D6EAF8","#0eaeee","#0979a6","#05455f","#000000"]
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


net = 3
nets = [net] * 9
rsms = [i for i in range(1, 10)]
files = [{'net': nets[0], 'rsm': rsms[0]}, {'net': nets[1], 'rsm': rsms[1]}, {'net': nets[2], 'rsm': rsms[2]},
          {'net': nets[3], 'rsm': rsms[3]}, {'net': nets[4], 'rsm': rsms[4]}, {'net': nets[5], 'rsm': rsms[5]},
          {'net': nets[6], 'rsm': rsms[6]}, {'net': nets[7], 'rsm': rsms[7]}, {'net': nets[8], 'rsm': rsms[8]}]


arrays = []
vmin, vmax = 1, -1
for i in range(9):
    n_net, n_rsm = files[i]['net'], files[i]['rsm']
    rsm = np.load('RSM/{}_net{}_rsm{}.npy'.format(dataset, n_net, n_rsm))

    if net == 3:
        rsm = rsm[[1, 3, 5], :, :]
    else:
        rsm = rsm[[0,2,3],:,:]

    array = np.mean(rsm, axis=0)
    array = array[:5,:5]
    arrays.append(array)

    vmin = min(vmin, array.min())
    vmax = max(vmax, array.max())

    # quantize array
    quantization(array)

print('vmax = {}\nvmin = {}'.format(vmax, vmin))

fontsize = 18

fig, axs = plt.subplots(nrows=3, ncols=4, gridspec_kw=dict(width_ratios=[1,1,1,0.1]))

for i in range(3):
    for j in range(3):

        plt.axes(axs[i][j])
        array = arrays[i*3 + j]

        df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
        sn.heatmap(df_cm, vmin=0, vmax=1, cbar=False, cmap=cmap)  # cmap = 'crest'

        if j == 0:
            plt.yticks(fontsize=fontsize)
        else:
            axs[i][j].axes.get_yaxis().set_visible(False)

        if i == 2 and j == 1:
            plt.xlabel('Tasks', fontsize=fontsize)
        if i == 1 and j == 0:
            plt.ylabel('Tasks', fontsize=fontsize)

        plt.title('Net{} - {}'.format(net, i*3 + j + 1), fontsize=fontsize)
        plt.xticks( fontsize=fontsize)

    # plot the color bar
    if i == 1:
        plt.axes(axs[i][-1])
        fig.colorbar(axs[i][0].collections[0], cax=axs[i][-1])
        plt.yticks(fontsize=fontsize)
    else:
        axs[i][-1].axis('off') # do not show color bar


fig.set_size_inches(9, 8)
plt.subplots_adjust(
    left=0.075,
    bottom=0.09,
    right=0.92,
    top=0.94,
    wspace=0.20,
    hspace=0.7,
)
fig.show()
fig.savefig("ConfusionMatrix_{}_9*9_net{}.pdf".format(dataset, net))

