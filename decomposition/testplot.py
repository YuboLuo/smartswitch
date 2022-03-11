import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('_mpl-gallery')
#
# # make data:
# np.random.seed(10)
# D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
#
# # plot
# fig, ax = plt.subplots()
# VP = ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
#                 showmeans=False, showfliers=False,
#                 medianprops={"color": "white", "linewidth": 0.5},
#                 boxprops={"facecolor": "C0", "edgecolor": "white",
#                           "linewidth": 0.5},
#                 whiskerprops={"color": "C0", "linewidth": 1.5},
#                 capprops={"color": "C0", "linewidth": 1.5})
#
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

a = 0.3596
k1 = -0.0266
k2 = -0.0566

x = np.linspace(0, 100)
y1 = a*2 * (1 - np.exp(k1 * x))
y2 = a * (1 - np.exp(k2 * x))
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'y')
plt.show()