import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.grid(axis='y', linestyle=':', zorder = 0)

x = 0

width = 0.75
ratio = 0.8
i = 0.75

time = (1.099, 5.86, 9.77)
energy = (0.5, 0.99, 1.67)
time_overhead = (0.46, 0.78, 1.071)
energy_overhead = (0.03, 0.06, 0.076)

ax.bar(i, time[0], width*ratio,color='#1f77b4', label='Inference', edgecolor='#353337', zorder = 2)
ax.bar(2*i, time[1], width*ratio,color='#1f77b4', edgecolor='#353337', zorder = 2)
ax.bar(3*i, time[2], width*ratio,color='#1f77b4', edgecolor='#353337', zorder = 2)

ax.bar(i, time_overhead[0], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=time[0])
ax.bar(2*i, time_overhead[1], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=time[1])
ax.bar(3*i, time_overhead[2], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=time[2])

ax.bar(4*i+0.375, energy[0], width*ratio, color='#1f77b4', edgecolor='#353337', zorder = 2)
ax.bar(5*i+0.375, energy[1], width*ratio, color='#1f77b4', edgecolor='#353337', zorder = 2)
ax.bar(6*i+0.375, energy[2], width*ratio, color='#1f77b4', edgecolor='#353337', zorder = 2)


ax.bar(4*i+0.375, energy_overhead[0], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=energy[0])
ax.bar(5*i+0.375, energy_overhead[1], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=energy[1])
ax.bar(6*i+0.375, energy_overhead[2], width*ratio, hatch='///', fill=False, edgecolor='#353337', zorder = 2, bottom=energy[2])

ax.set_xticks([i,2*i,3*i,4*i+0.375,5*i+0.375,6*i+0.375], fontsize=15)
ax.set_xticklabels([r'$\tau_1$', r'$\tau_2$', r'$\tau_3$', r'$\tau_1$', r'$\tau_2$', r'$\tau_3$'])
ax.tick_params(labelsize=13)
ax.set_yticks(ticks=np.arange(6)*2, fontsize=15)


# bbox_to_anchor = (x0, y0, width, height)
# legend = plt.legend(bbox_to_anchor=(0, 0.95, 1,1), loc=3, shadow=False,mode='expand',ncol=6,fontsize='x-large',frameon=False)


plt.xlabel('     MSP430         RP2040    ', fontsize=15)
plt.ylabel('Time (s)',fontsize=15)

fig.set_size_inches(3.2, 2.6)
plt.subplots_adjust(
    left=0.1,
    bottom=0.22,
    right=0.992,
    top=0.85,
    wspace=0.2,
    hspace=0.2,
)
fig.show()

fig.savefig("io_time_overhead.pdf", bbox_inches='tight')