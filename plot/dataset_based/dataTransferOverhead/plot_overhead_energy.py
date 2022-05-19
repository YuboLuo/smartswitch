

import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "msp_external_memory.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('Sheet1')

values_fram = df.values[1:7, 1:]
values_flash = df.values[11:17, 1:]
values_infram = df.values[20:26, 1:]

fontsize = 13
linewidth = 2

fig, ax = plt.subplots()

# values_flash[2-3] is the energy consumed by both MSP and external chip
# ax.plot(range(6), values_flash[2], 'r--', label='exFlash write', linewidth=linewidth)
# ax.plot(range(6), values_flash[3], 'r-', label='exFlash read', linewidth=linewidth)
# ax.plot(range(6), values_fram[2], 'b--', label='exFRAM write', linewidth=linewidth)
# ax.plot(range(6), values_fram[3], 'b-', label='exFRAM read', linewidth=linewidth)

# values_flash[4-5] is the energy consumed by only external chip
ax.plot(range(6), values_flash[4], 'r--', label='exFlash write', linewidth=linewidth)
ax.plot(range(6), values_flash[5], 'r-', label='exFlash read', linewidth=linewidth)
ax.plot(range(6), values_fram[4], 'b--', label='exFRAM write', linewidth=linewidth)
ax.plot(range(6), values_fram[5], 'b-', label='exFRAM read', linewidth=linewidth)


ax.plot(range(6), values_infram[4], 'k--', label='inFRAM write', linewidth=linewidth)
ax.plot(range(6), values_infram[5], 'k-', label='inFRAM read', linewidth=linewidth)

legend = ax.legend(loc='upper right', fontsize = fontsize - 4)

positions = range(6)
labels = ("0.5", "1", "2", "4", "8", "16")
plt.yticks(fontsize=fontsize)
plt.xticks(positions, labels, fontsize=fontsize)
plt.xlabel('Clock Frequency (MHz)', fontsize=fontsize)
plt.ylabel('Energy Overhead (mJ)', fontsize=fontsize)

fig.set_size_inches(4, 3)
plt.subplots_adjust(
    left=0.16,
    bottom=0.175,
    right=0.971,
    top=0.93,
    wspace=1,
    hspace=0.5,
)

fig.show()
fig.savefig("bg_overhead_energy.pdf")
