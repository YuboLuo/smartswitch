

import matplotlib.pyplot as plt
import pandas as pd

# To open Workbook
file = "msp_external_memory.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('Sheet1')

values_fram = df.values[1:7,1:]
values_flash = df.values[11:17,1:]
values_infram = df.values[20:26, 1:]


fontsize = 13
linewidth = 2

fig, ax = plt.subplots()

ax.plot(range(6), values_flash[0], 'r--', label = 'exFlash write', linewidth = linewidth)
ax.plot(range(6), values_flash[1], 'r-', label = 'exFlash read', linewidth = linewidth)

ax.plot(range(6), values_fram[0], 'b--', label = 'exFRAM write', linewidth = linewidth)
ax.plot(range(6), values_fram[1], 'b-', label = 'exFRAM read', linewidth = linewidth)

ax.plot(range(6), values_infram[0], 'k--', label = 'inFRAM write', linewidth = linewidth)
ax.plot(range(6), values_infram[1], 'k-', label = 'inFRAM read', linewidth = linewidth)

legend = ax.legend(loc='upper right', fontsize = fontsize - 4)

positions = range(6)
labels = ("0.5", "1", "2", "4", "8", "16")
plt.yticks(fontsize = fontsize)
plt.xticks(positions, labels, fontsize = fontsize)
plt.xlabel('Clock Frequency (MHz)', fontsize = fontsize)
plt.ylabel('Time Overhead (ms)', fontsize = fontsize)

fig.set_size_inches(4, 3)
plt.subplots_adjust(
    left = 0.22,
    bottom = 0.167,
    right = 0.971,
    top = 0.93,
    wspace=1,
    hspace=0.5,
)

fig.show()
fig.savefig("bg_overhead_time.pdf")

