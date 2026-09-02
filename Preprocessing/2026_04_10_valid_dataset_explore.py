#!/usr/bin/env python3
"""

"""
#%% Import packages
print("Import packages")
# %matplotlib qt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print("Done")

#%% Explore Validation-set: 93,363
print("Open validation-set") # 93,363
f="2026_04_10_valid_dataset.pkl"
df = pd.read_pickle(f)
print(f"Spectrum count:{len(df)}")
print(df.columns)
#%% head
df.head()

#%% Distribution of "SpectrumScore"
import matplotlib.pyplot as plt
import numpy as np

# 计算统计量
mean_score = df["SpectrumScore"].mean()
median_score = df["SpectrumScore"].median()
q25_score = df["SpectrumScore"].quantile(0.25)
q75_score = df["SpectrumScore"].quantile(0.75)
n_spectra = df["SpectrumScore"].count()

# 画 histogram
ax = df["SpectrumScore"].hist(bins=60, color="lightblue")

plt.xlabel("SpectrumScore")
plt.ylabel("Count")
plt.title("SpectrumScore Distribution of validation-set (2026_04_10)")

plt.xticks(np.arange(0, 6.6, 0.5))

# 标出平均数、中位数、25% 和 75%
plt.axvline(mean_score,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_score:.2f}")

plt.axvline(median_score,
            color="tab:purple",
            linestyle="-",
            linewidth=2,
            label=f"Median = {median_score:.2f}")

plt.axvline(q25_score,
            color="tab:green",
            linestyle=":",
            linewidth=2,
            label=f"25% = {q25_score:.2f}")

plt.axvline(q75_score,
            color="tab:orange",
            linestyle=":",
            linewidth=2,
            label=f"75% = {q75_score:.2f}")

# 在 plot 里面写总数和统计值
textstr = (
    f"Total spectra = {n_spectra}\n"
    f"Mean = {mean_score:.2f}\n"
    f"Median = {median_score:.2f}\n"
    f"25% = {q25_score:.2f}\n"
    f"75% = {q75_score:.2f}"
)

plt.legend(
    title=f"Total spectra = {len(df["SpectrumScore"])}",
    loc="best",
)
plt.grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.2
)
plt.show()

#%% Distribution of ratio 
import numpy as np

ratio = df['filtered_ration_of_1s']

mean_ratio = ratio.mean()
median_ratio = ratio.median()
q1_ratio = ratio.quantile(0.25)
q3_ratio = ratio.quantile(0.75)

plt.figure(figsize=(12,6))

plt.hist(
        ratio,
        bins=40,
        range=(0, 1),      # 固定 histogram 范围
        edgecolor='black',
        alpha=0.7
    )

plt.axvline(mean_ratio, linestyle='-', linewidth=2,color="tab:red",
            label=f'Mean={mean_ratio:.3f}')

plt.axvline(median_ratio, linestyle='--', linewidth=2,color="tab:purple",
            label=f'Median={median_ratio:.3f}')

plt.axvline(q1_ratio, linestyle=':', linewidth=2,
            label=f'Q1={q1_ratio:.3f}')

plt.axvline(q3_ratio, linestyle=':', linewidth=2,
            label=f'Q3={q3_ratio:.3f}')

# 设置 x 轴范围
plt.xlim(0, 1)

# 每 0.05 一个刻度
plt.xticks(np.arange(0, 1.05, 0.05))

plt.xlabel('Ratio')
plt.ylabel('Sample Count')
plt.title('Ratio Distribution of validation set(2026_04_10)')
plt.legend(
    title=f"Total spectra = {len(ratio)}",
    loc="best",
)

plt.tight_layout()
plt.show()

#%% Distribution of seq_len 
import numpy as np

seq_len = df['filtered_sum_of_1s'] + df['filtered_sum_of_0s']

mean_seq_len = seq_len.mean()
median_seq_len = seq_len.median()
q1_seq_len = seq_len.quantile(0.25)
q3_seq_len = seq_len.quantile(0.75)

plt.figure(figsize=(12,6))

plt.hist(
        seq_len,
        bins = np.linspace(0, 200, 21),  # 20个bin
        # range=(0, 1),      # 固定 histogram 范围
        edgecolor='black',
        alpha=0.7
    )

plt.axvline(mean_seq_len, linestyle='-', linewidth=2,color="tab:red",
            label=f'Mean={mean_seq_len:.3f}')

plt.axvline(median_seq_len, linestyle='--', linewidth=2,color="tab:purple",
            label=f'Median={median_seq_len:.3f}')

plt.axvline(q1_seq_len, linestyle=':', linewidth=2,
            label=f'Q1={q1_seq_len:.3f}')

plt.axvline(q3_seq_len, linestyle=':', linewidth=2,
            label=f'Q3={q3_seq_len:.3f}')

# 设置 x 轴范围
plt.xlim(0, 200)

# 每 0.05 一个刻度
plt.xticks(np.arange(0, 201, 10))

plt.xlabel('seq_len')
plt.ylabel('Sample Count')
plt.title('seq_len Distribution of validation-set (2026_04_10)')
plt.legend(
    title=f"Total spectra = {len(seq_len)}",
    loc="best",
)

plt.tight_layout()
plt.show()

#%% heatmap: distribution of ratio and seq_len
import seaborn as sns # Making heatmap

ratio_bins = pd.cut(
                    df['filtered_ration_of_1s'],
                    bins=np.linspace(0, 1, 21),
                    include_lowest=True
                    )
#print(f"ratio_bins:{ratio_bins}")

seq_bins = pd.cut(
                df['filtered_sum_of_1s'] + df['filtered_sum_of_0s'], # the number of peaks,
                bins = np.linspace(0, 200, 21),  # 20个bin
                include_lowest=True
            )
#print(f"seq_bins:{seq_bins}")

count_table = pd.crosstab(
                            seq_bins,
                            ratio_bins
                        )
count_table = count_table.iloc[::-1]

# Making the heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
            count_table,
            cmap='viridis'
            )

plt.title(
    f'Spectrum Count Distribution in validation-set (spectra count:{len(df['filtered_ration_of_1s'])})'
)

plt.xlabel('Ratio')
plt.ylabel('Sequence Length')

plt.show()

#%% 3D 
import matplotlib
#matplotlib.use('QtAgg')   # 必须放在 pyplot 前

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

y = df['filtered_sum_of_1s'] + df['filtered_sum_of_0s'] # the number of peaks
x = df['filtered_ration_of_1s'] # the ratio of '1' peaks

# 二维分箱
hist, xedges, yedges = np.histogram2d(
                                        x,
                                        y,
                                        bins=(20,20)
                                    )

# 网格中心
xpos, ypos = np.meshgrid(
    xedges[:-1],
    yedges[:-1],
    indexing='ij'
)

xpos = xpos.ravel()
ypos = ypos.ravel()

zpos = np.zeros_like(xpos)

dx = np.diff(xedges)[0]
dy = np.diff(yedges)[0]

dz = hist.ravel()
#################################################
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

ax.set_xlabel('ratio', labelpad=12)
ax.set_ylabel('seq_len', labelpad=12)
ax.set_zlabel('Spectrum Count', labelpad=12)

ax.tick_params(axis='both', labelsize=10)
ax.tick_params(axis='z', labelsize=10)

# 初始视角，可按需要调整
ax.view_init(elev=28, azim=-55)

# 避免 3D 图形比例过于扁平
ax.set_box_aspect((1.2, 1, 0.7))

plt.tight_layout()
plt.show()


#%% greater than 1000,  
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ratio_bins = pd.cut(
    df['filtered_ration_of_1s'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

seq_bins = pd.cut(
    df['filtered_sum_of_1s'] + df['filtered_sum_of_0s'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

count_table = pd.crosstab(seq_bins, ratio_bins)
count_table = count_table.iloc[::-1]

plt.figure(figsize=(12, 8))

ax = sns.heatmap(
    count_table,
    cmap='viridis'
)

# 对数量 >= 1000 的格子画红色边框
for row, col in zip(*np.where(count_table.values >= 1000)):
    ax.add_patch(
        Rectangle(
            (col, row),   # 左下角坐标
            1, 1,         # 宽、高：一个 heatmap 格子
            fill=False,
            edgecolor='red',
            linewidth=2
        )
    )
    
# 图例
legend_patch = Patch(
    facecolor='none',
    edgecolor='red',
    linewidth=2,
    label='Spectrum Count ≥ 1000'
)

ax.legend(
    handles=[legend_patch],
    loc='best',
    bbox_to_anchor=(1.02, 1)
)

plt.title(
    f"Spectrum Count Distribution in validation-set "
    f"(spectra count: {len(df['filtered_ration_of_1s'])})"
)
plt.xlabel('Ratio')
plt.ylabel('Sequence Length')

plt.show()