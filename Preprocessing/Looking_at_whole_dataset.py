#!/usr/bin/env python3
"""
Explore the whole dataset

"""
#%% Import packages
print("Import packages")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print("Done")
#%% 查看当前工作目录:/home/huimin 设置程序运行根目录
import os 
from pathlib import Path
print(f"Get current working directory:{os.getcwd()}") 
# 设置程序运行根目录,自动生成的
# 获取当前script所在目录
current_dir = Path(__file__).resolve().parent
print("current_dir:",current_dir)       # current_dir: /home/huimin/Prepocessing

# 获取 parent_dir 
parent_dir="/home/huimin"
print(f"parent_dir:{parent_dir}") # parent_dir:  /home/huimin

# 设置 program 的根目录为 parent_dir
os.chdir(parent_dir)

#%% pkl文件所在目录
date="2026_04_07"
dir="Preprocessing_output/preprocessing_output_2026_04_07/"
file_name="*_tensor_dataset.pkl"
path = dir+file_name
#%% 找到所有pkl文件
import glob
files = glob.glob(path)
print(f"files:{files}")

#%% 合并所有dataframe
dfs = []

for f in files:
    df = pd.read_pickle(f)
    dfs.append(df)

# 合并所有dataframe
merged_df = pd.concat(dfs, 
                      ignore_index=True, #pandas会重新生成新的index
                      )

print(merged_df.shape)

#%% explor the whole spectrum
df = merged_df
print(df.columns)
df.shape
# 'ScanNr', 
# 'rtinseconds', 
# 'pepmass', 
# 'charge', 
# 'm/z', 
# 'intensity',
# 'DenovoPeptideSequence', 
# 'SpectrumScore', 
# 'clean_seq', 
# 'ion_mass',
###############################
# 'label', 
# 'sum_of_peaks', 
# 'sum_of_1s', 
# 'sum_of_0s',
# 'ration_of_1s', ',
####################################
# 'protonated_mass', 
# 'filtered_mz', 
# 'filtered_intensity',
# 'filtered_label', 
###########################################
# 'filtered_sum_of_labels', 
# 'filtered_sum_of_1s',
# 'filtered_sum_of_0s',
# 'filtered_ration_of_1s',  
# 'filtered_changed'

#%% Heatmap of the whole dataset: distribution of ratio and seq_len
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
    f"Spectrum Count Distribution in the whole dataset"
    f"(spectra count:{len(df['filtered_ration_of_1s'])})"
)

plt.xlabel('Ratio')
plt.ylabel('Sequence Length')

plt.show()

#%% (New) sample count heatmap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 0. 为了和 heatmap 完全一致，先筛选有效数据
#    （如果你的 df 本身没有 NaN，也没有超出 0~1 / 0~200 的值，
#     这一步可以看作基本等于 df）
# =========================================================
df['ratio']=df['filtered_ration_of_1s']
df['seq_len']=df['filtered_sum_of_1s'] + df['filtered_sum_of_0s']

heatmap_df = df.dropna(subset=['ratio', 'seq_len']).copy()

heatmap_df = heatmap_df[
    (heatmap_df['ratio'] >= 0) & (heatmap_df['ratio'] <= 1) &
    (heatmap_df['seq_len'] >= 0) & (heatmap_df['seq_len'] <= 200)
].copy()

# =========================================================
# 1. 分箱
# =========================================================
ratio_edges = np.linspace(0, 1, 21)     # 20 bins
seq_edges = np.linspace(0, 200, 21)     # 20 bins

ratio_bins = pd.cut(
    heatmap_df['ratio'],
    bins=ratio_edges,
    include_lowest=True
)

seq_bins = pd.cut(
    heatmap_df['seq_len'],
    bins=seq_edges,
    include_lowest=True
)

# =========================================================
# 2. 计算每个格子的 sample count
# =========================================================
count_table = pd.crosstab(seq_bins, ratio_bins)

# seq_len 倒序显示（大的在上面）
count_table = count_table.iloc[::-1]

# =========================================================
# 3. 画 heatmap
# =========================================================
plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    count_table,
    cmap='viridis',
    annot=False,
    cbar_kws={'label': 'Sample Count'}
)

# =========================================================
# 4. 在 sample 数量 >= 10000 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 10000: # (1.6%)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                'X',
                ha='center',
                va='center',
                color='white',
                fontsize=14,
                fontweight='bold'
            )

# =========================================================
# 5. 加四个象限的辅助线
#    ratio = 0.2
#    seq_len = 70
# =========================================================
# 竖线位置：ratio=0.2 对应的 bin 边界位置
x_line = np.where(np.isclose(ratio_edges, 0.2))[0][0]   # 通常是 4

# 横线位置：由于 count_table 已经倒序，需要转换
y_line = (len(seq_edges) - 1) - np.where(np.isclose(seq_edges, 70))[0][0]  # 通常是 13

ax.axvline(
    x=x_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

ax.axhline(
    y=y_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

# =========================================================
# 6. 计算四个象限的 sample count 和 percentage
#    按你之前的定义：
#    I   = 左下  = ratio <= 0.2 且 seq_len <= 70
#    II  = 右下  = ratio > 0.2  且 seq_len <= 70
#    III = 右上  = ratio > 0.2  且 seq_len > 70
#    IV  = 左上  = ratio <= 0.2 且 seq_len > 70
# =========================================================
q1_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] <= 70)]   # I
q2_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] <= 70)]   # II
q3_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] > 70)]    # III
q4_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] > 70)]    # IV

total_samples = len(heatmap_df)

q1_count = len(q1_df)
q2_count = len(q2_df)
q3_count = len(q3_df)
q4_count = len(q4_df)

q1_pct = q1_count / total_samples * 100
q2_pct = q2_count / total_samples * 100
q3_pct = q3_count / total_samples * 100
q4_pct = q4_count / total_samples * 100

print(f"Quadrant I   sample count = {q1_count} ({q1_pct:.2f}%)")
print(f"Quadrant II  sample count = {q2_count} ({q2_pct:.2f}%)")
print(f"Quadrant III sample count = {q3_count} ({q3_pct:.2f}%)")
print(f"Quadrant IV  sample count = {q4_count} ({q4_pct:.2f}%)")

# =========================================================
# 7. 在图上标出四个象限的 sample count 和 percentage
#    用坐标轴相对坐标，更稳定
# =========================================================
textbox_style = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.78,
    edgecolor='none'
)

# I 左下
ax.text(
    0.18, 0.12,
    f'I\nn={q1_count}\n({q1_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# II 右下
ax.text(
    0.68, 0.12,
    f'II\nn={q2_count}\n({q2_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# III 右上
ax.text(
    0.68, 0.82,
    f'III\nn={q3_count}\n({q3_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# IV 左上
ax.text(
    0.18, 0.82,
    f'IV\nn={q4_count}\n({q4_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# =========================================================
# 8. 标题、坐标轴、图例
# =========================================================
plt.title(
    'Distribution of whole dataset Spectra by Ratio_of_1 and Peak_amount',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Ratio_of_1', fontsize=14)
plt.ylabel('Peak_amount', fontsize=14)

legend_elements = [
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='white',
        markeredgecolor='white',
        markersize=10,
        markeredgewidth=2,
        label='Spectra count ≥ 10000 (1.6%)'
    ),
    Line2D(
        [0], [0],
        linestyle='--',
        color='orange',
        linewidth=2,
        label='Quadrant boundaries: Ratio_of_1 = 0.2, Peak_amount = 70 '
    )
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes,
    fontsize=11,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()


#%% Open train-set
df_train = pd.read_pickle("MasterThesis/RunOnServer/2026_04_10_train_dataset.pkl")
print(df_train.shape) # (497935, 24)
#%% Open valid and test set
df_valid = pd.read_pickle("MasterThesis/RunOnServer/2026_04_10_valid_dataset.pkl")
print(df_valid.shape)
df_test = pd.read_pickle("MasterThesis/RunOnServer/2026_04_10_test_dataset.pkl")
print(df_test.shape)
#%% Combine three dataset

# Add three dataset into one list
df_list=[df_train, df_valid, df_test]

# 合并所有dataframe
full_df = pd.concat(df_list, 
                      ignore_index=True, #pandas会重新生成新的index
                      )
print(full_df.shape)


#%% train-set heatmap 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 0. 为了和 heatmap 完全一致，先筛选有效数据
#    （如果你的 df 本身没有 NaN，也没有超出 0~1 / 0~200 的值，
#     这一步可以看作基本等于 df）
# =========================================================
df_train['ratio']=df_train['filtered_ration_of_1s']
df_train['seq_len']=df_train['filtered_sum_of_1s'] + df_train['filtered_sum_of_0s']

heatmap_df = df_train.dropna(subset=['ratio', 'seq_len']).copy()

heatmap_df = heatmap_df[
    (heatmap_df['ratio'] >= 0) & (heatmap_df['ratio'] <= 1) &
    (heatmap_df['seq_len'] >= 0) & (heatmap_df['seq_len'] <= 200)
].copy()

# =========================================================
# 1. 分箱
# =========================================================
ratio_edges = np.linspace(0, 1, 21)     # 20 bins
seq_edges = np.linspace(0, 200, 21)     # 20 bins

ratio_bins = pd.cut(
    heatmap_df['ratio'],
    bins=ratio_edges,
    include_lowest=True
)

seq_bins = pd.cut(
    heatmap_df['seq_len'],
    bins=seq_edges,
    include_lowest=True
)

# =========================================================
# 2. 计算每个格子的 sample count
# =========================================================
count_table = pd.crosstab(seq_bins, ratio_bins)

# seq_len 倒序显示（大的在上面）
count_table = count_table.iloc[::-1]

# =========================================================
# 3. 画 heatmap
# =========================================================
plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    count_table,
    cmap='viridis',
    annot=False,
    cbar_kws={'label': 'Sample Count'}
)

# =========================================================
# 4. 在 sample 数量 >= 8000 (1.6%) 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 8000: # (1.6%)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                'X',
                ha='center',
                va='center',
                color='white',
                fontsize=14,
                fontweight='bold'
            )

# =========================================================
# 5. 加四个象限的辅助线
#    ratio = 0.2
#    seq_len = 70
# =========================================================
# 竖线位置：ratio=0.2 对应的 bin 边界位置
x_line = np.where(np.isclose(ratio_edges, 0.2))[0][0]   # 通常是 4

# 横线位置：由于 count_table 已经倒序，需要转换
y_line = (len(seq_edges) - 1) - np.where(np.isclose(seq_edges, 70))[0][0]  # 通常是 13

ax.axvline(
    x=x_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

ax.axhline(
    y=y_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

# =========================================================
# 6. 计算四个象限的 sample count 和 percentage
#    按你之前的定义：
#    I   = 左下  = ratio <= 0.2 且 seq_len <= 70
#    II  = 右下  = ratio > 0.2  且 seq_len <= 70
#    III = 右上  = ratio > 0.2  且 seq_len > 70
#    IV  = 左上  = ratio <= 0.2 且 seq_len > 70
# =========================================================
q1_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] <= 70)]   # I
q2_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] <= 70)]   # II
q3_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] > 70)]    # III
q4_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] > 70)]    # IV

total_samples = len(heatmap_df)

q1_count = len(q1_df)
q2_count = len(q2_df)
q3_count = len(q3_df)
q4_count = len(q4_df)

q1_pct = q1_count / total_samples * 100
q2_pct = q2_count / total_samples * 100
q3_pct = q3_count / total_samples * 100
q4_pct = q4_count / total_samples * 100

print(f"Quadrant I   sample count = {q1_count} ({q1_pct:.2f}%)")
print(f"Quadrant II  sample count = {q2_count} ({q2_pct:.2f}%)")
print(f"Quadrant III sample count = {q3_count} ({q3_pct:.2f}%)")
print(f"Quadrant IV  sample count = {q4_count} ({q4_pct:.2f}%)")

# =========================================================
# 7. 在图上标出四个象限的 sample count 和 percentage
#    用坐标轴相对坐标，更稳定
# =========================================================
textbox_style = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.78,
    edgecolor='none'
)

# I 左下
ax.text(
    0.18, 0.12,
    f'I\nn={q1_count}\n({q1_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# II 右下
ax.text(
    0.68, 0.12,
    f'II\nn={q2_count}\n({q2_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# III 右上
ax.text(
    0.68, 0.82,
    f'III\nn={q3_count}\n({q3_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# IV 左上
ax.text(
    0.18, 0.82,
    f'IV\nn={q4_count}\n({q4_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# =========================================================
# 8. 标题、坐标轴、图例
# =========================================================
plt.title(
    'Distribution of train-set Spectra by Ratio_of_1 and Peak_amount',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Ratio_of_1', fontsize=14)
plt.ylabel('Peak_amount', fontsize=14)

legend_elements = [
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='white',
        markeredgecolor='white',
        markersize=10,
        markeredgewidth=2,
        label='Spectra count ≥ 8000 (1.6%)'
    ),
    Line2D(
        [0], [0],
        linestyle='--',
        color='orange',
        linewidth=2,
        label='Quadrant boundaries: Ratio_of_1 = 0.2, Peak_amount = 70 '
    )
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes,
    fontsize=11,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()


#%% valid-set heatmap 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 0. 为了和 heatmap 完全一致，先筛选有效数据
#    （如果你的 df 本身没有 NaN，也没有超出 0~1 / 0~200 的值，
#     这一步可以看作基本等于 df）
# =========================================================
df_valid['ratio'] =  df_valid['filtered_ration_of_1s']
df_valid['seq_len'] =df_valid['filtered_sum_of_1s'] + df_valid['filtered_sum_of_0s']

heatmap_df = df_valid.dropna(subset=['ratio', 'seq_len']).copy()

heatmap_df = heatmap_df[
    (heatmap_df['ratio'] >= 0) & (heatmap_df['ratio'] <= 1) &
    (heatmap_df['seq_len'] >= 0) & (heatmap_df['seq_len'] <= 200)
].copy()

# =========================================================
# 1. 分箱
# =========================================================
ratio_edges = np.linspace(0, 1, 21)     # 20 bins
seq_edges = np.linspace(0, 200, 21)     # 20 bins

ratio_bins = pd.cut(
    heatmap_df['ratio'],
    bins=ratio_edges,
    include_lowest=True
)

seq_bins = pd.cut(
    heatmap_df['seq_len'],
    bins=seq_edges,
    include_lowest=True
)

# =========================================================
# 2. 计算每个格子的 sample count
# =========================================================
count_table = pd.crosstab(seq_bins, ratio_bins)

# seq_len 倒序显示（大的在上面）
count_table = count_table.iloc[::-1]

# =========================================================
# 3. 画 heatmap
# =========================================================
plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    count_table,
    cmap='viridis',
    annot=False,
    cbar_kws={'label': 'Sample Count'}
)

# =========================================================
# 4. 在 sample 数量 >= 1500 (1.6%) 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 1500: # (1.6%)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                'X',
                ha='center',
                va='center',
                color='white',
                fontsize=14,
                fontweight='bold'
            )

# =========================================================
# 5. 加四个象限的辅助线
#    ratio = 0.2
#    seq_len = 70
# =========================================================
# 竖线位置：ratio=0.2 对应的 bin 边界位置
x_line = np.where(np.isclose(ratio_edges, 0.2))[0][0]   # 通常是 4

# 横线位置：由于 count_table 已经倒序，需要转换
y_line = (len(seq_edges) - 1) - np.where(np.isclose(seq_edges, 70))[0][0]  # 通常是 13

ax.axvline(
    x=x_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

ax.axhline(
    y=y_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

# =========================================================
# 6. 计算四个象限的 sample count 和 percentage
#    按你之前的定义：
#    I   = 左下  = ratio <= 0.2 且 seq_len <= 70
#    II  = 右下  = ratio > 0.2  且 seq_len <= 70
#    III = 右上  = ratio > 0.2  且 seq_len > 70
#    IV  = 左上  = ratio <= 0.2 且 seq_len > 70
# =========================================================
q1_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] <= 70)]   # I
q2_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] <= 70)]   # II
q3_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] > 70)]    # III
q4_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] > 70)]    # IV

total_samples = len(heatmap_df)

q1_count = len(q1_df)
q2_count = len(q2_df)
q3_count = len(q3_df)
q4_count = len(q4_df)

q1_pct = q1_count / total_samples * 100
q2_pct = q2_count / total_samples * 100
q3_pct = q3_count / total_samples * 100
q4_pct = q4_count / total_samples * 100

print(f"Quadrant I   sample count = {q1_count} ({q1_pct:.2f}%)")
print(f"Quadrant II  sample count = {q2_count} ({q2_pct:.2f}%)")
print(f"Quadrant III sample count = {q3_count} ({q3_pct:.2f}%)")
print(f"Quadrant IV  sample count = {q4_count} ({q4_pct:.2f}%)")

# =========================================================
# 7. 在图上标出四个象限的 sample count 和 percentage
#    用坐标轴相对坐标，更稳定
# =========================================================
textbox_style = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.78,
    edgecolor='none'
)

# I 左下
ax.text(
    0.18, 0.12,
    f'I\nn={q1_count}\n({q1_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# II 右下
ax.text(
    0.68, 0.12,
    f'II\nn={q2_count}\n({q2_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# III 右上
ax.text(
    0.68, 0.82,
    f'III\nn={q3_count}\n({q3_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# IV 左上
ax.text(
    0.18, 0.82,
    f'IV\nn={q4_count}\n({q4_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# =========================================================
# 8. 标题、坐标轴、图例
# =========================================================
plt.title(
    'Distribution of validation-set Spectra by Ratio_of_1 and Peak_amount',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Ratio_of_1', fontsize=14)
plt.ylabel('Peak_amount', fontsize=14)

legend_elements = [
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='white',
        markeredgecolor='white',
        markersize=10,
        markeredgewidth=2,
        label='Spectra count ≥ 1500 (1.6%)'
    ),
    Line2D(
        [0], [0],
        linestyle='--',
        color='orange',
        linewidth=2,
        label='Quadrant boundaries: Ratio_of_1 = 0.2, Peak_amount = 70 '
    )
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes,
    fontsize=11,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()


#%% test-set heatmap 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# 0. 为了和 heatmap 完全一致，先筛选有效数据
#    （如果你的 df 本身没有 NaN，也没有超出 0~1 / 0~200 的值，
#     这一步可以看作基本等于 df）
# =========================================================
df_test['ratio'] =  df_test['filtered_ration_of_1s']
df_test['seq_len'] =df_test['filtered_sum_of_1s'] + df_test['filtered_sum_of_0s']

heatmap_df = df_test.dropna(subset=['ratio', 'seq_len']).copy()

heatmap_df = heatmap_df[
    (heatmap_df['ratio'] >= 0) & (heatmap_df['ratio'] <= 1) &
    (heatmap_df['seq_len'] >= 0) & (heatmap_df['seq_len'] <= 200)
].copy()

# =========================================================
# 1. 分箱
# =========================================================
ratio_edges = np.linspace(0, 1, 21)     # 20 bins
seq_edges = np.linspace(0, 200, 21)     # 20 bins

ratio_bins = pd.cut(
    heatmap_df['ratio'],
    bins=ratio_edges,
    include_lowest=True
)

seq_bins = pd.cut(
    heatmap_df['seq_len'],
    bins=seq_edges,
    include_lowest=True
)

# =========================================================
# 2. 计算每个格子的 sample count
# =========================================================
count_table = pd.crosstab(seq_bins, ratio_bins)

# seq_len 倒序显示（大的在上面）
count_table = count_table.iloc[::-1]

# =========================================================
# 3. 画 heatmap
# =========================================================
plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    count_table,
    cmap='viridis',
    annot=False,
    cbar_kws={'label': 'Sample Count'}
)

# =========================================================
# 4. 在 sample 数量 >= 500 (1.6%) 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 500: # (1.6%)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                'X',
                ha='center',
                va='center',
                color='white',
                fontsize=14,
                fontweight='bold'
            )

# =========================================================
# 5. 加四个象限的辅助线
#    ratio = 0.2
#    seq_len = 70
# =========================================================
# 竖线位置：ratio=0.2 对应的 bin 边界位置
x_line = np.where(np.isclose(ratio_edges, 0.2))[0][0]   # 通常是 4

# 横线位置：由于 count_table 已经倒序，需要转换
y_line = (len(seq_edges) - 1) - np.where(np.isclose(seq_edges, 70))[0][0]  # 通常是 13

ax.axvline(
    x=x_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

ax.axhline(
    y=y_line,
    color='orange',
    linestyle='--',
    linewidth=2.2,
    zorder=10
)

# =========================================================
# 6. 计算四个象限的 sample count 和 percentage
#    按你之前的定义：
#    I   = 左下  = ratio <= 0.2 且 seq_len <= 70
#    II  = 右下  = ratio > 0.2  且 seq_len <= 70
#    III = 右上  = ratio > 0.2  且 seq_len > 70
#    IV  = 左上  = ratio <= 0.2 且 seq_len > 70
# =========================================================
q1_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] <= 70)]   # I
q2_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] <= 70)]   # II
q3_df = heatmap_df[(heatmap_df['ratio'] > 0.2)  & (heatmap_df['seq_len'] > 70)]    # III
q4_df = heatmap_df[(heatmap_df['ratio'] <= 0.2) & (heatmap_df['seq_len'] > 70)]    # IV

total_samples = len(heatmap_df)

q1_count = len(q1_df)
q2_count = len(q2_df)
q3_count = len(q3_df)
q4_count = len(q4_df)

q1_pct = q1_count / total_samples * 100
q2_pct = q2_count / total_samples * 100
q3_pct = q3_count / total_samples * 100
q4_pct = q4_count / total_samples * 100

print(f"Quadrant I   sample count = {q1_count} ({q1_pct:.2f}%)")
print(f"Quadrant II  sample count = {q2_count} ({q2_pct:.2f}%)")
print(f"Quadrant III sample count = {q3_count} ({q3_pct:.2f}%)")
print(f"Quadrant IV  sample count = {q4_count} ({q4_pct:.2f}%)")

# =========================================================
# 7. 在图上标出四个象限的 sample count 和 percentage
#    用坐标轴相对坐标，更稳定
# =========================================================
textbox_style = dict(
    boxstyle='round',
    facecolor='white',
    alpha=0.78,
    edgecolor='none'
)

# I 左下
ax.text(
    0.18, 0.12,
    f'I\nn={q1_count}\n({q1_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# II 右下
ax.text(
    0.68, 0.12,
    f'II\nn={q2_count}\n({q2_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# III 右上
ax.text(
    0.68, 0.82,
    f'III\nn={q3_count}\n({q3_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# IV 左上
ax.text(
    0.18, 0.82,
    f'IV\nn={q4_count}\n({q4_pct:.2f}%)',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=13,
    fontweight='bold',
    bbox=textbox_style
)

# =========================================================
# 8. 标题、坐标轴、图例
# =========================================================
plt.title(
    'Distribution of test-set Spectra by Ratio_of_1 and Peak_amount',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Ratio_of_1', fontsize=14)
plt.ylabel('Peak_amount', fontsize=14)

legend_elements = [
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='white',
        markeredgecolor='white',
        markersize=10,
        markeredgewidth=2,
        label='Spectra count ≥ 500 (1.6%)'
    ),
    Line2D(
        [0], [0],
        linestyle='--',
        color='orange',
        linewidth=2,
        label='Quadrant boundaries: Ratio_of_1 = 0.2, Peak_amount = 70 '
    )
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes,
    fontsize=11,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()

