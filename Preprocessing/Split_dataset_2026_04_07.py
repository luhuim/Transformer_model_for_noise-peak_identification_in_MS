#/usr/bin/python3

#%% Import packages
print("Import packages")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print("Done")

#%% Checking current directory, and setting the root directory
import os 
from pathlib import Path
print(f"Get current working directory:{os.getcwd()}") 

# Getting the directory of the current script
current_dir = Path(__file__).resolve().parent
print("current_dir:",current_dir)       # current_dir: /home/huimin/Prepocessing

# get parent directory, 'parent_dir' 
parent_dir=current_dir.parent
print(f"Root directory = parent_dir:{parent_dir}") # parent_dir:  /home/huimin

# set the root directory as 'parent_dir'
os.chdir("/home/huimin/")

#%% The directory of .pkl files 
date="2026_04_07"
dir="Preprocessing_output/preprocessing_output_2026_04_07/"
file_name="*_tensor_dataset.pkl"
path = dir+file_name # The path: ~/Preprocessing_output/preprocessing_output_2026_04_07/

#%% Finding all pkl files
import glob
files = glob.glob(path)
print(f"files:{files}")

#%% Loading pkl files as DataFrames, and combining all DataFrames into one df
dfs = []

# loading pkl files, one by one
for f in files:
    df = pd.read_pickle(f)
    dfs.append(df) # add this dataframe into a list, called 'dfs'

# Combine all dataframes
merged_df = pd.concat(dfs, 
                      ignore_index=True, # Pandas will generate new index
                      )

print(merged_df.shape)

#%% The dataframe of original dataset 
df = merged_df
df.columns

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

#%% Distribution of label-1 peak proportion of original dataset
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


plt.xlim(0, 1)


plt.xticks(np.arange(0, 1.05, 0.05))

plt.xlabel('Ratio')
plt.ylabel('Sample Count')
plt.title("Ratio Distribution of the whole dataset")
plt.legend(
    title=f"Total spectra = {len(ratio)}",
    loc="best",
)

plt.tight_layout()
plt.show()

#%% Split the dataset into 'high-score group' and 'low score group'
# rank the dataframe in descent order, according to 'SpectrumScore' 
df_sorted = df.sort_values(by='SpectrumScore', ascending=False).reset_index(drop=True)

# index of top 50%
split_idx = len(df_sorted) // 2

df_top_50 = df_sorted.iloc[:split_idx].copy()      # top 50%, high-score group

df_bottom_50 = df_sorted.iloc[split_idx:].copy()   # 50-100%, low-score group

print('Lenth of both DataFrame:',len(df_top_50), len(df_bottom_50))
print("Top 50% DataFrame, lowest score:",df_top_50['SpectrumScore'].min())
print("Botom 50% DataFrame, highest score:",df_bottom_50['SpectrumScore'].max())

#%% High-score group : peak amount distribution, visualization
import numpy as np

seq_len = df_top_50['filtered_sum_of_1s'] + df_top_50['filtered_sum_of_0s']

mean_seq_len = seq_len.mean()
median_seq_len = seq_len.median()
q1_seq_len = seq_len.quantile(0.25)
q3_seq_len = seq_len.quantile(0.75)

plt.figure(figsize=(12,6))

plt.hist(
        seq_len,
        bins = np.linspace(0, 200, 21),  # 20 bins
        # range=(0, 1),      # histogram scale
        edgecolor='black',
        alpha=0.7
    )

plt.axvline(mean_seq_len, 
            linestyle='-', 
            linewidth=2,
            color="tab:red",
            label=f'Mean={mean_seq_len:.3f}')

plt.axvline(median_seq_len, 
            linestyle='--', 
            linewidth=2,
            color="tab:purple",
            label=f'Median={median_seq_len:.3f}')

plt.axvline(q1_seq_len, 
            linestyle=':', 
            linewidth=2,
            label=f'Q1={q1_seq_len:.3f}')

plt.axvline(q3_seq_len, 
            linestyle=':', 
            linewidth=2,
            label=f'Q3={q3_seq_len:.3f}')

# x-axis
plt.xlim(0, 200)

# every 0.05 in X axis
plt.xticks(np.arange(0, 201, 10))

plt.xlabel('seq_len')
plt.ylabel('Sample Count')
plt.title('seq_len Distribution of top 50% SpectrumScore')
plt.legend(
    title=f"Total spectra = {len(seq_len)}",
    loc="best",
)

plt.tight_layout()
plt.show()


#%% Low score group : peak_amount distribution, visualization 

import numpy as np

seq_len = df_bottom_50['filtered_sum_of_1s'] + df_bottom_50['filtered_sum_of_0s']

mean_seq_len = seq_len.mean()
median_seq_len = seq_len.median()
q1_seq_len = seq_len.quantile(0.25)
q3_seq_len = seq_len.quantile(0.75)

plt.figure(figsize=(12,6))

plt.hist(
        seq_len,
        bins = np.linspace(0, 200, 21),  # 20 bins
        # range=(0, 1),      
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

# X axis
plt.xlim(0, 200)

# every 0.05 in x-axis
plt.xticks(np.arange(0, 201, 10))

plt.xlabel('seq_len')
plt.ylabel('Sample Count')
plt.title('seq_len Distribution of bottom 50% SpectrumScore')
plt.legend(
    title=f"Total spectra = {len(seq_len)}",
    loc="best",
)

plt.tight_layout()
plt.show()

#%% The difference of peak amount distribution between high-score and low-score group
import numpy as np
import matplotlib.pyplot as plt

# Use the same bins like other histogram 
bins = np.linspace(0, 200, 21)

seq_len_top = (
    df_top_50['filtered_sum_of_1s'] +
    df_top_50['filtered_sum_of_0s']
)

seq_len_bottom = (
    df_bottom_50['filtered_sum_of_1s'] +
    df_bottom_50['filtered_sum_of_0s']
)

# Obtain sample counts of every bin 
top_counts, _ = np.histogram(seq_len_top, bins=bins)
bottom_counts, _ = np.histogram(seq_len_bottom, bins=bins)

# The count difference of every bin
diff_counts = top_counts - bottom_counts

# Center position of bin 
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = np.diff(bins)[0]

plt.figure(figsize=(12, 6))

plt.bar(
    bin_centers,
    diff_counts,
    width=bin_width * 0.85,
    edgecolor='black'
)

plt.axhline(0, color='black', linewidth=1)

plt.xlabel('seq_len')
plt.ylabel('Count Difference (Top 50% - Bottom 50%)')
plt.title('Difference of seq_len Distribution by SpectrumScore Group')

plt.xticks(np.arange(0, 201, 10))
plt.xlim(0, 200)

plt.tight_layout()
plt.show()

#%% Comparing distribution of peak amount, between high score group and low score group.  

# Compare whether the overall peak amount distrubution of the two independent group is different.
# Performe chi square test on histogram bins, in order to test every "seq_len" bin is different or not.

from scipy.stats import chi2_contingency
import numpy as np

# Two rows represent high score group / low score group
# Every column represent one seq_len bin
contingency_table = np.vstack([top_counts, bottom_counts])

print(f"top_counts:{len(top_counts)}\n{top_counts}")
print(f"bottom_counts:{len(bottom_counts)}\n{bottom_counts}")

# remove empty bin
nonzero_bins = contingency_table.sum(axis=0) > 0
contingency_table = contingency_table[:, nonzero_bins]

# chi2_contingency() tests whether the observed category frequencies differ significantly across groups.
chi2, p_value, dof, expected = chi2_contingency( contingency_table, correction=False)

# Cramér's V is used to quantify the effect size of the association.
n_total = contingency_table.sum()
n_rows, n_cols = contingency_table.shape

cramers_v = np.sqrt( chi2 / (n_total * min(n_rows - 1, n_cols - 1)) )

# print result
print(f"Chi-square statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4e}")
print(f"Cramér's V: {cramers_v:.4f}") # huge effect size

if p_value < 0.05:
    print("p_value < 0.05, two seq_len bin distribution statistically significant difference.")

#%% Mann–Whitney U test: 不依赖bin的整体distribution比较
#   Mann–Whitney U test 通常用于比较两组独立样本的分布或位置差异

df_top_50 = df_top_50.copy()
df_bottom_50 = df_bottom_50.copy()

# 转换为数值；无法转换的内容会变成 NaN
for data in [df_top_50, df_bottom_50]:
    data['filtered_sum_of_0s'] = pd.to_numeric(
        data['filtered_sum_of_0s'],
        errors='coerce'
    )

    data['filtered_sum_of_1s'] = pd.to_numeric(
        data['filtered_sum_of_1s'],
        errors='coerce'
    )

# 然后重新生成 seq_len：
seq_len_top = (
    df_top_50['filtered_sum_of_1s'] +
    df_top_50['filtered_sum_of_0s']
                ).dropna().to_numpy(dtype=float)

seq_len_bottom = (
    df_bottom_50['filtered_sum_of_1s'] +
    df_bottom_50['filtered_sum_of_0s']
                ).dropna().to_numpy(dtype=float)

# 再运行 Mann–Whitney U test：
from scipy.stats import mannwhitneyu

u_stat, p_mwu = mannwhitneyu(
    seq_len_top,
    seq_len_bottom,
    alternative='two-sided',
    method='auto'
)

# rank-biserial correlation：方向和效应量
n_top = len(seq_len_top)
n_bottom = len(seq_len_bottom)

rank_biserial = (
    2 * u_stat / (n_top * n_bottom) - 1
)

print(f"Mann-Whitney U: {u_stat:.3f}")
print(f"p-value: {p_mwu}")
print(f"Rank-biserial correlation: {rank_biserial:.4f}")
# U = 79744373062, p < 0.001, 说明两组的 seq_len 整体秩分布存在显著差异
# Rank-biserial correlation = 0.6467:
#               较大的正效应，表示你传入的第一个组 seq_len_top 通常比 seq_len_bottom 更大

#%% High-score group "df_top_50": ratio distribution
import numpy as np

ratio = df_top_50['filtered_ration_of_1s']

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
plt.title('Ratio Distribution of SpectrumScore top 50% ')
plt.legend(
            title=f"Total spectra = {len(ratio)}",
            loc="best",
)

plt.tight_layout()
plt.show()

#%% 50-100% SpectrumScore "df_bottom_50": ratio distribution
import numpy as np

ratio = df_bottom_50['filtered_ration_of_1s']

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
plt.title('Ratio Distribution of SpectrumScore bottom 50% ')
plt.legend(
            title=f"Total spectra = {len(ratio)}",
            loc="best",
)

plt.tight_layout()
plt.show()

#%% 差值柱状图: Ratio Distribution
import numpy as np
import matplotlib.pyplot as plt

# 与前面 histogram 使用完全相同的 bins
bins=np.linspace(0, 1, 41)

ratio_top = ( df_top_50['filtered_ration_of_1s'] )
ratio_bottom = ( df_bottom_50['filtered_ration_of_1s'])

# 获取每个 bin 的样本数
top_counts, _ = np.histogram(ratio_top, bins=bins)
bottom_counts, _ = np.histogram(ratio_bottom, bins=bins)

# 每个 bin 的差异
diff_counts = top_counts - bottom_counts

# bin 中心位置
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = np.diff(bins)[0]

plt.figure(figsize=(12, 6))

plt.bar(
    bin_centers,
    diff_counts,
    width=bin_width * 0.85,
    edgecolor='black'
)

plt.axhline(0, color='black', linewidth=1)

plt.xlabel('ratio')
plt.ylabel('Count Difference (Top 50% - Bottom 50%)')
plt.title('Difference of ratio Distribution by SpectrumScore Group')

# 设置 x 轴范围
plt.xlim(0, 1)
# 每 0.05 一个刻度
plt.xticks(np.arange(0, 1.05, 0.05))

plt.tight_layout()
plt.show()

#%% compare two distributions: ratio
# 对histogram bins做卡方检验, 检验每个"ratio" bin的比例分布是否不同
from scipy.stats import chi2_contingency
import numpy as np

# 两行分别代表 top 50% / bottom 50%
# 每一列代表一个 seq_len bin
contingency_table = np.vstack([top_counts, bottom_counts])

print(f"top_counts:{len(top_counts)}\n{top_counts}")
print(f"bottom_counts:{len(bottom_counts)}\n{bottom_counts}")

# 删除两组都为 0 的空 bin
nonzero_bins = contingency_table.sum(axis=0) > 0
contingency_table = contingency_table[:, nonzero_bins]

# 卡方齐性检验: chi2_contingency() 正是对contingency table中各类别频数是否独立/分布是否一致进行检验
chi2, p_value, dof, expected = chi2_contingency( contingency_table, correction=False)

# Cramér's V：效应量
n_total = contingency_table.sum()
n_rows, n_cols = contingency_table.shape

cramers_v = np.sqrt( chi2 / (n_total * min(n_rows - 1, n_cols - 1)) )

# print result
print(f"Chi-square statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4e}")
print(f"Cramér's V: {cramers_v:.4f}") # 这是较大的效应量,意味着差异不只是因为样本量大

if p_value < 0.05:
    print("p_value < 0.05, 两组的 'ratio' bin 分布存在 statistically significant difference.")


# Chi-square statistic: 39653.135
# Degrees of freedom: 15
# p-value: 0.0000e+00
# Cramér's V: 0.2524
#%% Mann–Whitney U test: 不依赖bin的整体distribution比较
#   Mann–Whitney U test 通常用于比较两组独立样本的分布或位置差异

df_top_50 = df_top_50.copy()
df_bottom_50 = df_bottom_50.copy()

# 转换为数值；无法转换的内容会变成 NaN
for data in [df_top_50, df_bottom_50]:
    data['filtered_ration_of_1s'] = pd.to_numeric(
        data['filtered_ration_of_1s'],
        errors='coerce'
    )

# 然后重新生成 seq_len：
ratio_top = (
    df_top_50['filtered_ration_of_1s'] 
                ).dropna().to_numpy(dtype=float)

ratio_bottom = (
    df_bottom_50['filtered_ration_of_1s']
                ).dropna().to_numpy(dtype=float)

# 再运行 Mann–Whitney U test：
from scipy.stats import mannwhitneyu

u_stat, p_mwu = mannwhitneyu(
    ratio_top,
    ratio_bottom,
    alternative='two-sided',
    method='auto'
)

# rank-biserial correlation：方向和效应量
n_top = len(ratio_top)
n_bottom = len(ratio_bottom)

rank_biserial = (
    2 * u_stat / (n_top * n_bottom) - 1
)

print(f"Mann-Whitney U: {u_stat:.3f}")
print(f"p-value: {p_mwu:.6e}")
print(f"Rank-biserial correlation: {rank_biserial:.4f}")

# Mann-Whitney U: 35368031790.500
# p-value: 0.000000e+00
# Rank-biserial correlation: -0.2696

#%% (New) Heatmap of high-score group
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
df_top_50['ratio'] = df_top_50['filtered_ration_of_1s']
df_top_50['seq_len']=df_top_50['filtered_sum_of_1s'] + df_top_50['filtered_sum_of_0s']

heatmap_df = df_top_50.dropna(subset=['ratio', 'seq_len']).copy()

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
# 4. 在 sample 数量 >= 5000 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 5000: # (1.6%)
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
    'Distribution of top 50% Spectra by Ratio_of_1 and Peak_amount',
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
        label='Spectra count ≥ 5000 (1.6%)'
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



#%% (New) Heatmap of Low-score group
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
df_bottom_50['ratio'] = df_bottom_50['filtered_ration_of_1s']
df_bottom_50['seq_len']=df_bottom_50['filtered_sum_of_1s'] + df_bottom_50['filtered_sum_of_0s']

heatmap_df = df_bottom_50.dropna(subset=['ratio', 'seq_len']).copy()

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
# 4. 在 sample 数量 >= 5000 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 5000: # (1.6%)
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
    'Distribution of bottom 50% Spectra by Ratio_of_1 and Peak_amount',
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
        label='Spectra count ≥ 5000 (1.6%)'
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

#%% Define the function to split the original dataset into train-set(80%) and test-set (20%)
import numpy as np
import pandas as pd

def split_train_test_by_ratio_distribution(
    df,
    ratio_col='filtered_ration_of_1s',
    score_col='SpectrumScore',
    test_size=0.2,
    n_bins=40,
    random_state=42):

    df = df.copy()

    # 1. 去掉 ratio 或 score 缺失的样本
    df = df.dropna(subset=[ratio_col, score_col]).copy()

    # 2. 找 SpectrumScore 前 50% 的阈值
    score_threshold = df[score_col].quantile(0.5)

    # 如果 SpectrumScore 越大越好，用 >=
    eligible_for_test = df[score_col] >= score_threshold

    # 3. 按 filtered_ration_of_1s 分箱
    # bins 固定在 0 到 1，和你的 histogram 一致
    df['ratio_bin'] = pd.cut(
                                df[ratio_col],
                                bins=np.linspace(0, 1, n_bins + 1),
                                include_lowest=True
                            )

    # 4. 目标 test-set 总数量
    n_test_total = int(round(len(df) * test_size))

    test_indices = []

    rng = np.random.default_rng(random_state)

    # 5. 每个 bin 按总体比例抽 test samples
    bin_counts = df['ratio_bin'].value_counts().sort_index()

    for bin_label, total_bin_count in bin_counts.items():
        if total_bin_count == 0:
            continue

        # 这个 bin 在总体中占多少比例
        bin_ratio = total_bin_count / len(df)

        # 这个 bin 需要抽多少 test samples
        n_test_bin = int(round(n_test_total * bin_ratio))

        # 当前 bin 中，且 SpectrumScore 在前 50% 的样本
        candidates = df[
            (df['ratio_bin'] == bin_label) &
            (eligible_for_test)
        ]

        # 如果候选数量不足，只能抽已有的
        n_sample = min(n_test_bin, len(candidates))

        if n_sample > 0:
            sampled_idx = rng.choice(
                candidates.index.to_numpy(),
                size=n_sample,
                replace=False
            )
            test_indices.extend(sampled_idx)

    # 6. 如果因为某些 bin 候选不足，test-set 数量不够，从剩余 eligible 样本中补齐
    test_indices = list(set(test_indices))

    remaining_needed = n_test_total - len(test_indices)

    if remaining_needed > 0:
        remaining_candidates = df[
            eligible_for_test &
            (~df.index.isin(test_indices))
        ]

        n_extra = min(remaining_needed, len(remaining_candidates))

        if n_extra > 0:
            extra_idx = rng.choice(
                remaining_candidates.index.to_numpy(),
                size=n_extra,
                replace=False
            )
            test_indices.extend(extra_idx)

    # 7. 生成 train / test
    test_df = df.loc[test_indices].copy()
    train_df = df.drop(index=test_indices).copy()

    # 8. 删除临时分箱列
    train_df = train_df.drop(columns=['ratio_bin'])
    test_df = test_df.drop(columns=['ratio_bin'])

    return train_df, test_df

#%% Use function to splite the original dataset into train-set(80%) and test-set 
train_df, test_df = split_train_test_by_ratio_distribution(
    df,
    ratio_col='filtered_ration_of_1s',
    score_col='SpectrumScore',
    test_size=0.2,
    n_bins=40,
    random_state=42
)

print(len(train_df), len(test_df))
print(len(test_df) / len(df))

#%% Check whether spectrum in test_df are all from high-score group

score_threshold = df['SpectrumScore'].quantile(0.5)

print(test_df['SpectrumScore'].min())
print(score_threshold)

assert (test_df['SpectrumScore'] >= score_threshold).all()

#%% Split test-set into validation-set(15%) and test-set(5%)
from sklearn.model_selection import train_test_split

validation_set_df, test_set_df = train_test_split(
                                test_df,
                                test_size=0.25,
                                )

#%% Distribution of label-1 peak proportionn in train-set (80%)

ratio = train_df['filtered_ration_of_1s']

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

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.05, 0.05))
plt.xlabel('Ratio')

plt.ylabel('Sample Count')

plt.title("Ratio Distribution of train-set (80%)")
plt.legend(
            title=f"Total spectra = {len(ratio)}",
            loc="best",)
plt.tight_layout()
plt.show()


#%% Distribution of Label-1 peak proportionn in validation-set (15%)

ratio = validation_set_df['filtered_ration_of_1s']

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


plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.05, 0.05))
plt.xlabel('Ratio')

plt.ylabel('Sample Count')
plt.title("Ratio Distribution of Validation-set (15%), SpectrumScore >= 2.380")
plt.legend(
    title=f"Total spectra = {len(ratio)}",
    loc="best",
)

plt.tight_layout()
plt.show()

#%% Distribution of label-1 peak proportionn in test-set (5%)
ratio = test_set_df['filtered_ration_of_1s']

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

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.05, 0.05))
plt.xlabel('Ratio')

plt.ylabel('Sample Count')
plt.title("Ratio Distribution of Test-set (5%), SpectrumScore >= 2.380")
plt.legend(
    title=f"Total spectra = {len(ratio)}",
    loc="best",
)
plt.tight_layout()
plt.show()

#%% Save the dataset in pkl file: train_df, validation_set_df, test_set_df

# train_df
train_df.to_pickle("Preprocessing_output/preprocessing_output_2026_04_07/2026_04_10_train_dataset.pkl")

# validation_set_df
validation_set_df.to_pickle("Preprocessing_output/preprocessing_output_2026_04_07/2026_04_10_valid_dataset.pkl")

# test_set_df
test_set_df.to_pickle("Preprocessing_output/preprocessing_output_2026_04_07/2026_04_10_test_dataset.pkl")

