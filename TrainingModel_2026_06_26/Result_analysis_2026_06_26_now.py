#!/usr/bin/env python3
#%% Import package
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import numpy as np

#%% Validation set metrics
df_2 = pd.read_csv("Valid_set_result_over_epochs.tsv",sep = '\t')
print(df_2)

#%% 绘制折线图
plt.figure(figsize=(8,5))

plt.plot(df_2['epoch'], df_2['recall'], marker='o',label='Average recall')
plt.plot(df_2['epoch'], df_2['precision'],marker='s', label='Average precision')

# 添加标题和标签
plt.title('Recall and Precision of Validation-set over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')

# 设置纵坐标范围 0~1
plt.ylim(0, 1)
# 设置 y轴刻度为 0.1 间隔
plt.yticks(np.arange(0.1, 1.1, 0.1))

# 强制横坐标显示所有 epoch
plt.xticks(df_2['epoch'])

# 显示图例
plt.legend()

# 网格线也是 0.1 间隔
plt.grid(True, which='major', linestyle='--')

# 显示图像
plt.show()

#%% 插入新的一行 df_3
import pandas as pd

# 新的一行
new_row = pd.DataFrame({
    "Unnamed: 0": [-1],      # 先随便填，后面重新编号
    "epoch": [19],
    "recall": [0.495],
    "precision": [0.426],
    "learning_rate": [1e-4]   # 根据你的实际学习率填写
})

# 插入到第一行
df_3 = pd.concat([new_row, df_2], ignore_index=True)

# 重新设置 Unnamed: 0（如果你需要）
df_3 ["Unnamed: 0"] = range(len(df_3))

print(df_3)

#%% 绘制折线图 df_3 recall and precision
# continued training: 从已有 checkpoint 继续训练)
plt.figure(figsize=(8, 5))

# ================= Recall =================

# Epoch 19: starting point
plt.scatter(
    19,
    0.495,
    color='green',
    marker='o',
    s=50,
    zorder=5,
    label='Recall at starting checkpoint (Epoch 19)'
)

# Epochs 20–24 画折线
plt.plot(
    df_3[df_3['epoch'] >= 20]['epoch'],
    df_3[df_3['epoch'] >= 20]['recall'],
    marker='o',
    label='Average recall (Epochs 20-39)'
)

# 19->20画虚线 (Dashed connection from Epoch 19 to Epoch 20)
plt.plot(
    [19, 20],
    [0.495, 0.481401],
    linestyle='--',
    color='gray'
)

# ================= Precision =================

# Epoch 19: starting point
plt.scatter(
    19,
    0.426,
    color='green',
    marker='s',
    s=40,
    zorder=5,
    label='Precision at starting checkpoint (Epoch 19)'
)

# Epochs 20–24 画折线
plt.plot(
    df_3[df_3['epoch'] >= 20]['epoch'],
    df_3[df_3['epoch'] >= 20]['precision'],
    marker='s',
    label='Average precision (Epochs 20-39)'
)

# Epoch 19->20 画虚线 (Dashed connection from Epoch 19 to Epoch 20)
plt.plot(
    [19, 20],
    [0.426, 0.409411],
    linestyle='--',
    color='gray'
)

# ================= Figure settings =================

plt.title('Validation-set Recall and Precision During Continued Training')
plt.xlabel('Epoch')
plt.ylabel('Score')

plt.ylim(0, 1)
plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.xticks(df_3['epoch'])

plt.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#%% 绘制learning rate折线图 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(10, 5))

# Epoch 19 是 starting checkpoint，不属于本次实际训练
# 因此只绘制 Epoch 20–39 的 learning rate
lr_df = df_3[df_3["epoch"] >= 20].copy()

# 按 epoch 排序，避免数据顺序影响折线
lr_df = lr_df.sort_values("epoch").reset_index(drop=True)

# =====================================================
# Learning-rate step plot
# =====================================================
plt.step(
    lr_df["epoch"],
    lr_df["learning_rate"],
    where="post",
    marker="o",
    markersize=5,
    linewidth=2,
    label="Learning rate (Epochs 20–39)"
)

# =====================================================
# Starting checkpoint: Epoch 19
# =====================================================
plt.axvline(
    x=19,
    color="green",
    linestyle="--",
    linewidth=2,
    label="Starting checkpoint (Epoch 19)"
)

# =====================================================
# Automatically identify learning-rate reductions
# =====================================================
lr_reduction_rows = lr_df[
    lr_df["learning_rate"].diff() < 0
]

for _, row in lr_reduction_rows.iterrows():

    reduction_epoch = row["epoch"]
    new_lr = row["learning_rate"]

    # Vertical dotted line at each reduction
    plt.axvline(
        x=reduction_epoch,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8
    )

    # Label the new learning rate
    plt.annotate(
        f"LR = {new_lr:.2g}",
        xy=(reduction_epoch, new_lr),
        xytext=(reduction_epoch + 0.3, new_lr + 5e-6),
        arrowprops=dict(
            arrowstyle="->",
            color="gray"
        ),
        fontsize=9
    )

# =====================================================
# Titles and axis labels
# =====================================================
plt.title("Learning Rate Schedule During Continued Training")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

# Show Epoch 19–39 on the x-axis
plt.xticks(
    range(19, 40),
    rotation=45
)

plt.xlim(18.6, 39.4)

# Y-axis range
plt.ylim(0, 1.15e-4)

# Use the actual learning-rate levels as ticks
lr_ticks = sorted(lr_df["learning_rate"].unique())
plt.yticks([0] + lr_ticks)

# Scientific notation
plt.gca().yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, position: "0" if value == 0 else f"{value:.2e}"
    )
)

plt.legend()
plt.grid(
    True,
    which="major",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()
plt.show()



#%%  Train set
df_3 = pd.read_csv("Train_set_result_over_epochs.tsv",sep = '\t')
print(df_3)
#%% 绘制折线图
plt.figure(figsize=(8,5))

# 绘制曲线
plt.plot(
    df_3['epoch'],          # x axis columns
    df_3['average_loss'],   # y axis columns
    marker='o',             # marker: bullet point
    label='Average loss',   # 
    )

# 找最大值和最小值
max_idx = df_3['average_loss'].idxmax()
min_idx = df_3['average_loss'].idxmin()

max_x = df_3.loc[max_idx, 'epoch']
max_y = df_3.loc[max_idx, 'average_loss']

min_x = df_3.loc[min_idx, 'epoch']
min_y = df_3.loc[min_idx, 'average_loss']

# 标出最大值
plt.scatter(max_x, max_y, s=100, label='Max',color='red')
plt.text(max_x, max_y,
         f'Max: {max_y:.3f}',
         fontsize=10,
         verticalalignment='bottom',
         color='red'
         )

# 标出最小值
plt.scatter(min_x, min_y, s=100, label='Min', color='green')
plt.text(min_x, min_y,
         f'Min: {min_y:.3f}',
         fontsize=10,
         verticalalignment='top',
         color='green')

# 添加标题和标签
plt.title('Average loss in training phase over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average loss')

# y轴坐标范围 0~1
plt.ylim(0, 1)
# y轴刻度为 0.1 间隔
plt.yticks(np.arange(0.1, 1.1, 0.1))

# x轴刻度: 强制横坐标显示所有 epoch
plt.xticks(df_3['epoch'])

# 显示图例
plt.legend()

# 网格线也是 0.1 间隔
plt.grid(True, which='major', linestyle='--')

# 显示图像
plt.show()
#%% Epoch 24
tsv_file_name="Epoch:24_Precision_Recall_DataFrame.tsv"
df = pd.read_csv( tsv_file_name ,sep = '\t') 

print(tsv_file_name)
print("Recall average:",df["recall"].mean())
print("Precision average:",df["precision"].mean())
#%% Validation-set explore: seq_len distribution
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

seq = df['seq_len']

mean_val = seq.mean()
median_val = seq.median()
q1 = seq.quantile(0.25)
q3 = seq.quantile(0.75)

max_val = seq.max()
min_val = seq.min()
n_samples = len(seq)

plt.figure(figsize=(12,6))

plt.hist(
    seq,
    bins=50,
    edgecolor='black',
    alpha=0.7
)

plt.axvline(
        mean_val, 
        linestyle='-', 
        linewidth=2,
        label=f'Mean={mean_val:.3f}', 
        color="tab:red",)

plt.axvline(
        median_val, 
        linestyle='--', 
        linewidth=2,
        label = f'Median={median_val:.3f}',
        color="tab:purple",)

plt.axvline(
        q1, 
        linestyle=':', 
        linewidth=2,
        label=f'Q1={q1:.3f}',
        color="tab:green",)

plt.axvline(
        q3, 
        linestyle=':', 
        linewidth=2,
        label=f'Q3={q3:.3f}',
        color="tab:orange",)


# 获取已有legend对象
handles, labels = plt.gca().get_legend_handles_labels()

# 添加文本信息（无图形）
handles.extend([
    Line2D([], [], color='none', label=f'Min = {min_val}'),
    Line2D([], [], color='none', label=f'Max = {max_val}'),
    Line2D([], [], color='none', label=f'Spectrum count = {n_samples:,}')
])

# 设置x轴刻度更密
plt.xticks(np.arange(0, seq.max()+10, 10))  # 每10一个刻度

plt.xlabel('Sequence Length')
plt.ylabel('Sample Count')
plt.title('Sequence length distribution of validation-set')
plt.legend(handles=handles, loc='best')

plt.tight_layout()
plt.show()
#%% Validation-set explore: ratio distribution

ratio = df['ratio'].dropna()

ratio_mean = ratio.mean()
ratio_median = ratio.median()
ratio_q1 = ratio.quantile(0.25)
ratio_q3 = ratio.quantile(0.75)

max_ratio = ratio.max()
min_ratio = ratio.min()
n_samples = len(ratio)

plt.figure(figsize=(12,6))

plt.hist(
    ratio,
    bins= np.linspace(0, 1, 41),
    edgecolor='black',
    alpha=0.7
)

plt.axvline(ratio_mean, 
            linestyle='-', 
            linewidth=2,
            label=f'Mean={ratio_mean:.3f}',
            color="tab:red",
            )

plt.axvline(ratio_median, 
            linestyle='--', 
            linewidth=2,
            label=f'Median={ratio_median:.3f}',
            color="tab:purple",)

plt.axvline(ratio_q1, 
            linestyle=':', 
            linewidth=2,
            label=f'Q1={ratio_q1:.3f}',
            color="tab:green",)

plt.axvline(ratio_q3, 
            linestyle=':', 
            linewidth=2,
            label=f'Q3={ratio_q3:.3f}',
            color="tab:orange",
            )

# 获取已有legend对象
handles, labels = plt.gca().get_legend_handles_labels()

# 添加文本信息（无图形）
handles.extend([
    Line2D([], [], color='none', label=f'Min = {min_ratio:.3f}'),
    Line2D([], [], color='none', label=f'Max = {max_ratio:.3f}'),
    Line2D([], [], color='none', label=f'Spectrum count = {n_samples:,}')
])

# 设置x轴刻度更密
plt.xticks(np.arange(0, 1.05, 0.05))  # 每0.05一个刻度

plt.xlabel('Ratio')
plt.ylabel('Sample Count')
plt.title('Label-1 proportion distribution of validation-set')
plt.legend(handles=handles, loc='best')

plt.tight_layout()
plt.show()

#%% 3D 柱状图  X seq_len, Y ratio, Z amount
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

y = df['seq_len']
x = df['ratio']

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

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(
    xpos,
    ypos,
    zpos,
    dx,
    dy,
    dz,
    shade=True
)

ax.set_ylabel('seq_len')
ax.set_xlabel('ratio')
ax.set_zlabel('Sample Count')

plt.show()

#%% Heatmap(ratio × seq_len) making 20×20 的二维网格
ratio_bins = pd.cut(
                    df['ratio'],
                    bins=np.linspace(0, 1, 21),
                    include_lowest=True
                    )
#print(f"ratio_bins:{ratio_bins}")

seq_bins = pd.cut(
                    df['seq_len'],
                    bins=np.linspace(0, 200, 21),  # 20个bin
                    include_lowest=True
)
#print(f"seq_bins:{seq_bins}")

count_table = pd.crosstab(
                            seq_bins,
                            ratio_bins
                        )

count_table = count_table.iloc[::-1]

# Making heatmap
import seaborn as sns

plt.figure(figsize=(12,8))

sns.heatmap(
            count_table,
            cmap='viridis'
            )

plt.title(
    'Sample Count Distribution'
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
# 4. 在 sample 数量 >= 1500 的格子打白色 "X"
# =========================================================
for row_idx in range(count_table.shape[0]):
    for col_idx in range(count_table.shape[1]):
        cell_count = count_table.iloc[row_idx, col_idx]

        if pd.notna(cell_count) and cell_count >= 1500:
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
    'Distribution of Validation-Set Spectra by Label-1 proportion and Peak_amount',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Label-1 proportion', fontsize=14)
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
        label='Spectra count ≥ 1500'
    ),
    Line2D(
        [0], [0],
        linestyle='--',
        color='orange',
        linewidth=2,
        label='Quadrant boundaries: Ratio_of_1 = 0.2, Peak_amount = 70'
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


#%% Recall heatmap 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ratio分箱
ratio_bins = pd.cut(
    df['ratio'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

# seq_len分箱
seq_bins = pd.cut(
    df['seq_len'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

# 计算每个格子的平均Recall
recall_table = pd.pivot_table(
    df,
    values='recall',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# seq_len倒序显示
recall_table = recall_table.iloc[::-1]

plt.figure(figsize=(14,10))

sns.heatmap(
    recall_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label':'Mean Recall'}
)

plt.xlabel('Ratio', fontsize=16)
plt.ylabel('Sequence Length', fontsize=16)

plt.title(
    'Epoch 22: Mean Recall Across Ratio and Sequence Length',
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

#%% Precision heatmap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ratio分箱
ratio_bins = pd.cut(
    df['ratio'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

# seq_len分箱
seq_bins = pd.cut(
    df['seq_len'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

# 计算每个格子的平均Recall
precision_table = pd.pivot_table(
    df,
    values='precision',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# seq_len倒序显示
precision_table = precision_table.iloc[::-1]

plt.figure(figsize=(14,10))

sns.heatmap(
    precision_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label':'Mean Recall'}
)

plt.xlabel('Ratio', fontsize=16)
plt.ylabel('Sequence Length', fontsize=16)

plt.title(
    'Epoch 22: Mean Precison Across Ratio and Sequence Length',
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout()
plt.show() 

#%% Recall heatmap: red box 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 整个 dataset 的平均 recall
dataset_mean_recall = df['recall'].mean()
print(f'Dataset mean recall: {dataset_mean_recall:.3f}')

# ratio 分箱
ratio_bins = pd.cut(
    df['ratio'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

# seq_len 分箱
seq_bins = pd.cut(
    df['seq_len'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

# 计算每个格子的平均 recall
recall_table = pd.pivot_table(
    df,
    values='recall',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# seq_len 倒序显示
recall_table = recall_table.iloc[::-1]

plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    recall_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label': 'Mean Recall'}
)

for row_idx in range(recall_table.shape[0]):
    for col_idx in range(recall_table.shape[1]):

        cell_recall = recall_table.iloc[row_idx, col_idx]

        # 排除空格子，并标记低于 dataset mean recall 的格子
        if pd.notna(cell_recall) and cell_recall < dataset_mean_recall:
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5
                )
            )

plt.xlabel('Ratio', fontsize=16)
plt.ylabel('Sequence Length', fontsize=16)

plt.title(
    'Epoch 22: Mean Recall Across Ratio and Sequence Length',
    fontsize=20,
    fontweight='bold'
)

# 在图的右上角标出整个 dataset 的平均 recall
ax.text(
    0.98,
    0.98,
    f'Dataset mean recall = {dataset_mean_recall:.3f}',
    transform=ax.transAxes,
    horizontalalignment='right',
    verticalalignment='top',
    fontsize=14,
    bbox={
        'boxstyle': 'round',
        'facecolor': 'white',
        'alpha': 0.85
    }
)

legend_element = Line2D(
    [0],
    [0],
    color='red',
    linewidth=2,
    label=f'Mean recall < dataset mean ({dataset_mean_recall:.3f})'
)

ax.legend(
    handles=[legend_element],
    loc='upper left',
    bbox_to_anchor=(0, 1.08),
    fontsize=12,
    frameon=True
)

plt.tight_layout()
plt.show()
#%% Recall Heatmap: red box; "X": sample count >= 1500
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 整个 dataset 的平均 recall
dataset_mean_recall = df['recall'].mean()
print(f'Validation-set mean recall: {dataset_mean_recall:.3f}')

# ratio 分箱
ratio_bins = pd.cut(
    df['ratio'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

# seq_len 分箱
seq_bins = pd.cut(
    df['seq_len'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

# 计算每个格子的平均 recall
recall_table = pd.pivot_table(
    df,
    values='recall',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# 计算每个格子的 sample 数量
count_table = pd.pivot_table(
    df,
    values='recall',   # 用任意非空列都可以
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='count'
)

# seq_len 倒序显示
recall_table = recall_table.iloc[::-1] # 每个格子的平均 recall
count_table  = count_table.iloc[::-1]   # 每个格子的 sample 数量

plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    recall_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label': 'Mean Recall'}
)

for row_idx in range(recall_table.shape[0]):
    for col_idx in range(recall_table.shape[1]):

        cell_recall = recall_table.iloc[row_idx, col_idx]
        cell_count = count_table.iloc[row_idx, col_idx]

        # 排除空格子，并标记低于 dataset mean recall 的格子
        if pd.notna(cell_recall) and cell_recall < dataset_mean_recall:
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5
                )
            )

        # sample 数量 >= 1500 的格子打上 X
        if pd.notna(cell_count) and cell_count >= 1500:
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

# ratio = 0.2 的竖线
ax.axvline(
    x=4,
    color='black',
    linestyle='--',
    linewidth=2.5,
    zorder=10
)

# sequence length = 70 的横线
ax.axhline(
    y=13,
    color='black',
    linestyle='--',
    linewidth=2.5,
    zorder=10
)

# =========================
# 1. 定义四个象限
# =========================
q1_df = df[(df['ratio'] <= 0.2) & (df['seq_len'] <= 70)]   # I 左下
q2_df = df[(df['ratio'] > 0.2)  & (df['seq_len'] <= 70)]   # II 右下
q3_df = df[(df['ratio'] > 0.2)  & (df['seq_len'] > 70)]    # III 右上
q4_df = df[(df['ratio'] <= 0.2) & (df['seq_len'] > 70)]    # IV 左上

# =========================
# 2. 计算四个象限的平均 recall
# =========================
q1_mean_recall = q1_df['recall'].mean()
q2_mean_recall = q2_df['recall'].mean()
q3_mean_recall = q3_df['recall'].mean()
q4_mean_recall = q4_df['recall'].mean()

print(f"Quadrant I   mean recall = {q1_mean_recall:.4f}")
print(f"Quadrant II  mean recall = {q2_mean_recall:.4f}")
print(f"Quadrant III mean recall = {q3_mean_recall:.4f}")
print(f"Quadrant IV  mean recall = {q4_mean_recall:.4f}")

# 如果你还想看每个象限有多少样本
total_samples = len(df)

print(
    f"Quadrant I   sample count = {len(q1_df)} "
    f"({len(q1_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant II  sample count = {len(q2_df)} "
    f"({len(q2_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant III sample count = {len(q3_df)} "
    f"({len(q3_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant IV  sample count = {len(q4_df)} "
    f"({len(q4_df) / total_samples * 100:.2f}%)"
)

# ================================
# 3. 在图上写每个象限的 mean recall
# ================================
ax.text(
    0.18, 0.12,
    f'I\nRecall={q1_mean_recall:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.68, 0.12,
    f'II\nRecall={q2_mean_recall:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.68, 0.82,
    f'III\nRecall={q3_mean_recall:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.18, 0.82,
    f'IV\nRecall={q4_mean_recall:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)


# ================================
# X,Y轴, label; legend
# ================================

plt.xlabel('Label-1 proportion', fontsize=16)
plt.ylabel('Peak_amount', fontsize=16)

plt.title(
    'Epoch 24: Mean Recall Across Label-1 proportion and Peak_amount',
    fontsize=20,
    fontweight='bold'
)

legend_elements = [
    # Validation-set mean recall, 平均recall
    Patch(
        facecolor='white',
        edgecolor='gray',
        label=f'Validation-set mean recall = {dataset_mean_recall:.3f}'
    ),

    # 红框
    Patch(
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        label=f'Cell mean recall < dataset mean ({dataset_mean_recall:.3f})'
    ),

    # X 标记
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='black',
        markersize=10,
        markeredgewidth=2,
        label='Sample count ≥ 1500'
    )
]

ax.legend(
    handles=legend_elements,
    # Location: 把 legend 的右上角放在坐标轴内部的 (0.98, 0.98)
    loc='upper right',           # legend 的右上角
    bbox_to_anchor=(0.98, 0.98), # 坐标轴内部的 (0.98, 0.98)
    bbox_transform=ax.transAxes,
    # Font
    fontsize=12,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()


#%% Precision Heatmap: red box; "X": sample count >= 1500
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 整个 dataset 的平均 recall
dataset_mean_precision = df['precision'].mean()
print(f'Validation-set mean precision: {dataset_mean_precision:.3f}')

# ratio 分箱
ratio_bins = pd.cut(
    df['ratio'],
    bins=np.linspace(0, 1, 21),
    include_lowest=True
)

# seq_len 分箱
seq_bins = pd.cut(
    df['seq_len'],
    bins=np.linspace(0, 200, 21),
    include_lowest=True
)

# 计算每个格子的平均 precision
precision_table = pd.pivot_table(
    df,
    values='precision',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# 计算每个格子的 sample 数量
count_table = pd.pivot_table(
    df,
    values='precision',   # 用任意非空列都可以
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='count'
)

# seq_len 倒序显示
precision_table = precision_table.iloc[::-1] # 每个格子的平均 recall
count_table  = count_table.iloc[::-1]   # 每个格子的 sample 数量

plt.figure(figsize=(14, 10))

ax = sns.heatmap(
    precision_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label': 'Mean Precision'}
)

for row_idx in range(precision_table.shape[0]):
    for col_idx in range(precision_table.shape[1]):

        cell_precision = precision_table.iloc[row_idx, col_idx]
        cell_count = count_table.iloc[row_idx, col_idx]

        # 排除空格子，并标记低于 dataset mean recall 的格子
        if pd.notna(cell_precision) and cell_precision < dataset_mean_precision:
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5
                )
            )

        # sample 数量 >= 1500 的格子打上 X
        if pd.notna(cell_count) and cell_count >= 1500:
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

# ratio = 0.2 的竖线
ax.axvline(
    x=4,
    color='black',
    linestyle='--',
    linewidth=2.5,
    zorder=10
)

# sequence length = 70 的横线
ax.axhline(
    y=13,
    color='black',
    linestyle='--',
    linewidth=2.5,
    zorder=10
)

# =========================
# 1. 定义四个象限
# =========================
q1_df = df[(df['ratio'] <= 0.2) & (df['seq_len'] <= 70)]   # I 左下
q2_df = df[(df['ratio'] > 0.2)  & (df['seq_len'] <= 70)]   # II 右下
q3_df = df[(df['ratio'] > 0.2)  & (df['seq_len'] > 70)]    # III 右上
q4_df = df[(df['ratio'] <= 0.2) & (df['seq_len'] > 70)]    # IV 左上

# =========================
# 2. 计算四个象限的平均 precision
# =========================
q1_mean_precision = q1_df['precision'].mean()
q2_mean_precision = q2_df['precision'].mean()
q3_mean_precision = q3_df['precision'].mean()
q4_mean_precision = q4_df['precision'].mean()

print(f"Quadrant I   mean precision = {q1_mean_precision:.4f}")
print(f"Quadrant II  mean precision = {q2_mean_precision:.4f}")
print(f"Quadrant III mean precision = {q3_mean_precision:.4f}")
print(f"Quadrant IV  mean precision = {q4_mean_precision:.4f}")
print("--------------------------------------------------------")
# 如果你还想看每个象限有多少样本
total_samples = len(df)

print(
    f"Quadrant I   sample count = {len(q1_df)} "
    f"({len(q1_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant II  sample count = {len(q2_df)} "
    f"({len(q2_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant III sample count = {len(q3_df)} "
    f"({len(q3_df) / total_samples * 100:.2f}%)"
)

print(
    f"Quadrant IV  sample count = {len(q4_df)} "
    f"({len(q4_df) / total_samples * 100:.2f}%)"
)

# ================================
# 3. 在图上写每个象限的 mean precision
# ================================
ax.text(
    0.18, 0.12,
    f'I\nPrecision={q1_mean_precision:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.68, 0.12,
    f'II\nPrecision={q2_mean_precision:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.68, 0.82,
    f'III\nPrecision={q3_mean_precision:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

ax.text(
    0.18, 0.82,
    f'IV\nPrecision={q4_mean_precision:.3f}',
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=14,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)

# ================================
# X,Y轴, label; legend
# ================================
plt.xlabel('Label-1 proportion', fontsize=16)
plt.ylabel('Peak_amount', fontsize=16)

plt.title(
    'Epoch 24: Mean Precision Across Label-1 proportion and Peak amount',
    fontsize=20,
    fontweight='bold'
)

legend_elements = [
    # Validation-set mean precision, 平均precision
    Patch(
        facecolor='white',
        edgecolor='gray',
        label=f'Validation-set mean precision = {dataset_mean_precision:.3f}'
    ),

    # 红框
    Patch(
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        label=f'Cell mean precision < dataset mean ({dataset_mean_precision:.3f})'
    ),

    # X 标记
    Line2D(
        [0], [0],
        marker='x',
        linestyle='None',
        color='black',
        markersize=10,
        markeredgewidth=2,
        label='Sample count ≥ 1500'
    )
]

ax.legend(
    handles=legend_elements,
    # Location: 把 legend 的右上角放在坐标轴内部的 (0.98, 0.98)
    loc='upper right',           # legend 的右上角
    bbox_to_anchor=(0.98, 0.98), # 坐标轴内部的 (0.98, 0.98)
    bbox_transform=ax.transAxes,
    # Font
    fontsize=12,
    frameon=True,
    ncol=1
)

plt.tight_layout()
plt.show()
