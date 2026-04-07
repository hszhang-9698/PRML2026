import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 设置中文支持 (非常关键)
# 尝试查找系统中常用的中文字体。如果不成功，请根据你的系统配置手动设置字体。
# 常见的 Windows 字体：'SimHei', 'SimSun'
# 常见的 macOS 字体：'Heiti TC', 'PingFang SC'
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'DejaVu Sans'] # 尝试按顺序加载
mpl.rcParams['axes.unicode_minus'] = False # 处理负号显示问题

# 2. 定义数据生成函数 (基于你的要求)
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    # 拼接两类数据 (C0 和 C1)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 添加高斯噪声
    np.random.seed(42) # 设置随机种子以获得可复现的结果
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y_labels

# 3. 生成训练集 (1000条) 和 测试集 (500条)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2) # 500*2 = 1000
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)   # 250*2 = 500

# 4. 数据预处理 (对SVM和特定可视化至关重要)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 定义并训练要可视化的 SVM 分类器
classifiers = {
    "RBF-SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Poly-SVM": SVC(kernel='poly', degree=3, C=1.0)
}

# 6. 用于可视化的专门绘图函数
def plot_classification_results(ax, X, y_true, y_pred, title):
    """
    在指定的坐标轴上绘制3D分类结果散点图，


    """
    # 找到不同类型的点
    correct_c0 = (y_true == 0) & (y_pred == 0)
    correct_c1 = (y_true == 1) & (y_pred == 1)
    incorrect = (y_true != y_pred)

    # 绘制正确分类的点 (半透明圆点)
    ax.scatter(X[correct_c0, 0], X[correct_c0, 1], X[correct_c0, 2], c='blue', marker='o', s=10, alpha=0.6)
    ax.scatter(X[correct_c1, 0], X[correct_c1, 1], X[correct_c1, 2], c='red', marker='o', s=10, alpha=0.6)

    # 绘制错误分类的点 (突出显示的黄色星号)
    ax.scatter(X[incorrect, 0], X[incorrect, 1], X[incorrect, 2], c='gold', marker='*', s=40, alpha=1.0, edgecolors='black')

    # 坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围以匹配示例 (取决于你的数据缩放)
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.25)

    # 设置内部标题 (使用中文)
    ax.set_title(title, fontsize=12, pad=15)

    # 创建自定义中文图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='C0 分类正确'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='C1 分类正确'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=10, markeredgecolor='black', label='分类错误')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="分类结果", fontsize=10, title_fontsize=11)

    # 调整视角以获得最佳3D效果 (与示例一致)
    ax.view_init(elev=20, azim=110)

# 7. 创建主绘图布局 (2x1的3D子图)
fig = plt.figure(figsize=(10, 16))

# ---- 子图1: SVM RBF 核 ----
ax1 = fig.add_subplot(211, projection='3d')
# 训练 RBF 分类器
clf_rbf = classifiers["RBF-SVM"]
clf_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = clf_rbf.predict(X_test_scaled)
# 绘图
plot_classification_results(ax1, X_test_scaled, y_test, y_pred_rbf, "SVM RBF")

# ---- 子图2: SVM 多项式核 ----
ax2 = fig.add_subplot(212, projection='3d')
# 训练多项式分类器
clf_poly = classifiers["Poly-SVM"]
clf_poly.fit(X_train_scaled, y_train)
y_pred_poly = clf_poly.predict(X_test_scaled)
# 绘图
plot_classification_results(ax2, X_test_scaled, y_test, y_pred_poly, "SVM ")

# 8. 添加外部中文文本标签 (完全一致)
# 文本： "图 4: SVM_RBF 核分类结果图"
fig.text(0.5, 0.52, "SVM_RBF", ha='center', fontsize=14, fontfamily='serif')
# 文本： "图 5: SVM_多项式核分类结果图"
fig.text(0.5, 0.04, "SVM_多项式", ha='center', fontsize=14, fontfamily='serif')
# 文本： "分类正确"
fig.text(0.5, 0.02, "分类正确", ha='center', fontsize=12)

# 调整子图布局以适应外部文本
plt.tight_layout(pad=6.0)

# 显示图像
plt.show()

# ---- 额外的：如果你还需要计算性能并打印结果 ----
print("\n---- 分类性能评估 ----")
## 重新定义所有分类器 (包括你之前提到的决策树和AdaBoost)
all_classifiers = {
    "Decision Tree (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "AdaBoost (DT)": AdaBoostClassifier(n_estimators=50, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', C=1.0),
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "SVM (Poly)": SVC(kernel='poly', degree=3, C=1.0)
}

print(f"{'算法':<20} | {'准确率':<10}")
print("-" * 35)

for name, clf in all_classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<20} | {acc:.4f}")