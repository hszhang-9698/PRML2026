import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 设置中文支持
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'DejaVu Sans'] # 尝试按顺序加载
mpl.rcParams['axes.unicode_minus'] = False # 处理负号显示问题

# 2. 定义数据生成函数
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
X_train_raw, y_train = make_moons_3d(n_samples=500, noise=0.2) # 500*2 = 1000
X_test_raw, y_test = make_moons_3d(n_samples=250, noise=0.2)   # 250*2 = 500

# 4. 数据预处理 (对SVM和特定可视化至关重要)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 5. 定义所有分类器
classifiers = [
    ("Decision Tree (depth=10)", DecisionTreeClassifier(max_depth=10, random_state=42)),
    # 这里AdaBoost使用默认的浅层决策树基分类器
    ("AdaBoost (DT)", AdaBoostClassifier(n_estimators=50, random_state=42)),
    ("SVM (Linear)", SVC(kernel='linear', C=1.0)),
    ("SVM (RBF)", SVC(kernel='rbf', C=1.0, gamma='scale')),
    ("SVM (Poly)", SVC(kernel='poly', degree=3, C=1.0))
]

# 训练并记录预测结果和准确率
results = []
print(f"{'Algorithm':<25} | {'Accuracy':<10}")
print("-" * 40)

for name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, y_pred, acc))
    print(f"{name:<25} | {acc:.4f}")

# 6. 用于可视化的专门绘图函数
def plot_classification_results(ax, X, y_true, y_pred, title):
    """
    在指定的坐标轴上绘制3D分类结果散点图，
    使用蓝色圆点表示C0正确，红色圆点表示C1正确，黄色星号表示分类错误。
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

# 7. 创建主绘图布局 (2x3的3D子图)
fig = plt.figure(figsize=(15, 20))

# 循环绘制五种算法的结果
for i, (name, y_pred, acc) in enumerate(results):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    # 格式化中文标题
    title_cn = f"{name} 分类性能可视化 (准确率: {acc:.4f})"
    plot_classification_results(ax, X_test, y_test, y_pred, title_cn)

# 隐藏最后一个多余的子图
fig.delaxes(fig.add_subplot(2, 3, 6, projection='3d'))

# 8. 添加外部中文文本标签 (完全一致)
fig.suptitle("3D Make Moons", fontsize=18, y=0.98, fontfamily='serif')

# 文本： "图 [X]: [算法名称] 分类结果图" 和 "分类正确"
fig.text(0.5, 0.50, f"AdaBoost", ha='center', fontsize=14, fontfamily='serif')
fig.text(0.5, 0.03, f"SVM ", ha='center', fontsize=14, fontfamily='serif')

# 调整子图布局以适应外部文本和标题
plt.tight_layout(pad=6.0, rect=[0, 0.05, 1, 0.96])

# 显示图像
plt.show()