import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

#  1. 数据加载与检查
file_path = 'Data4Regression.xlsx'
if not os.path.exists(file_path):
    print(f"❌ 错误：找不到文件 {file_path}", flush=True)
    exit()

try:
    train_df = pd.read_excel(file_path, sheet_name=0)
    test_df = pd.read_excel(file_path, sheet_name=1)
    X_train = train_df.iloc[:, 0].values.reshape(-1, 1)
    y_train = train_df.iloc[:, 1].values
    X_test  = test_df.iloc[:, 0].values.reshape(-1, 1)
    y_test  = test_df.iloc[:, 1].values
    print("✅ 数据加载成功！", flush=True)
except Exception as e:
    print(f"❌ 读取失败: {e}", flush=True)
    exit()

#2. 定义 7 种模型框架
models = {
    "Linear (OLS)": LinearRegression(),
    "Polynomial (Deg=6)": make_pipeline(PolynomialFeatures(6), LinearRegression()),
    "Bayesian Poly": make_pipeline(PolynomialFeatures(6), BayesianRidge()),
    "Kernel Ridge (RBF)": KernelRidge(kernel='rbf', alpha=0.1, gamma=1.0),
    "SVR (RBF)": SVR(kernel='rbf', C=1e3, gamma=0.1),
    "Original MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42),
    "Optimized MLP": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.005, max_iter=10000, random_state=42)
    )
}

# 3. 训练与预测
X_plot = np.linspace(min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max()), 1000)[:, None]
results = []
plot_preds = {}

print("正在训练所有模型...", flush=True)
for name, model in models.items():
    model.fit(X_train, y_train)
    # 存储绘图数据
    plot_preds[name] = model.predict(X_plot)
    # 计算指标
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    results.append({
        "name": name,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    })

#4. 可视化对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))


colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

#Training Set
ax1.scatter(X_train, y_train, color='black', s=20, alpha=0.3, label='Train Data Points')
for (name, y_p), color in zip(plot_preds.items(), colors):
    ax1.plot(X_plot, y_p, label=name, color=color, linewidth=1.5)
ax1.set_title("Training Set: All 7 Models Fitting", fontsize=15)
ax1.legend(loc='upper left', fontsize='small', ncol=1)
ax1.grid(True, alpha=0.3)

#Test Set
ax2.scatter(X_test, y_test, color='red', marker='x', s=30, alpha=0.5, label='Test Data Points')
for (name, y_p), color in zip(plot_preds.items(), colors):
    ax2.plot(X_plot, y_p, label=name, color=color, linewidth=1.5)
ax2.set_title("Test Set: All 7 Models Generalization", fontsize=15)
ax2.legend(loc='upper left', fontsize='small', ncol=1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("\n" + "="*95)
header = f"{'Algorithm':<22} | {'Train MSE':<12} | {'Test MSE':<12} | {'Train R2':<10} | {'Test R2':<10}"
print(header)
print("-" * 95)
for res in results:
    print(f"{res['name']:<22} | {res['train_mse']:<12.5f} | {res['test_mse']:<12.5f} | {res['train_r2']:<10.4f} | {res['test_r2']:<10.4f}")
print("="*95)