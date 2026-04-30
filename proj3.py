import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --------------------------
# 1. 数据预处理模块
# --------------------------
def load_and_preprocess(file_path):
    # 加载数据集
    df = pd.read_csv(file_path)

    # 如果 CSV 已经包含 date 列，直接解析为时间索引
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('date', inplace=True)
        df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)

    # 填充缺失值
    # 目标值 pollution 或 pm2.5 若缺失则删除该行，保证标签真实
    target_col = 'pm2.5' if 'pm2.5' in df.columns else 'pollution'
    df.dropna(subset=[target_col], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 风向特征独热编码 (One-Hot Encoding)
    if 'wnd_dir' in df.columns:
        df = pd.get_dummies(df, columns=['wnd_dir'])
    elif 'cbwd' in df.columns:
        df = pd.get_dummies(df, columns=['cbwd'])

    return df

def create_dataset(data, window_size=24):
    """
    构造滑动窗口数据集
    X: (N, 24, features), y: (N, 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :])
        # 假设 pm2.5 是数据表的第一列
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# --------------------------
# 2. LSTM 模型定义
# --------------------------
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(AirQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True 使输入维度为 [batch, seq_len, feature]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 仅取最后一个时间步的输出
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --------------------------
# 3. 训练与评估主流程
# --------------------------
def main():
    # 数据加载 (请确保文件名正确)
    # 原始数据集下载后通常命名为 raw.csv 或 PRSA_data_...
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'LSTM-Multivariate_pollution.csv')
    try:
        data_df = load_and_preprocess(data_path)
    except Exception as e:
        print("加载数据失败：", e)
        return

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_df.values)

    # 构造时序样本
    window_size = 24
    X, y = create_dataset(scaled_data, window_size)

    # 划分训练集与测试集 (8:2)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为 PyTorch 张量
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=False
    )

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AirQualityLSTM(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 30
    losses = []
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    # --------------------------
    # 4. 结果预测与逆归一化
    # --------------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    # 逆归一化 (仅针对 PM2.5 列)
    # 注意：逆归一化需要构造一个和原始特征数相同的矩阵
    prediction_copies = np.repeat(y_pred, scaled_data.shape[1], axis=1)
    y_pred_inv = scaler.inverse_transform(prediction_copies)[:, 0]

    test_actual_copies = np.repeat(y_test.reshape(-1, 1), scaled_data.shape[1], axis=1)
    y_test_inv = scaler.inverse_transform(test_actual_copies)[:, 0]

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"\n测试集评估结果:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # --------------------------
    # 5. 可视化
    # --------------------------
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    # 预测对比
    plt.subplot(1, 2, 2)
    plt.plot(y_test_inv[:200], label='True', alpha=0.7)
    plt.plot(y_pred_inv[:200], label='Pred', alpha=0.7)
    plt.title('PM2.5 Prediction (First 200 hours)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()