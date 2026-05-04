import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_paths(S0, mu, sigma, T, N_steps, N_paths):
    """
    几何布朗运动 (GBM) 仿真引擎
    S0: 初始价格, mu: 年化漂移率, sigma: 年化波动率
    T: 总时长(年), N_steps: 时间步数, N_paths: 模拟路径数
    """
    dt = T / N_steps
    # 生成标准正态分布随机数矩阵 (步数, 路径数)
    Z = np.random.standard_normal((N_steps, N_paths))
    
    # 初始化价格矩阵 (步数+1, 路径数)
    S = np.zeros((N_steps + 1, N_paths))
    S[0] = S0
    
    # 利用矩阵运算加速路径生成
    # S_t = S_{t-1} * exp((mu - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z)
    for t in range(1, N_steps + 1):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
        
    return S


# 设置仿真参数
days = 63            # 模拟未来 3 个月（约 63 个交易日）
years = days / 252   # 转化为年化时间
num_paths = 50       # 生成 50 条随机路径

# 加载参数
df_params = pd.read_csv("calibrated_parameters.csv", index_col=0)
    
# 选取股票进行仿真
tickers = df_params.index
for ticker in tickers:
    p = df_params.loc[ticker]
    
    paths = generate_gbm_paths(
        S0=p['S0'], 
        mu=p['Mu'], 
        sigma=p['Sigma'], 
        T=years, 
        N_steps=days, 
        N_paths=num_paths
    )
    
    # 4. 可视化
    plt.figure(figsize=(12, 6))

    # 将价格除以初始价格，转化为百分比路径
    normalized_paths = paths / p['S0']
    plt.plot(normalized_paths, linewidth=1, alpha=0.6)
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Start (100%)')
    plt.ylabel("Simulated Normalized Price (S_t / S_0)")


    # plt.plot(paths, linewidth=1, alpha=0.7)
    # plt.axhline(p['S0'], color='black', linestyle='--', label='Initial Price')
    plt.title(f"Monte Carlo Simulation: {ticker} ({days} Days, {num_paths} Paths)")
    plt.xlabel("Trading Days")
    # plt.ylabel("Simulated Price")
    plt.grid(True, alpha=0.3)
    
    filename = f"{ticker}_GBM_paths.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close()