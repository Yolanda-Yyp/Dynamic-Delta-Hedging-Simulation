import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSM_Engine import BSM_Engine
from GBM_Simulation import generate_gbm_paths

def run_hedging_simulation(paths, K, T_total, r, sigma, fee_rate, hedge_freq=1):
    """
    执行批量动态对冲仿真
    paths: GBM 生成的价格路径矩阵 (N_steps + 1, N_paths)
    hedge_freq: 调仓频率 (1 = Daily, 5 = Weekly)
    """
    N_steps = paths.shape[0] - 1
    N_paths = paths.shape[1]
    dt = T_total / N_steps
    
    # === 第 0 天：初始建仓 ===
    S0 = paths[0, :] # 取出所有路径的第一天价格
    
    # 卖出期权收到权利金
    C0 = BSM_Engine.calculate_price(S0, K, T_total, r, sigma, "call")
    
    # 计算需要买入的初始股票份额 (Delta)
    delta_old = BSM_Engine.calculate_delta(S0, K, T_total, r, sigma, "call")
    
    # 买入股票并扣除手续费
    stock_cost = delta_old * S0
    fee_accumulated = stock_cost * fee_rate
    
    # 初始化现金账户
    bank = C0 - stock_cost - fee_accumulated
    
    # === 第 1 天 到 到期前 1 天：动态循环 ===
    for t in range(1, N_steps):
        S_t = paths[t, :]
        T_remaining = T_total - t * dt
        
        # 现金账户隔夜产生利息 (复利)
        bank = bank * np.exp(r * dt)
        
        # 判断今天是否需要调仓 (比如 hedge_freq=5 就是每 5 天调仓一次)
        if t % hedge_freq == 0:
            delta_new = BSM_Engine.calculate_delta(S_t, K, T_remaining, r, sigma, "call")
            
            # 计算需要买卖的差额
            trade_shares = delta_new - delta_old
            trade_value = trade_shares * S_t
            
            # 扣除手续费 (买卖都要交钱，取绝对值)
            daily_fee = np.abs(trade_value) * fee_rate
            fee_accumulated += daily_fee
            
            # 更新账户
            bank = bank - trade_value - daily_fee
            delta_old = delta_new
            
    # === 到期日 (Maturity)：结算 ===
    S_T = paths[-1, :]
    bank = bank * np.exp(r * dt) # 最后一天的利息
    
    # 1. 清仓卖出所有股票
    liquidation_value = delta_old * S_T
    liquidation_fee = liquidation_value * fee_rate
    fee_accumulated += liquidation_fee
    bank = bank + liquidation_value - liquidation_fee
    
    # 2. 结算期权 (客户行权则扣款)
    option_payoff = np.maximum(S_T - K, 0)
    bank = bank - option_payoff
    
    # 最终的 PnL 就是现金账户的余额
    return bank, fee_accumulated

if __name__ == "__main__":
    print("🚀 启动 Week 2: 全球三巨头动态对冲仿真引擎...")
    
    # 1. 加载所有真实的校准参数
    df = pd.read_csv("calibrated_parameters.csv", index_col=0)
    
    # 2. 设定全场通用的实验环境
    days = 63
    T = days / 252
    r = 0.04
    fee = 0.002        # 设定千分之二的手续费
    N_paths = 1000     # 每只股票生成 1000 个平行宇宙
    
    # ⭐ 新增：创建一个空列表，用来收集所有股票的对冲结果
    results_list = []
    
    # 3. 开启批量测试循环
    for ticker in df.index:
        p = df.loc[ticker]
        K = p['S0'] # 设定为平值期权
        
        # 固定随机数种子，保证实验的可重复性
        np.random.seed(42) 
        
        # 针对当前股票生成专属路径
        paths = generate_gbm_paths(p['S0'], p['Mu'], p['Sigma'], T, days, N_paths)
        
        # 跑每日调仓
        pnl_daily, fees_daily = run_hedging_simulation(paths, K, T, r, p['Sigma'], fee, hedge_freq=1)
        # 跑每周调仓
        pnl_weekly, fees_weekly = run_hedging_simulation(paths, K, T, r, p['Sigma'], fee, hedge_freq=5)
        
        # 4. 打印这只股票的专属成绩单 (保留控制台输出，方便实时查看)
        print("-" * 55)
        print(f"📊 {ticker} (Sigma: {p['Sigma']*100:.2f}%) 对冲性能分析 (样本: {N_paths})")
        print("-" * 55)
        print("【每日对冲 Daily (Freq=1)】")
        print(f"平均最终 PnL:     {pnl_daily.mean():.4f}")
        print(f"PnL 标准差 (风险): {pnl_daily.std():.4f}")
        print(f"平均累计手续费:   {fees_daily.mean():.4f}")
        
        print("\n【每周对冲 Weekly (Freq=5)】")
        print(f"平均最终 PnL:     {pnl_weekly.mean():.4f}")
        print(f"PnL 标准差 (风险): {pnl_weekly.std():.4f}")
        print(f"平均累计手续费:   {fees_weekly.mean():.4f}\n")
        
        # ⭐ 新增：将计算结果打包成字典，存入列表中
        results_list.append({
            "Ticker": ticker,
            "Volatility_Sigma": round(p['Sigma'], 4),
            "Daily_Mean_PnL": round(pnl_daily.mean(), 4),
            "Daily_PnL_StdDev": round(pnl_daily.std(), 4),
            "Daily_Mean_Fee": round(fees_daily.mean(), 4),
            "Weekly_Mean_PnL": round(pnl_weekly.mean(), 4),
            "Weekly_PnL_StdDev": round(pnl_weekly.std(), 4),
            "Weekly_Mean_Fee": round(fees_weekly.mean(), 4)
        })
        
    # ⭐ 新增：循环结束后，将列表转换为 DataFrame 并导出为 CSV
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("Hedging_Results.csv", index=False)
    
    print("✅ 所有实验跑完！")
    print("📁 核心实证对比数据已成功保存至: Hedging_Results.csv")