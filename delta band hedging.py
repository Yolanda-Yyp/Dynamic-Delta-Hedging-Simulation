import numpy as np
import pandas as pd
from BSM_Engine import BSM_Engine
from GBM_Simulation import generate_gbm_paths

def run_hedging_simulation(paths, K, T_total, r, sigma, fee_rate, hedge_freq=None, delta_band=None):
    """
    升级版：支持按频率 (hedge_freq) 或 按阈值 (delta_band) 调仓
    """
    N_steps = paths.shape[0] - 1
    N_paths = paths.shape[1]
    dt = T_total / N_steps
    
    # === 第 0 天：初始建仓 ===
    S0 = paths[0, :] 
    C0 = BSM_Engine.calculate_price(S0, K, T_total, r, sigma, "call")
    delta_old = BSM_Engine.calculate_delta(S0, K, T_total, r, sigma, "call")
    
    stock_cost = delta_old * S0
    fee_accumulated = stock_cost * fee_rate
    bank = C0 - stock_cost - fee_accumulated
    
    # 统计交易次数 (用于评估活跃度)
    trade_count = np.ones(N_paths) 
    
    # === 动态调仓期 ===
    for t in range(1, N_steps):
        S_t = paths[t, :]
        T_remaining = T_total - t * dt
        bank = bank * np.exp(r * dt)
        
        # 每天都计算一下理论上新的 Delta 是多少
        delta_new = BSM_Engine.calculate_delta(S_t, K, T_remaining, r, sigma, "call")
        
        # 策略 1: 阈值对冲 (Delta-Band)
        if delta_band is not None:
            # 判断每条路径的 Delta 偏移量是否超过了阈值
            trade_mask = np.abs(delta_new - delta_old) > delta_band
            
            # 只有触发阈值的路径才进行交易计算，否则交易量为 0
            trade_shares = np.where(trade_mask, delta_new - delta_old, 0.0)
            trade_value = trade_shares * S_t
            daily_fee = np.abs(trade_value) * fee_rate
            
            # 更新账户和计数器
            fee_accumulated += daily_fee
            bank = bank - trade_value - daily_fee
            trade_count += np.where(trade_mask, 1, 0)
            
            # 只有触发交易的路径，才更新它的 delta_old
            delta_old = np.where(trade_mask, delta_new, delta_old)
            
        # 策略 2: 定期对冲 (Daily/Weekly)
        elif hedge_freq is not None:
            if t % hedge_freq == 0:
                trade_shares = delta_new - delta_old
                trade_value = trade_shares * S_t
                daily_fee = np.abs(trade_value) * fee_rate
                
                fee_accumulated += daily_fee
                bank = bank - trade_value - daily_fee
                trade_count += 1
                delta_old = delta_new

    # === 到期日结算 ===
    S_T = paths[-1, :]
    bank = bank * np.exp(r * dt) 
    
    liquidation_value = delta_old * S_T
    liquidation_fee = liquidation_value * fee_rate
    fee_accumulated += liquidation_fee
    trade_count += 1
    
    bank = bank + liquidation_value - liquidation_fee
    option_payoff = np.maximum(S_T - K, 0)
    bank = bank - option_payoff
    
    return bank, fee_accumulated, trade_count


if __name__ == "__main__":
    print("🚀 启动终极实验：Daily vs Weekly vs Delta-Band...")
    
    df = pd.read_csv("calibrated_parameters.csv", index_col=0)
    days, T, r, fee, N_paths = 63, 63/252, 0.04, 0.002, 1000
    results_list = []
    
    for ticker in df.index:
        p = df.loc[ticker]
        K = p['S0']
        np.random.seed(42) 
        paths = generate_gbm_paths(p['S0'], p['Mu'], p['Sigma'], T, days, N_paths)
        
        # 跑三种策略
        pnl_d, fees_d, trades_d = run_hedging_simulation(paths, K, T, r, p['Sigma'], fee, hedge_freq=1)
        pnl_w, fees_w, trades_w = run_hedging_simulation(paths, K, T, r, p['Sigma'], fee, hedge_freq=5)
        # ⭐ 新增：跑阈值对冲实验，设定容忍度为 0.05
        pnl_b, fees_b, trades_b = run_hedging_simulation(paths, K, T, r, p['Sigma'], fee, delta_band=0.05)
        
        print("-" * 55)
        print(f"📊 {ticker} 终极对冲性能对比")
        print(f"【Daily】 Mean PnL: {pnl_d.mean():.4f} | Risk: {pnl_d.std():.4f} | 交易次数: {trades_d.mean():.1f}")
        print(f"【Weekly】Mean PnL: {pnl_w.mean():.4f} | Risk: {pnl_w.std():.4f} | 交易次数: {trades_w.mean():.1f}")
        print(f"【Band】  Mean PnL: {pnl_b.mean():.4f} | Risk: {pnl_b.std():.4f} | 交易次数: {trades_b.mean():.1f}")
        
        results_list.append({
            "Ticker": ticker,
            "Strategy": "Daily",
            "Mean_PnL": round(pnl_d.mean(), 4),
            "Risk_StdDev": round(pnl_d.std(), 4),
            "Mean_Fee": round(fees_d.mean(), 4),
            "Avg_Trades": round(trades_d.mean(), 1)
        })
        results_list.append({
            "Ticker": ticker,
            "Strategy": "Weekly",
            "Mean_PnL": round(pnl_w.mean(), 4),
            "Risk_StdDev": round(pnl_w.std(), 4),
            "Mean_Fee": round(fees_w.mean(), 4),
            "Avg_Trades": round(trades_w.mean(), 1)
        })
        results_list.append({
            "Ticker": ticker,
            "Strategy": "Delta-Band (0.05)",
            "Mean_PnL": round(pnl_b.mean(), 4),
            "Risk_StdDev": round(pnl_b.std(), 4),
            "Mean_Fee": round(fees_b.mean(), 4),
            "Avg_Trades": round(trades_b.mean(), 1)
        })
        
    pd.DataFrame(results_list).to_csv("Ultimate_Hedging_Results.csv", index=False)
    print("\n✅ 终极实验跑完！数据已保存至 Ultimate_Hedging_Results.csv")