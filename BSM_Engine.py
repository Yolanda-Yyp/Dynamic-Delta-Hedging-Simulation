import numpy as np
from scipy.stats import norm

class BSM_Engine:
    """
    Black-Scholes-Merton 定价与希腊字母计算引擎
    """
    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type="call"):
        """
        计算期权的 Delta 值
        S: 当前价格, K: 行权价, T: 到期剩余时间(年), r: 无风险利率, sigma: 波动率
        """
        if T <= 0: # 已经到期
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    @staticmethod
    def calculate_price(S, K, T, r, sigma, option_type="call"):
        """
        计算期权理论价格
        """
        if T <= 0:
            return max(0, S - K) if option_type == "call" else max(0, K - S)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

# --- 测试演示 ---
if __name__ == "__main__":
    # 模拟一个典型的平值期权场景
    S_test = 100    # 当前股价
    K_test = 100    # 行权价 (平值 ATM)
    T_test = 0.25   # 3个月到期
    r_test = 0.04   # 4% 无风险利率
    v_test = 0.30   # 30% 年化波动率
    
    price = BSM_Engine.calculate_price(S_test, K_test, T_test, r_test, v_test)
    delta = BSM_Engine.calculate_delta(S_test, K_test, T_test, r_test, v_test)
    
    print("-" * 30)
    print("BSM 引擎单元测试 (平值看涨期权)")
    print("-" * 30)
    print(f"理论价格: {price:.4f}")
    print(f"Delta 值: {delta:.4f} (意味着对冲 1 份期权需要买入 {delta:.2f} 份股票)")