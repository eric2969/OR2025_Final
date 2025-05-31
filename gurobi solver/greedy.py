import pandas as pd
import sys, time, os

# === 全域參數 ===
μ = 6        # 處理速率（輛/分鐘）
L = 20       # 每台卡車最多載車數
T_num = 30   # 卡車數
max_dispatch = T_num * L
location = sys.argv[1]
formatted_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
if not os.path.exists("results"):
    os.makedirs("results")

# === 讀取資料 ===
df = pd.read_csv(f"assets/gurobi_demand_table_{location}.csv")
times = sorted(df["interval_time"].unique())
stations = sorted(df["sno"].unique())
sna_map = df.drop_duplicates("sno").set_index("sno")["sna"].to_dict()

# 站點資訊
capacity = df.drop_duplicates("sno").set_index("sno")["total"].to_dict()
B = {s: int(0.35 * capacity[s]) for s in stations}  # 初始車量

# 儲存結果
dispatch_result = []
hide_result = []

# === 每期處理 ===
for t in times:
    cur = df[df["interval_time"] == t]
    
    # 分為缺車（要補）與溢車（要收）
    need = []
    surplus = []
    for _, row in cur.iterrows():
        b_need = row.demand_borrow - B[row.sno]
        r_over = row.demand_return - (capacity[row.sno] - B[row.sno])
        if b_need > 0:
            need.append((row.sno, b_need))
        if r_over > 0:
            surplus.append((row.sno, r_over))
    
    # 依據需求排序
    need.sort(key=lambda x: -x[1])
    surplus.sort(key=lambda x: -x[1])
    
    remaining_dispatch = max_dispatch

    for s_from, extra in surplus:
        if remaining_dispatch <= 0:
            break
        for i, (s_to, shortage) in enumerate(need):
            if remaining_dispatch <= 0:
                break
            move = min(extra, shortage, remaining_dispatch)
            if move <= 0:
                continue
            # 更新現況
            B[s_from] -= move
            B[s_to] += move
            extra -= move
            need[i] = (s_to, shortage - move)
            remaining_dispatch -= move
            dispatch_result.append(dict(
                time=t,
                from_sno=s_from,
                from_sna=sna_map[s_from],
                to_sno=s_to,
                to_sna=sna_map[s_to],
                quantity=move
            ))

    # 貪婪藏車／釋放
    for _, row in cur.iterrows():
        sno = row.sno
        # 如果退車仍溢出，藏車
        over = row.demand_return - (capacity[sno] - B[sno])
        if over > 0:
            hide = min(over, int(0.4 * capacity[sno]))
            B[sno] -= hide
            hide_result.append(dict(
                time=t,
                sno=sno,
                sna=sna_map[sno],
                hide=hide,
                release=0
            ))
        # 如果借車仍不足，釋放
        lack = row.demand_borrow - B[sno]
        if lack > 0:
            release = min(lack, int(0.4 * capacity[sno]))
            B[sno] += release
            hide_result.append(dict(
                time=t,
                sno=sno,
                sna=sna_map[sno],
                hide=0,
                release=release
            ))

# === 輸出結果 ===
pd.DataFrame(dispatch_result).to_csv(f"./result/greedy_dispatch-{location}_{formatted_time}.csv", index=False)
pd.DataFrame(hide_result).to_csv(f"./result/greedy_hide-{location}_{formatted_time}.csv", index=False)
print("✅ 貪婪演算法結果已輸出為 CSV 檔案")

# === 成本估算（基於貪婪法結果） ===
μ = 6
α = 1
β = 0.04

wait_cost = 0
dispatch_cost = 0
hide_cost = 0

# 計算等待成本
for t in times:
    cur = df[df["interval_time"] == t]
    for _, row in cur.iterrows():
        sno = row.sno
        cap = capacity[sno]
        b = B[sno]
        wait_borrow = max(row.demand_borrow - b, 0) / μ
        wait_return = max(row.demand_return - (cap - b), 0) / μ
        wait_cost += wait_borrow + wait_return

# 計算調度與藏車／釋放成本
dispatch_cost = α * sum(r["quantity"] for r in dispatch_result)
hide_cost = β * sum(r["hide"] + r["release"] for r in hide_result)
total_cost = wait_cost + dispatch_cost + hide_cost

# 顯示結果
print("\n=== 成本明細（貪婪法） ===")
print(f"⏱️ 借還車等待成本: {wait_cost:.2f}")
print(f"🚚 調度成本: {dispatch_cost:.2f}")
print(f"📦 藏車/釋放成本: {hide_cost:.2f}")
print(f"🎯 成本總和: {total_cost:.2f}")