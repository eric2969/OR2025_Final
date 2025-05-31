from gurobipy import *
import pandas as pd
import sys, os, time

# === 全域參數設定 ===
μ = 6        # 處理速率（輛/分鐘）
α = 1      # 調度成本權重
β = 0.04     # 藏車／釋放成本權重
L = 20       # 每輛卡車可載車數
T_num = 30   # 卡車數
max_visit = 3  # 每台卡車最多拜訪站點數
K = T_num * L  # 每期最大調度數量
location = sys.argv[1]
limit_time = int(sys.argv[2]) if len(sys.argv) > 2 else 600  # 最大運行時間（秒）
if not os.path.exists("results"):
    os.makedirs("results")

# === 讀取資料 ===
start_time = time.time()
df = pd.read_csv(f"assets/gurobi_demand_table_{location}.csv")

# 索引與對應關係
times = sorted(df["interval_time"].unique())
stations = sorted(df["sno"].unique())
sna_map = df.drop_duplicates("sno").set_index("sno")["sna"].to_dict()

S = range(len(stations))
T = range(len(times))
s_index = {s: i for i, s in enumerate(stations)}
t_index = {t: i for i, t in enumerate(times)}

# 資料轉換
C = {s_index[row.sno]: row.total for _, row in df.drop_duplicates("sno").iterrows()}
D_borrow = {(s_index[row.sno], t_index[row.interval_time]): row.demand_borrow for _, row in df.iterrows()}
D_return = {(s_index[row.sno], t_index[row.interval_time]): row.demand_return for _, row in df.iterrows()}
B0 = {i: int(0.35 * C[i]) for i in S}
max_hide_per_station = {i: int(0.4 * C[i]) for i in S}

# === 模型建立 ===
m = Model("YouBike_Multiperiod")
m.setParam("OutputFlag", 0)
m.setParam("TimeLimit", limit_time)   # 最多跑 30 分鐘
m.setParam("MIPGap", 0.05)  # 允許 1% 誤差內解即可接受

x = m.addVars(S, S, T, vtype=GRB.CONTINUOUS, name="x")
v = m.addVars(S, S, T, vtype=GRB.BINARY, name="v")
h_in = m.addVars(S, T, vtype=GRB.INTEGER, name="h_in")
h_out = m.addVars(S, T, vtype=GRB.INTEGER, name="h_out")
B = m.addVars(S, T, lb=0, name="B")
W_borrow = m.addVars(S, T, lb=0, name="W_borrow")
W_return = m.addVars(S, T, lb=0, name="W_return")

# 延遲時間設定
delay = 2  # 調度延遲時間（期數）

# === 限制式 ===
for t in T:
    for i in S:
        if t == 0:
            inflow = 0
        else:
            inflow = quicksum(x[j, i, t - delay] for j in S if j != i) if t - delay >= 0 else 0

        outflow = quicksum(x[i, j, t] for j in S if j != i)

        if t == 0:
            m.addConstr(B[i, t] == B0[i] + h_out[i, t] - h_in[i, t])
        else:
            m.addConstr(B[i, t] == B[i, t - 1] + inflow - outflow + h_out[i, t] - h_in[i, t])

        m.addConstr(B[i, t] <= C[i])
        m.addConstr(W_borrow[i, t] >= (D_borrow.get((i, t), 0) - B[i, t]) / μ)
        m.addConstr(W_return[i, t] >= (D_return.get((i, t), 0) - (C[i] - B[i, t])) / μ)
        m.addConstr(h_in[i, t] <= max_hide_per_station[i])
        m.addConstr(h_in[i, t] * h_out[i, t] == 0)
        m.addConstr(h_out[i, t] <= quicksum(h_in[i, τ] - h_out[i, τ] for τ in range(t + 1)))

for t in T:
    m.addConstr(quicksum(x[i, j, t] for i in S for j in S if i != j) <= K)

# 拜訪次數與調度變數連結
M = L
for i in S:
    for j in S:
        if i != j:
            for t in T:
                m.addConstr(x[i, j, t] <= M * v[i, j, t])

# 總拜訪次數限制
m.addConstr(quicksum(v[i, j, t] for i in S for j in S if i != j for t in T) <= T_num * max_visit)

# === 目標函數 ===
m.setObjective(
    quicksum(W_borrow[i, t] + W_return[i, t] for i in S for t in T) +
    α * quicksum(x[i, j, t] for i in S for j in S if i != j for t in T) +
    β * quicksum(h_in[i, t] + h_out[i, t] for i in S for t in T),
    GRB.MINIMIZE
)

m.optimize()

# === 結果輸出 ===
dispatch_records = []
hide_records = []

for t in T:
    time = times[t]
    for i in S:
        s1 = stations[i]
        for j in S:
            if i != j and x[i, j, t].X > 0.5:
                s2 = stations[j]
                dispatch_records.append(dict(
                    from_sno=s1, to_sno=s2,
                    from_sna=sna_map[s1], to_sna=sna_map[s2],
                    time=time, quantity=int(x[i, j, t].X)
                ))
        if h_in[i, t].X > 0.5 or h_out[i, t].X > 0.5:
            hide_records.append(dict(
                sno=s1, sna=sna_map[s1],
                time=time,
                hide=int(h_in[i, t].X),
                release=int(h_out[i, t].X)
            ))

pd.DataFrame(dispatch_records).to_csv(f"./results/gurobi_dispatch-{location}.csv", index=False)
pd.DataFrame(hide_records).to_csv(f"./results/gurobi_hide-{location}.csv", index=False)
print("✅ 結果已輸出為 CSV 檔案")

# === 統計與列印總結資訊 ===
total_cost = m.ObjVal
total_dispatch = sum(x[i, j, t].X for i in S for j in S if i != j for t in T)
total_hide = sum(h_in[i, t].X for i in S for t in T)
total_release = sum(h_out[i, t].X for i in S for t in T)
end_time = time.time()

with open(f"./results/gurobi_summary-{location}.txt", "w", encoding="utf-8") as f:
    f.write("=== 結果總結 ===\n")
    f.write(f"🎯 總成本 (Objective): {total_cost:.2f}\n")
    f.write(f"🚚 總調度數量: {int(total_dispatch)}\n")
    f.write(f"📦 總藏車數量: {int(total_hide)}\n")
    f.write(f"🔓 總釋放數量: {int(total_release)}\n")
    f.write(f"⏱️ 運行時間: {end_time - start_time:.2f} 秒\n")

print("=== 結果總結 ===")
print(f"🎯 總成本 (Objective): {total_cost:.2f}")
print(f"🚚 總調度數量: {int(total_dispatch)}")
print(f"📦 總藏車數量: {int(total_hide)}")
print(f"🔓 總釋放數量: {int(total_release)}")
print(f"⏱️ 運行時間: {end_time - start_time:.2f} 秒")