from gurobipy import *
import pandas as pd

# === 全域參數設定 ===
μ = 6        # 處理速率（輛/分鐘）
α = 0.1      # 調度成本權重
β = 0.02     # 藏車／釋放成本權重
L = 20       # 每輛卡車可載車數
T_num = 30    # 卡車數
K = T_num * L  # 每期最大調度數量

# === 讀取資料 ===
df = pd.read_csv("gurobi_demand_table_daan.csv")

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
B0 = {s: min(D_return.get((s, 0), 0), C[s]) for s in S}
max_hide_per_station = {i: int(0.4 * C[i]) for i in S}

# === 模型建立 ===
m = Model("YouBike_Multiperiod")
m.setParam("OutputFlag", 0)

x = m.addVars(S, S, T, vtype=GRB.INTEGER, name="x")
h_in = m.addVars(S, T, vtype=GRB.INTEGER, name="h_in")
h_out = m.addVars(S, T, vtype=GRB.INTEGER, name="h_out")
B = m.addVars(S, T, lb=0, name="B")
W_borrow = m.addVars(S, T, lb=0, name="W_borrow")
W_return = m.addVars(S, T, lb=0, name="W_return")

# === 限制式 ===
for t in T:
    for i in S:
        if t == 0:
            m.addConstr(B[i, t] == B0[i] + h_out[i, t] - h_in[i, t])
        else:
            m.addConstr(B[i, t] == B[i, t-1]
                        + quicksum(x[j, i, t-1] for j in S if j != i)
                        - quicksum(x[i, j, t-1] for j in S if j != i)
                        + h_out[i, t] - h_in[i, t])
        m.addConstr(B[i, t] <= C[i])
        m.addConstr(W_borrow[i, t] >= (D_borrow.get((i, t), 0) - B[i, t]) / μ)
        m.addConstr(W_return[i, t] >= (D_return.get((i, t), 0) - (C[i] - B[i, t])) / μ)
        m.addConstr(h_in[i, t] <= max_hide_per_station[i])
        m.addConstr(h_in[i, t] * h_out[i, t] == 0)
        m.addConstr(h_out[i, t] <= quicksum(h_in[i, τ] - h_out[i, τ] for τ in range(t+1)))

for t in T:
    m.addConstr(quicksum(x[i, j, t] for i in S for j in S if i != j) <= K)

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

pd.DataFrame(dispatch_records).to_csv("dispatch_result.csv", index=False)
pd.DataFrame(hide_records).to_csv("hide_result.csv", index=False)
print("✅ 結果已輸出為 dispatch_result.csv 與 hide_result.csv")
