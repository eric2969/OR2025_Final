from gurobipy import *
import pandas as pd
import os

# === 全域參數設定 ===
μ = 6         # 處理速率（輛/分鐘）
α = 1         # 調度成本權重
β = 0.04      # 藏車／釋放成本權重
L = 20        # 每輛卡車可載車數
T_num = 30    # 卡車數
max_visit = 3 # 每台卡車最多拜訪站點數
K = T_num * L # 每期最大調度數量
limit_time = 120  # 每段最多運行時間（秒）

# === 載入資料（處理整個台北市，不過濾 sarea）===
df = pd.read_csv("./assets/gurobi_demand_table.csv")

# 整理時間與站點
times = sorted(df["interval_time"].unique())
stations = sorted(df["sno"].unique())
sna_map = df.drop_duplicates("sno").set_index("sno")["sna"].to_dict()

# 索引對應
S = range(len(stations))
s_index = {s: i for i, s in enumerate(stations)}
stations_map = {i: s for s, i in s_index.items()}
C = {s_index[row.sno]: row.total for _, row in df.drop_duplicates("sno").iterrows()}
max_hide_per_station = {i: int(0.4 * C[i]) for i in S}
B0 = {i: int(0.35 * C[i]) for i in S}

# 時間切段（每 12 個半小時 = 6 小時一段）
interval_per_batch = 12
batches = [times[i:i + interval_per_batch] for i in range(0, len(times), interval_per_batch)]

# 建立輸出資料夾
if not os.path.exists("results"):
    os.makedirs("results")

dispatch_records = []
hide_records = []
prev_B = B0.copy()

for batch_id, batch_times in enumerate(batches):
    print(f"🟢 處理第 {batch_id + 1} 段，共 {len(batch_times)} 個時段...")

    T = range(len(batch_times))
    t_index = {t: i for i, t in enumerate(batch_times)}
    D_borrow = {(s_index[row.sno], t_index[row.interval_time]): row.demand_borrow
                for _, row in df[df["interval_time"].isin(batch_times)].iterrows()}
    D_return = {(s_index[row.sno], t_index[row.interval_time]): row.demand_return
                for _, row in df[df["interval_time"].isin(batch_times)].iterrows()}

    # 模型
    m = Model(f"YouBike_Batch_{batch_id + 1}")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", limit_time)
    m.setParam("MIPGap", 0.05)

    x = m.addVars(S, S, T, vtype=GRB.CONTINUOUS, name="x")
    v = m.addVars(S, S, T, vtype=GRB.BINARY, name="v")
    h_in = m.addVars(S, T, vtype=GRB.INTEGER, name="h_in")
    h_out = m.addVars(S, T, vtype=GRB.INTEGER, name="h_out")
    B = m.addVars(S, T, lb=0, name="B")
    W_borrow = m.addVars(S, T, lb=0, name="W_borrow")
    W_return = m.addVars(S, T, lb=0, name="W_return")

    delay = 2

    for t in T:
        for i in S:
            inflow = quicksum(x[j, i, t - delay] for j in S if j != i) if t - delay >= 0 else 0
            outflow = quicksum(x[i, j, t] for j in S if j != i)

            if t == 0:
                m.addConstr(B[i, t] == prev_B[i] + h_out[i, t] - h_in[i, t])
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

    for i in S:
        for j in S:
            if i != j:
                for t in T:
                    m.addConstr(x[i, j, t] <= L * v[i, j, t])

    m.addConstr(quicksum(v[i, j, t] for i in S for j in S if i != j for t in T) <= T_num * max_visit)

    m.setObjective(
        quicksum(W_borrow[i, t] + W_return[i, t] for i in S for t in T) +
        α * quicksum(x[i, j, t] for i in S for j in S if i != j for t in T) +
        β * quicksum(h_in[i, t] + h_out[i, t] for i in S for t in T),
        GRB.MINIMIZE
    )

    m.optimize()

    for t in T:
        time = batch_times[t]
        for i in S:
            s1 = stations_map[i]
            for j in S:
                if i != j and x[i, j, t].X > 0.5:
                    s2 = stations_map[j]
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

    # 傳遞期末庫存
    prev_B = {i: int(B[i, len(T) - 1].X) for i in S}
    print(f"✅ 第 {batch_id + 1} 段完成")

# 輸出 CSV
pd.DataFrame(dispatch_records).to_csv(f"./results/gurobi_dispatch-full.csv", index=False)
pd.DataFrame(hide_records).to_csv(f"./results/gurobi_hide-full.csv", index=False)
print("🎉 所有區段已完成並輸出結果")