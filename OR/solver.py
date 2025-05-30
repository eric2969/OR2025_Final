from gurobipy import *
import pandas as pd

# ========================
# 全域參數（可依需求修改）
# ========================
μ = 10                    # 平均處理速率（輛/分鐘）
α = 0.15                    # 每輛調度成本
β = 0.03                    # 每輛隱藏或釋放成本
L = 20                   # 每輛卡車可載車數
T_num = 30                # 可用卡車數量
K = T_num * L          # 每期最多調度輛數

# ========================
# 讀取資料
# ========================
df = pd.read_csv("gurobi_demand_table.csv")

# 僅取單一時段（例：00:00）做單期模型
target_time = "00:00"
df = df[df["interval_time"] == target_time]

stations = df["sno"].tolist()
n = len(stations)

# 索引對照
s_index = {sno: i for i, sno in enumerate(stations)}
I = range(n)

# 參數
B0 = {i: min(df.iloc[i]["demand_return"], df.iloc[i]["total"]) for i in I}  # 初始車量假設為平均還車需求
C = {i: df.iloc[i]["total"] for i in I}
D_borrow = {i: df.iloc[i]["demand_borrow"] for i in I}
D_return = {i: df.iloc[i]["demand_return"] for i in I}

# ========================
# 建立 Gurobi 模型
# ========================
m = Model("YouBike_Redistribution")

# 決策變數
x = m.addVars(I, I, vtype=GRB.INTEGER, name="x")  # 從 i → j 的調度
h_in = m.addVars(I, vtype=GRB.INTEGER, name="h_in")    # 隱藏
h_out = m.addVars(I, vtype=GRB.INTEGER, name="h_out")  # 釋放
B = m.addVars(I, lb=0, name="B")                       # 最終車輛數
W_borrow = m.addVars(I, lb=0.0, name="W_borrow")       # 借車等待時間
W_return = m.addVars(I, lb=0.0, name="W_return")       # 還車等待時間

# Final bike balance
for i in I:
    m.addConstr(B[i] == B0[i]
                + quicksum(x[j, i] for j in I if j != i)
                - quicksum(x[i, j] for j in I if j != i)
                + h_out[i] - h_in[i])

# 等待時間估計（線性化）
for i in I:
    m.addConstr(W_borrow[i] >= (D_borrow[i] - B[i]) / μ)
    m.addConstr(W_return[i] >= (D_return[i] - (C[i] - B[i])) / μ)

# 容量與限制
m.addConstrs((B[i] <= C[i] - h_in[i] / 2 for i in I), name="capacity_with_hin")
m.addConstrs((B[i] >= 0 for i in I), name="non_negative")

# 限制總調度數量（不得超過總調度上限 K 與卡車總運能）
m.addConstr(
    quicksum(x[i, j] for i in I for j in I if i != j) <= min(K, T_num * L),
    name="dispatch_capacity"
)

# 每站最多可藏車數（犧牲最多 20% 停車格、可藏兩倍）
m.addConstrs(
    (h_in[i] <= int(0.4 * C[i]) for i in I),
    name="max_hide_per_station"
)

# 目標函數
m.setObjective(
    quicksum(W_borrow[i] + W_return[i] for i in I) +
    α * quicksum(x[i, j] for i in I for j in I if i != j) +
    β * quicksum(h_in[i] + h_out[i] for i in I),
    GRB.MINIMIZE
)

# ========================
# 求解與輸出
# ========================
m.optimize()

# print("\n===== 解結果摘要 =====")
# for i in I:
#     print(f"站點 {df.iloc[i]['sna']} | 借車等待：{W_borrow[i].X:.2f} 分 | 還車等待：{W_return[i].X:.2f} 分 | 調整後車輛：{B[i].X:.1f}")

print("\n🚚 總調度數量：", sum(x[i,j].X for i in I for j in I if i != j))
print("📦 總隱藏數量：", sum(h_in[i].X + h_out[i].X for i in I))
print("🎯 目標值（總成本）：", m.ObjVal)

# # 調度決策變數
# print("\n===== 調度決策 x[i,j]（非零者）=====")
# for i in I:
#     for j in I:
#         if i != j and x[i, j].X > 0.5:
#             print(f"從 {df.iloc[i]['sna']} → {df.iloc[j]['sna']}: {x[i, j].X:.0f} 輛")

# # 隱藏／釋放變數
# print("\n===== 隱藏／釋放決策（非零者）=====")
# for i in I:
#     if h_in[i].X > 0.5 or h_out[i].X > 0.5:
#         print(f"{df.iloc[i]['sna']}: 隱藏 {h_in[i].X:.0f} 輛, 釋放 {h_out[i].X:.0f} 輛")
