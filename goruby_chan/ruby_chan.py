# youbike_sensitivity_v4.py
# ------------------------------------------------------------
#  • 自動偵測站點檔案，否則隨機產生 100 站
#  • dense grids, full CSV export, 3-D surface plots
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ========== ① 站點資料載入 ==========
DATA_FILE = "stations.xlsx"        # ← 改成你的檔名即可 (.csv / .xlsx / .json 均可)

if os.path.isfile(DATA_FILE):
    print(f"[INFO] Loading station data from {DATA_FILE}")
    ext = os.path.splitext(DATA_FILE)[1].lower()
    if ext == ".csv":
        stations_df = pd.read_csv(DATA_FILE)
    elif ext in (".xlsx", ".xls"):
        stations_df = pd.read_excel(DATA_FILE)
    elif ext == ".json":
        stations_df = pd.read_json(DATA_FILE)
    else:
        raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON.")

    # 期待欄位：capacity, borrow_demand, return_demand
    I = stations_df.index
    B0      = {i: 0 for i in I}
    C       = dict(zip(I, stations_df["capacity"]))
    Db_true = dict(zip(I, stations_df["borrow_demand"]))
    Dr_true = dict(zip(I, stations_df["return_demand"]))

else:
    print(f"[WARN] {DATA_FILE} not found — generate random 100-station demo.")
    np.random.seed(97)
    I = range(100)
    B0      = {i: 0 for i in I}
    C       = {i: np.random.randint(35, 55) for i in I}
    Db_true = {i: np.random.randint(15, 30) for i in I}
    Dr_true = {i: np.random.randint(15, 30) for i in I}

print(f"[INFO] Using {len(I)} stations.")

# ========== ② 單次求解函式（與 v3 相同） ==========
def solve_once(mu, alpha, beta, K, H, L, numTruck):
    m = Model()
    m.Params.LogToConsole = 0

    x    = {(i, j): m.addVar(vtype=GRB.INTEGER) for i in I for j in I if i != j}
    hin  = {i: m.addVar(vtype=GRB.INTEGER) for i in I}
    hout = {i: m.addVar(vtype=GRB.INTEGER) for i in I}
    B    = {i: m.addVar() for i in I}
    Wb   = {i: m.addVar() for i in I}
    Wr   = {i: m.addVar() for i in I}

    for i in I:
        m.addConstr(B[i] == B0[i]
                    + quicksum(x[j, i] for j in I if j != i)
                    - quicksum(x[i, j] for j in I if j != i)
                    + hin[i] - hout[i])
        m.addConstr(B[i] <= C[i]);  m.addConstr(B[i] >= 0)
        m.addConstr(Wb[i] >= (Db_true[i] - B[i]) / mu);  m.addConstr(Wb[i] >= 0)
        m.addConstr(Wr[i] >= (Dr_true[i] - (C[i] - B[i])) / mu);  m.addConstr(Wr[i] >= 0)

    m.addConstr(quicksum(x.values()) <= K)
    m.addConstr(quicksum(hin[i] + hout[i] for i in I) <= H)
    m.addConstr(quicksum(x.values()) <= numTruck * L)

    m.setObjective(quicksum(Wb[i] + Wr[i] for i in I)
                   + alpha * quicksum(x.values())
                   + beta  * quicksum(hin[i] + hout[i] for i in I),
                   GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return np.nan, np.nan
    obj = m.ObjVal
    W_mean = sum(Wb[i].X + Wr[i].X for i in I) / (2 * len(I))
    return obj, W_mean

# ========== ③ 靈敏度網格（與 v3 相同但密度已拉高） ==========
base = dict(mu=1.5, alpha=0.02, beta=0.01, K=100, H=40, L=25, numTruck=4)

mu_vals    = np.linspace(1.0, 3.0, 9)
K_vals     = np.linspace(50, 150, 11, dtype=int)
alpha_vals = np.linspace(0.005, 0.045, 9)
beta_vals  = np.linspace(0.005, 0.045, 9)

records = []

# ---- (A) μ × K ----
for mu in mu_vals:
    for K in K_vals:
        params = base | dict(mu=mu, K=int(K))
        obj, Wm = solve_once(**params)
        records.append(params | dict(pair="mu_K", X=mu, Y=K,
                                     objective=obj, mean_wait=Wm))

# ---- (B) α × β ----
for alpha in alpha_vals:
    for beta in beta_vals:
        params = base | dict(alpha=float(alpha), beta=float(beta))
        obj, Wm = solve_once(**params)
        records.append(params | dict(pair="alpha_beta", X=alpha, Y=beta,
                                     objective=obj, mean_wait=Wm))

df = pd.DataFrame(records)

# ========== ④ 列印 & 儲存 ==========
print("\n===== first 5 rows =====")
print(df.head().to_string(index=False))
print("\n===== last 5 rows =====")
print(df.tail().to_string(index=False))

csv_path = "youbike_sensitivity_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n>>> All {len(df)} scenarios saved to {csv_path}")

# ========== ⑤ 3-D 曲面圖 ==========
def plot_surface(sub, z_col, title, xlab, ylab, zlab):
    Xu, Yu = np.sort(sub["X"].unique()), np.sort(sub["Y"].unique())
    X, Y = np.meshgrid(Xu, Yu)
    Z = sub.pivot(index="Y", columns="X", values=z_col).values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    ax.set_title(title)
    ax.set_xlabel(xlab);  ax.set_ylabel(ylab);  ax.set_zlabel(zlab)
    plt.tight_layout();  plt.show()

pair_muK = df[df["pair"] == "mu_K"]
plot_surface(pair_muK, "objective",  "Objective vs μ & K",
             "Service rate μ (bike/min)", "Redistribution cap K (bike)", "Total objective")
plot_surface(pair_muK, "mean_wait",  "Avg wait vs μ & K",
             "Service rate μ (bike/min)", "Redistribution cap K (bike)", "Average wait (min)")

pair_ab = df[df["pair"] == "alpha_beta"]
plot_surface(pair_ab, "objective",  "Objective vs α & β",
             "Redistribution cost α", "Hiding cost β", "Total objective")
plot_surface(pair_ab, "mean_wait",  "Avg wait vs α & β",
             "Redistribution cost α", "Hiding cost β", "Average wait (min)")
