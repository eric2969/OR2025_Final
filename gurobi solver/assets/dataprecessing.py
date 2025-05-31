import pandas as pd

# 讀取原始 CSV
df = pd.read_csv("gurobi_demand_table.csv")

# 篩選出大同區資料
df_datong = df[df["sarea"] == "大同區"]

# 儲存為新的 CSV 檔案
datong_path = "gurobi_demand_table_datong.csv"
df_datong.to_csv(datong_path, index=False)

datong_path