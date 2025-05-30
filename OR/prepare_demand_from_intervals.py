import pandas as pd
import glob
import os

# 讀取所有 interval_*.csv 並加入時間欄位
data_folder = "interval_outputs"
interval_files = sorted(glob.glob(os.path.join(data_folder, "interval_*.csv")))

records = []

for filepath in interval_files:
    time_str = os.path.splitext(os.path.basename(filepath))[0].split("_")[1]  # e.g., 0030
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    df = pd.read_csv(filepath)
    df["interval_time"] = f"{hour:02d}:{minute:02d}"
    records.append(df)

# 合併所有時段資料
all_data = pd.concat(records, ignore_index=True)

# 清理站名（移除前綴 "YouBike2.0_"）
all_data["sna"] = all_data["sna"].str.replace("YouBike2.0_", "", regex=False)

# 建立需求資料：每站每時段估算需求
demand_df = all_data[[
    "sno", "sna", "sarea", "interval_time",
    "total", "available_rent_bikes", "available_return_bikes"
]].copy()

# 假設：可還車位下降代表借出 → 借車需求，反之為還車需求
# 這裡使用 available_return_bikes 作為估算的借車需求
# 使用 available_rent_bikes 作為估算的還車需求
demand_df["demand_borrow"] = demand_df["available_return_bikes"]
demand_df["demand_return"] = demand_df["available_rent_bikes"]

# 最終僅保留 Gurobi 所需欄位
demand_df = demand_df[[
    "sno", "sna", "sarea", "interval_time", "total",
    "demand_borrow", "demand_return"
]]

# 儲存成 CSV 檔案供 Gurobi 模型載入
demand_df.to_csv("gurobi_demand_table.csv", index=False, encoding="utf-8-sig")
print("✔ 已產生 gurobi_demand_table.csv")
