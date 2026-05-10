#%%
from pathlib import Path

import pandas as pd


BASE = Path(__file__).parent.parent
DATA = BASE / "data"

CLUSTER_CSV = DATA / "cluster.csv"
RECOMMEND_CSV = DATA / "final_moa_recommend.csv"


#%%
cluster_df = pd.read_csv(CLUSTER_CSV, dtype={"행정동코드": str})
recommend_df = pd.read_csv(RECOMMEND_CSV)

cluster_df["행정동코드"] = cluster_df["행정동코드"].astype(str).str.zfill(10)


#%%
result_df = recommend_df.merge(
    cluster_df[["행정동코드", "자치구", "행정동", "cluster", "cluster_name"]],
    on=["자치구", "행정동"],
    how="left",
)


#%%
missing_df = result_df[result_df["cluster"].isna()]
if not missing_df.empty:
    print("[매칭 실패 행정동]")
    print(missing_df[["순위", "자치구", "행정동"]].to_string(index=False))
    print()


#%%
print("[final_moa_recommend 행정동별 cluster]")
print(
    result_df[
        ["순위", "행정동코드", "자치구", "행정동", "cluster", "cluster_name"]
    ].to_string(index=False)
)
