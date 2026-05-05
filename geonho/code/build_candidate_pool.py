"""
build_candidate_pool.py
=======================
랜덤 시뮬레이션 후보군 필터링 (조건 2 제거 최종본)

[조건]
  1. 기존 모아센터 위치 행정동 제외
  2. (제거) 저층주거수 필터 — HVI가 이미 주거취약도 대리변수로 포함됨
  3. Final_Score_4 / Demand_4 / Supply 결측 제외
  4. Demand_4 하위 10% 제외

실행: cd code && python build_candidate_pool.py
결과: data/output/candidate_pool.csv
      data/output/filter_log.txt
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data"
OUT   = DATA / "output"
OUT.mkdir(exist_ok=True)

SCORE_CSV = DATA / "final_score_4.csv"
MOA_CSV   = DATA / "moa_center.csv"
SHP_PATH  = DATA / "hdong_data" / "bnd_dong_11_2025_2Q" / "bnd_dong_11_2025_2Q.shp"

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

# ══════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════
log("=" * 60)
log("[0] 데이터 로드")

score_df = pd.read_csv(SCORE_CSV, dtype={"행정동코드": str})
score_df["행정동코드"] = score_df["행정동코드"].astype(str).str.zfill(10)
log(f"  final_score_4 : {len(score_df)}행")

# ══════════════════════════════════════════════════════════
# 조건 3 — MCI 지표 결측 제거
# ══════════════════════════════════════════════════════════
log("\n[조건 3] Final_Score_4 / Demand_4 / Supply 결측 제거")
before = len(score_df)
score_df = score_df.dropna(subset=["Final_Score_4", "Demand_4", "Supply"])
log(f"  제거 {before - len(score_df)}행 → 잔존 {len(score_df)}행")

# ══════════════════════════════════════════════════════════
# 조건 1 — 기존 모아센터 행정동 제외
# ══════════════════════════════════════════════════════════
log("\n[조건 1] 기존 모아센터 위치 행정동 제외")

moa_df = pd.read_csv(MOA_CSV)
log(f"  모아센터 {len(moa_df)}개소")

shp = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
adm_col = next(c for c in shp.columns if "CD" in c.upper())
nm_col  = next(c for c in shp.columns if "NM" in c.upper())
shp["행정동코드"] = shp[adm_col].astype(str).str.zfill(10)

moa_gdf = gpd.GeoDataFrame(
    moa_df,
    geometry=gpd.points_from_xy(moa_df["lon"], moa_df["lat"]),
    crs="EPSG:4326"
)
moa_joined = gpd.sjoin(
    moa_gdf,
    shp[["행정동코드", nm_col, "geometry"]],
    how="left", predicate="within"
)

log("\n  [모아센터 → 행정동 매핑]")
for _, r in moa_joined.iterrows():
    code = r.get("행정동코드", "❌ 미매핑")
    nm   = r.get(nm_col, "")
    log(f"    {str(r['name']):<28} → {code}  {nm}")

moa_codes = set(moa_joined["행정동코드"].dropna())
log(f"\n  제외 행정동 {len(moa_codes)}개")

before = len(score_df)
score_df = score_df[~score_df["행정동코드"].isin(moa_codes)]
log(f"  제거 {before - len(score_df)}행 → 잔존 {len(score_df)}행")

# ══════════════════════════════════════════════════════════
# 조건 4 — Demand_4 하위 10% 제외
# ══════════════════════════════════════════════════════════
log("\n[조건 4] Demand_4 하위 10% 제외")
d_thresh = score_df["Demand_4"].quantile(0.10)
log(f"  컷오프: {d_thresh:.4f}")

before = len(score_df)
score_df = score_df[score_df["Demand_4"] >= d_thresh].copy()
log(f"  제거 {before - len(score_df)}행 → 잔존 {len(score_df)}행")

# 조건 5 — HVI=0 또는 excluded 행정동 제외
# 조건 5 — HVI=0 또는 excluded 행정동 제외
HVI_CSV = BASE.parent / "Index_data" / "B068_HVI.csv"
hvi_df = pd.read_csv(HVI_CSV, dtype=str)

hvi_zero_names = set(
    hvi_df[(hvi_df["HVI_index"].astype(float) == 0) | (hvi_df["reliability"] == "excluded")]
    .apply(lambda r: r["gu_name"].strip() + "_" + r["hdong_name"].strip(), axis=1)
    .tolist()
)

score_df["_name_key"] = score_df["자치구"].str.strip() + "_" + score_df["행정동"].str.strip()

log(f"\n[조건 5] HVI=0 또는 excluded 행정동 제외")
log(f"  해당 행정동: {sorted(hvi_zero_names)}")

before = len(score_df)
score_df = score_df[~score_df["_name_key"].isin(hvi_zero_names)].copy()
score_df = score_df.drop(columns=["_name_key"])
log(f"  제거 {before - len(score_df)}행 → 잔존 {len(score_df)}행")

# ══════════════════════════════════════════════════════════
# S4 플래그 확인 (리스트 확정 후 업데이트)
# ══════════════════════════════════════════════════════════
# ※ 가양3동은 신영 파트에서 HVI=0 필터 후 대체 예정
#   확정되면 아래 리스트 업데이트
S4_DONGS = [
    "등촌3동", "중계2·3동", "가양2동", "신원동", "구로4동",
    "창신2동", "신길4동", "청림동", "암사1동",
    "번2동", "번3동", "송천동", "천호3동",
    "TBD"   # 가양3동 대체 후보 — 신영 파트 확정 후 교체
]
score_df["is_S4"] = score_df["행정동"].isin(S4_DONGS)
n_s4 = score_df["is_S4"].sum()
log(f"\n[S4 확인] 랜덤 풀 포함: {n_s4}개")
missing = [d for d in S4_DONGS if d != "TBD" and d not in score_df["행정동"].values]
if missing:
    log(f"  ※ 풀에 없는 S4 동: {missing}")

# ══════════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════════
out_cols = [c for c in
            ["행정동코드", "자치구", "행정동", "Demand_4", "Supply",
             "Final_Score_4", "is_S4"]
            if c in score_df.columns]

score_df[out_cols].to_csv(OUT / "candidate_pool.csv", index=False, encoding="utf-8-sig")
(OUT / "filter_log.txt").write_text("\n".join(LOG), encoding="utf-8")

log(f"\n[완료] 최종 후보군: {len(score_df)}개 행정동")
log(f"  → {OUT / 'candidate_pool.csv'}")
