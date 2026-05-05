"""
simulation.py
=============
모아센터 신규 입지 시뮬레이션 (S0~S4)

[입력 파일]
  data/final_score_4.csv
  data/moa_center.csv
  data/output/candidate_pool.csv
  data/final_moa_recommend.csv      ← 신영 파트 최종 14개 행정동 (자치구, 행정동 컬럼)
  data/hdong_data/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp

[시나리오]
  S0: 기존 14개소
  S1: 기존 14 + 랜덤 14개 (1,000회 반복) ← 메인 3 baseline
  S2: 기존 14 + Demand_4 상위 14개
  S3: 기존 14 + Final_Score_4 상위 14개 (위치제약 없음)
  S4: 기존 14 + 최종 추천 14개 (신영 파트 결과)

[평가지표]
  메인 1: 서비스 공백지역(기존 14개소 기준 최근접거리 상위 30%) 평균/중앙값/90%분위 거리 감소
  메인 3: S1 1,000회 분포에서 S4 백분위
  보조  : 전체 행정동 기준 거리 CDF

실행: cd code && python simulation.py
결과: data/output/simulation_result.csv
      data/output/simulation_summary.txt
      data/output/figures/ (히스토그램, CDF)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
for fname in fm.findSystemFonts():
    if "AppleGothic" in fname or "NanumGothic" in fname or "Malgun" in fname:
        plt.rcParams["font.family"] = fm.FontProperties(fname=fname).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data"
OUT   = DATA / "output"
FIG   = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SCORE_CSV     = DATA / "final_score_4.csv"
MOA_CSV       = DATA / "moa_center.csv"
POOL_CSV      = OUT  / "candidate_pool.csv"
RECOMMEND_CSV = DATA / "final_moa_recommend.csv"
SHP_PATH      = DATA / "hdong_data" / "bnd_dong_11_2025_2Q" / "bnd_dong_11_2025_2Q.shp"

N_RANDOM  = 1000   # S1 반복 횟수
N_NEW     = 14     # 신규 개소 수
RANDOM_SEED = 42

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

# ══════════════════════════════════════════════════════════
# 0. 데이터 로드 & representative_point 생성
# ══════════════════════════════════════════════════════════
log("=" * 60)
log("[0] 데이터 로드 및 대표점 생성")

score_df = pd.read_csv(SCORE_CSV, dtype={"행정동코드": str})
score_df["행정동코드"] = score_df["행정동코드"].astype(str).str.zfill(10)

shp = gpd.read_file(SHP_PATH).to_crs(epsg=5179)
adm_col = next(c for c in shp.columns if "CD" in c.upper())
nm_col  = next(c for c in shp.columns if "NM" in c.upper())
shp["행정동코드"] = shp[adm_col].astype(str).str.zfill(10)

# representative_point (polygon 내부 보장)
shp["rep_point"] = shp.geometry.representative_point()
shp["rep_x"] = shp["rep_point"].x
shp["rep_y"] = shp["rep_point"].y

# score_df에 대표점 좌표 병합
score_df = score_df.merge(
    shp[["행정동코드", "rep_x", "rep_y"]],
    on="행정동코드", how="left"
)
n_missing = score_df[["rep_x","rep_y"]].isna().any(axis=1).sum()
log(f"  행정동 수: {len(score_df)}  /  대표점 미매핑: {n_missing}개")

# 기존 모아센터 좌표 (EPSG:5179 변환)
moa_df = pd.read_csv(MOA_CSV)
moa_gdf = gpd.GeoDataFrame(
    moa_df,
    geometry=gpd.points_from_xy(moa_df["lon"], moa_df["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=5179)
moa_gdf["x"] = moa_gdf.geometry.x
moa_gdf["y"] = moa_gdf.geometry.y
EXISTING_CENTERS = moa_gdf[["x","y"]].values
log(f"  기존 모아센터: {len(EXISTING_CENTERS)}개소")

# 후보 풀
pool_df = pd.read_csv(POOL_CSV, dtype={"행정동코드": str})
pool_df = pool_df.merge(
    score_df[["행정동코드","rep_x","rep_y"]],
    on="행정동코드", how="left"
).dropna(subset=["rep_x","rep_y"])
log(f"  랜덤 후보 풀: {len(pool_df)}개 행정동")

# S4 최종 추천 행정동
recommend_df = pd.read_csv(RECOMMEND_CSV)
log(f"  S4 최종 추천: {len(recommend_df)}개 행정동")
log(f"  {list(recommend_df['행정동'])}")

# S4 대표점
s4_df = score_df.merge(recommend_df[["행정동"]], on="행정동", how="inner")
if len(s4_df) < N_NEW:
    log(f"  ⚠️  S4 매핑 {len(s4_df)}개 (기대 {N_NEW}개) — recommend.csv 행정동명 확인 필요")
S4_CENTERS = s4_df[["rep_x","rep_y"]].values

# S2 Demand 상위 14개 (후보 풀 내에서)
s2_df = pool_df.nlargest(N_NEW, "Demand_4")
S2_CENTERS = s2_df[["rep_x","rep_y"]].values

# S3 Final_Score_4 상위 14개 (후보 풀 내에서, 위치제약 없음)
s3_df = pool_df.nlargest(N_NEW, "Final_Score_4")
S3_CENTERS = s3_df[["rep_x","rep_y"]].values

log(f"\n  S2 Demand 상위 14: {list(s2_df['행정동'])}")
log(f"  S3 MCI 상위 14:    {list(s3_df['행정동'])}")

# ══════════════════════════════════════════════════════════
# 1. 서비스 공백지역 정의 (메인 1·3 기준)
# ══════════════════════════════════════════════════════════
log("\n[1] 서비스 공백지역 정의")

eval_df = score_df.dropna(subset=["rep_x","rep_y"]).copy()
eval_coords = eval_df[["rep_x","rep_y"]].values

# S0 기준 최근접 거리 계산
tree_s0 = cKDTree(EXISTING_CENTERS)
dist_s0, _ = tree_s0.query(eval_coords)
eval_df["dist_s0"] = dist_s0

# 공백지역 = 기존 센터 기준 최근접거리 2km 초과 행정동 (3.5절 기준과 통일)
GAP_THRESHOLD = 2000  # 2km (단위: m, EPSG:5179)
gap_df = eval_df[eval_df["dist_s0"] > GAP_THRESHOLD].copy()
GAP_COORDS = gap_df[["rep_x","rep_y"]].values

log(f"  공백지역 기준 거리: 2km 초과")
log(f"  공백지역 행정동 수: {len(gap_df)}개")

# ══════════════════════════════════════════════════════════
# 2. 거리 계산 함수
# ══════════════════════════════════════════════════════════
def calc_gap_stats(new_centers):
    """공백지역 기준 최근접 센터 거리 통계"""
    all_centers = np.vstack([EXISTING_CENTERS, new_centers])
    tree = cKDTree(all_centers)
    dists, _ = tree.query(GAP_COORDS)
    return {
        "mean":   dists.mean(),
        "median": np.median(dists),
        "p90":    np.percentile(dists, 90),
        "over2km": (dists > 2000).sum() / len(dists),
    }

def calc_all_stats(new_centers):
    """전체 행정동 기준 최근접 센터 거리 통계 (CDF용)"""
    all_centers = np.vstack([EXISTING_CENTERS, new_centers])
    tree = cKDTree(all_centers)
    dists, _ = tree.query(eval_coords)
    return dists

# ══════════════════════════════════════════════════════════
# 3. S0 기저선
# ══════════════════════════════════════════════════════════
log("\n[2] S0 기저선 계산")
tree_s0_gap = cKDTree(EXISTING_CENTERS)
dists_s0_gap, _ = tree_s0_gap.query(GAP_COORDS)
s0_stats = {
    "mean":    dists_s0_gap.mean(),
    "median":  np.median(dists_s0_gap),
    "p90":     np.percentile(dists_s0_gap, 90),
    "over2km": (dists_s0_gap > 2000).sum() / len(dists_s0_gap),
}
log(f"  S0 공백지역 평균거리: {s0_stats['mean']/1000:.2f}km")

# ══════════════════════════════════════════════════════════
# 4. S2 / S3 / S4 계산
# ══════════════════════════════════════════════════════════
log("\n[3] S2 / S3 / S4 계산")
s2_stats = calc_gap_stats(S2_CENTERS)
s3_stats = calc_gap_stats(S3_CENTERS)
s4_stats = calc_gap_stats(S4_CENTERS) if len(S4_CENTERS) == N_NEW else None

for name, st in [("S2", s2_stats), ("S3", s3_stats), ("S4", s4_stats)]:
    if st:
        log(f"  {name} 평균거리: {st['mean']/1000:.2f}km  "
            f"중앙값: {st['median']/1000:.2f}km  "
            f"P90: {st['p90']/1000:.2f}km  "
            f"2km초과: {st['over2km']:.1%}")

# ══════════════════════════════════════════════════════════
# 5. S1 랜덤 1,000회 (메인 3)
# ══════════════════════════════════════════════════════════
log(f"\n[4] S1 랜덤 {N_RANDOM}회 반복")
rng = np.random.default_rng(RANDOM_SEED)
pool_coords = pool_df[["rep_x","rep_y"]].values
pool_idx    = np.arange(len(pool_coords))

random_means = []
for i in range(N_RANDOM):
    sampled_idx = rng.choice(pool_idx, size=N_NEW, replace=False)
    sampled_centers = pool_coords[sampled_idx]
    st = calc_gap_stats(sampled_centers)
    random_means.append(st["mean"])
    if (i+1) % 200 == 0:
        log(f"  {i+1}/{N_RANDOM} 완료...")

random_means = np.array(random_means)
log(f"  S1 분포: 평균={random_means.mean()/1000:.2f}km  "
    f"std={random_means.std()/1000:.2f}km")

# S4 백분위
if s4_stats:
    s4_mean = s4_stats["mean"]
    pct = (random_means < s4_mean).mean() * 100
    log(f"\n  ★ S4는 S1 분포의 하위 {pct:.1f}% (거리 기준)")
    log(f"    → 랜덤 배치보다 공백지역 평균거리가 짧음: 상위 {100-pct:.1f}%")

# ══════════════════════════════════════════════════════════
# 6. 결과 저장
# ══════════════════════════════════════════════════════════
log("\n[5] 결과 저장")

# 요약 테이블
results = []
s0_mean = s0_stats["mean"]
for name, st in [("S0(기존14)", s0_stats),
                 ("S2(수요기준)", s2_stats),
                 ("S3(MCI단독)", s3_stats),
                 ("S4(최종제안)", s4_stats)]:
    if st is None:
        continue
    results.append({
        "시나리오": name,
        "평균거리(km)":  round(st["mean"]/1000, 3),
        "중앙값(km)":   round(st["median"]/1000, 3),
        "P90(km)":      round(st["p90"]/1000, 3),
        "2km초과비율":  round(st["over2km"], 4),
        "평균거리감소(km)": round((s0_mean - st["mean"])/1000, 3),
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "simulation_result.csv", index=False, encoding="utf-8-sig")
log(f"  simulation_result.csv 저장")
log(f"\n{result_df.to_string(index=False)}")

# ══════════════════════════════════════════════════════════
# 7. 시각화
# ══════════════════════════════════════════════════════════
log("\n[6] 시각화")

# [그림 1] 메인 3 — 랜덤 분포 히스토그램 + S4 수직선
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(random_means / 1000, bins=50, color="#4A90D9", alpha=0.7,
        edgecolor="white", label=f"랜덤 배치 {N_RANDOM}회")
if s4_stats:
    ax.axvline(s4_stats["mean"] / 1000, color="#E74C3C", linewidth=2.5,
               label=f"S4 MCI 제안 ({s4_stats['mean']/1000:.2f}km)")
ax.axvline(s0_stats["mean"] / 1000, color="#888", linewidth=1.5,
           linestyle="--", label=f"S0 기존 ({s0_stats['mean']/1000:.2f}km)")
ax.set_xlabel("서비스 공백지역 평균 최근접 거리 (km)")
ax.set_ylabel("빈도")
ax.set_title("랜덤 배치 vs MCI 제안 — 공백지역 접근성 비교")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "hist_random_vs_s4.png", dpi=150)
plt.close()
log("  hist_random_vs_s4.png 저장")

# [그림 2] 보조 — 전체 행정동 기준 거리 CDF
fig, ax = plt.subplots(figsize=(9, 5))
scenario_colors = {
    "S0 기존": ("#888",   "--", EXISTING_CENTERS[:0]),   # dummy
    "S2 수요": ("#F39C12", "-",  S2_CENTERS),
    "S3 MCI단독": ("#27AE60", "-",  S3_CENTERS),
    "S4 최종제안": ("#E74C3C", "-",  S4_CENTERS),
}

# S0
d0 = calc_all_stats(np.empty((0, 2)))  # 신규 없음
# S0는 EXISTING_CENTERS만
tree_s0_all = cKDTree(EXISTING_CENTERS)
d0, _ = tree_s0_all.query(eval_coords)
xs = np.sort(d0) / 1000
ax.plot(xs, np.linspace(0, 1, len(xs)), color="#888", linestyle="--",
        linewidth=1.5, label="S0 기존")

for label, (color, ls, centers) in scenario_colors.items():
    if label == "S0 기존":
        continue
    if len(centers) == 0:
        continue
    d = calc_all_stats(centers)
    xs = np.sort(d) / 1000
    ax.plot(xs, np.linspace(0, 1, len(xs)), color=color, linestyle=ls,
            linewidth=2, label=label)

for km in [1, 2, 3]:
    ax.axvline(km, color="#ddd", linewidth=0.8, linestyle=":")
ax.set_xlabel("최근접 모아센터 거리 (km)")
ax.set_ylabel("누적 비율")
ax.set_title("시나리오별 최근접 센터 거리 누적분포 (전체 행정동)")
ax.set_xlim(0, 8)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "cdf_all_scenarios.png", dpi=150)
plt.close()
log("  cdf_all_scenarios.png 저장")

# 로그 저장
(OUT / "simulation_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
log(f"\n[완료] 결과: {OUT}")