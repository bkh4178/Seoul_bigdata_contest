"""
check_dong_mapping.py
=====================
행정동별_주택수.csv  vs  final_score_4.csv 행정동명 매핑 진단

실행: cd code && python check_dong_mapping.py
결과: data/output/mapping_check.txt
"""

import pandas as pd
from pathlib import Path

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data"
OUT   = DATA / "output"
OUT.mkdir(exist_ok=True)

SCORE_CSV   = DATA / "final_score_4.csv"
HOUSING_CSV = DATA / "행정동별_주택수.csv"

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

# ── final_score_4 로드 ──
score_df = pd.read_csv(SCORE_CSV, dtype={"행정동코드": str})
score_df["행정동코드"] = score_df["행정동코드"].astype(str).str.zfill(10)
log(f"final_score_4 행 수: {len(score_df)}")

# ── 주택수 파일 로드 (헤더 4행 스킵) ──
raw = pd.read_csv(HOUSING_CSV, header=None, skiprows=4, encoding="utf-8-sig", dtype=str)
raw = raw.rename(columns={0: "시도", 1: "구", 2: "행정동명"})

# 서울시 + 소계 제거
raw = raw[raw["시도"].str.strip() == "서울시"].copy()
raw = raw[~raw["행정동명"].str.strip().isin(["소계", "합계", "전체", "nan", ""])].copy()
raw = raw[raw["행정동명"].notna()].copy()
log(f"주택수 파일 행정동 수: {len(raw)}")

# ── 정규화 함수 ──
def norm(s):
    if not isinstance(s, str): return ""
    return s.replace(".", "·").replace(" ", "").strip()

# ── 매핑 키 생성 ──
raw["_key"]   = raw["구"].str.strip() + "_" + raw["행정동명"].apply(norm)
score_df["_key"] = score_df["자치구"].str.strip() + "_" + score_df["행정동"].apply(norm)

housing_keys = set(raw["_key"].tolist())
score_keys   = set(score_df["_key"].tolist())

matched    = score_keys & housing_keys
only_score = score_keys - housing_keys   # score에만 있음 (주택수 파일에 없음)
only_hous  = housing_keys - score_keys   # 주택수에만 있음 (score에 없음)

log(f"\n{'='*50}")
log(f"매핑 성공: {len(matched)}개")
log(f"score에만 있음 (주택수 미매핑): {len(only_score)}개")
log(f"주택수에만 있음 (score 미포함): {len(only_hous)}개")

log(f"\n{'='*50}")
log("[score에만 있음 — 주택수 파일에서 못 찾은 행정동]")
for k in sorted(only_score):
    gu, dong = k.split("_", 1) if "_" in k else (k, "")
    log(f"  {gu} {dong}")

log(f"\n{'='*50}")
log("[주택수에만 있음 — score에 없는 행정동 (참고용)]")
for k in sorted(only_hous):
    gu, dong = k.split("_", 1) if "_" in k else (k, "")
    log(f"  {gu} {dong}")

# 저장
(OUT / "mapping_check.txt").write_text("\n".join(LOG), encoding="utf-8")
log(f"\n결과 저장: {OUT / 'mapping_check.txt'}")
