#%%
from pathlib import Path

import pandas as pd


BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT = DATA / "output"
OUT.mkdir(exist_ok=True)

SCORE_CSV = DATA / "final_score_4.csv"
OUTPUT_CSV = OUT / "demand_supply_all_dong.csv"


#%%
def minmax_scale(series):
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return series * 0
    return (series - min_value) / (max_value - min_value)


def print_saved_csv(csv_path):
    saved_df = pd.read_csv(csv_path, dtype={"행정동코드": str})

    saved_df = saved_df.sort_values("점수", ascending=False)

    print("\n[CSV 불러오기] 순서대로 행정동 출력")
    for _, row in saved_df.iterrows():
        gu = row["자치구"]
        dong = row["행정동"]
        score = row["점수"]
        print(f"{gu} {dong} - {score:.4f}")


#%%
def build_score_df():
    score_df = pd.read_csv(SCORE_CSV, dtype={"행정동코드": str})
    score_df["행정동코드"] = score_df["행정동코드"].astype(str).str.zfill(10)

    required_cols = ["행정동코드", "자치구", "행정동", "Demand_4", "Supply"]
    missing_cols = [col for col in required_cols if col not in score_df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    score_df = score_df.dropna(subset=["Demand_4", "Supply"]).copy()
    score_df["Demand_Supply_Score"] = score_df["Demand_4"] * score_df["Supply"]
    score_df["Sqrt_Demand_Supply_Score"] = score_df["Demand_Supply_Score"] ** 0.5
    score_df["Scaled_Sqrt_Demand_Supply_Score"] = minmax_scale(
        score_df["Sqrt_Demand_Supply_Score"]
    )
    score_df["점수"] = score_df["Scaled_Sqrt_Demand_Supply_Score"] * 100

    result_df = score_df.sort_values(
        "점수",
        ascending=False,
    ).copy()
    return result_df


#%%
def save_result_df(result_df):
    output_df = result_df.rename(
        columns={
            "Demand_4": "수요",
            "Supply": "공급",
        }
    )
    out_cols = ["행정동코드", "자치구", "행정동", "수요", "공급", "점수"]

    output_df[out_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[완료] sqrt(Demand_4 x Supply) min-max 기준 전체 {len(output_df)}개 행정동")
    print(output_df[out_cols].to_string(index=False))
    print(f"\n저장 경로: {OUTPUT_CSV}")

    print_saved_csv(OUTPUT_CSV)


#%%
def plot_distribution(result_df):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "분포 시각화를 보려면 matplotlib가 필요합니다. "
            "현재 실행 환경에 matplotlib를 설치한 뒤 다시 실행하세요."
        ) from exc

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(result_df["Demand_Supply_Score"], bins=30, edgecolor="black")
    axes[0].set_title("Demand_4 x Supply")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("count")

    axes[1].hist(result_df["Sqrt_Demand_Supply_Score"], bins=30, edgecolor="black")
    axes[1].set_title("sqrt(Demand_4 x Supply)")
    axes[1].set_xlabel("sqrt score")
    axes[1].set_ylabel("count")

    axes[2].hist(
        result_df["Scaled_Sqrt_Demand_Supply_Score"],
        bins=30,
        edgecolor="black",
    )
    axes[2].set_title("MinMax scaled sqrt score")
    axes[2].set_xlabel("scaled score")
    axes[2].set_ylabel("count")

    plt.tight_layout()
    plt.show()


#%%
def main():
    result_df = build_score_df()
    save_result_df(result_df)
    try:
        plot_distribution(result_df)
    except ModuleNotFoundError as exc:
        print(f"\n[시각화 생략] {exc}")


#%%
if __name__ == "__main__":
    main()
