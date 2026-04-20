"""
HVI 시각화 코드 (로컬 실행용)
- 월별 CSV 로드 → 13개 HVI=0 동 추가 → 시각화 4종 + 최종 CSV 저장
- 실행 전: pip install pandas geopandas folium matplotlib seaborn
"""

import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ────────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 (본인 환경에 맞게 수정) ────────────────────────
DATA_DIR   = './data/HVI_final'
SHP_PATH   = './bnd_dong_11_2025_2Q.shp'
OUTPUT_DIR = './output/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 월별 CSV 로드 ─────────────────────────────────────────────
MONTHS = ['202206', '202207', '202208']

dfs_raw = {}
for m in MONTHS:
    path = os.path.join(DATA_DIR, f'HVI_bnd_dong_{m}_high_match.csv')
    if os.path.exists(path):
        dfs_raw[m] = pd.read_csv(path, dtype={'hdong_code': str})
        print(f"[{m}] {len(dfs_raw[m])}개 행정동 로드")
    else:
        print(f"[{m}] 파일 없음: {path}")


# ════════════════════════════════════════════════════════════════
# 13개 HVI=0 처리 행정동 정의
# 사유: 연립·다세대 실질적 부재 (아파트단지/상업지역/재건축철거)
# ════════════════════════════════════════════════════════════════
ZERO_DONGS = [
    # 처음부터 B068 없음 + 직접 확인 (4개)
    {'hdong_code': '11020520', 'hdong_name': '소공동',   'gu_name': '중구',   'exclusion_reason': '상업지역 밀집, 연립·다세대 없음'},
    {'hdong_code': '11150690', 'hdong_name': '신정6동',  'gu_name': '양천구', 'exclusion_reason': '직접 확인 — 연립·다세대 없음'},
    {'hdong_code': '11240780', 'hdong_name': '잠실7동',  'gu_name': '송파구', 'exclusion_reason': '아파트단지 밀집'},
    {'hdong_code': '11250700', 'hdong_name': '둔촌1동',  'gu_name': '강동구', 'exclusion_reason': '둔촌주공 재건축으로 2022년 기준 건물 대부분 철거'},
    # 그룹 3 — 건축물대장 n_building < 5 + 직접 확인 (9개)
    {'hdong_code': '11150680', 'hdong_name': '가양3동',  'gu_name': '강서구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11240690', 'hdong_name': '문정2동',  'gu_name': '송파구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11220560', 'hdong_name': '반포본동', 'gu_name': '서초구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11110720', 'hdong_name': '상계8동',  'gu_name': '노원구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11110730', 'hdong_name': '상계9동',  'gu_name': '노원구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11240590', 'hdong_name': '오륜동',   'gu_name': '송파구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11240790', 'hdong_name': '잠실2동',  'gu_name': '송파구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11240760', 'hdong_name': '잠실6동',  'gu_name': '송파구', 'exclusion_reason': '아파트/산업단지 확인'},
    {'hdong_code': '11020530', 'hdong_name': '을지로동', 'gu_name': '중구',   'exclusion_reason': '상업지역, 데이터 2건으로 대표성 없음'},
]

def add_zero_dongs(df):
    """13개 HVI=0 동을 df에 추가 (이미 있으면 스킵)"""
    existing_codes = set(df['hdong_code'].tolist())
    rows = []
    for d in ZERO_DONGS:
        if d['hdong_code'] not in existing_codes:
            rows.append({
                'hdong_code':         d['hdong_code'],
                'hdong_name':         d['hdong_name'],
                'gu_name':            d['gu_name'],
                'HVI_score':          0.0,
                'HVI_index':          0,
                'HVI_rank':           None,
                'HVI_grade':          '해당없음',
                'n_axes':             0,
                'reliability':        'excluded',
                'imputation_flag':    'excluded',
                'exclusion_reason':   d['exclusion_reason'],
                'low_ratio_combined': None,
                'old_ratio':          None,
                'unit_density':       None,
                'score_low':          None,
                'score_old':          None,
                'score_density':      None,
                'low_ratio_imputed':  None,
                'density_imputed':    None,
            })
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# 전체 월에 0처리 동 추가
dfs = {}
for m, df_raw in dfs_raw.items():
    dfs[m] = add_zero_dongs(df_raw.copy())
    n_added = len(dfs[m]) - len(df_raw)
    print(f"[{m}] 0처리 동 {n_added}개 추가 → 총 {len(dfs[m])}개")

# 대표 월
BASE_MONTH = '202208'
df = dfs[BASE_MONTH].copy()

# ── shapefile 로드 ────────────────────────────────────────────
gdf = gpd.read_file(SHP_PATH, encoding='cp949')
gdf = gdf.rename(columns={'ADM_CD': 'hdong_code', 'ADM_NM': 'hdong_name'})
gdf = gdf.to_crs(epsg=4326)
gdf_hvi = gdf.merge(df, on='hdong_code', how='left')


# ════════════════════════════════════════════════════════════════
# 1. Folium 코로플레스 지도
#    - 분석 동: YlOrRd 색상
#    - 0처리 동: 회색 + 툴팁에 제외 사유
# ════════════════════════════════════════════════════════════════
def make_folium_map(gdf_hvi, month_str, output_dir):
    m = folium.Map(location=[37.56, 126.97], zoom_start=11, tiles='CartoDB positron')

    # 분석 대상 동
    gdf_valid = gdf_hvi[
        gdf_hvi['HVI_index'].notna() & (gdf_hvi['HVI_grade'] != '해당없음')
    ].copy()
    folium.Choropleth(
        geo_data=gdf_valid.__geo_interface__,
        data=gdf_valid[['hdong_code', 'HVI_index']],
        columns=['hdong_code', 'HVI_index'],
        key_on='feature.properties.hdong_code',
        fill_color='YlOrRd',
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name='HVI 지수 (0~100)',
        nan_fill_color='lightgray',
        nan_fill_opacity=0.3,
    ).add_to(m)

    # 0처리 동 — 회색 레이어
    gdf_zero = gdf_hvi[gdf_hvi['HVI_grade'] == '해당없음'].copy()
    if len(gdf_zero) > 0:
        folium.GeoJson(
            gdf_zero,
            name='분석제외(연립·다세대 부재)',
            style_function=lambda x: {
                'fillColor': '#aaaaaa',
                'color': '#888888',
                'weight': 0.8,
                'fillOpacity': 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['hdong_name_x', 'gu_name', 'exclusion_reason'],
                aliases=['행정동', '자치구', '제외 사유'],
            )
        ).add_to(m)

    # 분석 동 툴팁
    folium.GeoJson(
        gdf_valid,
        style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
        tooltip=folium.GeoJsonTooltip(
            fields=['hdong_name_x', 'gu_name', 'HVI_index', 'HVI_grade', 'reliability'],
            aliases=['행정동', '자치구', 'HVI지수', '등급', '신뢰도'],
            localize=True
        )
    ).add_to(m)

    folium.LayerControl().add_to(m)
    out = os.path.join(output_dir, f'HVI_choropleth_{month_str}.html')
    m.save(out)
    print(f"[저장] {out}")

make_folium_map(gdf_hvi, BASE_MONTH, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 2. 구별 HVI 평균 bar chart (0처리 동 제외 후 계산)
# ════════════════════════════════════════════════════════════════
def make_gu_barchart(df, month_str, output_dir):
    df_valid = df[df['HVI_grade'] != '해당없음'].copy()
    gu_mean = (df_valid.groupby('gu_name')['HVI_score']
               .mean()
               .sort_values(ascending=False)
               .reset_index())

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#d73027' if v >= gu_mean['HVI_score'].quantile(0.75)
              else '#fee08b' if v >= gu_mean['HVI_score'].median()
              else '#91bfdb'
              for v in gu_mean['HVI_score']]

    bars = ax.bar(gu_mean['gu_name'], gu_mean['HVI_score'], color=colors, edgecolor='white')
    ax.set_title(f'자치구별 평균 HVI 점수 ({month_str})', fontsize=14, fontweight='bold')
    ax.set_ylabel('평균 HVI 점수')
    plt.xticks(rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, gu_mean['HVI_score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, f'HVI_gu_barchart_{month_str}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_gu_barchart(df, BASE_MONTH, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 3. 3축 산점도 (0처리 동 제외)
# ════════════════════════════════════════════════════════════════
def make_scatter(df, month_str, output_dir):
    valid = df[df['HVI_grade'] != '해당없음'].dropna(
        subset=['low_ratio_combined', 'old_ratio', 'HVI_score']
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        valid['low_ratio_combined'] * 100,
        valid['old_ratio'] * 100,
        c=valid['HVI_score'],
        s=valid['HVI_score'] * 1.5 + 10,
        cmap='YlOrRd', alpha=0.7,
        edgecolors='white', linewidths=0.5
    )
    plt.colorbar(sc, ax=ax, label='HVI 점수')

    top10 = valid.nlargest(10, 'HVI_score')
    for _, row in top10.iterrows():
        ax.annotate(row['hdong_name'],
                    (row['low_ratio_combined']*100, row['old_ratio']*100),
                    fontsize=7, ha='left', xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('저가비율 (%)', fontsize=12)
    ax.set_ylabel('노후도 비율 (%)', fontsize=12)
    ax.set_title(f'저가비율 vs 노후도 (버블 크기: HVI 점수) ({month_str})', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'HVI_scatter_{month_str}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_scatter(df, BASE_MONTH, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 4. 월별 HVI 상위 20개 동 변화 (0처리 동 제외)
# ════════════════════════════════════════════════════════════════
def make_monthly_trend(dfs, output_dir):
    if len(dfs) < 2:
        print("월별 데이터 2개 이상 필요")
        return

    last_month = sorted(dfs.keys())[-1]
    df_last = dfs[last_month][dfs[last_month]['HVI_grade'] != '해당없음']
    top20_dongs = df_last.nlargest(20, 'HVI_score')['hdong_name'].tolist()

    monthly_data = []
    for m, df_m in sorted(dfs.items()):
        df_m_valid = df_m[df_m['HVI_grade'] != '해당없음']
        tmp = df_m_valid[df_m_valid['hdong_name'].isin(top20_dongs)][
            ['hdong_name', 'HVI_score']].copy()
        tmp['month'] = m
        monthly_data.append(tmp)

    df_trend = pd.concat(monthly_data)

    fig, ax = plt.subplots(figsize=(14, 7))
    for dong in top20_dongs:
        sub = df_trend[df_trend['hdong_name'] == dong]
        if len(sub) > 0:
            ax.plot(sub['month'], sub['HVI_score'], marker='o', label=dong, linewidth=1.5)

    ax.set_title('HVI 상위 20개 동 월별 변화', fontsize=13, fontweight='bold')
    ax.set_xlabel('기준 월')
    ax.set_ylabel('HVI 점수')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'HVI_monthly_trend_top20.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_monthly_trend(dfs, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 최종 CSV 저장 (0처리 동 포함 완전한 산출물)
# ════════════════════════════════════════════════════════════════
EXPORT_COLS = [
    'hdong_code', 'hdong_name', 'gu_name',
    'low_ratio_combined', 'low_ratio_imputed',
    'old_ratio',
    'unit_density', 'density_imputed',
    'score_low', 'score_old', 'score_density',
    'HVI_score', 'HVI_index', 'HVI_rank', 'HVI_grade', 'n_axes',
    'imputation_flag', 'reliability', 'exclusion_reason'
]

for m, df_full in dfs.items():
    for col in EXPORT_COLS:
        if col not in df_full.columns:
            df_full[col] = None
    out = os.path.join(DATA_DIR, f'HVI_bnd_dong_{m}_final.csv')
    df_full[EXPORT_COLS].to_csv(out, index=False, encoding='utf-8-sig')
    print(f"[저장] {out} ({len(df_full)}개 행정동)")

print("\n✅ 완료. 결과물 위치:", OUTPUT_DIR)