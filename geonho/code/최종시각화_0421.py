"""
HVI 시각화 코드 (로컬 실행용)
- 월별 CSV 로드 → 3개월 평균 → 13개 HVI=0 동 추가 → 최종 CSV 저장 → 시각화 4종
- 실행 전: pip install pandas geopandas folium matplotlib seaborn
"""
#%%
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ────────────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 (본인 환경에 맞게 수정) ────────────────────────
DATA_DIR   = '../data/HVI_all'
SHP_PATH   = '../data/hdong_data/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp'
OUTPUT_DIR = '../data/output/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 월별 CSV 로드 ─────────────────────────────────────────────
MONTHS = ['202206', '202207', '202208']

dfs_raw = {}
for m in MONTHS:
    path = os.path.join(DATA_DIR, f'HVI_f_{m}.xls')
    if os.path.exists(path):
        dfs_raw[m] = pd.read_csv(path, dtype={'hdong_code': str})
        print(f"[{m}] {len(dfs_raw[m])}개 행정동 로드")
    else:
        print(f"[{m}] 파일 없음: {path}")

if not dfs_raw:
    raise FileNotFoundError("로드된 월별 파일이 없습니다. DATA_DIR 경로를 확인하세요.")

#%%
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

ZERO_DONG_CODES = {d['hdong_code'] for d in ZERO_DONGS}


# ════════════════════════════════════════════════════════════════
# 3개월 평균 집계 → df_final 생성
# 수치 컬럼은 평균, 범주 컬럼은 최빈값(마지막 월 우선)
# ════════════════════════════════════════════════════════════════
NUMERIC_COLS = [
    'low_ratio_combined', 'low_ratio_imputed',
    'old_ratio',
    'unit_density', 'density_imputed',
    'score_low', 'score_old', 'score_density',
    'HVI_score', 'HVI_index',
]
META_COLS = ['hdong_code', 'hdong_name', 'gu_name',
             'HVI_grade', 'n_axes', 'imputation_flag',
             'reliability']

# 각 월 df를 리스트로 모아 concat
all_months_df = pd.concat(
    [df.assign(month=m) for m, df in dfs_raw.items()],
    ignore_index=True
)

# 수치 컬럼 평균
numeric_avg = (
    all_months_df
    .groupby('hdong_code')[NUMERIC_COLS]
    .mean()
    .reset_index()
)

# 범주 컬럼: 마지막 월(202208) 기준으로 가져오기
last_month_key = sorted(dfs_raw.keys())[-1]
meta = dfs_raw[last_month_key][['hdong_code'] + [c for c in META_COLS if c != 'hdong_code']].copy()

df_final = meta.merge(numeric_avg, on='hdong_code', how='outer')

# HVI_rank 재계산 (평균값 기준)
df_valid_for_rank = df_final[~df_final['hdong_code'].isin(ZERO_DONG_CODES)].copy()
df_valid_for_rank['HVI_rank'] = df_valid_for_rank['HVI_score'].rank(
    ascending=False, method='min'
).astype('Int64')
df_final = df_final.merge(
    df_valid_for_rank[['hdong_code', 'HVI_rank']],
    on='hdong_code', how='left'
)

print(f"\n[평균 집계 완료] {len(df_final)}개 행정동")

#%%
# ════════════════════════════════════════════════════════════════
# 13개 HVI=0 동 추가
# ════════════════════════════════════════════════════════════════
def add_zero_dongs(df):
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
        print(f"[0처리 동 추가] {len(rows)}개 추가 → 총 {len(df)}개")
    return df

df_final = add_zero_dongs(df_final)
df_final['HVI_index'] = df_final['HVI_index'].round().astype('Int64')

# ════════════════════════════════════════════════════════════════
# 최종 CSV 저장 (3개월 평균 기반, 0처리 동 포함)
# ════════════════════════════════════════════════════════════════
EXPORT_COLS = [
    'hdong_code', 'hdong_name', 'gu_name',
    'low_ratio_combined', 'low_ratio_imputed',
    'old_ratio',
    'unit_density', 'density_imputed',
    'score_low', 'score_old', 'score_density',
    'HVI_score', 'HVI_index', 'HVI_rank', 'HVI_grade', 'n_axes',
    'imputation_flag', 'reliability'
]
for col in EXPORT_COLS:
    if col not in df_final.columns:
        df_final[col] = None

FINAL_CSV_PATH = os.path.join(DATA_DIR, 'B068_HVI_full.csv')
df_final[EXPORT_COLS].to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"\n[최종 CSV 저장] {FINAL_CSV_PATH} ({len(df_final)}개 행정동)")

#%%
# ── shapefile 로드 및 merge ───────────────────────────────────
gdf = gpd.read_file(SHP_PATH, encoding='cp949')
gdf = gdf.rename(columns={'ADM_CD': 'hdong_code', 'ADM_NM': 'hdong_name_shp'})
gdf = gdf.to_crs(epsg=4326)
gdf_hvi = gdf.merge(df_final, on='hdong_code', how='left')


# ════════════════════════════════════════════════════════════════
# 1. Folium 코로플레스 지도
#    - 분석 동: YlOrRd 색상
#    - 0처리 동: 회색 + 툴팁에 제외 사유
#    - 모아센터 위치: 파란 마커 오버레이 (데이터 있을 경우)
# ════════════════════════════════════════════════════════════════
def make_folium_map(gdf_hvi, output_dir):
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
                fields=['hdong_name', 'gu_name', 'exclusion_reason'],
                aliases=['행정동', '자치구', '제외 사유'],
            )
        ).add_to(m)

    # 분석 동 툴팁
    folium.GeoJson(
        gdf_valid,
        name='HVI 분석 동',
        style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
        tooltip=folium.GeoJsonTooltip(
            fields=['hdong_name', 'gu_name', 'HVI_index', 'HVI_grade', 'reliability'],
            aliases=['행정동', '자치구', 'HVI지수', '등급', '신뢰도'],
            localize=True
        )
    ).add_to(m)

    # ── 모아센터 마커 (CSV 파일 있을 경우) ──────────────────
    # 모아센터 CSV 형식: lat, lon, name 컬럼 필요
    # 서울 열린데이터광장에서 다운로드 후 경로 지정
    MOA_CENTER_PATH = '../data/moa_center.csv'
    if os.path.exists(MOA_CENTER_PATH):
        df_moa = pd.read_csv(MOA_CENTER_PATH, encoding='utf-8-sig')
        moa_group = folium.FeatureGroup(name='모아센터 현황', show=True)
        for _, row in df_moa.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=7,
                color='#1a6faf',
                fill=True,
                fill_color='#1a6faf',
                fill_opacity=0.85,
                popup=folium.Popup(row.get('name', '모아센터'), max_width=200),
                tooltip=row.get('name', '모아센터'),
            ).add_to(moa_group)
        moa_group.add_to(m)
        print(f"[모아센터] {len(df_moa)}개 마커 추가")
    else:
        print(f"[모아센터] 파일 없음 ({MOA_CENTER_PATH}) — 마커 생략")

    folium.LayerControl().add_to(m)
    out = os.path.join(output_dir, 'HVI_choropleth_final.html')
    m.save(out)
    print(f"[저장] {out}")

make_folium_map(gdf_hvi, OUTPUT_DIR)

#%%
# ════════════════════════════════════════════════════════════════
# 2. 구별 HVI 평균 bar chart (0처리 동 제외 후 계산)
# ════════════════════════════════════════════════════════════════
def make_gu_barchart(df, output_dir):
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
    ax.set_title('자치구별 평균 HVI 점수 (2022년 6~8월 평균)', fontsize=14, fontweight='bold')
    ax.set_ylabel('평균 HVI 점수')
    plt.xticks(rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, gu_mean['HVI_score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, 'HVI_gu_barchart_final.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_gu_barchart(df_final, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 3. 3축 산점도 (0처리 동 제외)
# ════════════════════════════════════════════════════════════════
def make_scatter(df, output_dir):
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
    ax.set_title('저가비율 vs 노후도 (버블 크기: HVI 점수) — 3개월 평균', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'HVI_scatter_final.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_scatter(df_final, OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════
# 4. 월별 HVI 상위 20개 동 변화 (월별 원본 dfs 사용, 0처리 동 제외)
# ════════════════════════════════════════════════════════════════
def make_monthly_trend(dfs_raw, df_final, output_dir):
    if len(dfs_raw) < 2:
        print("월별 데이터 2개 이상 필요")
        return

    # 최종 평균 기준 상위 20개 동
    top20_dongs = (
        df_final[df_final['HVI_grade'] != '해당없음']
        .nlargest(20, 'HVI_score')['hdong_name']
        .tolist()
    )

    monthly_data = []
    for m, df_m in sorted(dfs_raw.items()):
        df_m_valid = df_m[~df_m['hdong_code'].isin(ZERO_DONG_CODES)]
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

    ax.set_title('HVI 상위 20개 동 월별 변화 (3개월 평균 기준 상위 선정)', fontsize=13, fontweight='bold')
    ax.set_xlabel('기준 월')
    ax.set_ylabel('HVI 점수')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'HVI_monthly_trend_top20.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {out}")

make_monthly_trend(dfs_raw, df_final, OUTPUT_DIR)

print("\n✅ 완료. 결과물 위치:", OUTPUT_DIR)
print(f"   최종 CSV: {FINAL_CSV_PATH}")

#%%
# 슬림 버전 CSV (지수 + 신뢰도만)
SLIM_COLS = ['hdong_code', 'gu_name', 'hdong_name', 'HVI_index', 'reliability']
SLIM_CSV_PATH = os.path.join(DATA_DIR, 'B068_HVI.csv')
df_final[SLIM_COLS].to_csv(SLIM_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"[슬림 CSV 저장] {SLIM_CSV_PATH} ({len(df_final)}개 행정동)")