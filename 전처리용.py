import pandas as pd
import numpy as np
from pandarallel import pandarallel


from pyproj import Transformer

# 행안부 표준인 EPSG:5174(중부원점)에서 WGS84(위경도)로 변환기 설정
transformer = Transformer.from_crs("epsg:5174", "epsg:4326", always_xy=True)

def get_age(x, ages_dict, global_dict, min_y, max_y, target_cols):
    import pandas as pd
    
    y1, y2 = x['인허가일자'].year, x['폐업일자'].year
    sy = max(min_y, min(max_y, int(y1))) if pd.notna(y1) else min_y
    ey = max(min_y, min(max_y, int(y2))) if pd.notna(y2) else max_y
    
    years = range(sy, ey + 1)
    s1, s2 = x['시군구1'], x['시군구2']
    
    vals = [ages_dict[(s1, y)] for y in years if (s1, y) in ages_dict]
    if vals:
        return pd.Series([sum(col) / len(vals) for col in zip(*vals)], index=target_cols)
        
    vals = [ages_dict[(s2, y)] for y in years if (s2, y) in ages_dict]
    if vals:
        return pd.Series([sum(col) / len(vals) for col in zip(*vals)], index=target_cols)
        
    vals = [global_dict[y] for y in years if y in global_dict]
    if vals:
        return pd.Series([sum(col) / len(vals) for col in zip(*vals)], index=target_cols)
        
    return pd.Series([float('nan')] * len(target_cols), index=target_cols)

# --------------------------------

name = '서울특별시'  # 전처리할 지역 이름
output_name = '문화_노래연습장업'  # 처리할 파일 이름

# --------------------------------

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True,shm_size_mb=4000)
    
    data = pd.read_csv(f'./{name}/{output_name}.csv',encoding='cp949')
    data['인허가일자'] = pd.to_datetime(data['인허가일자'],errors='coerce')
    data['폐업일자'] = pd.to_datetime(data['폐업일자'],errors='coerce')
    
    today = pd.Timestamp.now().normalize()
    data['폐업일자']=data['폐업일자'].fillna(today+pd.DateOffset(years=2))

    data = data.dropna(subset=['인허가일자'])
    data = data[~((data['폐업일자'].isna()) & (data['인허가일자'].dt.year > 2026-4)) & (data['인허가일자'].dt.year >= 2000)]
    data = data[(data['폐업일자'] - data['인허가일자']).dt.days > 30]

    data = data.dropna(subset=['좌표정보(X)','좌표정보(Y)','지번주소'])

    data['경도'], data['위도'] = transformer.transform(data['좌표정보(X)'].values, data['좌표정보(Y)'].values)

    data['시군구1'] = data['지번주소'].str.split().str[0:3].str.join(" ").replace(r'\d', '', regex=True)
    data['시군구2'] = data['지번주소'].str.split().str[0:2].str.join(" ")
    
    # 날짜 구간(Period) 경계값 설정
    td_2015 = pd.read_csv(f'./평균거래가2015_{name}.csv', index_col='시군구')['㎡당단가(만원)'].to_dict()
    td_2020 = pd.read_csv(f'./평균거래가2020_{name}.csv', index_col='시군구')['㎡당단가(만원)'].to_dict()
    td_2025 = pd.read_csv(f'./평균거래가2025_{name}.csv', index_col='시군구')['㎡당단가(만원)'].to_dict()
    p15 = data['시군구1'].map(td_2015).fillna(data['시군구2'].map(td_2015))
    p20 = data['시군구1'].map(td_2020).fillna(data['시군구2'].map(td_2020))
    p25 = data['시군구1'].map(td_2025).fillna(data['시군구2'].map(td_2025))

    start = data['인허가일자']
    end = data['폐업일자']

    # 겹치는 일수 계산 함수 (벡터화)
    half_life_years = 3
    decay_rate = np.log(2) / (half_life_years * 365.25)

    # 2. 겹치는 일수 및 거리 기반 가중치 계산 함수 (벡터화)
    def get_weighted_overlap(st, ed, p_st, p_ed, decay_rate):
        o_st = np.maximum(st, pd.Timestamp(p_st))
        o_ed = np.minimum(ed, pd.Timestamp(p_ed))
        days = np.maximum(0, (o_ed - o_st).dt.days + 1)
        
        midpoint = o_st + (o_ed - o_st) / 2
        
        distance = np.maximum(0, (midpoint - st).dt.days)
        
        # 거리에 따른 지수 감쇠 (멀어질수록 1에서 0에 가까워짐)
        weight_multiplier = np.exp(-decay_rate * distance)
        
        # 최종 가중치: 실제 겹치는 일수 * 거리에 따른 페널티
        weighted_days = days * weight_multiplier
        
        return days, weighted_days

    # 3. 각 기간별 일수(d) 및 가중치(w) 계산
    d15, w15 = get_weighted_overlap(start, end, '2000-01-01', '2015-12-31', decay_rate)
    d20, w20 = get_weighted_overlap(start, end, '2016-01-01', '2020-12-31', decay_rate)
    d25, w25 = get_weighted_overlap(start, end, '2021-01-01', '2100-12-31', decay_rate)

    total_days = d15 + d20 + d25
    total_weights = w15 + w20 + w25 # 순수 일수가 아닌 가중치의 합

    # 4. 결측치 조건 마스크 (유효성 검사는 기존 '일수(d)' 기준 유지)
    valid_mask = (
        ((d15 == 0) | p15.notna()) &
        ((d20 == 0) | p20.notna()) &
        ((d25 == 0) | p25.notna()) &
        (total_days > 0)
    )

    # 5. 가중 평균 단가 계산 (d15, d20 대신 w15, w20 사용)
    data['㎡당단가(만원)'] = np.where(
        valid_mask,
        (p15.fillna(0) * w15 + p20.fillna(0) * w20 + p25.fillna(0) * w25) / np.maximum(total_weights, 1e-9),
        np.nan
    )

    ages = pd.read_csv('연령통계.csv')
    min_y, max_y = ages['연도'].min(), ages['연도'].max()
    target_cols = ages.columns.difference(['시군구', '연도']).to_list()
    
    ages_dict = ages.set_index(['시군구', '연도'])[target_cols].apply(tuple, axis=1).to_dict()
    global_dict = ages.groupby('연도')[target_cols].mean().apply(tuple, axis=1).to_dict()
    
    ages_results = data.parallel_apply(get_age, axis=1, args=(ages_dict,global_dict, min_y, max_y, target_cols))
    data = pd.concat([data, ages_results], axis=1)

    data = data[['인허가일자','폐업일자','경도','위도','㎡당단가(만원)']+target_cols]

    # [수정 6] 에러가 나는 코드 오타 수정 (parallel_apply -> dropna)
    data = data.dropna(subset=['㎡당단가(만원)'])

    data.info()
    data.to_csv(f'{output_name}_{name}.csv', index=False, encoding='utf-8-sig')