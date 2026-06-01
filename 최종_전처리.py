import pandas as pd
import glob
from sklearn.neighbors import BallTree
import taichi as ti
import numpy as np

# GPU 초기화
ti.init(arch=ti.gpu)

# --- (haversine_kernel과 haversine_matrix_taichi 함수는 이전과 동일하게 이 자리에 위치) ---
@ti.kernel
def haversine_kernel(
    lat1: ti.types.ndarray(dtype=ti.f32, ndim=1), 
    lon1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    lat2: ti.types.ndarray(dtype=ti.f32, ndim=1), 
    lon2: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    for i, j in out:
        pi = 3.141592653589793
        r_lat1 = lat1[i] * (pi / 180.0)
        r_lon1 = lon1[i] * (pi / 180.0)
        r_lat2 = lat2[j] * (pi / 180.0)
        r_lon2 = lon2[j] * (pi / 180.0)

        dlat = r_lat2 - r_lat1
        dlon = r_lon2 - r_lon1

        a = ti.sin(dlat / 2.0)**2 + ti.cos(r_lat1) * ti.cos(r_lat2) * ti.sin(dlon / 2.0)**2
        a_clamped = ti.max(0.0, ti.min(1.0, a))
        out[i, j] = 6371.0 * 2.0 * ti.atan2(ti.sqrt(a_clamped), ti.sqrt(1.0 - a_clamped))

def haversine_matrix_taichi(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float32).flatten()
    lon1 = np.asarray(lon1, dtype=np.float32).flatten()
    lat2 = np.asarray(lat2, dtype=np.float32).flatten()
    lon2 = np.asarray(lon2, dtype=np.float32).flatten()
    
    N, M = len(lat1), len(lat2)
    out = np.empty((N, M), dtype=np.float32)
    haversine_kernel(lat1, lon1, lat2, lon2, out)
    return out

if __name__ == '__main__':
    
    sta = pd.read_csv('역데이터.csv')
    # Taichi 커널 입력을 위해 1차원 배열로 준비
    sta_lat_flat = sta['역위도'].values
    sta_lon_flat = sta['역경도'].values
    # 나중에 가중치 계산시 브로드캐스팅을 위해 (1, M) 형태 유지
    sta_pop = sta['일평균'].values[np.newaxis, :].astype(np.float32)

    big = pd.read_csv('대규모점포.csv')
    big['인허가일자'] = pd.to_datetime(big['인허가일자'], errors='coerce')
    big['폐업일자'] = pd.to_datetime(big['폐업일자'], errors='coerce')
    big_opened_ts = big['인허가일자'].values.astype('datetime64[D]').astype(np.int64)[np.newaxis, :]
    big_closed_ts = big['폐업일자'].values.astype('datetime64[D]').astype(np.int64)[np.newaxis, :]
    
    # Taichi 커널 입력을 위해 1차원 배열로 준비
    big_lat_flat = big['위도'].values
    big_lon_flat = big['경도'].values
    
    results_list=[]
    R = 6371.0
    radius_500m_rad = 0.5 / R
    
    for name in ['음식','제과','미용','주점','헬스','오락','편의점','카페']:
        print(f"[{name}] 데이터 처리 시작...")
        path = f'#{name}/*.csv'
        file_list = glob.glob(path)
        data = pd.concat([pd.read_csv(file) for file in file_list], axis=0)
        
        data['인허가일자'] = pd.to_datetime(data['인허가일자'],errors='coerce')
        data['폐업일자'] = pd.to_datetime(data['폐업일자'],errors='coerce')
        data['생존일수'] = (data['폐업일자'] - data['인허가일자']).dt.days

        data_lat_flat = data['위도'].values
        data_lon_flat = data['경도'].values
        
        
        dist_matrix_sta = haversine_matrix_taichi(data_lat_flat, data_lon_flat, sta_lat_flat, sta_lon_flat)
        # 가중치 계산 (NumPy 벡터화 연산)
        weights = sta_pop / (1.0 + dist_matrix_sta / 2.0) ** 2
        data['유동인구'] = weights.sum(axis=1)

        print(f"[{name}] 생존일수 및 대규모점포 거리 계산중...")
        opened_ts_data = data['인허가일자'].values.astype('datetime64[D]').astype(np.int64)[:, np.newaxis]
        
        dist_to_big = haversine_matrix_taichi(data_lat_flat, data_lon_flat, big_lat_flat, big_lon_flat)
        
        valid_big_mask = (big_opened_ts <= opened_ts_data) & (big_closed_ts > opened_ts_data)
        dist_to_big = np.where(valid_big_mask, dist_to_big, 100.0)
        data['대규모점포_최단거리'] = np.min(dist_to_big, axis=1)

        data.insert(0, '업종',name)
        
        # 3. 동종업종수 계산 (기존 cKDTree 유지)
        coords_rad = np.radians(data[['위도', '경도']].values)
        tree = BallTree(coords_rad, metric='haversine')
        neighbors_list = tree.query_radius(coords_rad, r=radius_500m_rad)
        
        opened_ts = data['인허가일자'].values.astype('datetime64[D]').astype(np.int64)
        closed_ts = data['폐업일자'].values.astype('datetime64[D]').astype(np.int64)
        competitor_counts = np.zeros(len(data), dtype=np.int32)
        
        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) <= 1:
                continue
            idx = np.array(neighbors)
            is_competitor = (opened_ts[idx] <= opened_ts[i]) & (closed_ts[idx] > opened_ts[i]) & (idx != i)
            competitor_counts[i] = np.sum(is_competitor)

        data['동종업종수'] = competitor_counts
        
        results_list.append(data)
        
    result = pd.concat(results_list, axis=0, ignore_index=True)
    
    # -----------------------------------------------
    
    print("총업종수 계산중...")
    coords_rad = np.radians(result[['위도', '경도']].values)
    tree = BallTree(coords_rad, metric='haversine')
    neighbors_list = tree.query_radius(coords_rad, r=radius_500m_rad)
    
    opened_ts = result['인허가일자'].values.astype('datetime64[D]').astype(np.int64)
    closed_ts = result['폐업일자'].values.astype('datetime64[D]').astype(np.int64)
    competitor_counts = np.zeros(len(result), dtype=np.int32)
    
    chunk_size = 50000
    for start in range(0, len(result), chunk_size):
        end = min(start + chunk_size, len(result))
        chunk_neighbors = tree.query_radius(coords_rad[start:end], r=radius_500m_rad)
        
        for i, neighbors in enumerate(chunk_neighbors, start=start):
            if len(neighbors) <= 1:
                continue
            idx = np.array(neighbors)
            idx = idx[idx != i]
            is_open = (opened_ts[idx] <= opened_ts[i]) & (closed_ts[idx] > opened_ts[i])
            competitor_counts[i] = np.sum(is_open)
        
        print(f"  {end}/{len(result)} 완료")

    result['총업종수'] = competitor_counts
    
    # -----------------------------------------------
    
    result_years = pd.to_datetime(result['인허가일자']).dt.to_period('Q').astype(str)
    
    cost = pd.read_csv('소비자물가지수.csv')
    cost_dict = dict(zip(cost['TIME'].astype(str), cost['DATA_VALUE']))
    result['물가'] = result_years.map(cost_dict)
    
    cost = pd.read_csv('전기료.csv')
    cost_dict = dict(zip(cost['TIME'].astype(str), cost['DATA_VALUE']))
    result['전기료'] = result_years.map(cost_dict)
    
    cost = pd.read_csv('가스비.csv')
    cost_dict = dict(zip(cost['TIME'].astype(str), cost['DATA_VALUE']))
    result['가스비'] = result_years.map(cost_dict)

    result = result.dropna(subset=['물가', '전기료','가스비'])

    ecos = pd.read_csv('ecos_data.csv')
    ecos['TIME'] = pd.to_datetime(ecos['TIME'],format='%Y%m')
    ecos['기준금리'] = ecos['DATA_VALUE']

    result['year_month'] = result['인허가일자'].dt.to_period('M')
    ecos['year_month'] = ecos['TIME'].dt.to_period('M')

    result = pd.merge(result, ecos, on='year_month', how='inner')
    
    result['평균연령']=result['평균연령'].round(2)

    result = result[['업종','경도','위도','㎡당단가(만원)','총인구수','평균연령','인구변화율','유동인구','물가','전기료','가스비','대규모점포_최단거리','기준금리','동종업종수','총업종수','생존일수']]
    result.to_csv('#결과.csv', index=False, encoding='utf-8-sig')
    print("처리 완료 및 파일 저장 성공")