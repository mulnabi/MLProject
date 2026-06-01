import pandas as pd

from pyproj import Transformer

# 행안부 표준인 EPSG:5174(중부원점)에서 WGS84(위경도)로 변환기 설정
transformer = Transformer.from_crs("epsg:5174", "epsg:4326", always_xy=True)

# --------------------------------
result = pd.DataFrame()
for name in ['경기도','서울특별시','인천광역시','충청북도','충청남도']:
    output_name = '생활_대규모점포'  # 처리할 파일 이름


        
    data = pd.read_csv(f'./{name}/{output_name}.csv',encoding='cp949')
    data['인허가일자'] = pd.to_datetime(data['인허가일자'],errors='coerce')
    data['폐업일자'] = pd.to_datetime(data['폐업일자'],errors='coerce')
    
    today = pd.Timestamp.now().normalize()
    data['폐업일자']=data['폐업일자'].fillna(today+pd.DateOffset(years=2))

    data = data.dropna(subset=['인허가일자'])
    data = data[(data['폐업일자'] - data['인허가일자']).dt.days > 30]

    data = data.dropna(subset=['좌표정보(X)','좌표정보(Y)'])

    data['경도'], data['위도'] = transformer.transform(data['좌표정보(X)'].values, data['좌표정보(Y)'].values)
    
    data = data[['인허가일자','폐업일자','경도','위도']]

    data.info()
    result = pd.concat([result, data], axis=0, ignore_index=True)
result.to_csv(f'./대규모점포.csv', index=False, encoding='utf-8-sig')