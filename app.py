import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib


import sys


# 1. Haversine 공식을 이용한 두 좌표 간의 거리 계산 함수 (단위: km)
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # 지구 반지름 (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2)**2
    return r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

# 2. 카카오 API를 활용한 역지오코딩
def get_address_from_coords(lat, lon):
    KAKAO_API_KEY = "1dd78a4b7627d8212526be45a7a7e0fe" 
    url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={lon}&y={lat}"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        response = requests.get(url, headers=headers).json()
        return response['documents'][1]['address_name']
    except:
        return "주소 확인 불가"
    
def get_sigungu_code_from_coords(lat, lon):
    KAKAO_API_KEY = "1dd78a4b7627d8212526be45a7a7e0fe" 
    url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={lon}&y={lat}"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        response = requests.get(url, headers=headers).json()
        return response['documents'][1]['code'][:5]
    except:
        return "코드 확인 불가"

# 3. 소상공인시장진흥공단 API 호출 함수 (반경 내 상가업소 조회)
def get_nearby_stores_api(lat, lon, radius=500):
    API_KEY = "fe8d33d65644ac0b445f09784f017364ea502f4b1d2e74037f78e3442b72c1e4"
    
    url = "http://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
    
    params = {
        'ServiceKey': API_KEY,
        'pageNo': '1',
        'numOfRows': '1000',  # 가져올 주변 상가 최대 개수
        'radius': radius,    # 반경 (미터 단위)
        'cx': lon,           # 경도 (x좌표)
        'cy': lat,           # 위도 (y좌표)
        'type': 'json'       # 응답 데이터 형태 설정
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('body', {}).get('items')
            if items:
                df = pd.DataFrame(items)
                
                conditions = [
                    (df['indsMclsNm'] == '주점'),
                    (df['indsSclsNm'] == '카페'),
                    df['indsSclsNm'].isin(['빵/도넛', '떡/한과']),
                    (df['indsLclsNm'] == '음식'),
                    (df['indsMclsNm'] == '이용·미용'),
                    df['indsSclsNm'].isin(['스포츠/운동', '실내운동시설']),
                    (df['indsMclsNm'].isin(['유원지·오락'])),
                    (df['indsSclsNm'] == '편의점'),
                ]
                choices = ['주점', '카페', '제과', '음식', '미용', '헬스', '오락', '편의점']

                df['업종'] = np.select(conditions, choices, default='')
                df = df[df['업종'] != ''].reset_index(drop=True)
                df = df[['업종', 'lon', 'lat']]
                # lat, lon 값을 float(실수형)으로 변환
                df['lat'] = df['lat'].astype(float)
                df['lon'] = df['lon'].astype(float)
                return df
        return pd.DataFrame(columns=['업종', 'lon', 'lat'])
        
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return None
    
    
def get_commercial_sales_data(lawd_cd, deal_ymd):
    API_KEY = "fe8d33d65644ac0b445f09784f017364ea502f4b1d2e74037f78e3442b72c1e4"
    
    url = "https://apis.data.go.kr/1613000/RTMSDataSvcNrgTrade/getRTMSDataSvcNrgTrade"

    params = {
        'serviceKey': API_KEY,          # 일반 인증키 (Decoding 키 추천)
        'LAWD_CD': str(lawd_cd),        # 시군구코드 5자리 (예: '11680')
        'DEAL_YMD': str(deal_ymd),      # 계약연월 6자리 (예: '202312')
        'numOfRows': '1000',            # 한 번에 불러올 데이터 수 (넉넉하게 설정)
        'pageNo': '1',
        '_type': 'json'                 # JSON 형식으로 강제 요청
    }

    print(f"지역코드[{lawd_cd}], 계약월[{deal_ymd}] 데이터를 요청합니다...")

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                print("JSON 파싱 에러: 응답이 XML로 왔거나 서비스 장애입니다.")
                print(response.text)
                return None
            try:
                total_count = data['response']['body']['totalCount']
                if total_count == 0:
                    print("해당 조건에 거래된 내역이 없습니다.")
                    return pd.DataFrame()

                items = data['response']['body']['items']['item']

                # 거래가 1건일 경우 딕셔너리로 오고, 여러 건이면 리스트로 오기 때문에 처리해줍니다.
                if isinstance(items, dict):
                    items = [items]

                df = pd.DataFrame(items)

                print("✅ 데이터 불러오기 성공!")
                return df

            except KeyError as e:
                print(f"응답 구조가 예상과 다릅니다: {e}")
                print(data)
                return None
        else:
            print(f"HTTP 통신 에러: {response.status_code}")
            return None

    except Exception as e:
        print(f"시스템 에러: {e}")
        return None


SERVICE_ID = "999b4e643e55440c9506"
SECURITY_KEY = "cd18af2bfd2842babd47"

def get_access_token(service_id, security_key):
    url = "https://sgisapi.mods.go.kr/OpenAPI3/auth/authentication.json"
    params={
        "consumer_key": service_id,
        "consumer_secret": security_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # HTTP 에러 발생 시 예외 처리 확인
        data = response.json()

        if data.get("errCd") == 0:
            print("Access Token 발급 성공!")
            return data["result"]["accessToken"]
        else:
            print(f"토큰 발급 실패: {data.get('errMsg')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"통신 에러 발생: {e}")
        return None

def get_census_data(date,address):
    print(address)
    address = address.split()
    def get_odcloud_uddi(date):
        url = "https://infuser.odcloud.kr/oas/docs?namespace=15097972/v1"
        response = requests.get(url)
        if response.status_code == 200:
            paths = response.json().get('paths', {})
            dict_path = {}
            for path, details in paths.items():
                month = details.get('get', {}).get('summary', '')[-8:-2]
                dict_path[month] = path
            while True:
                YYYYMM=date.strftime('%Y%m')
                if dict_path.get(YYYYMM) is not None:
                    return [dict_path[(date-pd.DateOffset(years=i,months=1)).strftime('%Y%m')]for i in range(3)]
                date=date-pd.DateOffset(months=1)
        return None
    
    uddi_list=get_odcloud_uddi(date)
    searchNo=0
    def get_data(uddi,no=0,searchMode=True):
        nonlocal searchNo
        API_KEY = "fe8d33d65644ac0b445f09784f017364ea502f4b1d2e74037f78e3442b72c1e4"
        headers = {
            "Authorization": f"Infuser {API_KEY}"
        }
        perPage=1000
        if searchMode:
            searchNo=no
        else:
            perPage=500
            no=searchNo+no+1
            
        params ={
            "page": no,
            "perPage": perPage,
            "returnType": "JSON"
        }
        
        response = requests.get("https://api.odcloud.kr/api"+uddi, params=params, headers=headers)
        if response.status_code == 200:
            return pd.DataFrame(response.json()['data'])
        else:
            print(f"❌ HTTP 통신 에러: {response.status_code}")
            print("에러 내용:", response.text)
    
    AGE_WEIGHTS = np.repeat(np.arange(60),2)
    POP_COLS    = [f"{i}세{g}" for i in range(60) for g in ['남자', '여자']]
    
    start=True
    result_list=[]
    for uddi in uddi_list:
        dataNN=False
        data_list=[]
        for no in range(100):
            df = get_data(uddi,no,searchMode=dataNN==False)
            if df is None:
                return None
            if len(df.index) == 0:
                break
            
            df=df[df['시도명'] == address[0]]
            
            sigungu=df['시군구명'].str.split()
            df=df[sigungu.str[0] == address[1]]
            df = df.reset_index(drop=True)
            sigungu = df['시군구명'].str.split()
            
            length = len(df.index)
            if dataNN==False and length == 0:
                continue
            else :
                dataNN=True
            if length == 0:
                break
            print(f"✅ {address} 데이터 로드 성공!")
            
            if address[2]:
                df_buf=df[sigungu.str[1] == address[2]]
                if len(df_buf.index)>0:
                    df = df_buf.reset_index(drop=True)
            pop_matrix = df[POP_COLS].astype(int)

            df['평균연령'] = (pop_matrix @ AGE_WEIGHTS) / df['계'].astype(int)
            df['총인구수']=df['계'].astype(int)
            
            df=df[['총인구수','평균연령']]
            data_list.append(df)
        data = pd.concat(data_list, axis=0, ignore_index=True)
            
        data=data.mean(axis=0)
        result_list.append(data)
    result = result_list[0]
    
    result['인구변화율']=(result_list[0]['총인구수']/result_list[1]['총인구수']+result_list[1]['총인구수']/result_list[2]['총인구수'])/2
    
    return result

def get_ecos_data(date, cycle='Q', code1='901Y009', code2='0'):
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/I1UN5RLR648PT4ST9SVI/json/kr/1/1/{code1}/{cycle}/{date}/{date}/{code2}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            print(data)
            if "RESULT" in data and data["RESULT"]["CODE"] != "OK":
                print(f"API 오류: {data['RESULT']['MESSAGE']}")
            else:
                df = pd.DataFrame(data["StatisticSearch"]["row"])
                return float(df['DATA_VALUE'].values[0])
        except requests.exceptions.JSONDecodeError:
            print("오류: 서버가 정상적인 JSON 데이터를 반환하지 않았다.")
            print("--- 원본 데이터 확인 ---")
            print(response.text[:500])
    else:
        print(f"HTTP 에러 발생: {response.status_code}")

class MO(nn.Module):
    def __init__(self,cat_l,num_l):
        super(MO, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=cat_l, embedding_dim=3)
        self.fc0 = nn.Linear(3+num_l, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.025)
    def forward(self, x):
        x_cat=x[:,0].long()
        x_cont=x[:,1:]
        emb = self.embedding(x_cat)
        x = torch.cat([emb,x_cont], dim=1)
        x =              self.act(self.fc0(x))
        x1= x
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))+x1
        x = self.dropout(self.act(self.fc3(x)))
        x = self.dropout(self.act(self.fc4(x)))
        x = self.dropout(self.act(self.fc5(x)))
        x = self.dropout(self.act(self.fc6(x)))
        x = self.fc7(x)
        return F.softplus(x)

# 모델 불러오기

# from collections import OrderedDict

le = joblib.load("label_encoder.joblib")
STDscaler=joblib.load("standard_scaler.joblib")
MMscaler=joblib.load("minmax_scaler.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MO(cat_l=len(le.classes_),num_l=14).to(device)
state_dict = torch.load('survival_model.pth', map_location=torch.device('cpu'))


def predict(data):
    print(data)
    data['업종'] = le.transform(data['업종'])
    cat_l = len(le.classes_)

    cols = ['업종'] + [col for col in data.columns if col != '업종']
    data = data[cols]
    
    data["㎡당단가(만원)"] = np.log1p(data["㎡당단가(만원)"])
    data["유동인구"]       = np.sqrt(data["유동인구"])
    data['동종업종수']     = np.log1p(data['동종업종수'])
    data['대규모점포_최단거리'] = 1/(1+np.log1p(data['대규모점포_최단거리'])/2)
    
    target_cols = ['경도','위도','㎡당단가(만원)','평균연령','인구변화율','물가','전기료','가스비','기준금리']
    data[target_cols] = STDscaler.fit_transform(data[target_cols])

    target_cols = ['유동인구','대규모점포_최단거리','총인구수','동종업종수','총업종수']
    data[target_cols] = MMscaler.fit_transform(data[target_cols])
    
    single_input = torch.tensor(data.values, dtype=torch.float32)

    with torch.no_grad():
        output = model(single_input)
    return output.item() * 3650

# ==========================================
# 대시보드 UI 구성
# ==========================================
from folium import MacroElement
from jinja2 import Template
class SingleClickMarker(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template("""
            {% macro script(this, kwargs) %}
                var clickMarker = null;
                {{this._parent.get_name()}}.on('click', function(e) {
                    if (clickMarker) {
                        {{this._parent.get_name()}}.removeLayer(clickMarker);
                    }
                    clickMarker = L.marker(e.latlng).addTo({{this._parent.get_name()}});
                });
            {% endmacro %}
        """)

st.title("지역 상권 폐업일 예측 대시보드")
st.write("지도에서 분석을 원하는 상권 위치를 클릭하세요.")

# 중심 좌표 설정 및 지도 객체 생성 (예: 대전광역시 서구청 부근)
m = folium.Map(location=[36.3504, 127.3845], zoom_start=14)
m.add_child(SingleClickMarker())

map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])

lob=st.selectbox("예측할 업종은 선택해 주세요", ["음식", "제과", "카페", "미용", "주점", "헬스", "오락", "편의점"])

if "previous_lob" not in st.session_state:
    st.session_state["previous_lob"] = "음식"

if (map_data or lob != st.session_state["previous_lob"]) and map_data.get("last_clicked") :
    st.session_state["previous_lob"] = lob
    
    click_lat = map_data["last_clicked"]["lat"]
    click_lon = map_data["last_clicked"]["lng"]
    
    print(lob)
    # 클릭한 좌표를 실제 주소로 변환
    address = get_address_from_coords(click_lat, click_lon)
    st.subheader(f"📍 선택된 지역: {address} (위도: {click_lat:.4f}, 경도: {click_lon:.4f})")
    
    with st.spinner("데이터 수집 및 전처리 중입니다..."):
        df = get_nearby_stores_api(click_lat, click_lon, radius=500)
        
        if df is None:
            st.info("오류가 발생하였습니다.")
            sys.exit()
        
        sigungu_code=get_sigungu_code_from_coords(click_lat, click_lon)
        
        date=pd.Timestamp.now()
        
        YYYYMM=date.strftime('%Y%m')
        LandDeal=get_commercial_sales_data(sigungu_code,'202406')
        LandDeal['dealAmount']=LandDeal['dealAmount'].astype(str).str.replace(',', '').astype(int)
        LandDeal['buildingAr'] = LandDeal['buildingAr'].astype(float)
        LandDeal['㎡당단가(만원)'] = LandDeal['dealAmount'] / LandDeal['buildingAr']
        LandDealAmount=LandDeal['㎡당단가(만원)'].mean()
        
        census=get_census_data(date=date,address=address)
        
        YYYYQ = str((date-pd.DateOffset(months=3)).to_period('Q'))
        price=get_ecos_data(YYYYQ)
        electricity=get_ecos_data(YYYYQ,code2='D051')
        gas=get_ecos_data(YYYYQ,code2='D052')
        YYYYMM=(date-pd.DateOffset(months=1)).strftime('%Y%m')
        interest=get_ecos_data(YYYYMM,cycle='M', code1='722Y001',code2='0101000')
        
        
    
    inputdata=pd.DataFrame([{
    '업종': lob,
    '위도': click_lat,
    '경도': click_lon,
    '㎡당단가(만원)': LandDealAmount,
    '총인구수': census['총인구수'],
    '평균연령': census['평균연령'],
    '인구변화율': census['인구변화율'],
    '유동인구': census['총인구수']*.1,
    '물가': price,
    '전기료': electricity,
    '가스비': gas,
    '대규모점포_최단거리': 1.5,
    '기준금리': interest,
    '동종업종수': (df['업종'] == lob).sum(),
    '총업종수': len(df)
    }])
    # ML 모델 예측
    with st.spinner("AI 모델이 예측을 수행 중입니다..."):
        # survival_days = 600
        survival_days=predict(inputdata)-400
        if survival_days/365>5.5:
            st.success(f"✅ **해당 구역의 소상공인 평균 생존기간: {(survival_days-200)/365:.1f}~{(survival_days+200)/365:.1f}년**")
        elif survival_days/365>4:
            st.warning(f"⚠️ **해당 구역의 소상공인 평균 생존기간: {(survival_days-200)/365:.1f}~{(survival_days+200)/365:.1f}년**")
        else:
            st.error(f"☠️ **해당 구역의 소상공인 평균 생존기간: {(survival_days-200)/365:.1f}~{(survival_days+200)/365:.1f}년**")
        
        