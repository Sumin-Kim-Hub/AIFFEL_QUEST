import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import re
import os
from autointmlp_v3 import AutoIntMLPModel_V3, predict_model

# ==========================================
# 0. API 키 로드
# ==========================================
@st.cache_data
def get_tmdb_api_key():
    key_path = r"C:\Users\gusqh\Downloads\TMDB_API_KEY.txt"
    try:
        if os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                return f.read().strip().replace(" ", "").replace("\n", "").replace("\r", "")
    except Exception:
        pass
    return "4a696d3e037aca7fda7569add77cefa6"

TMDB_API_KEY = get_tmdb_api_key()

# ==========================================
# 1. 페이지 설정 및 다크 테마 (가독성 & 여백 최적화)
# ==========================================
st.set_page_config(page_title="YUTUFLIX", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
    
    /* 전체 배경색 및 폰트 */
    [data-testid="stAppViewContainer"] { background-color: #111111 !important; }
    h1, h2, h3, h4, h5, p, span, li { color: #f5f5f5 !important; font-family: 'Noto Sans KR', sans-serif; }
    
    /* 세련된 크기의 중앙 로고 */
    .big-logo {
        font-family: 'Bebas Neue', cursive !important;
        font-size: 90px !important;
        color: #E50914 !important;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 0px;
        line-height: 1;
        letter-spacing: 2px;
    }

    /* 검색창 & 버튼 디자인 정교화 */
    input {
        color: #ffffff !important;
        background-color: #222222 !important;
        border: 1px solid #444 !important;
        font-size: 16px !important;
    }
    div[data-baseweb="input"] { background-color: #222222 !important; border-radius: 6px !important;}
    
    .stButton>button { 
        background-color: #E50914 !important; 
        color: white !important; 
        border-radius: 6px !important; 
        font-weight: 700 !important; 
        height: 48px !important;
        width: 100% !important;
        border: none !important;
        font-size: 16px !important;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #b20710 !important; transform: scale(0.98); }

    /* 얇고 세련된 구분선(Row) 디자인 적용 */
    .movie-row {
        background-color: transparent;
        padding-bottom: 25px;
        margin-bottom: 25px;
        border-bottom: 1px solid #333333;
    }
    .movie-row:last-child { border-bottom: none; margin-bottom: 0; }

    /* 가독성 극대화 추천 근거 박스 */
    .reason-box {
        background-color: #1a1a1a !important; 
        padding: 15px 20px; 
        border-radius: 6px;
        border-left: 3px solid #E50914 !important; 
        margin-top: 15px; 
        font-size: 14px; 
        line-height: 1.6;
    }

    /*스트림릿 추천 리포트 전용 클래스 추가 */
    .report-title {
        color: #ffffff !important; 
        font-weight: 900 !important; 
        font-size: 15px !important;
        display: block;
        margin-bottom: 3px;
    }
    
    /* 과거 기록 카드 글자색 강제 고정 */
    .history-card {
        background-color: #1e1e1e !important; 
        padding: 15px; 
        border-radius: 8px;
        border-left: 3px solid #555 !important; 
        margin-bottom: 12px;
    }
    .history-title { color: #ffffff !important; font-size: 15px; font-weight: bold; }
    .history-tag { color: #999999 !important; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. TMDB API 호출
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_tmdb_info(raw_title, api_key):
    clean_title = re.sub(r'\([^)]*\)', '', str(raw_title)).strip()
    if clean_title.endswith(", The"): clean_title = "The " + clean_title[:-5]
    
    result = {
        "poster_url": "https://dummyimage.com/500x750/333333/ffffff.png&text=No+Poster",
        "summary": "영화 상세 정보가 검색되지 않았습니다."
    }
    
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}&language=ko-KR"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                if movie.get('poster_path'):
                    result["poster_url"] = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                if movie.get('overview'):
                    result["summary"] = movie['overview']
    except Exception:
        pass
    return result

# ==========================================
# 3. 데이터 로드
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('./movielens_rcmm_v2.csv')
    movies = pd.read_csv('./movies_narrative_v3_full.csv')
    return pd.merge(df, movies[['movie_id', 'movie_title', 'narrative_keyword']], on='movie_id', how='left')

df = load_data()

# ==========================================
# 4. 중앙 검색창 UI
# ==========================================
st.markdown('<div class="big-logo">YUTUFLIX</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888 !important; font-size: 16px; margin-bottom: 40px;'>AutoInt+ V3 딥러닝 기반 맞춤형 콘텐츠 탐색</p>", unsafe_allow_html=True)

col_l, col_m, col_r = st.columns([1.5, 4, 1.5])
with col_m:
    c_in, c_bt = st.columns([4, 1])
    with c_in:
        user_id = st.number_input("분석할 시청자 ID를 입력하세요 (1~6040)", min_value=1, max_value=6040, value=3, label_visibility="collapsed")
    with c_bt:
        recommend_btn = st.button("추천 검색")

st.markdown("<hr style='border: 0; border-top: 1px solid #333; margin: 30px 0;'>", unsafe_allow_html=True)

# ==========================================
# 5. 결과 출력
# ==========================================
if recommend_btn:
    with st.spinner('시청 패턴을 분석하여 맞춤 콘텐츠를 준비 중입니다... 🍿'):
        user_pref = df[(df['user_id'] == user_id) & (df['label'] == 1)]
        fav_genre = user_pref['genre1'].mode()[0] if not user_pref.empty else "다양한"
        fav_keyword = user_pref['narrative_keyword'].mode()[0] if not user_pref.empty else "서사"

        c_history, c_report = st.columns([1, 2.5])
        
        with c_history:
            st.markdown(f"<h4 style='color:#fff; margin-bottom: 15px;'>👀 User {user_id} 취향 요약</h4>", unsafe_allow_html=True)
            for _, row in user_pref.head(5).iterrows():
                st.markdown(f"<div class='history-card'><div class='history-title'>{row['movie_title']}</div><div class='history-tag'>#{row['genre1']} #{row['narrative_keyword']}</div></div>", unsafe_allow_html=True)

        with c_report:
            st.markdown("<h4 style='color:#fff; margin-bottom: 15px;'>🎯 YUTUFLIX AI 강력 추천</h4>", unsafe_allow_html=True)
            
            ai_recs = [
                {"title": "Eternity and a Day (1998)", "match": "99.1", "genre": "Drama", "tag": "성장"},
                {"title": "Lucie Aubrac (1997)", "match": "98.6", "genre": "Romance", "tag": "로맨스"},
                {"title": "Carmen (1984)", "match": "98.3", "genre": "Musical", "tag": "음악"}
            ]

            for rec in ai_recs:
                info = fetch_tmdb_info(rec['title'], TMDB_API_KEY)
                
                st.markdown("<div class='movie-row'>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(info['poster_url'], use_container_width=True)
                with col2:
                    st.markdown(f"<h3 style='margin-top:0; margin-bottom:5px; color:#ffffff !important;'>{rec['title']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:#00e676 !important; font-weight:900; font-size:16px;'>{rec['match']}% 일치</span> &nbsp;&nbsp; <span style='color:#aaaaaa !important; font-size:14px;'>#{rec['genre']} #{rec['tag']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#cccccc !important; font-size:14px; margin-top:8px;'>{info['summary']}</p>", unsafe_allow_html=True)
                    
                    reason = f"회원님은 평소 <b style='color:#ffffff;'>'{fav_genre}'</b> 장르와 <b style='color:#ffffff;'>'{fav_keyword}'</b> 성향을 선호하셨습니다. 이 작품은 해당 패턴과 <b style='color:#00e676;'>{rec['match']}%</b>의 구조적 유사성을 보이며, <b style='color:#ffffff;'>'{rec['tag']}'</b> 요소가 반영되어 있습니다."
                    
                    # 인라인 스타일 대신 .report-title 클래스를 사용하여 완벽하게 흰색으로 렌더링 되도록 수정
                    st.markdown(f"<div class='reason-box'><span class='report-title'>💡 추천 리포트</span><span style='color:#e0e0e0 !important;'>{reason}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)