"""
로보틱스 및 AI 논문 트렌드 분석 대시보드
Streamlit을 사용한 인터랙티브 시각화 애플리케이션
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="로보틱스 & AI 논문 트렌드 분석",
    page_icon="🤖",
    layout="wide"
)

# NLTK 불용어 다운로드 (처음 실행 시)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOPWORDS = set(stopwords.words('english'))
except:
    # NLTK가 없을 경우 기본 불용어 리스트 사용
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
        'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }


@st.cache_data
def load_data():
    """CSV 파일에서 데이터를 로드합니다."""
    try:
        # 두 개의 파일을 읽어서 합침 (pd.concat)
        df1 = pd.read_csv("arxiv_data_part1.csv.gz", encoding='utf-8-sig')
        df2 = pd.read_csv("arxiv_data_part2.csv.gz", encoding='utf-8-sig')
        
        # 위아래로 붙이기
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Published_Date를 datetime으로 변환
        df['Published_Date'] = pd.to_datetime(df['Published_Date'], errors='coerce')
        # 연도 컬럼 추가 (정수로 변환)
        df['Year'] = df['Published_Date'].dt.year
        
        # 결측값 제거
        df = df.dropna(subset=['Year', 'Abstract'])
        
        # Year를 정수로 변환하고 유효한 연도만 필터링 (1900-2100 사이)
        df['Year'] = df['Year'].astype(float).fillna(0).astype(int)
        df = df[(df['Year'] >= 1900) & (df['Year'] <= 2100)]
        
        # Year가 0인 경우 제거 (유효하지 않은 날짜)
        df = df[df['Year'] > 0]
        
        return df
    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e}\n먼저 collector.py를 실행하여 데이터를 수집해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.stop()


# 언어별 텍스트 매핑
TEXTS = {
    'ko': {
        'title': '로보틱스 & AI 논문 트렌드 분석 대시보드',
        'settings': '⚙️ 설정',
        'language': '🌐 언어 선택',
        'clear_cache': '🔄 캐시 클리어 & 새로고침',
        'clear_cache_help': '캐시를 클리어하고 페이지를 새로고침합니다',
        'clear_cache_caption': '💡 캐시 누적으로 목록의 순서가 안맞을때 누르세요',
        'category_selection': '📂 카테고리 그룹 선택',
        'select_category_warning': '⚠️ 최소 1개 이상의 카테고리 그룹을 선택해주세요.',
        'year_range': '연도 범위 선택',
        'data_year': '데이터 연도',
        'top_n_keywords': '분석할 Top N 키워드 개수',
        'keyword_source': '키워드 추출 소스',
        'keyword_source_help': 'Abstract: 논문 초록에서 키워드 추출\nTitle: 논문 제목에서 키워드 추출',
        'analyze_start': '🚀 분석 시작',
        'select_category_error': '카테고리 그룹을 선택해주세요!',
        'extracting_keywords': '에서 키워드 추출 중...',
        'data_summary': '📊 데이터 요약',
        'selected_papers': '선택된 논문 수',
        'total_papers': '전체 논문 수',
        'papers': '총 논문 수',
        'year_range_label': '연도 범위',
        'selected_groups': '선택된 그룹',
        'groups': '개',
        'included_categories': '포함 카테고리',
        'categories': '개',
        'keyword_source_label': '키워드 소스',
        'select_category_info': '👆 카테고리를 선택하고 \'분석 시작\' 버튼을 클릭하세요.',
        'start_analysis_info': '👈 사이드바에서 카테고리를 선택하고 \'분석 시작\' 버튼을 클릭하여 분석을 시작하세요.',
        'no_data_warning': '선택된 카테고리에 해당하는 데이터가 없습니다.',
        'no_keywords_warning': '키워드를 추출할 수 없습니다. 다른 설정을 시도해보세요.',
        'tab_trend': '📈 연도별 흐름 (Trend)',
        'tab_heatmap': '🔥 히트맵 (Heatmap)',
        'tab_bump': '🏆 순위 경쟁 (Bump Chart)',
        'tab_hype': '🚀 하이프 사이클 (Hype Cycle)',
        'trend_header': '연도별 주요 키워드 트렌드',
        'trend_desc': '시간에 따른 주요 키워드의 논문 수 변화를 시각화합니다.',
        'trend_chart_title': '연도별 주요 키워드 트렌드 (Streamgraph)',
        'trend_xaxis': '연도',
        'trend_yaxis': '논문 수',
        'keyword_stats': '키워드별 통계',
        'total_papers_col': '총 논문 수',
        'avg_papers_col': '평균 논문 수',
        'max_papers_col': '최대 논문 수',
        'heatmap_header': '연도별 키워드 빈도 히트맵',
        'heatmap_desc': '연도와 키워드의 교차점에서 논문 수를 색상 농도로 표현합니다.',
        'heatmap_title': '연도별 키워드 빈도 히트맵',
        'heatmap_xaxis': '연도',
        'heatmap_yaxis': '키워드',
        'heatmap_colorbar': '논문 수',
        'bump_header': '연도별 키워드 순위 변화',
        'bump_desc': '시간에 따른 키워드의 순위 변화를 추적합니다.',
        'bump_title': '연도별 키워드 순위 변화 (Top {})',
        'bump_xaxis': '연도',
        'bump_yaxis': '순위',
        'hype_header': '하이프 사이클 스타일 차트',
        'hype_desc': '기술 성장률과 총 언급량을 기준으로 키워드를 배치합니다.',
        'hype_xaxis_label': 'X축',
        'hype_yaxis_label': 'Y축',
        'hype_size_label': '점 크기',
        'hype_xaxis_desc': '기술 성장률 (전년 대비 증가율)',
        'hype_yaxis_desc': '기술 언급량 (Total Volume)',
        'hype_size_desc': '최신 연도 언급량',
        'hype_title': '하이프 사이클 스타일 차트',
        'hype_xaxis': '기술 성장률 (전년 대비 증가율, %)',
        'hype_yaxis': '기술 언급량 (Total Volume)',
        'hype_colorbar': '최신 연도<br>언급량',
        'hype_hover_total': '총 언급량',
        'hype_hover_growth': '성장률',
        'hype_hover_latest': '최신 연도 언급량',
        'hype_no_data': '데이터가 부족하여 차트를 생성할 수 없습니다.',
        'footer_notes': '📝 참고사항',
        'footer_data_source': '데이터는 arXiv API를 통해 수집되었습니다.',
        'footer_keyword_source': '키워드는 논문 제목(Title)과 초록(Abstract)에서 추출되었으며, 불필요한 단어는 제외되었습니다.',
        'footer_period': '데이터 수집기간 : 2021~2025년, (만든이 : Minsu Hwang)',
        'papers_count': '건',
        'categories_count': '개 카테고리'
    },
    'en': {
        'title': 'Robotics & AI Paper Trend Analysis Dashboard',
        'settings': '⚙️ Settings',
        'language': '🌐 Language',
        'clear_cache': '🔄 Clear Cache & Refresh',
        'clear_cache_help': 'Clear cache and refresh the page',
        'clear_cache_caption': '💡 Click when category list order is incorrect due to cache accumulation',
        'category_selection': '📂 Category Group Selection',
        'select_category_warning': '⚠️ Please select at least one category group.',
        'year_range': 'Year Range Selection',
        'data_year': 'Data Year',
        'top_n_keywords': 'Top N Keywords to Analyze',
        'keyword_source': 'Keyword Extraction Source',
        'keyword_source_help': 'Abstract: Extract keywords from paper abstracts\nTitle: Extract keywords from paper titles',
        'analyze_start': '🚀 Start Analysis',
        'select_category_error': 'Please select a category group!',
        'extracting_keywords': 'Extracting keywords from...',
        'data_summary': '📊 Data Summary',
        'selected_papers': 'Selected Papers',
        'total_papers': 'Total Papers',
        'papers': 'Total Papers',
        'year_range_label': 'Year Range',
        'selected_groups': 'Selected Groups',
        'groups': 'groups',
        'included_categories': 'Included Categories',
        'categories': 'categories',
        'keyword_source_label': 'Keyword Source',
        'select_category_info': '👆 Please select categories and click the \'Start Analysis\' button.',
        'start_analysis_info': '👈 Please select categories in the sidebar and click the \'Start Analysis\' button to begin analysis.',
        'no_data_warning': 'No data available for the selected categories.',
        'no_keywords_warning': 'Unable to extract keywords. Please try different settings.',
        'tab_trend': '📈 Trend Over Time',
        'tab_heatmap': '🔥 Heatmap',
        'tab_bump': '🏆 Ranking Competition (Bump Chart)',
        'tab_hype': '🚀 Hype Cycle',
        'trend_header': 'Annual Keyword Trends',
        'trend_desc': 'Visualizes changes in the number of papers for major keywords over time.',
        'trend_chart_title': 'Annual Major Keyword Trends (Streamgraph)',
        'trend_xaxis': 'Year',
        'trend_yaxis': 'Number of Papers',
        'keyword_stats': 'Keyword Statistics',
        'total_papers_col': 'Total Papers',
        'avg_papers_col': 'Average Papers',
        'max_papers_col': 'Max Papers',
        'heatmap_header': 'Annual Keyword Frequency Heatmap',
        'heatmap_desc': 'Represents the number of papers at the intersection of years and keywords using color intensity.',
        'heatmap_title': 'Annual Keyword Frequency Heatmap',
        'heatmap_xaxis': 'Year',
        'heatmap_yaxis': 'Keyword',
        'heatmap_colorbar': 'Number of Papers',
        'bump_header': 'Annual Keyword Ranking Changes',
        'bump_desc': 'Tracks changes in keyword rankings over time.',
        'bump_title': 'Annual Keyword Ranking Changes (Top {})',
        'bump_xaxis': 'Year',
        'bump_yaxis': 'Rank',
        'hype_header': 'Hype Cycle Style Chart',
        'hype_desc': 'Places keywords based on technology growth rate and total mentions.',
        'hype_xaxis_label': 'X-axis',
        'hype_yaxis_label': 'Y-axis',
        'hype_size_label': 'Point Size',
        'hype_xaxis_desc': 'Technology Growth Rate (Year-over-Year Increase, %)',
        'hype_yaxis_desc': 'Technology Mentions (Total Volume)',
        'hype_size_desc': 'Latest Year Mentions',
        'hype_title': 'Hype Cycle Style Chart',
        'hype_xaxis': 'Technology Growth Rate (Year-over-Year Increase, %)',
        'hype_yaxis': 'Technology Mentions (Total Volume)',
        'hype_colorbar': 'Latest Year<br>Mentions',
        'hype_hover_total': 'Total Mentions',
        'hype_hover_growth': 'Growth Rate',
        'hype_hover_latest': 'Latest Year Mentions',
        'hype_no_data': 'Insufficient data to generate chart.',
        'footer_notes': '📝 Notes',
        'footer_data_source': 'Data was collected through the arXiv API.',
        'footer_keyword_source': 'Keywords were extracted from paper titles and abstracts, with unnecessary words excluded.',
        'footer_period': 'Data Collection Period: 2021~2025, (Created by: Minsu Hwang)',
        'papers_count': 'papers',
        'categories_count': 'categories'
    }
}

def get_language():
    """현재 선택된 언어를 반환합니다."""
    if 'language' not in st.session_state:
        st.session_state.language = 'ko'
    return st.session_state.language

def get_text(key):
    """언어에 맞는 텍스트를 반환합니다."""
    lang = get_language()
    return TEXTS[lang].get(key, key)

# 카테고리명 한글 매핑
CATEGORY_NAMES_KR = {
    "cs.AI": "인공지능",
    "cs.LG": "머신러닝",
    "cs.NE": "신경망",
    "stat.ML": "통계학습",
    "math.OC": "최적화",
    "math.ST": "통계이론",
    "math.PR": "확률론",
    "cs.CV": "컴퓨터비전",
    "eess.IV": "영상처리",
    "eess.SP": "신호처리",
    "cs.MM": "멀티미디어",
    "cs.SD": "음성처리",
    "physics.optics": "광학",
    "cs.RO": "로보틱스",
    "eess.SY": "시스템제어",
    "cs.MA": "다중에이전트",
    "cs.AR": "하드웨어",
    "physics.app-ph": "응용물리",
    "cs.DB": "데이터베이스",
    "cs.IR": "정보검색",
    "cs.DC": "분산컴퓨팅",
    "cs.PF": "성능분석",
    "cs.IT": "정보이론",
    "cs.SE": "소프트웨어공학",
    "cs.OS": "운영체제",
    "cs.PL": "프로그래밍언어",
    "cs.SC": "기호계산",
    "cs.CE": "컴퓨터공학",
    "cs.HC": "인간컴퓨터상호작용",
    "cs.CY": "컴퓨터와사회",
    "cs.SI": "사회정보학",
    "econ.GN": "일반경제학",
    "econ.TH": "경제이론",
    "cs.DS": "자료구조",
    "cs.LO": "논리학",
    "cs.CC": "계산복잡도",
    "math.NA": "수치해석",
    "math.DS": "동역학시스템",
    "physics.comp-ph": "계산물리",
    "physics.flu-dyn": "유체역학",
    "physics.plasm-ph": "플라즈마물리",
    "cond-mat.mtrl-sci": "재료과학",
    "cond-mat.stat-mech": "통계역학",
    "q-bio.NC": "신경계산",
    "q-bio.QM": "정량적방법",
    "q-bio.BM": "생체분자",
    "q-bio.MN": "분자네트워크",
    "q-fin.TR": "거래이론",
    "q-fin.PM": "포트폴리오관리",
    "q-fin.RM": "리스크관리",
    "econ.EM": "계량경제학"
}

# 카테고리 그룹 매핑
CATEGORY_GROUPS = {
    "1️⃣ AI / Machine Learning": {
        "categories": ["cs.AI", "cs.LG", "cs.NE", "stat.ML", "math.OC", "math.ST", "math.PR"],
        "description": "인공지능, 머신러닝, 신경망, 통계, 최적화"
    },
    "2️⃣ Computer Vision / Perception": {
        "categories": ["cs.CV", "eess.IV", "eess.SP", "cs.MM", "cs.SD", "physics.optics"],
        "description": "컴퓨터 비전, 영상 처리, 신호 처리, 멀티미디어"
    },
    "3️⃣ Robotics / Control / Autonomous Systems": {
        "categories": ["cs.RO", "eess.SY", "cs.MA", "cs.AR", "physics.app-ph"],
        "description": "로보틱스, 제어 시스템, 다중 에이전트, 하드웨어"
    },
    "4️⃣ Data / Information Systems": {
        "categories": ["cs.DB", "cs.IR", "cs.DC", "cs.PF", "cs.IT"],
        "description": "데이터베이스, 정보 검색, 분산 컴퓨팅, 정보 이론"
    },
    "5️⃣ Software / Systems Engineering": {
        "categories": ["cs.SE", "cs.OS", "cs.PL", "cs.SC", "cs.CE"],
        "description": "소프트웨어 공학, 운영체제, 프로그래밍 언어"
    },
    "6️⃣ Human / Society / Interaction": {
        "categories": ["cs.HC", "cs.CY", "cs.SI", "econ.GN", "econ.TH"],
        "description": "인간-컴퓨터 상호작용, 사회정보학, 경제학"
    },
    "7️⃣ Optimization / Theory / Foundations": {
        "categories": ["cs.DS", "cs.LO", "cs.CC", "math.NA", "math.DS"],
        "description": "자료구조, 알고리즘, 논리학, 계산 복잡도, 수치해석"
    },
    "8️⃣ Physics-based Modeling / Simulation": {
        "categories": ["physics.comp-ph", "physics.flu-dyn", "physics.plasm-ph", 
                      "cond-mat.mtrl-sci", "cond-mat.stat-mech"],
        "description": "계산 물리, 유체역학, 플라즈마, 재료 과학"
    },
    "9️⃣ Bio / Neuro-inspired Computing": {
        "categories": ["q-bio.NC", "q-bio.QM", "q-bio.BM", "q-bio.MN"],
        "description": "신경 계산, 생체 분자, 분자 네트워크"
    },
    "🔟 Finance / Economics / Risk": {
        "categories": ["q-fin.TR", "q-fin.PM", "q-fin.RM", "econ.EM"],
        "description": "거래 이론, 포트폴리오 관리, 리스크 관리, 계량 경제학"
    }
}


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda x: hash(tuple(x.shape))})
def get_all_categories(df):
    """데이터에서 모든 고유 카테고리를 추출하고 그룹 정보를 반환합니다."""
    all_categories = set()
    for categories_str in df['Category'].dropna():
        # 쉼표로 구분된 카테고리 분리
        categories = [cat.strip() for cat in str(categories_str).split(',')]
        all_categories.update(categories)
    
    # 그룹별로 사용 가능한 카테고리 정리 및 논문 수 계산
    # CATEGORY_GROUPS의 순서를 유지하기 위해 OrderedDict 사용
    from collections import OrderedDict
    available_groups = OrderedDict()
    
    # CATEGORY_GROUPS의 정의된 순서대로 처리 (중복 완전 차단)
    # 리스트로 변환하여 순서 보장
    category_group_items = list(CATEGORY_GROUPS.items())
    processed_groups = set()
    
    for group_name, group_info in category_group_items:
        # 엄격한 중복 체크
        if group_name in processed_groups:
            continue
        if group_name in available_groups:
            continue
        processed_groups.add(group_name)
        
        available_cats = [cat for cat in group_info["categories"] if cat in all_categories]
        if available_cats:  # 데이터에 있는 카테고리만 포함
            # 해당 그룹의 카테고리를 포함하는 논문 수 계산
            group_mask = df['Category'].apply(
                lambda x: any(cat in str(x) for cat in available_cats) if pd.notna(x) else False
            )
            paper_count = group_mask.sum()
            
            # 최종 중복 체크 후 추가
            if group_name not in available_groups:
                available_groups[group_name] = {
                    "categories": available_cats,
                    "description": group_info["description"],
                    "paper_count": paper_count
                }
    
    # 최종 검증: 중복이 없는지 확인
    final_groups = OrderedDict()
    for group_name, group_info in available_groups.items():
        if group_name not in final_groups:
            final_groups[group_name] = group_info
    
    return sorted(list(all_categories)), final_groups


def filter_by_categories(df, selected_groups):
    """선택된 그룹에 해당하는 논문만 필터링합니다."""
    if not selected_groups:
        return df
    
    # 선택된 그룹의 모든 카테고리 수집
    selected_categories = set()
    for group_name in selected_groups:
        if group_name in CATEGORY_GROUPS:
            selected_categories.update(CATEGORY_GROUPS[group_name]["categories"])
    
    if not selected_categories:
        return df
    
    # 각 논문의 Category 컬럼에 선택된 카테고리가 하나라도 포함되어 있는지 확인
    mask = df['Category'].apply(
        lambda x: any(cat in str(x) for cat in selected_categories) if pd.notna(x) else False
    )
    return df[mask].copy()


# 제외할 bigram 키워드 (stopwords)
EXCLUDED_BIGRAMS = {
    'https github',
    'available https',
    'open source',
    'publicly available',
    'success rate',
    'code available',
    'success rates',
    'decision making',
    'large scale',
    'fine tuned',
    'domain specific',
    'source code',
    'latent space',
    'medical imaging'
}

# 중복 용어 통합 매핑 (복수형/변형 → 단수형/표준형)
KEYWORD_NORMALIZATION = {
    # 복수형 → 단수형
    'neural networks': 'neural network',
    'point clouds': 'point cloud',
    'autonomous vehicles': 'autonomous driving',
    # 불완전한 용어 → 완전한 용어
    'deep reinforcement': 'deep reinforcement learning',
    'simulation real': 'sim-to-real',
    'large language': 'large language model',
    'vision language': 'vision-language model',
    'fine tuning': 'fine-tuning'
}


def normalize_keywords(keywords):
    """
    중복된 용어들을 하나로 통합합니다.
    
    Parameters:
    -----------
    keywords : list
        키워드 리스트
    
    Returns:
    --------
    list
        정규화된 키워드 리스트
    """
    if not keywords:
        return []
    
    normalized = []
    seen_normalized = set()
    
    for keyword in keywords:
        # 정규화 매핑 확인
        normalized_keyword = KEYWORD_NORMALIZATION.get(keyword, keyword)
        
        # 이미 정규화된 키워드가 있으면 스킵
        if normalized_keyword in seen_normalized:
            continue
        
        # 정규화된 키워드 추가
        normalized.append(normalized_keyword)
        seen_normalized.add(normalized_keyword)
    
    return normalized


def filter_keywords(keywords):
    """
    추출된 키워드를 정제하고 중복을 제거합니다.
    
    Parameters:
    -----------
    keywords : list
        추출된 키워드 리스트
    
    Returns:
    --------
    list
        정제된 키워드 리스트
    """
    if not keywords:
        return []
    
    # 제외할 bigram 키워드 필터링
    filtered = [kw for kw in keywords if kw not in EXCLUDED_BIGRAMS]
    
    # 제외할 일반적인 단어들 (추가 필터링)
    exclude_words = {
        'paper', 'propose', 'proposed', 'method', 'methods', 'approach', 'algorithm',
        'system', 'systems', 'model', 'models', 'framework', 'technique', 'solution', 'problem',
        'task', 'tasks', 'result', 'results', 'experiment', 'experimental',
        'evaluation', 'performance', 'improve', 'improved', 'novel', 'new',
        'present', 'presented', 'show', 'demonstrate', 'demonstrated',
        'effective', 'efficient', 'accurate', 'better', 'significantly',
        'compared', 'comparison', 'state', 'art', 'baseline', 'existing',
        'work', 'works', 'study', 'research', 'application', 'applications',
        'data', 'dataset', 'datasets', 'image', 'images', 'video', 'videos',
        'using', 'used', 'use', 'based', 'provide', 'provides', 'enables',
        'enable', 'allows', 'allow', 'achieve', 'achieves', 'obtain',
        'obtained', 'different', 'various', 'multiple', 'multi', 'several', 'many',
        'important', 'significant', 'recent', 'recently', 'current',
        'currently', 'previous', 'previously', 'future', 'potential',
        'challenging', 'challenge', 'challenges', 'difficult', 'difficulty',
        'complex', 'complexity', 'simple', 'simpler', 'easy', 'easier',
        # 로보틱스/AI 분야에서 너무 일반적인 무의미한 키워드
        'robot', 'robots', 'robotic', 'control', 'planning', 'motion',
        'human', 'environments', 'object', 'high',
        # 추가 불용어
        'however', 'also', 'first', 'introduce', 'address', 'across',
        'level', 'information', 'experiments', 'approaches'
    }
    
    # 숫자만 포함된 키워드 제거
    filtered = []
    for kw in keywords:
        # 단일 단어인 경우
        if ' ' not in kw:
            # 제외 단어에 포함되지 않고, 숫자만이 아닌 경우
            if kw not in exclude_words and not kw.isdigit():
                filtered.append(kw)
        else:
            # bigram인 경우 - 각 단어가 모두 제외 단어가 아닌 경우만 포함
            words = kw.split()
            if not any(w in exclude_words for w in words):
                filtered.append(kw)
    
    return filtered


def remove_duplicate_keywords(keywords):
    """
    중복 키워드를 제거합니다.
    bigram만 사용하므로 단순히 중복 제거만 수행합니다.
    
    Parameters:
    -----------
    keywords : list
        키워드 리스트 (bigram만)
    
    Returns:
    --------
    list
        중복이 제거된 키워드 리스트
    """
    if not keywords:
        return []
    
    # bigram만 사용하므로 중복 제거만 수행
    return list(set(keywords))


def extract_keywords(text, ngram_range=(2, 2), max_features=150):
    """
    텍스트에서 키워드를 추출합니다.
    
    Parameters:
    -----------
    text : str or pd.Series
        추출할 텍스트
    ngram_range : tuple
        n-gram 범위 (기본값: (2, 2) - bigram만)
    max_features : int
        최대 특징 개수
    
    Returns:
    --------
    list
        추출된 키워드 리스트
    """
    if isinstance(text, pd.Series):
        text = ' '.join(text.astype(str))
    
    # 확장된 불용어 리스트
    extended_stopwords = STOPWORDS | {
        'paper', 'propose', 'proposed', 'method', 'methods', 'approach', 'approaches', 'algorithm',
        'system', 'systems', 'model', 'models', 'framework', 'technique', 'solution', 'problem',
        'task', 'tasks', 'result', 'results', 'experiment', 'experiments', 'experimental',
        'evaluation', 'performance', 'improve', 'improved', 'novel', 'new',
        'present', 'presented', 'show', 'demonstrate', 'demonstrated',
        'effective', 'efficient', 'accurate', 'better', 'significantly',
        'compared', 'comparison', 'state', 'art', 'baseline', 'existing',
        'work', 'works', 'study', 'research', 'application', 'applications',
        'using', 'used', 'use', 'based', 'provide', 'provides', 'enables',
        'enable', 'allows', 'allow', 'achieve', 'achieves', 'obtain',
        'obtained', 'different', 'various', 'multiple', 'multi', 'several', 'many',
        'important', 'significant', 'recent', 'recently', 'current',
        'currently', 'previous', 'previously', 'future', 'potential',
        # 로보틱스/AI 분야에서 너무 일반적인 무의미한 키워드
        'robot', 'robots', 'robotic', 'control', 'planning', 'motion',
        'human', 'environments', 'object', 'high',
        # 추가 불용어
        'however', 'also', 'first', 'introduce', 'address', 'across', 
        'level', 'information'
    }
    
    # CountVectorizer 사용 (bigram만)
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),  # bigram만 사용
        max_features=max_features,
        stop_words=list(extended_stopwords),
        lowercase=True,
        token_pattern=r'\b[a-z]{4,}\b'  # 최소 4글자 이상의 단어만
    )
    
    try:
        vectorizer.fit([text])
        keywords = vectorizer.get_feature_names_out().tolist()
        
        # 키워드 정제
        keywords = filter_keywords(keywords)
        
        # 중복 용어 통합
        keywords = normalize_keywords(keywords)
        
        # 중복 제거
        keywords = remove_duplicate_keywords(keywords)
        
        return keywords
    except:
        return []


@st.cache_data
def get_top_keywords_by_year(df, top_n=20, source='abstract'):
    """
    연도별 상위 키워드를 추출합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        논문 데이터프레임
    top_n : int
        상위 N개 키워드
    source : str
        키워드 추출 소스 ('abstract' 또는 'title')
    
    Returns:
    --------
    pd.DataFrame
        연도별 키워드 빈도 데이터프레임
    """
    # 소스 컬럼 선택
    source_column = 'Abstract' if source.lower() == 'abstract' else 'Title'
    
    # 전체 텍스트에서 키워드 추출 (bigram만)
    all_text = ' '.join(df[source_column].astype(str))
    all_keywords = extract_keywords(all_text, ngram_range=(2, 2), max_features=200)
    
    # 제외할 bigram 키워드 제거
    all_keywords = [kw for kw in all_keywords if kw not in EXCLUDED_BIGRAMS]
    
    # 연도별 키워드 빈도 계산
    year_keyword_counts = []
    
    # 연도를 정수로 변환하고 유효한 연도만 필터링
    df['Year'] = df['Year'].astype(float).fillna(0).astype(int)
    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2100) & (df['Year'] > 0)]
    
    for year in sorted(df['Year'].unique()):
        year = int(year)  # 정수로 변환
        year_df = df[df['Year'] == year]
        year_text = ' '.join(year_df[source_column].astype(str))
        
        # 해당 연도의 키워드 빈도 계산 (bigram만)
        vectorizer = CountVectorizer(
            ngram_range=(2, 2),  # bigram만 사용
            vocabulary=all_keywords,
            lowercase=True,
            token_pattern=r'\b[a-z]{4,}\b'  # 최소 4글자 이상
        )
        
        try:
            X = vectorizer.fit_transform([year_text])
            feature_names = vectorizer.get_feature_names_out()
            counts = X.toarray()[0]
            
            for keyword, count in zip(feature_names, counts):
                # 제외할 bigram 키워드 필터링
                if keyword in EXCLUDED_BIGRAMS:
                    continue
                
                # 최소 빈도 필터링 (2회 이상)
                if count >= 2:
                    # 키워드 정규화 적용
                    normalized_keyword = KEYWORD_NORMALIZATION.get(keyword, keyword)
                    year_keyword_counts.append({
                        'Year': year,
                        'Keyword': normalized_keyword,
                        'Count': int(count)
                    })
        except:
            continue
    
    keyword_df = pd.DataFrame(year_keyword_counts)
    
    # Year를 정수로 변환 (소수점 완전 제거)
    if len(keyword_df) > 0:
        keyword_df['Year'] = keyword_df['Year'].astype(float).fillna(0).astype(int)
        keyword_df = keyword_df[(keyword_df['Year'] >= 1900) & (keyword_df['Year'] <= 2100) & (keyword_df['Year'] > 0)]
    
    # 정규화된 키워드로 그룹화하여 카운트 합산
    if len(keyword_df) > 0:
        keyword_df = keyword_df.groupby(['Year', 'Keyword'])['Count'].sum().reset_index()
    else:
        return pd.DataFrame(columns=['Year', 'Keyword', 'Count'])
    
    # 전체 기간 동안 상위 N개 키워드 선택
    total_counts = keyword_df.groupby('Keyword')['Count'].sum().sort_values(ascending=False)
    top_keywords = total_counts.head(top_n).index.tolist()
    
    # 상위 키워드만 필터링
    keyword_df = keyword_df[keyword_df['Keyword'].isin(top_keywords)]
    
    # Year를 다시 정수로 확실히 변환 (소수점 완전 제거)
    keyword_df['Year'] = keyword_df['Year'].astype(float).astype(int)
    
    return keyword_df


def create_trend_chart(keyword_df, selected_years, lang='ko'):
    """연도별 키워드 트렌드 차트 생성 (Streamgraph/Area Chart)"""
    # Year를 정수로 변환
    keyword_df = keyword_df.copy()
    keyword_df['Year'] = keyword_df['Year'].astype(float).fillna(0).astype(int)
    
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot_table(
        index='Year',
        columns='Keyword',
        values='Count',
        fill_value=0
    )
    
    # 인덱스를 정수로 변환 (소수점 완전 제거)
    pivot_df.index = pivot_df.index.astype(float).astype(int)
    pivot_df = pivot_df.sort_index()
    
    # x축 데이터를 정수 리스트로 변환
    x_data = [int(x) for x in pivot_df.index]
    
    fig = go.Figure()
    
    # 각 키워드별로 영역 차트 추가
    for keyword in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=pivot_df[keyword].values,
            mode='lines',
            name=keyword,
            stackgroup='one',
            fill='tonexty' if len(fig.data) > 0 else 'tozeroy'
        ))
    
    texts = TEXTS[lang]
    fig.update_layout(
        title=texts['trend_chart_title'],
        xaxis_title=texts['trend_xaxis'],
        yaxis_title=texts['trend_yaxis'],
        hovermode='x unified',
        height=500,
        xaxis=dict(
            type='linear',
            tickmode='linear',
            dtick=1,
            tickformat='d'  # 정수 형식으로 표시
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_heatmap(keyword_df, selected_years, lang='ko'):
    """연도별 키워드 히트맵 생성"""
    # Year를 정수로 변환
    keyword_df = keyword_df.copy()
    keyword_df['Year'] = keyword_df['Year'].astype(float).fillna(0).astype(int)
    
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot_table(
        index='Keyword',
        columns='Year',
        values='Count',
        fill_value=0
    )
    
    # 컬럼을 정수로 변환 (소수점 완전 제거)
    pivot_df.columns = pivot_df.columns.astype(float).astype(int)
    pivot_df = pivot_df.sort_index(axis=1)
    
    # x축 데이터를 정수 리스트로 변환
    x_data = [int(x) for x in pivot_df.columns]
    
    texts = TEXTS[lang]
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=x_data,
        y=pivot_df.index,
        colorscale='Viridis',
        text=pivot_df.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=texts['heatmap_colorbar'])
    ))
    
    fig.update_layout(
        title=texts['heatmap_title'],
        xaxis_title=texts['heatmap_xaxis'],
        yaxis_title=texts['heatmap_yaxis'],
        height=600,
        xaxis=dict(
            type='linear',
            tickmode='linear',
            dtick=1,
            tickformat='d'  # 정수 형식으로 표시
        ),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_bump_chart(keyword_df, selected_years, top_n=10, lang='ko'):
    """연도별 키워드 순위 변화 차트 생성"""
    # Year를 정수로 변환
    keyword_df = keyword_df.copy()
    keyword_df['Year'] = keyword_df['Year'].astype(float).fillna(0).astype(int)
    
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 연도별 순위 계산
    rankings = []
    for year in sorted(filtered_df['Year'].unique()):
        year = int(year)  # 정수로 변환
        year_df = filtered_df[filtered_df['Year'] == year]
        year_df_sorted = year_df.sort_values('Count', ascending=False).head(top_n)
        
        for rank, (idx, row) in enumerate(year_df_sorted.iterrows(), 1):
            rankings.append({
                'Year': year,
                'Keyword': row['Keyword'],
                'Rank': rank,
                'Count': row['Count']
            })
    
    rank_df = pd.DataFrame(rankings)
    
    # Year를 정수로 변환 (소수점 완전 제거)
    if len(rank_df) > 0:
        rank_df['Year'] = rank_df['Year'].astype(float).astype(int)
    
    # 상위 N개 키워드만 선택 (전체 기간 동안 평균 순위 기준)
    if len(rank_df) > 0:
        avg_ranks = rank_df.groupby('Keyword')['Rank'].mean().sort_values()
        top_keywords = avg_ranks.head(top_n).index.tolist()
        rank_df = rank_df[rank_df['Keyword'].isin(top_keywords)]
    else:
        top_keywords = []
    
    fig = go.Figure()
    
    # 각 키워드별로 선 그래프 추가
    for keyword in top_keywords:
        keyword_data = rank_df[rank_df['Keyword'] == keyword].sort_values('Year')
        # Year를 정수로 변환 (소수점 완전 제거)
        keyword_data['Year'] = keyword_data['Year'].astype(float).astype(int)
        x_data = [int(x) for x in keyword_data['Year']]
        fig.add_trace(go.Scatter(
            x=x_data,
            y=keyword_data['Rank'].values,
            mode='lines+markers',
            name=keyword,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    texts = TEXTS[lang]
    fig.update_layout(
        title=texts['bump_title'].format(top_n),
        xaxis_title=texts['bump_xaxis'],
        yaxis_title=texts['bump_yaxis'],
        xaxis=dict(
            type='linear',
            tickmode='linear',
            dtick=1,
            tickformat='d'  # 정수 형식으로 표시
        ),
        yaxis=dict(autorange="reversed"),  # 1위가 위에 오도록
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_hype_cycle_chart(keyword_df, selected_years, lang='ko'):
    """하이프 사이클 스타일 차트 생성"""
    # Year를 정수로 변환
    keyword_df = keyword_df.copy()
    keyword_df['Year'] = keyword_df['Year'].astype(float).fillna(0).astype(int)
    
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 키워드별 통계 계산
    keyword_stats = []
    
    for keyword in filtered_df['Keyword'].unique():
        keyword_data = filtered_df[filtered_df['Keyword'] == keyword].sort_values('Year')
        
        if len(keyword_data) < 2:
            continue
        
        # 총 언급량 (Total Volume)
        total_volume = keyword_data['Count'].sum()
        
        # 최신 연도 언급량
        latest_year = int(keyword_data['Year'].max())
        latest_count = keyword_data[keyword_data['Year'] == latest_year]['Count'].values[0]
        
        # 성장률 계산 (전년 대비 증가율)
        # Year를 정수로 변환 (소수점 완전 제거)
        keyword_data['Year'] = keyword_data['Year'].astype(float).astype(int)
        years = sorted([int(y) for y in keyword_data['Year'].unique()])
        if len(years) >= 2:
            prev_year = int(years[-2])
            current_year = int(years[-1])
            
            prev_count = keyword_data[keyword_data['Year'] == prev_year]['Count'].values[0]
            current_count = keyword_data[keyword_data['Year'] == current_year]['Count'].values[0]
            
            if prev_count > 0:
                growth_rate = ((current_count - prev_count) / prev_count) * 100
            else:
                growth_rate = 0 if current_count == 0 else 100
        else:
            growth_rate = 0
        
        keyword_stats.append({
            'Keyword': keyword,
            'Total_Volume': total_volume,
            'Growth_Rate': growth_rate,
            'Latest_Count': latest_count
        })
    
    stats_df = pd.DataFrame(keyword_stats)
    
    if len(stats_df) == 0:
        return None
    
    # 상위 30개 키워드만 선택 (Total Volume 기준)
    stats_df = stats_df.nlargest(30, 'Total_Volume')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stats_df['Growth_Rate'],
        y=stats_df['Total_Volume'],
        mode='markers+text',
        text=stats_df['Keyword'],
        textposition="middle right",
        marker=dict(
            size=stats_df['Latest_Count'] * 2,
            color=stats_df['Latest_Count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=TEXTS[lang]['hype_colorbar']),
            sizemode='diameter',
            sizemin=5,
            sizeref=stats_df['Latest_Count'].max() / 50
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      f"{TEXTS[lang]['hype_hover_total']}: %{{y}}<br>" +
                      f"{TEXTS[lang]['hype_hover_growth']}: %{{x:.1f}}%<br>" +
                      f"{TEXTS[lang]['hype_hover_latest']}: %{{marker.size}}<extra></extra>"
    ))
    
    texts = TEXTS[lang]
    fig.update_layout(
        title=texts['hype_title'],
        xaxis_title=texts['hype_xaxis'],
        yaxis_title=texts['hype_yaxis'],
        height=600,
        hovermode='closest'
    )
    
    return fig


def main():
    """메인 애플리케이션"""
    # 언어 선택 초기화
    if 'language' not in st.session_state:
        st.session_state.language = 'ko'
    
    lang = get_language()
    texts = TEXTS[lang]
    
    st.title(f"🤖 {texts['title']}")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'keyword_df' not in st.session_state:
        st.session_state.keyword_df = None
    
    # 데이터 로드
    df = load_data()
    
    # 카테고리 그룹 선택 (맨 위로 이동)
    st.sidebar.markdown(f"### {texts['category_selection']}")
    all_categories, available_groups = get_all_categories(df)
    
    # available_groups에서 중복 완전 제거 (방어적 프로그래밍)
    if isinstance(available_groups, dict):
        # OrderedDict를 일반 dict로 변환 후 다시 OrderedDict로 (중복 제거)
        from collections import OrderedDict
        unique_groups = OrderedDict()
        for group_name, group_info in available_groups.items():
            if group_name not in unique_groups:
                unique_groups[group_name] = group_info
        available_groups = unique_groups
    
    # 기본 선택: 로보틱스, AI, CV 관련 그룹
    default_selected_groups = []
    seen_default = set()
    for group_name in available_groups.keys():
        if group_name in seen_default:
            continue
        seen_default.add(group_name)
        if any(cat in ['cs.RO', 'cs.AI', 'cs.CV'] for cat in available_groups[group_name]["categories"]):
            if group_name not in default_selected_groups:
                default_selected_groups.append(group_name)
    
    selected_groups = []
    displayed_groups = set()  # 이미 표시된 그룹 추적 (중복 완전 차단)
    
    # CATEGORY_GROUPS의 정의된 순서대로 표시 (중복 완전 차단)
    # 리스트로 변환하여 순서 보장
    category_group_list = list(CATEGORY_GROUPS.keys())
    
    for idx, group_name in enumerate(category_group_list):
        # 엄격한 중복 체크
        if group_name in displayed_groups:
            continue
        if group_name not in available_groups:
            continue  # 데이터에 없는 그룹은 건너뛰기
        
        # 한 번 더 확인
        if group_name in displayed_groups:
            continue
        
        displayed_groups.add(group_name)
        group_info = available_groups[group_name]
        
        # 그룹 체크박스 (논문 수 포함) - 인덱스를 포함한 완전히 고유한 key 사용
        paper_count = group_info.get("paper_count", 0)
        unique_key = f"cat_group_{idx}_{group_name.replace(' ', '_').replace('/', '_')}"
        papers_text = texts['papers_count'] if lang == 'ko' else 'papers'
        count_text = f"{paper_count:,}{papers_text}" if lang == 'ko' else f"{paper_count:,} {papers_text}"
        is_selected = st.sidebar.checkbox(
            f"{group_name} ({count_text})",
            value=group_name in default_selected_groups,
            key=unique_key,
            help=group_info["description"]
        )
        if is_selected:
            if group_name not in selected_groups:  # 중복 선택 방지
                selected_groups.append(group_name)
            
            # 선택된 그룹의 카테고리 목록 표시 (접기/펼치기) - 컴팩트하게
            categories_text = f"{len(group_info['categories'])}{texts['categories_count']}" if lang == 'ko' else f"{len(group_info['categories'])} {texts['categories_count']}"
            with st.sidebar.expander(f"  └ {categories_text}", expanded=False):
                # 카테고리를 2열로 표시하여 컴팩트하게
                categories = group_info['categories']
                num_cols = 2
                
                # 카테고리 텍스트를 미리 준비
                category_texts = []
                for cat in categories:
                    if lang == 'ko':
                        kr_name = CATEGORY_NAMES_KR.get(cat, "")
                        if kr_name:
                            category_texts.append(f"{cat} ({kr_name})")
                        else:
                            category_texts.append(cat)
                    else:
                        category_texts.append(cat)
                
                # 2열로 나누어 표시 (행간 간격 최소화)
                for i in range(0, len(category_texts), num_cols):
                    cols = st.sidebar.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(category_texts):
                            # markdown을 사용하여 작은 텍스트로 표시 (행간 최소화)
                            cols[j].markdown(f"<small>{category_texts[i+j]}</small>", unsafe_allow_html=True)
    
    if not selected_groups:
        st.sidebar.warning(texts['select_category_warning'])
    
    st.sidebar.markdown("---")
    
    # 연도 범위 선택 (정수로 변환하여 소수점 제거)
    # Year를 정수로 확실히 변환 (소수점 완전 제거)
    df['Year'] = df['Year'].astype(float).fillna(0).astype(int)
    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2100) & (df['Year'] > 0)]
    
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    
    # min과 max가 같을 때 슬라이더 오류 방지
    if min_year == max_year:
        selected_years = (min_year, max_year)
        st.sidebar.info(f"{texts['data_year']}: {min_year}")
    else:
        selected_years = st.sidebar.slider(
            texts['year_range'],
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1
        )
        # 선택된 연도를 정수로 변환 (소수점 완전 제거)
        selected_years = (int(selected_years[0]), int(selected_years[1]))
    
    # Top N 키워드 개수 선택
    top_n = st.sidebar.slider(
        texts['top_n_keywords'],
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    # 키워드 추출 소스 선택
    keyword_source = st.sidebar.radio(
        texts['keyword_source'],
        options=['Abstract', 'Title'],
        index=0,
        help=texts['keyword_source_help']
    )
    
    # 분석 시작 버튼
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button(texts['analyze_start'], type="primary", use_container_width=True)
    
    if analyze_button:
        if not selected_groups:
            st.sidebar.error(texts['select_category_error'])
        else:
            st.session_state.analysis_started = True
            # 선택된 그룹으로 필터링
            st.session_state.filtered_df = filter_by_categories(df, selected_groups)
            # 키워드 추출
            with st.spinner(f"{keyword_source} {texts['extracting_keywords']}"):
                st.session_state.keyword_df = get_top_keywords_by_year(
                    st.session_state.filtered_df, 
                    top_n=top_n, 
                    source=keyword_source.lower()
                )
    
    # 데이터 요약 정보 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {texts['data_summary']}")
    if st.session_state.analysis_started and st.session_state.filtered_df is not None:
        filtered_count = len(st.session_state.filtered_df)
        papers_text = texts['papers_count'] if lang == 'ko' else 'papers'
        st.sidebar.metric(texts['selected_papers'], f"{filtered_count:,}{papers_text}" if lang == 'ko' else f"{filtered_count:,} {papers_text}")
        st.sidebar.metric(texts['total_papers'], f"{len(df):,}{papers_text}" if lang == 'ko' else f"{len(df):,} {papers_text}")
        st.sidebar.metric(texts['year_range_label'], f"{min_year} - {max_year}")
        groups_text = texts['groups'] if lang == 'en' else '개'
        st.sidebar.info(f"{texts['selected_groups']}: {len(selected_groups)}{groups_text}")
        # 선택된 그룹의 카테고리 수집
        selected_cats = set()
        for group_name in selected_groups:
            if group_name in CATEGORY_GROUPS:
                selected_cats.update(CATEGORY_GROUPS[group_name]["categories"])
        categories_text = texts['categories'] if lang == 'en' else '개'
        st.sidebar.caption(f"{texts['included_categories']}: {len(selected_cats)}{categories_text}")
        st.sidebar.info(f"{texts['keyword_source_label']}: **{keyword_source}**")
    else:
        papers_text = texts['papers_count'] if lang == 'ko' else 'papers'
        st.sidebar.metric(texts['papers'], f"{len(df):,}{papers_text}" if lang == 'ko' else f"{len(df):,} {papers_text}")
        st.sidebar.metric(texts['year_range_label'], f"{min_year} - {max_year}")
        st.sidebar.info(texts['select_category_info'])
    
    # 사이드바 설정 (맨 아래로 이동)
    st.sidebar.markdown("---")
    st.sidebar.header(texts['settings'])
    
    # 언어 선택 (설정 내부)
    language_options = {
        '한국어': 'ko',
        'English': 'en'
    }
    selected_lang_name = [k for k, v in language_options.items() if v == lang][0]
    new_lang_name = st.sidebar.selectbox(
        texts['language'],
        options=list(language_options.keys()),
        index=list(language_options.keys()).index(selected_lang_name),
        key='lang_select'
    )
    
    # 언어가 변경되면 세션 상태 업데이트 및 새로고침
    if language_options[new_lang_name] != lang:
        st.session_state.language = language_options[new_lang_name]
        st.rerun()
    
    # 언어 재설정 (변경 후)
    lang = get_language()
    texts = TEXTS[lang]
    
    # 캐시 클리어 버튼
    if st.sidebar.button(texts['clear_cache'], use_container_width=True, help=texts['clear_cache_help']):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.caption(texts['clear_cache_caption'])
    
    # 분석이 시작되지 않았으면 안내 메시지 표시
    if not st.session_state.analysis_started:
        st.info(texts['start_analysis_info'])
        st.stop()
    
    # 필터링된 데이터가 없으면 중지
    if st.session_state.filtered_df is None or len(st.session_state.filtered_df) == 0:
        st.warning(texts['no_data_warning'])
        st.stop()
    
    # 키워드 데이터가 없으면 중지
    if st.session_state.keyword_df is None or len(st.session_state.keyword_df) == 0:
        st.warning(texts['no_keywords_warning'])
        st.stop()
    
    keyword_df = st.session_state.keyword_df
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        texts['tab_trend'],
        texts['tab_heatmap'],
        texts['tab_bump'],
        texts['tab_hype']
    ])
    
    # Tab 1: 연도별 흐름
    with tab1:
        st.header(texts['trend_header'])
        st.markdown(texts['trend_desc'])
        
        trend_fig = create_trend_chart(keyword_df, selected_years, lang)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # 통계 테이블
        st.subheader(texts['keyword_stats'])
        filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
        keyword_summary = filtered_df.groupby('Keyword')['Count'].agg(['sum', 'mean', 'max']).round(1)
        keyword_summary.columns = [texts['total_papers_col'], texts['avg_papers_col'], texts['max_papers_col']]
        keyword_summary = keyword_summary.sort_values(texts['total_papers_col'], ascending=False)
        st.dataframe(keyword_summary, use_container_width=True)
    
    # Tab 2: 히트맵
    with tab2:
        st.header(texts['heatmap_header'])
        st.markdown(texts['heatmap_desc'])
        
        heatmap_fig = create_heatmap(keyword_df, selected_years, lang)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Tab 3: 순위 경쟁
    with tab3:
        st.header(texts['bump_header'])
        st.markdown(texts['bump_desc'])
        
        bump_fig = create_bump_chart(keyword_df, selected_years, top_n=min(top_n, 15), lang=lang)
        st.plotly_chart(bump_fig, use_container_width=True)
    
    # Tab 4: 하이프 사이클
    with tab4:
        st.header(texts['hype_header'])
        st.markdown(texts['hype_desc'])
        st.markdown(f"- **{texts['hype_xaxis_label']}**: {texts['hype_xaxis_desc']}")
        st.markdown(f"- **{texts['hype_yaxis_label']}**: {texts['hype_yaxis_desc']}")
        st.markdown(f"- **{texts['hype_size_label']}**: {texts['hype_size_desc']}")
        
        hype_fig = create_hype_cycle_chart(keyword_df, selected_years, lang)
        if hype_fig:
            st.plotly_chart(hype_fig, use_container_width=True)
        else:
            st.warning(texts['hype_no_data'])
    
    # 푸터
    st.markdown("---")
    st.markdown(f"### {texts['footer_notes']}")
    st.markdown(f"- {texts['footer_data_source']}")
    st.markdown(f"- {texts['footer_keyword_source']}")
    st.markdown(f"- {texts['footer_period']}")


if __name__ == "__main__":
    main()

