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
def load_data(file_path='robotics_papers.csv'):
    """CSV 파일에서 데이터를 로드합니다."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # Published_Date를 datetime으로 변환
        df['Published_Date'] = pd.to_datetime(df['Published_Date'], errors='coerce')
        # 연도 컬럼 추가
        df['Year'] = df['Published_Date'].dt.year
        # 결측값 제거
        df = df.dropna(subset=['Year', 'Abstract'])
        return df
    except FileNotFoundError:
        st.error(f"파일 '{file_path}'을 찾을 수 없습니다. 먼저 collector.py를 실행하여 데이터를 수집해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.stop()


def extract_keywords(text, ngram_range=(1, 2), max_features=100):
    """
    텍스트에서 키워드를 추출합니다.
    
    Parameters:
    -----------
    text : str or pd.Series
        추출할 텍스트
    ngram_range : tuple
        n-gram 범위 (기본값: (1, 2) - unigram과 bigram)
    max_features : int
        최대 특징 개수
    
    Returns:
    --------
    list
        추출된 키워드 리스트
    """
    if isinstance(text, pd.Series):
        text = ' '.join(text.astype(str))
    
    # 로보틱스 관련 주요 키워드 (불용어에서 제외)
    robotics_keywords = {
        'slam', 'reinforcement', 'learning', 'manipulation', 'llm', 'gpt',
        'transformer', 'neural', 'network', 'deep', 'robot', 'robotic',
        'robotics', 'autonomous', 'navigation', 'planning', 'control',
        'perception', 'vision', 'sensor', 'actuator', 'grasp', 'grasping',
        'locomotion', 'humanoid', 'drone', 'uav', 'mobile', 'wheeled',
        'legged', 'quadruped', 'biped', 'arm', 'manipulator', 'end',
        'effector', 'trajectory', 'motion', 'dynamics', 'kinematics',
        'optimization', 'policy', 'imitation', 'demonstration', 'transfer',
        'simulation', 'real', 'world', 'benchmark', 'dataset'
    }
    
    # 불용어에서 로보틱스 키워드 제외
    custom_stopwords = STOPWORDS - robotics_keywords
    
    # CountVectorizer 사용
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=list(custom_stopwords),
        lowercase=True,
        token_pattern=r'\b[a-z]{3,}\b'  # 최소 3글자 이상의 단어만
    )
    
    try:
        vectorizer.fit([text])
        keywords = vectorizer.get_feature_names_out().tolist()
        return keywords
    except:
        return []


def get_top_keywords_by_year(df, top_n=20):
    """
    연도별 상위 키워드를 추출합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        논문 데이터프레임
    top_n : int
        상위 N개 키워드
    
    Returns:
    --------
    pd.DataFrame
        연도별 키워드 빈도 데이터프레임
    """
    # 전체 Abstract에서 키워드 추출
    all_text = ' '.join(df['Abstract'].astype(str))
    all_keywords = extract_keywords(all_text, ngram_range=(1, 2), max_features=200)
    
    # 연도별 키워드 빈도 계산
    year_keyword_counts = []
    
    for year in sorted(df['Year'].unique()):
        year_df = df[df['Year'] == year]
        year_text = ' '.join(year_df['Abstract'].astype(str))
        
        # 해당 연도의 키워드 빈도 계산
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            vocabulary=all_keywords,
            lowercase=True,
            token_pattern=r'\b[a-z]{3,}\b'
        )
        
        try:
            X = vectorizer.fit_transform([year_text])
            feature_names = vectorizer.get_feature_names_out()
            counts = X.toarray()[0]
            
            for keyword, count in zip(feature_names, counts):
                if count > 0:
                    year_keyword_counts.append({
                        'Year': year,
                        'Keyword': keyword,
                        'Count': int(count)
                    })
        except:
            continue
    
    keyword_df = pd.DataFrame(year_keyword_counts)
    
    # 전체 기간 동안 상위 N개 키워드 선택
    total_counts = keyword_df.groupby('Keyword')['Count'].sum().sort_values(ascending=False)
    top_keywords = total_counts.head(top_n).index.tolist()
    
    # 상위 키워드만 필터링
    keyword_df = keyword_df[keyword_df['Keyword'].isin(top_keywords)]
    
    return keyword_df


def create_trend_chart(keyword_df, selected_years):
    """연도별 키워드 트렌드 차트 생성 (Streamgraph/Area Chart)"""
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot_table(
        index='Year',
        columns='Keyword',
        values='Count',
        fill_value=0
    )
    
    fig = go.Figure()
    
    # 각 키워드별로 영역 차트 추가
    for keyword in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[keyword],
            mode='lines',
            name=keyword,
            stackgroup='one',
            fill='tonexty' if len(fig.data) > 0 else 'tozeroy'
        ))
    
    fig.update_layout(
        title='연도별 주요 키워드 트렌드 (Streamgraph)',
        xaxis_title='연도',
        yaxis_title='논문 수',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_heatmap(keyword_df, selected_years):
    """연도별 키워드 히트맵 생성"""
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot_table(
        index='Keyword',
        columns='Year',
        values='Count',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Viridis',
        text=pivot_df.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="논문 수")
    ))
    
    fig.update_layout(
        title='연도별 키워드 빈도 히트맵',
        xaxis_title='연도',
        yaxis_title='키워드',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_bump_chart(keyword_df, selected_years, top_n=10):
    """연도별 키워드 순위 변화 차트 생성"""
    filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
    
    # 연도별 순위 계산
    rankings = []
    for year in sorted(filtered_df['Year'].unique()):
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
    
    # 상위 N개 키워드만 선택 (전체 기간 동안 평균 순위 기준)
    avg_ranks = rank_df.groupby('Keyword')['Rank'].mean().sort_values()
    top_keywords = avg_ranks.head(top_n).index.tolist()
    rank_df = rank_df[rank_df['Keyword'].isin(top_keywords)]
    
    fig = go.Figure()
    
    # 각 키워드별로 선 그래프 추가
    for keyword in top_keywords:
        keyword_data = rank_df[rank_df['Keyword'] == keyword].sort_values('Year')
        fig.add_trace(go.Scatter(
            x=keyword_data['Year'],
            y=keyword_data['Rank'],
            mode='lines+markers',
            name=keyword,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f'연도별 키워드 순위 변화 (Top {top_n})',
        xaxis_title='연도',
        yaxis_title='순위',
        yaxis=dict(autorange="reversed"),  # 1위가 위에 오도록
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_hype_cycle_chart(keyword_df, selected_years):
    """하이프 사이클 스타일 차트 생성"""
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
        latest_year = keyword_data['Year'].max()
        latest_count = keyword_data[keyword_data['Year'] == latest_year]['Count'].values[0]
        
        # 성장률 계산 (전년 대비 증가율)
        years = sorted(keyword_data['Year'].unique())
        if len(years) >= 2:
            prev_year = years[-2]
            current_year = years[-1]
            
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
            colorbar=dict(title="최신 연도<br>언급량"),
            sizemode='diameter',
            sizemin=5,
            sizeref=stats_df['Latest_Count'].max() / 50
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      '총 언급량: %{y}<br>' +
                      '성장률: %{x:.1f}%<br>' +
                      '최신 연도 언급량: %{marker.size}<extra></extra>'
    ))
    
    fig.update_layout(
        title='하이프 사이클 스타일 차트',
        xaxis_title='기술 성장률 (전년 대비 증가율, %)',
        yaxis_title='기술 언급량 (Total Volume)',
        height=600,
        hovermode='closest'
    )
    
    return fig


def main():
    """메인 애플리케이션"""
    st.title("🤖 로보틱스 & AI 논문 트렌드 분석 대시보드")
    st.markdown("---")
    
    # 데이터 로드
    df = load_data()
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 연도 범위 선택
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    selected_years = st.sidebar.slider(
        "연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    # Top N 키워드 개수 선택
    top_n = st.sidebar.slider(
        "분석할 Top N 키워드 개수",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    # 데이터 요약 정보 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 데이터 요약")
    st.sidebar.metric("총 논문 수", len(df))
    st.sidebar.metric("연도 범위", f"{min_year} - {max_year}")
    
    # 키워드 추출 (캐싱)
    with st.spinner("키워드 추출 중..."):
        keyword_df = get_top_keywords_by_year(df, top_n=top_n)
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 연도별 흐름 (Trend)",
        "🔥 히트맵 (Heatmap)",
        "🏆 순위 경쟁 (Bump Chart)",
        "🚀 하이프 사이클 (Hype Cycle)"
    ])
    
    # Tab 1: 연도별 흐름
    with tab1:
        st.header("연도별 주요 키워드 트렌드")
        st.markdown("시간에 따른 주요 키워드의 논문 수 변화를 시각화합니다.")
        
        trend_fig = create_trend_chart(keyword_df, selected_years)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # 통계 테이블
        st.subheader("키워드별 통계")
        filtered_df = keyword_df[keyword_df['Year'].between(selected_years[0], selected_years[1])]
        keyword_summary = filtered_df.groupby('Keyword')['Count'].agg(['sum', 'mean', 'max']).round(1)
        keyword_summary.columns = ['총 논문 수', '평균 논문 수', '최대 논문 수']
        keyword_summary = keyword_summary.sort_values('총 논문 수', ascending=False)
        st.dataframe(keyword_summary, use_container_width=True)
    
    # Tab 2: 히트맵
    with tab2:
        st.header("연도별 키워드 빈도 히트맵")
        st.markdown("연도와 키워드의 교차점에서 논문 수를 색상 농도로 표현합니다.")
        
        heatmap_fig = create_heatmap(keyword_df, selected_years)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Tab 3: 순위 경쟁
    with tab3:
        st.header("연도별 키워드 순위 변화")
        st.markdown("시간에 따른 키워드의 순위 변화를 추적합니다.")
        
        bump_fig = create_bump_chart(keyword_df, selected_years, top_n=min(top_n, 15))
        st.plotly_chart(bump_fig, use_container_width=True)
    
    # Tab 4: 하이프 사이클
    with tab4:
        st.header("하이프 사이클 스타일 차트")
        st.markdown("기술 성장률과 총 언급량을 기준으로 키워드를 배치합니다.")
        st.markdown("- **X축**: 기술 성장률 (전년 대비 증가율)")
        st.markdown("- **Y축**: 기술 언급량 (Total Volume)")
        st.markdown("- **점 크기**: 최신 연도 언급량")
        
        hype_fig = create_hype_cycle_chart(keyword_df, selected_years)
        if hype_fig:
            st.plotly_chart(hype_fig, use_container_width=True)
        else:
            st.warning("데이터가 부족하여 차트를 생성할 수 없습니다.")
    
    # 푸터
    st.markdown("---")
    st.markdown("### 📝 참고사항")
    st.markdown("- 데이터는 arXiv API를 통해 수집되었습니다.")
    st.markdown("- 키워드는 논문 초록(Abstract)에서 추출되었습니다.")
    st.markdown("- 차트는 Plotly를 사용하여 인터랙티브하게 제공됩니다.")


if __name__ == "__main__":
    main()

