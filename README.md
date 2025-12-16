# 로보틱스 & AI 논문 트렌드 분석 대시보드

arXiv API를 사용하여 최근 5년간의 로보틱스(cs.RO) 및 AI(cs.AI) 논문 데이터를 수집하고, Streamlit으로 시각화하여 기술 트렌드를 분석하는 프로젝트입니다.

## 🚀 시작하기

### 1. 환경 설정

먼저 필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

### 2. 데이터 수집

`collector.py`를 실행하여 arXiv에서 논문 데이터를 수집합니다:

```bash
python collector.py
```

기본적으로 최근 5년간의 cs.RO 및 cs.AI 카테고리 논문 1000개를 수집합니다.

**옵션:**
- `--max_results`: 수집할 최대 논문 개수 지정 (기본값: 1000)
- `--query`: arXiv 검색 쿼리 지정 (기본값: "cat:cs.RO OR cat:cs.AI")

**예시:**
```bash
# 500개만 수집
python collector.py --max_results 500

# 다른 카테고리 검색
python collector.py --query "cat:cs.CV OR cat:cs.LG"
```

수집된 데이터는 `robotics_papers.csv` 파일로 저장됩니다.

### 3. 대시보드 실행

Streamlit 대시보드를 실행합니다:

```bash
streamlit run app.py
```

브라우저에서 자동으로 대시보드가 열립니다.

## 📊 대시보드 기능

### 사이드바 설정
- **연도 범위 선택**: 분석할 연도 범위를 슬라이더로 선택
- **Top N 키워드 개수**: 분석할 상위 키워드 개수 선택 (5~30개)
- **데이터 요약**: 총 논문 수 및 연도 범위 표시

### 탭 구성

#### 1. 📈 연도별 흐름 (Trend)
- Streamgraph 스타일의 영역 차트
- 시간에 따른 주요 키워드의 논문 수 변화 시각화
- 키워드별 통계 테이블 제공

#### 2. 🔥 히트맵 (Heatmap)
- X축: 연도, Y축: 키워드
- 논문 수를 색상 농도로 표현
- 연도별 키워드 빈도를 한눈에 파악 가능

#### 3. 🏆 순위 경쟁 (Bump Chart)
- 연도별 키워드 순위 변화를 선 그래프로 표현
- 상위 키워드의 순위 변동 추적
- 1위가 위에 오도록 Y축 역순 설정

#### 4. 🚀 하이프 사이클 (Hype Cycle)
- X축: 기술 성장률 (전년 대비 증가율)
- Y축: 기술 언급량 (Total Volume)
- 점 크기: 최신 연도 언급량
- 기술의 성장 단계를 시각적으로 분석

## 📁 프로젝트 구조

```
robotics_trend/
├── collector.py          # arXiv 데이터 수집 스크립트
├── app.py               # Streamlit 대시보드 애플리케이션
├── requirements.txt     # 필요한 라이브러리 목록
├── README.md           # 프로젝트 설명서
└── robotics_papers.csv # 수집된 논문 데이터 (생성됨)
```

## 🔧 기술 스택

- **Python 3.9+**
- **데이터 수집**: `arxiv` 라이브러리
- **데이터 처리**: `pandas`, `numpy`
- **시각화**: `plotly` (interactive charts)
- **UI**: `streamlit`
- **NLP**: `scikit-learn` (CountVectorizer), `nltk` (불용어 제거)

## 📝 데이터 구조

수집된 CSV 파일의 컬럼:
- `arXiv_ID`: 논문 arXiv ID
- `Title`: 논문 제목
- `First_Author`: 제1저자 이름
- `Category`: 논문 카테고리 (쉼표로 구분)
- `Abstract`: 논문 초록
- `Published_Date`: 출판일 (YYYY-MM-DD 형식)

## ⚠️ 주의사항

1. **데이터 수집 시간**: 논문 수에 따라 수집 시간이 오래 걸릴 수 있습니다. API 부하를 줄이기 위해 요청 간 짧은 대기 시간이 포함되어 있습니다.

2. **NLTK 데이터**: 처음 실행 시 NLTK 불용어 데이터를 자동으로 다운로드합니다.

3. **메모리 사용량**: 대량의 논문 데이터를 처리할 경우 메모리 사용량이 증가할 수 있습니다.

## 🐛 문제 해결

### 데이터 수집 오류
- 인터넷 연결을 확인하세요.
- arXiv API 서버 상태를 확인하세요.
- `--max_results` 값을 줄여서 다시 시도해보세요.

### 대시보드 실행 오류
- `robotics_papers.csv` 파일이 존재하는지 확인하세요.
- 필요한 라이브러리가 모두 설치되었는지 확인하세요: `pip install -r requirements.txt`

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

