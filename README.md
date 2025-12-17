# 로보틱스 & AI 논문 트렌드 분석 대시보드

arXiv API를 사용하여 로보틱스(cs.RO), AI(cs.AI), 컴퓨터 비전(cs.CV) 논문 데이터를 수집하고, Streamlit으로 시각화하여 기술 트렌드를 분석하는 프로젝트입니다.

## 🚀 시작하기

### 1. 환경 설정

먼저 필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

또는 `uv`를 사용하여 가상환경을 만들고 설치할 수 있습니다:

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 2. 데이터 수집

`collector_improved.py`를 실행하여 arXiv에서 논문 데이터를 수집합니다:

```bash
python collector_improved.py
```

**수집 전략:**
- **카테고리**: cs.RO (로보틱스), cs.AI (인공지능), cs.CV (컴퓨터 비전) 통합 검색
- **기간**: 2021년 ~ 2025년 (설정 파일에서 변경 가능)
- **분할 수집**: 10일 단위로 분할하여 수집 (HTTP 500 오류 방지)
- **병렬 처리**: 최대 6개 워커로 병렬 수집
- **자동 중복 제거**: arXiv_ID 기준으로 중복 제거

**설정 변경:**
`collector_improved.py` 파일에서 다음 변수를 수정할 수 있습니다:
- `TARGET_START_YEAR`: 시작 연도 (기본값: 2021)
- `TARGET_END_YEAR`: 종료 연도 (기본값: 2025)
- `MAX_WORKERS`: 병렬 워커 수 (기본값: 6)
- `QUERY`: 검색 쿼리 (기본값: "cat:cs.RO OR cat:cs.AI OR cat:cs.CV")

수집된 데이터는 `arxiv_mixed_trend_{START_YEAR}_{END_YEAR}.csv` 파일로 저장됩니다.

**참고**: 대용량 데이터의 경우 파일을 두 개로 분할하여 저장한 후, `arxiv_data_part1.csv.gz`와 `arxiv_data_part2.csv.gz`로 압축하여 저장하는 것을 권장합니다.

### 3. 대시보드 실행

Streamlit 대시보드를 실행합니다:

```bash
streamlit run app.py
```

브라우저에서 자동으로 대시보드가 열립니다.

## 📊 대시보드 기능

### 사이드바 설정
- **🔄 캐시 클리어 & 새로고침**: Streamlit 캐시를 클리어하고 페이지를 새로고침
- **📂 카테고리 그룹 선택**: 
  - 10개의 카테고리 그룹 중 선택 (AI/ML, Computer Vision, Robotics, 등)
  - 각 그룹의 논문 수 표시
  - 세부 카테고리 목록 확인 (한글 병기)
- **연도 범위 선택**: 분석할 연도 범위를 슬라이더로 선택
- **Top N 키워드 개수**: 분석할 상위 키워드 개수 선택 (5~30개)
- **키워드 추출 소스**: Abstract(초록) 또는 Title(제목)에서 키워드 추출 선택
- **🚀 분석 시작**: 선택한 설정으로 분석 시작
- **데이터 요약**: 총 논문 수, 선택된 논문 수, 연도 범위 표시

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
├── collector_improved.py    # arXiv 데이터 수집 스크립트 (10일 단위 분할 수집)
├── app.py                    # Streamlit 대시보드 애플리케이션
├── requirements.txt          # 필요한 라이브러리 목록
├── README.md                 # 프로젝트 설명서
├── arxiv_mixed_trend_*.csv   # 수집된 논문 데이터 (생성됨)
└── arxiv_data_part*.csv.gz   # 대시보드용 분할 데이터 파일 (선택사항)
```

## 🔧 기술 스택

- **Python 3.9+**
- **데이터 수집**: `arxiv` 라이브러리
- **데이터 처리**: `pandas`, `numpy`
- **시각화**: `plotly` (interactive charts)
- **UI**: `streamlit`
- **NLP**: `scikit-learn` (CountVectorizer), `nltk` (불용어 제거)
- **병렬 처리**: `concurrent.futures` (ThreadPoolExecutor)
- **진행률 표시**: `tqdm`

## 📝 데이터 구조

수집된 CSV 파일의 컬럼:
- `arXiv_ID`: 논문 arXiv ID
- `Title`: 논문 제목
- `First_Author`: 제1저자 이름
- `Category`: 논문 카테고리 (쉼표로 구분)
- `Abstract`: 논문 초록
- `Published_Date`: 출판일 (YYYY-MM-DD 형식)

## ⚠️ 주의사항

1. **데이터 수집 시간**: 
   - 대량의 데이터 수집 시 시간이 오래 걸릴 수 있습니다 (수 시간 소요 가능)
   - API 부하를 줄이기 위해 요청 간 랜덤 대기 시간(0.5~4초)이 포함되어 있습니다
   - 10일 단위 분할 수집으로 HTTP 500 오류를 방지합니다

2. **NLTK 데이터**: 처음 실행 시 NLTK 불용어 데이터를 자동으로 다운로드합니다.

3. **메모리 사용량**: 대량의 논문 데이터를 처리할 경우 메모리 사용량이 증가할 수 있습니다.

4. **파일 크기**: 
   - 수집된 데이터가 매우 클 수 있습니다 (수십만 건)
   - 필요시 파일을 분할하여 압축 저장하는 것을 권장합니다
   - 대시보드는 `arxiv_data_part1.csv.gz`와 `arxiv_data_part2.csv.gz` 두 파일을 읽습니다

5. **캐시 관리**: 
   - Streamlit 캐시로 인해 카테고리 목록이 중복 표시될 수 있습니다
   - 사이드바의 "🔄 캐시 클리어 & 새로고침" 버튼을 사용하여 해결할 수 있습니다

## 🐛 문제 해결

### 데이터 수집 오류
- **인터넷 연결 확인**: 안정적인 인터넷 연결이 필요합니다
- **arXiv API 서버 상태**: arXiv 서버가 정상 작동하는지 확인하세요
- **HTTP 500 오류**: 
  - 10일 단위 분할 수집이 자동으로 적용됩니다
  - `MAX_WORKERS` 값을 줄여서 다시 시도해보세요 (기본값: 6)
- **수집 기간 조정**: `TARGET_START_YEAR`와 `TARGET_END_YEAR`를 줄여서 테스트해보세요

### 대시보드 실행 오류
- **파일 없음 오류**: 
  - `arxiv_data_part1.csv.gz`와 `arxiv_data_part2.csv.gz` 파일이 존재하는지 확인하세요
  - 파일이 하나만 있는 경우, `load_data()` 함수를 수정하여 단일 파일을 읽도록 변경할 수 있습니다
- **라이브러리 설치**: 필요한 라이브러리가 모두 설치되었는지 확인하세요: `pip install -r requirements.txt`
- **캐시 문제**: 
  - 카테고리 목록이 중복 표시되거나 순서가 맞지 않는 경우
  - "🔄 캐시 클리어 & 새로고침" 버튼을 클릭하거나
  - 터미널에서 `rm -rf ~/.streamlit/cache` 실행 (macOS/Linux)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

