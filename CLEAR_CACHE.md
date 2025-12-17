# Streamlit 캐시 삭제 방법 (맥북)

## 방법 1: 터미널에서 캐시 삭제

```bash
# 프로젝트 디렉토리로 이동
cd /Users/gymnopedie/robotics_trend

# Streamlit 캐시 디렉토리 삭제
rm -rf .streamlit

# Python 캐시 파일 삭제 (프로젝트 내)
find . -type d -name "__pycache__" ! -path "./.venv/*" -exec rm -rf {} +

# 또는 한 번에 실행
rm -rf .streamlit && find . -type d -name "__pycache__" ! -path "./.venv/*" -exec rm -rf {} +
```

## 방법 2: Streamlit 앱 재시작

1. Streamlit 앱이 실행 중이면 `Ctrl+C`로 중지
2. 다시 실행: `streamlit run app.py`

## 방법 3: 브라우저 캐시 삭제

1. **Chrome/Safari**: `Cmd + Shift + Delete` → 캐시 삭제
2. 또는 **시크릿 모드**로 열기: `Cmd + Shift + N` (Chrome) 또는 `Cmd + Shift + N` (Safari)

## 방법 4: Streamlit 캐시 강제 클리어

Streamlit 앱 실행 중:
- 사이드바 하단의 "Clear cache" 버튼 클릭
- 또는 `Ctrl + C`로 앱 중지 후 재시작

## 방법 5: 코드에서 캐시 비활성화 (임시)

`@st.cache_data` 데코레이터를 주석 처리하거나 제거


