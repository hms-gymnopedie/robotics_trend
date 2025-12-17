import arxiv
import pandas as pd
from datetime import datetime
import calendar
import time
import random
import concurrent.futures
from tqdm import tqdm

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
TARGET_START_YEAR = 2021  # 양이 많으므로 최근 2-3년만 테스트 권장
TARGET_END_YEAR = 2025
MAX_WORKERS = 6
# [핵심] 3개 카테고리 통합 검색 (OR 연산)
QUERY = "cat:cs.RO OR cat:cs.AI OR cat:cs.CV"
# ---------------------------------------------------------

def fetch_period_data(args):
    """
    [Worker] 특정 기간(10일 단위)의 데이터를 수집
    """
    start_date, end_date, query = args
    
    time.sleep(random.uniform(0.5, 4.0))
    
    # 쿼리 조합
    full_query = f"({query}) AND submittedDate:[{start_date} TO {end_date}]"
    
    client = arxiv.Client(
        page_size=1000,
        delay_seconds=3.0,
        num_retries=5
    )
    
    search = arxiv.Search(
        query=full_query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    try:
        results = list(client.results(search))
        for paper in results:
            categories = paper.categories if hasattr(paper, 'categories') else []
            first_author = paper.authors[0].name if paper.authors else "Unknown"
            published_date = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown"
            
            papers.append({
                'arXiv_ID': paper.entry_id.split('/')[-1],
                'Title': paper.title.replace("\n", " "),
                'First_Author': first_author,
                'Category': ', '.join(categories),
                'Abstract': paper.summary.replace("\n", " "),
                'Published_Date': published_date
            })
    except Exception as e:
        return []
        
    return papers

def generate_tasks_by_10days(start_year, end_year, query):
    """
    1개월을 10일 단위(초순, 중순, 하순)로 쪼개서 태스크 생성
    이유: 3개 카테고리를 합치면 월간 1만건이 넘을 수 있음 -> HTTP 500 방지
    """
    tasks = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 미래 제외
            if datetime.now() < datetime(year, month, 1):
                continue
                
            _, last_day = calendar.monthrange(year, month)
            
            # 3등분 (1~10일, 11~20일, 21~말일)
            periods = [
                (f"{year}{month:02d}01000000", f"{year}{month:02d}10235959"),
                (f"{year}{month:02d}11000000", f"{year}{month:02d}20235959"),
                (f"{year}{month:02d}21000000", f"{year}{month:02d}{last_day:02d}235959")
            ]
            
            for start, end in periods:
                # 미래 시점의 '10일' 단위도 걸러내기 위해 간단 체크
                # (정밀하진 않아도 쿼리 결과가 0건이면 되므로 패스)
                tasks.append((start, end, query))
                
    return tasks

def main():
    start_total_time = time.time()
    
    print(f"🤖 [RO + AI + CV] 대규모 통합 수집기 가동")
    print(f"Target: {TARGET_START_YEAR} ~ {TARGET_END_YEAR}")
    print("전략: 10일 단위 분할 수집 (HTTP 500 방지)")
    print("-" * 50)

    # 1. 태스크 생성 (10일 단위)
    tasks = generate_tasks_by_10days(TARGET_START_YEAR, TARGET_END_YEAR, QUERY)
    print(f"총 {len(tasks)}개의 구간(Period) 작업이 예약되었습니다.")

    # 2. 병렬 처리
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(fetch_period_data, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="수집 진행률"):
            data = future.result()
            if data:
                all_results.extend(data)

    if not all_results:
        print("데이터 없음")
        return

    # 3. 저장
    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=['arXiv_ID'])
    df = df.sort_values(by='Published_Date', ascending=False)
    
    filename = f"arxiv_mixed_trend_{TARGET_START_YEAR}_{TARGET_END_YEAR}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print(f"✅ 수집 완료! 총 {len(df)}건")
    print(f"파일 저장: {filename}")

if __name__ == "__main__":
    main()