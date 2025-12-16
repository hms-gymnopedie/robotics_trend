"""
arXiv API를 사용하여 로보틱스 및 AI 논문 데이터를 수집하는 스크립트
"""
import arxiv
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import sys


def collect_papers(max_results=1000, query="cat:cs.RO OR cat:cs.AI"):
    """
    arXiv API를 사용하여 논문 데이터를 수집합니다.
    
    Parameters:
    -----------
    max_results : int
        수집할 최대 논문 개수 (기본값: 1000)
    query : str
        arXiv 검색 쿼리 (기본값: "cat:cs.RO OR cat:cs.AI")
    
    Returns:
    --------
    pd.DataFrame
        수집된 논문 데이터
    """
    # 5년 전 날짜 계산
    five_years_ago = datetime.now() - timedelta(days=5*365)
    
    # 날짜 필터 추가
    date_str = five_years_ago.strftime("%Y%m%d")
    query_with_date = f"{query} AND submittedDate:[{date_str}000000 TO {datetime.now().strftime('%Y%m%d')}235959]"
    
    print(f"검색 쿼리: {query_with_date}")
    print(f"최대 수집 개수: {max_results}")
    
    papers_data = []
    search = arxiv.Search(
        query=query_with_date,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    try:
        print("\n논문 수집 중...")
        for paper in tqdm(search.results(), total=max_results, desc="수집 진행"):
            try:
                # 카테고리 리스트 추출
                categories = paper.categories if hasattr(paper, 'categories') else []
                
                # 제1저자 추출
                first_author = paper.authors[0].name if paper.authors else "Unknown"
                
                # 출판일 포맷팅
                published_date = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown"
                
                papers_data.append({
                    'arXiv_ID': paper.entry_id.split('/')[-1],
                    'Title': paper.title,
                    'First_Author': first_author,
                    'Category': ', '.join(categories),
                    'Abstract': paper.summary,
                    'Published_Date': published_date
                })
                
                # API 부하 방지를 위한 짧은 대기
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n논문 처리 중 오류 발생: {e}")
                continue
        
        print(f"\n총 {len(papers_data)}개의 논문을 수집했습니다.")
        
    except Exception as e:
        print(f"\n데이터 수집 중 오류 발생: {e}")
        sys.exit(1)
    
    # DataFrame 생성
    df = pd.DataFrame(papers_data)
    
    return df


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='arXiv 논문 데이터 수집')
    parser.add_argument(
        '--max_results',
        type=int,
        default=1000,
        help='수집할 최대 논문 개수 (기본값: 1000)'
    )
    parser.add_argument(
        '--query',
        type=str,
        default="cat:cs.RO OR cat:cs.AI",
        help='arXiv 검색 쿼리 (기본값: "cat:cs.RO OR cat:cs.AI")'
    )
    
    args = parser.parse_args()
    
    # 데이터 수집
    df = collect_papers(max_results=args.max_results, query=args.query)
    
    # CSV 파일로 저장
    output_file = 'robotics_papers.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n데이터가 '{output_file}' 파일로 저장되었습니다.")
    print(f"저장된 데이터 형태: {df.shape}")
    print(f"\n컬럼: {list(df.columns)}")


if __name__ == "__main__":
    main()

