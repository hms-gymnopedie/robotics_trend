# Robotics & AI Paper Trend Analysis Dashboard

A project that collects paper data from arXiv API for Robotics (cs.RO), AI (cs.AI), and Computer Vision (cs.CV), and visualizes technology trends using Streamlit.

## 🚀 Getting Started

### 1. Environment Setup

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Or use `uv` to create a virtual environment and install:

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 2. Data Collection

Run `collector_improved.py` to collect paper data from arXiv:

```bash
python collector_improved.py
```

**Collection Strategy:**
- **Categories**: Combined search for cs.RO (Robotics), cs.AI (Artificial Intelligence), cs.CV (Computer Vision)
- **Period**: 2021 ~ 2025 (configurable in the settings file)
- **Split Collection**: Collect in 10-day intervals (prevents HTTP 500 errors)
- **Parallel Processing**: Parallel collection with up to 6 workers
- **Automatic Deduplication**: Removes duplicates based on arXiv_ID

**Configuration:**
You can modify the following variables in `collector_improved.py`:
- `TARGET_START_YEAR`: Start year (default: 2021)
- `TARGET_END_YEAR`: End year (default: 2025)
- `MAX_WORKERS`: Number of parallel workers (default: 6)
- `QUERY`: Search query (default: "cat:cs.RO OR cat:cs.AI OR cat:cs.CV")

Collected data is saved as `arxiv_mixed_trend_{START_YEAR}_{END_YEAR}.csv`.

**Note**: For large datasets, it is recommended to split the file into two parts and save them as `arxiv_data_part1.csv.gz` and `arxiv_data_part2.csv.gz`.

### 3. Dashboard Execution

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser.

## 📊 Dashboard Features

### Sidebar Settings

**Order:**
1. **📂 Category Group Selection**: 
   - Select from 10 category groups (AI/ML, Computer Vision, Robotics, etc.)
   - Shows paper count for each group
   - View detailed category list (with Korean annotations)
2. **Year Range Selection**: Select analysis year range using slider
3. **Top N Keywords**: Select number of top keywords to analyze (5~30)
4. **Keyword Extraction Source**: Choose to extract keywords from Abstract or Title
5. **🚀 Start Analysis**: Start analysis with selected settings
6. **Data Summary**: Shows total papers, selected papers, year range
7. **⚙️ Settings**:
   - **🌐 Language**: Choose between Korean / English
     - When language is changed, all menus, charts, and messages are immediately updated to the selected language
   - **🔄 Clear Cache & Refresh**: Clear Streamlit cache and refresh the page

### Tab Configuration

#### 1. 📈 Trend Over Time
- Streamgraph-style area chart
- Visualizes changes in the number of papers for major keywords over time
- Provides keyword statistics table

#### 2. 🔥 Heatmap
- X-axis: Year, Y-axis: Keyword
- Represents paper count using color intensity
- Easily identify keyword frequency by year

#### 3. 🏆 Ranking Competition (Bump Chart)
- Shows keyword ranking changes by year as line graph
- Tracks ranking fluctuations of top keywords
- Y-axis reversed so rank 1 appears at the top

#### 4. 🚀 Hype Cycle
- X-axis: Technology Growth Rate (Year-over-Year Increase)
- Y-axis: Technology Mentions (Total Volume)
- Point Size: Latest Year Mentions
- Visually analyze technology growth stages

## 📁 Project Structure

```
robotics_trend/
├── collector_improved.py    # arXiv data collection script (10-day interval split collection)
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Required library list
├── README.md                 # Project documentation (Korean)
├── README_EN.md              # Project documentation (English)
├── arxiv_mixed_trend_*.csv   # Collected paper data (generated)
└── arxiv_data_part*.csv.gz   # Split data files for dashboard (optional)
```

## 🔧 Technology Stack

- **Python 3.9+**
- **Data Collection**: `arxiv` library
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `plotly` (interactive charts)
- **UI**: `streamlit`
- **NLP**: `scikit-learn` (CountVectorizer), `nltk` (stopwords removal)
- **Parallel Processing**: `concurrent.futures` (ThreadPoolExecutor)
- **Progress Display**: `tqdm`

## 📝 Data Structure

Columns in the collected CSV file:
- `arXiv_ID`: Paper arXiv ID
- `Title`: Paper title
- `First_Author`: First author name
- `Category`: Paper categories (comma-separated)
- `Abstract`: Paper abstract
- `Published_Date`: Publication date (YYYY-MM-DD format)

## ⚠️ Notes

1. **Data Collection Time**: 
   - Large-scale data collection may take a long time (several hours possible)
   - Random delay between requests (0.5~4 seconds) is included to reduce API load
   - 10-day interval split collection prevents HTTP 500 errors

2. **NLTK Data**: NLTK stopwords data is automatically downloaded on first run.

3. **Memory Usage**: Processing large amounts of paper data may increase memory usage.

4. **File Size**: 
   - Collected data can be very large (hundreds of thousands of entries)
   - It is recommended to split and compress files when necessary
   - The dashboard reads two files: `arxiv_data_part1.csv.gz` and `arxiv_data_part2.csv.gz`

5. **Cache Management**: 
   - Category lists may appear duplicated due to Streamlit caching
   - Use the "🔄 Clear Cache & Refresh" button in the sidebar to resolve this

## 🐛 Troubleshooting

### Data Collection Errors
- **Check Internet Connection**: Stable internet connection is required
- **arXiv API Server Status**: Check if arXiv servers are functioning normally
- **HTTP 500 Error**: 
  - 10-day interval split collection is automatically applied
  - Try reducing the `MAX_WORKERS` value (default: 6)
- **Adjust Collection Period**: Try reducing `TARGET_START_YEAR` and `TARGET_END_YEAR` for testing

### Dashboard Execution Errors
- **File Not Found Error**: 
  - Check if `arxiv_data_part1.csv.gz` and `arxiv_data_part2.csv.gz` files exist
  - If only one file exists, you can modify the `load_data()` function to read a single file
- **Library Installation**: Make sure all required libraries are installed: `pip install -r requirements.txt`
- **Cache Issues**: 
  - If category lists appear duplicated or out of order
  - Click the "🔄 Clear Cache & Refresh" button or
  - Run `rm -rf ~/.streamlit/cache` in terminal (macOS/Linux)

## 📄 License

This project is provided for educational and research purposes.

