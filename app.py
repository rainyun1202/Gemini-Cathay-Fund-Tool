import streamlit as st
import requests
import pandas as pd
import urllib3
import logging
import io
import yfinance as yf
import plotly.express as px  # 新增：互動式繪圖套件
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

# === 設定區 ===
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="全球市場與基金分析", layout="wide")

class Config:
    """全域配置類別"""
    # --- 國泰基金設定 ---
    API_URL = "https://www.cathaybk.com.tw/cathaybk/service/newwealth/fund/chartservice.asmx/GetFundNavChart"
    BASE_URL = "https://www.cathaybk.com.tw/cathaybk/personal/investment/fund/details/?fundid={}"
    USER_AGENT = "Mozilla/5.0"
    TIMEOUT = 10
    DEFAULT_DATE_FROM = "1900/01/01"
    
    DEFAULT_FUND_IDS_LIST = [
        "00580030", "00400013", "00060004", "00100045", "00010144", "00120001",
        "00040097", "10340003", "10350005", "00060003", "00400029", "00100046",
        "00010074", "0074B059", "0012C007", "0012C004", "0012C033", "0012C035",
        "0012C008", "00100118", "00400156", "00400104", "00040052", "10020058",
        "10110022", "0074B065", "00100058", "00580062", "10310016", "00100063",
        "00560011", "00400072"
    ]

    # --- Yahoo Finance 市場指數設定 ---
    MARKET_TICKERS = {
        "比特幣 (BTC-USD)": "BTC-USD",
        "VIX 恐慌指數": "^VIX",
        "美國 10 年期公債殖利率": "^TNX",
        "美元指數 (DXY)": "DX-Y.NYB",
        "布蘭特原油": "BZ=F",
        "黃金期貨": "GC=F",
        "羅素 2000": "^RUT",
        "NASDAQ 指數": "^IXIC",
        "S&P 500": "^GSPC",
        "費城半導體": "^SOX",
        "上證指數": "000001.SS",
        "香港國企指數": "^HSCE"
    }


class FundScraper:
    """負責抓取國泰基金"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.USER_AGENT})
        self.session.verify = False 

    def fetch_nav(self, fund_id: str) -> Optional[pd.DataFrame]:
        target_url = Config.BASE_URL.format(fund_id)
        payload = {"req": {"Keys": [fund_id], "From": Config.DEFAULT_DATE_FROM}}
        headers = {"Referer": target_url}

        try:
            resp = self.session.post(Config.API_URL, json=payload, headers=headers, timeout=Config.TIMEOUT)
            resp.raise_for_status()
            data_json = resp.json()
            if not data_json.get('Data'): return None
            
            fund_info = data_json['Data'][0]
            df = pd.DataFrame(fund_info['data'], columns=['timestamp', 'NAV'])
            df['日期'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df['基金名稱'] = fund_info['name']
            df['URL'] = target_url
            return df[['日期', 'NAV', '基金名稱', 'URL']]
        except Exception as e:
            logger.error(f"基金 {fund_id} 失敗: {e}")
            return None

    def fetch_all(self, fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
        # 注意：為了配合 Caching，這裡移除了 progress_bar 的參數傳遞
        # 因為快取函數在背景執行時無法更新 UI 元件
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(self.fetch_nav, fid): fid for fid in fund_ids}
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    df = future.result()
                    if df is not None: results[fid] = df
                except Exception: pass
        return results


class MarketScraper:
    """負責抓取 Yahoo Finance 市場數據"""
    def fetch_history(self, name: str, ticker: str) -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            # 使用 "max" 抓取完整歷史數據
            hist = stock.history(period="max")
            
            if hist.empty: return None
            
            hist = hist.reset_index()
            hist['Date'] = hist['Date'].dt.date
            
            df = pd.DataFrame()
            df['日期'] = hist['Date']
            df['NAV'] = hist['Close']
            df['基金名稱'] = name
            df['URL'] = f"https://finance.yahoo.com/quote/{ticker}"
            
            return df[['日期', 'NAV', '基金名稱', 'URL']]
        except Exception as e:
            logger.error(f"市場指數 {name} 失敗: {e}")
            return None

    def fetch_all(self, market_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        results = {}
        for name, ticker in market_dict.items():
            df = self.fetch_history(name, ticker)
            if df is not None: results[name] = df
        return results


class FundAnalyzer:
    """負責計算邏輯"""
    @staticmethod
    def analyze_single(df: pd.DataFrame) -> Dict[str, Any]:
        df = df.sort_values('日期')
        fund_name = df['基金名稱'].iloc[0]
        url = df['URL'].iloc[0]
        latest = df.iloc[-1]
        
        hist_max_idx = df['NAV'].idxmax()
        hist_min_idx = df['NAV'].idxmin()

        one_year_ago = df['日期'].max() - timedelta(days=365)
        df_1y = df[df['日期'] >= one_year_ago]
        
        if df_1y.empty:
            max_1y, min_1y, max_1y_date, min_1y_date = None, None, None, None
        else:
            max_1y_idx = df_1y['NAV'].idxmax()
            min_1y_idx = df_1y['NAV'].idxmin()
            max_1y = df_1y.loc[max_1y_idx, 'NAV']
            max_1y_date = df_1y.loc[max_1y_idx, '日期']
            min_1y = df_1y.loc[min_1y_idx, 'NAV']
            min_1y_date = df_1y.loc[min_1y_idx, '日期']

        return {
            "基金名稱": fund_name,
            "基金連結": url,
            "最新價格": latest['NAV'],
            "最新價格日期": latest['日期'],
            "歷史最高價格": df.loc[hist_max_idx, 'NAV'],
            "歷史最高價格日期": df.loc[hist_max_idx, '日期'],
            "歷史最低價格": df.loc[hist_min_idx, 'NAV'],
            "歷史最低價格日期": df.loc[hist_min_idx, '日期'],
            "近一年最高價格": max_1y,
            "近一年最高價格日期": max_1y_date,
            "近一年最低價格": min_1y,
            "近一年最低價格日期": min_1y_date
        }

    @staticmethod
    def analyze_all(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        summary_list = []
        for df in data_map.values():
            summary_list.append(FundAnalyzer.analyze_single(df))
        return pd.DataFrame(summary_list)


class ExcelReport:
    """Excel 產生器"""
    @staticmethod
    def create_excel_bytes(summary_df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_df = summary_df.drop(columns=['基金連結'])
            display_df.to_excel(writer, index=False, header=False, sheet_name='Summary', startrow=1)

            workbook = writer.book
            worksheet = writer.sheets['Summary']
            
            ExcelReport._apply_styles(workbook, worksheet, display_df, summary_df)
            ExcelReport._set_columns_width(display_df, worksheet)
            
            worksheet.freeze_panes(1, 0)
        return output.getvalue()

    @staticmethod
    def _apply_styles(workbook, worksheet, display_df, original_df):
        base_font = 'Microsoft JhengHei'
        header_fmt = workbook.add_format({'bold': True, 'font_name': base_font, 'bg_color': '#DCE6F1', 'align': 'center', 'valign': 'vcenter', 'border': 1})
        text_fmt = workbook.add_format({'font_name': base_font, 'valign': 'top', 'border': 1})
        num_fmt = workbook.add_format({'font_name': base_font, 'valign': 'top', 'border': 1, 'num_format': '#,##0.00'}) 
        link_fmt = workbook.add_format({'font_color': 'blue', 'underline': 1, 'font_name': base_font, 'valign': 'top', 'border': 1})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': base_font, 'valign': 'top', 'border': 1})

        for col, val in enumerate(display_df.columns):
            worksheet.write(0, col, val, header_fmt)

        date_cols = [i for i, c in enumerate(display_df.columns) if '日期' in str(c) or 'Date' in str(c)]
        
        for i in range(len(display_df)):
            name = display_df.iat[i, 0]
            url = original_df.iloc[i]['基金連結']
            worksheet.write_url(i+1, 0, url, link_fmt, string=name)

            for j in range(1, len(display_df.columns)):
                val = display_df.iat[i, j]
                if j in date_cols and pd.notna(val):
                    if isinstance(val, (str, datetime, pd.Timestamp)): val = pd.to_datetime(val)
                    worksheet.write_datetime(i+1, j, val, date_fmt)
                elif isinstance(val, (int, float)):
                    worksheet.write_number(i+1, j, val, num_fmt)
                else:
                    worksheet.write(i+1, j, str(val), text_fmt)

    @staticmethod
    def _set_columns_width(df, worksheet):
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(lambda x: len(x.encode('utf-8'))).max(),
                len(str(col).encode('utf-8'))
            )
            width = min(max(max_len * 0.9, 10), 50)
            worksheet.set_column(i, i, width)

# === 【新增】 快取資料載入函式 ===
@st.cache_data(ttl=3600, show_spinner="正在自網路下載最新數據...")
def load_data_with_cache(target_markets: Dict[str, str], fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    這個函式會被 Streamlit 快取。
    只要輸入參數 (fund_ids, target_markets) 沒變，就會直接回傳上次的結果，不會重新下載。
    """
    all_data = {}
    
    # 下載市場資料
    if target_markets:
        market_scraper = MarketScraper()
        # 注意：為了 Cache 穩定，這裡不傳入 progress bar
        market_data = market_scraper.fetch_all(target_markets)
        all_data.update(market_data)
        
    # 下載基金資料
    if fund_ids:
        fund_scraper = FundScraper()
        fund_data = fund_scraper.fetch_all(fund_ids)
        all_data.update(fund_data)
        
    return all_data

# === 【新增】 繪圖邏輯函式 ===
def plot_normalized_trends(all_data: Dict[str, pd.DataFrame], selected_assets: List[str]):
    """繪製歸一化 (累積報酬率) 比較圖"""
    if not selected_assets:
        st.info("請從上方選單勾選至少一項資產進行比較。")
        return

    plot_data = []
    
    for name in selected_assets:
        if name in all_data:
            df = all_data[name].copy()
            df = df.sort_values('日期')
            
            # 過濾掉極端舊的資料，避免圖表拉太長，這裡預設取最近 3 年 (若不足則全取)
            start_date_limit = pd.to_datetime("today") - pd.DateOffset(years=3)
            df['日期'] = pd.to_datetime(df['日期']) # 確保日期格式
            df = df[df['日期'] >= start_date_limit]

            if not df.empty:
                # 歸一化邏輯：(當日價格 / 第一天價格 - 1) * 100
                first_nav = df['NAV'].iloc[0]
                df['累積報酬率(%)'] = ((df['NAV'] / first_nav) - 1) * 100
                df['資產名稱'] = name
                plot_data.append(df[['日期', '累積報酬率(%)', '資產名稱']])
    
    if not plot_data:
        st.warning("選取的資產在近三年內無足夠數據可供繪圖。")
        return

    # 合併所有資料
    final_df = pd.concat(plot_data)
    
    # 使用 Plotly 畫圖
    fig = px.line(
        final_df, 
        x="日期", 
        y="累積報酬率(%)", 
        color="資產名稱",
        title="近三年累積報酬率比較 (歸一化)",
        hover_data={"日期": "|%Y-%m-%d"},
        height=500
    )
    
    # 優化圖表樣式
    fig.update_layout(
        xaxis_title="",
        yaxis_title="累積報酬率 (%)",
        hovermode="x unified", # 滑鼠移過去顯示所有資產數值
        legend=dict(orientation="h", y=1.1) # 圖例放上面
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 全球市場與基金淨值戰情室")
    st.markdown("整合 **國泰基金** 與 **全球關鍵市場指標** 的自動化分析工具。")

    # === 側邊欄佈局 ===
    with st.sidebar:
        st.header("⚙️ 設定面板")
        
        with st.expander("🌍 全球市場指標", expanded=True):
            selected_markets = st.multiselect(
                "選擇關注市場指標",
                options=list(Config.MARKET_TICKERS.keys()),
                default=list(Config.MARKET_TICKERS.keys())
            )
            target_markets = {name: Config.MARKET_TICKERS[name] for name in selected_markets}

        with st.expander("🏦 國泰基金清單", expanded=True):
            default_ids = ",\n".join(Config.DEFAULT_FUND_IDS_LIST)
            fund_input = st.text_area(
                "基金代號 (每行一個)", 
                value=default_ids, 
                height=300, 
                help="請輸入基金代號，多筆請換行或用逗號分隔"
            )
            fund_ids = [x.strip() for x in fund_input.replace("\n", ",").split(",") if x.strip()]

    # === 主邏輯修改：使用 session_state 或直接執行 ===
    # 這裡我們將邏輯改為：使用者調整側邊欄 -> 點擊按鈕 -> 載入資料 (有快取) -> 顯示 Tabs
    
    if st.button("🚀 開始/更新 分析", type="primary"):
        st.session_state['has_run'] = True

    # 檢查是否已經按過按鈕 (讓畫面刷新時不會消失)
    if st.session_state.get('has_run'):
        
        # 1. 載入資料 (使用快取，速度快)
        # 注意：我們移除了進度條，改用 st.spinner (由装饰器處理)
        all_data = load_data_with_cache(target_markets, fund_ids)

        if not all_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return

        # 2. 建立分頁 (Tabs)
        tab1, tab2 = st.tabs(["📋 報表總覽", "📈 趨勢比較"])

        # === 分頁 1：原本的表格與 Excel 下載 ===
        with tab1:
            summary_df = FundAnalyzer.analyze_all(all_data)
            st.success(f"✅ 完成！共分析 {len(summary_df)} 筆標的")
            st.dataframe(summary_df)

            excel_data = ExcelReport.create_excel_bytes(summary_df)
            file_name = f"Global_Market_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
            
            st.download_button(
                label="📥 下載完整 Excel 報表",
                data=excel_data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # === 分頁 2：視覺化圖表 (新增功能) ===
        with tab2:
            st.subheader("📈 資產走勢 PK")
            st.caption("比較不同資產在相同時間區間內的漲跌幅表現 (近三年，起點歸零)")
            
            # 讓使用者選擇要畫哪些圖 (預設全選，但如果太多會很亂，建議選前 5 個)
            all_assets_list = list(all_data.keys())
            chart_selection = st.multiselect(
                "選擇要繪製的資產:",
                options=all_assets_list,
                default=all_assets_list[:5] # 預設只選前5個避免眼花
            )
            
            plot_normalized_trends(all_data, chart_selection)

if __name__ == "__main__":
    main()