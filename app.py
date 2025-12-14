import streamlit as st
import requests
import pandas as pd
import urllib3
import logging
import io
import yfinance as yf
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

    # 移除原本的 fetch_all，改為透過外部函式呼叫以支援快取
    def fetch_single_safe(self, fid):
        """單純為了 ThreadPool 設計的輔助函式"""
        return self.fetch_nav(fid)


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

# === 【第一階段優化】快取函式 ===
# 這些函式獨立於 Class 之外，因為 Streamlit 的快取裝飾器對 Class method 支援度較複雜
# ttl=3600 代表快取存活 1 小時，避免資料過舊
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_fund_data(fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """[快取] 抓取國泰基金資料"""
    scraper = FundScraper()
    results = {}
    # 為了顯示進度，我們這裡簡單模擬，實際上因為有快取，第二次執行會瞬間完成
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(scraper.fetch_single_safe, fid): fid for fid in fund_ids}
        for future in as_completed(future_to_id):
            fid = future_to_id[future]
            try:
                df = future.result()
                if df is not None: results[fid] = df
            except Exception: pass
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(market_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """[快取] 抓取市場資料"""
    scraper = MarketScraper()
    results = {}
    for name, ticker in market_dict.items():
        df = scraper.fetch_history(name, ticker)
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

# === 【第二階段優化】視覺化處理類別 ===
class Visualizer:
    @staticmethod
    def prepare_chart_data(all_data: Dict[str, pd.DataFrame], normalize=False) -> pd.DataFrame:
        """
        將多個 DataFrame 合併為適合繪圖的 Wide Format
        normalize: 是否將起點歸一化為 100% (方便比較漲跌幅)
        """
        # 1. 提取所有資料的 '日期' 和 'NAV'
        series_list = []
        for name, df in all_data.items():
            # 確保日期是 datetime 格式
            temp_df = df[['日期', 'NAV']].copy()
            temp_df['日期'] = pd.to_datetime(temp_df['日期'])
            temp_df = temp_df.set_index('日期')
            temp_df.columns = [name]
            series_list.append(temp_df)
        
        if not series_list:
            return pd.DataFrame()

        # 2. 合併 (Outer Join 以保留所有日期)
        chart_df = pd.concat(series_list, axis=1).sort_index()
        
        # 3. 填補空值 (Forward Fill: 假日沿用週五價格)
        chart_df = chart_df.fillna(method='ffill')
        
        # 4. 歸一化處理 (可選)
        if normalize:
            # 找到每一欄第一個非空值，將其設為基準點 (100)
            # 這樣可以比較不同價格區間的商品 (如比特幣 90000 vs 基金 10)
            return chart_df.apply(lambda x: x / x.first_valid_index() * 100 if x.first_valid_index() else x)
        
        return chart_df

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

    # 主畫面按鈕
    if st.button("🚀 開始分析", type="primary"):
        # 這裡的 Spinner 會在資料抓取時轉圈圈
        with st.spinner("正在連線至全球資料庫 (若為第一次抓取請稍候)..."):
            all_data = {}
            
            # 使用快取函式 (第一次慢，第二次快)
            if target_markets:
                market_data = get_cached_market_data(target_markets)
                all_data.update(market_data)
                
            if fund_ids:
                fund_data = get_cached_fund_data(fund_ids)
                all_data.update(fund_data)
        
        if not all_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return

        # 建立頁籤 (Tabs) 來區分「數據報表」與「趨勢圖表」
        tab1, tab2 = st.tabs(["📋 數據報表", "📈 趨勢圖表"])

        with tab1:
            st.success(f"✅ 分析完成！共 {len(all_data)} 筆標的")
            summary_df = FundAnalyzer.analyze_all(all_data)
            st.dataframe(summary_df.head(10))
            
            excel_data = ExcelReport.create_excel_bytes(summary_df)
            file_name = f"Global_Market_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
            st.download_button(
                label="📥 下載完整 Excel 報表",
                data=excel_data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with tab2:
            st.subheader("歷史走勢比較")
            
            # 控制項：選擇要畫的標的
            all_options = list(all_data.keys())
            selected_chart_items = st.multiselect(
                "選擇要繪製的項目 (建議 3-5 項)", 
                options=all_options,
                default=all_options[:3] if len(all_options) >= 3 else all_options
            )
            
            # 控制項：時間範圍
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                # 歸一化開關
                normalize_chart = st.checkbox("📈 歸一化比較 (以起點為 100%)", value=True, help="將不同價格單位的商品放在同一個起跑點比較漲跌幅")
            
            if selected_chart_items:
                # 篩選出要畫圖的 data
                chart_subset = {k: v for k, v in all_data.items() if k in selected_chart_items}
                
                # 準備繪圖資料
                chart_df = Visualizer.prepare_chart_data(chart_subset, normalize=normalize_chart)
                
                # 繪製互動式線圖
                st.line_chart(chart_df)
            else:
                st.info("請從上方選單選擇至少一個項目來繪製圖表")

if __name__ == "__main__":
    main()