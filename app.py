import streamlit as st
import requests
import pandas as pd
import urllib3
import logging
import io
import yfinance as yf  # 新增：Yahoo Finance 套件
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

    # --- Yahoo Finance 市場指數設定 (代號對照表) ---
    # 格式: "顯示名稱": "Yahoo代號"
    MARKET_TICKERS = {
        "比特幣 (BTC-USD)": "BTC-USD",
        "VIX 恐慌指數": "^VIX",
        "美國 10 年期公債殖利率": "^TNX",
        "美元指數 (DXY)": "DX-Y.NYB", # 或 ^DXY
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
    """負責抓取國泰基金 (維持原本邏輯)"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.USER_AGENT})
        self.session.verify = False 

    def fetch_nav(self, fund_id: str) -> Optional[pd.DataFrame]:
        # ... (維持原本的抓取邏輯不變) ...
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

    def fetch_all(self, fund_ids: List[str], progress_bar=None) -> Dict[str, pd.DataFrame]:
        results = {}
        total = len(fund_ids)
        completed = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(self.fetch_nav, fid): fid for fid in fund_ids}
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    df = future.result()
                    if df is not None: results[fid] = df
                except Exception: pass
                completed += 1
                if progress_bar: progress_bar.progress(completed / total, text=f"正在抓取基金... ({completed}/{total})")
        return results


class MarketScraper:
    """[新增] 負責抓取 Yahoo Finance 市場數據"""
    
    def fetch_history(self, name: str, ticker: str) -> Optional[pd.DataFrame]:
        try:
            # 抓取 2 年資料以確保能計算近一年高低點
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            
            if hist.empty:
                return None
            
            # === 關鍵步驟：資料清洗與格式化 ===
            # 我們要讓 Yahoo 的資料長得跟國泰基金的資料一模一樣
            # 1. 重設索引，將 Date 變成欄位
            hist = hist.reset_index()
            
            # 2. 挑選需要的欄位 (Date, Close) 並改名
            # 注意：Yahoo 的 Date 通常帶有時區，需要移除時區資訊以便對齊
            hist['Date'] = hist['Date'].dt.date
            
            # 建立目標格式 DataFrame
            df = pd.DataFrame()
            df['日期'] = hist['Date']
            df['NAV'] = hist['Close']  # 將收盤價視為淨值
            df['基金名稱'] = name       # 使用我們自定義的中文名稱
            df['URL'] = f"https://finance.yahoo.com/quote/{ticker}" # 偽造一個 Yahoo 連結
            
            return df[['日期', 'NAV', '基金名稱', 'URL']]
            
        except Exception as e:
            logger.error(f"市場指數 {name} 失敗: {e}")
            return None

    def fetch_all(self, market_dict: Dict[str, str], progress_bar=None) -> Dict[str, pd.DataFrame]:
        results = {}
        total = len(market_dict)
        completed = 0
        
        # 雖然 yfinance 支援批量下載，但為了配合我們的資料結構與錯誤處理，
        # 我們還是單支單支處理 (速度很快)
        for name, ticker in market_dict.items():
            df = self.fetch_history(name, ticker)
            if df is not None:
                results[name] = df # 這裡用名稱當 Key
            
            completed += 1
            if progress_bar:
                progress_bar.progress(completed / total, text=f"正在抓取市場指數... ({name})")
        
        return results


class FundAnalyzer:
    """負責計算邏輯 (完全不需要修改，因為輸入格式統一了)"""
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
            "名稱": fund_name, # 微調欄位名稱以通用化
            "連結": url,
            "最新價格": latest['NAV'],
            "最新日期": latest['日期'],
            "歷史最高": df.loc[hist_max_idx, 'NAV'],
            "歷史最高日": df.loc[hist_max_idx, '日期'],
            "歷史最低": df.loc[hist_min_idx, 'NAV'],
            "歷史最低日": df.loc[hist_min_idx, '日期'],
            "近一年最高": max_1y,
            "近一年最高日": max_1y_date,
            "近一年最低": min_1y,
            "近一年最低日": min_1y_date
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
            # 移除連結欄位用於顯示
            display_df = summary_df.drop(columns=['連結'])
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
        num_fmt = workbook.add_format({'font_name': base_font, 'valign': 'top', 'border': 1, 'num_format': '#,##0.00'}) # 加了小數點格式
        link_fmt = workbook.add_format({'font_color': 'blue', 'underline': 1, 'font_name': base_font, 'valign': 'top', 'border': 1})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': base_font, 'valign': 'top', 'border': 1})

        for col, val in enumerate(display_df.columns):
            worksheet.write(0, col, val, header_fmt)

        date_cols = [i for i, c in enumerate(display_df.columns) if '日' in str(c) or 'Date' in str(c)]
        
        for i in range(len(display_df)):
            name = display_df.iat[i, 0]
            url = original_df.iloc[i]['連結']
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
            # 簡單估算欄寬
            worksheet.set_column(i, i, 15)


def main():
    st.title("📊 全球市場與基金淨值戰情室")
    st.markdown("整合 **國泰基金** 與 **全球關鍵市場指標** 的自動化分析工具。")

    col1, col2 = st.columns(2)
    
    # 1. 基金設定
    with col1:
        st.subheader("🏦 國泰基金清單")
        default_ids = ",\n".join(Config.DEFAULT_FUND_IDS_LIST)
        fund_input = st.text_area("基金代號", value=default_ids, height=200)
        fund_ids = [x.strip() for x in fund_input.replace("\n", ",").split(",") if x.strip()]

    # 2. 市場指數設定 (使用多選選單)
    with col2:
        st.subheader("🌍 全球市場指標")
        selected_markets = st.multiselect(
            "選擇要關注的指標",
            options=list(Config.MARKET_TICKERS.keys()),
            default=list(Config.MARKET_TICKERS.keys())
        )
        # 轉回 Dict 格式以便處理
        target_markets = {name: Config.MARKET_TICKERS[name] for name in selected_markets}

    if st.button("🚀 開始全域分析", type="primary"):
        # 進度條共用
        bar = st.progress(0, text="初始化...")
        
        all_data = {}
        
        # A. 抓市場資料
        if target_markets:
            market_scraper = MarketScraper()
            market_data = market_scraper.fetch_all(target_markets, bar)
            all_data.update(market_data)
            
        # B. 抓基金資料
        if fund_ids:
            fund_scraper = FundScraper()
            fund_data = fund_scraper.fetch_all(fund_ids, bar)
            all_data.update(fund_data)
            
        bar.progress(100, text="分析中...")

        if not all_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return

        # C. 統一分析
        summary_df = FundAnalyzer.analyze_all(all_data)
        
        # D. 顯示與下載
        st.success(f"✅ 完成！共分析 {len(summary_df)} 筆標的")
        st.dataframe(summary_df)

        excel_data = ExcelReport.create_excel_bytes(summary_df)
        file_name = f"Global_Market_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        st.download_button(
            label="📥 下載完整 Excel 戰情報表",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()