import streamlit as st
import requests
import pandas as pd
import urllib3
import logging
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

# === 設定區 ===
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 設定 Logging (在 Streamlit 中，這會輸出到後台 Terminal)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定網頁標題與寬度佈局
st.set_page_config(page_title="國泰基金淨值分析", layout="wide")

class Config:
    """全域配置類別"""
    API_URL = "https://www.cathaybk.com.tw/cathaybk/service/newwealth/fund/chartservice.asmx/GetFundNavChart"
    BASE_URL = "https://www.cathaybk.com.tw/cathaybk/personal/investment/fund/details/?fundid={}"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    TIMEOUT = 10
    DEFAULT_DATE_FROM = "1900/01/01"
    
    # 這裡直接定義預設清單，方便網頁輸入框使用
    DEFAULT_FUND_IDS_LIST = [
        "00580030", "00400013", "00060004", "00100045", "00010144", "00120001",
        "00040097", "10340003", "10350005", "00060003", "00400029", "00100046",
        "00010074", "0074B059", "0012C007", "0012C004", "0012C033", "0012C035",
        "0012C008", "00100118", "00400156", "00400104", "00040052", "10020058",
        "10110022", "0074B065", "00100058", "00580062", "10310016", "00100063",
        "00560011", "00400072"
    ]


class FundScraper:
    """負責網路請求與資料抓取 (核心邏輯不變)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.USER_AGENT})
        self.session.verify = False 

    def fetch_nav(self, fund_id: str) -> Optional[pd.DataFrame]:
        target_url = Config.BASE_URL.format(fund_id)
        payload = {"req": {"Keys": [fund_id], "From": Config.DEFAULT_DATE_FROM}}
        headers = {"Referer": target_url}

        try:
            resp = self.session.post(
                Config.API_URL, json=payload, headers=headers, timeout=Config.TIMEOUT
            )
            resp.raise_for_status()
            data_json = resp.json()

            if not data_json.get('Data'):
                return None

            fund_info = data_json['Data'][0]
            df = pd.DataFrame(fund_info['data'], columns=['timestamp', 'NAV'])
            
            df['日期'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df['基金名稱'] = fund_info['name']
            df['URL'] = target_url
            
            return df[['日期', 'NAV', '基金名稱', 'URL']]

        except Exception as e:
            logger.error(f"取得基金 {fund_id} 失敗: {e}")
            return None

    def fetch_all_funds(self, fund_ids: List[str], progress_bar=None) -> Dict[str, pd.DataFrame]:
        """
        新增 progress_bar 參數，用來更新網頁上的進度條
        """
        results = {}
        total = len(fund_ids)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(self.fetch_nav, fid): fid for fid in fund_ids}
            
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[fid] = df
                except Exception as e:
                    logger.error(f"Error {fid}: {e}")
                
                # 更新進度條
                completed += 1
                if progress_bar:
                    progress_bar.progress(completed / total, text=f"正在抓取... ({completed}/{total})")
        
        return results


class FundAnalyzer:
    """負責計算邏輯 (核心邏輯不變)"""
    
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
            max_1y, min_1y = None, None
            max_1y_date, min_1y_date = None, None
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
    """Excel 產生器：修改為寫入 BytesIO 記憶體"""
    
    @staticmethod
    def create_excel_bytes(summary_df: pd.DataFrame) -> bytes:
        """產生 Excel 檔案並回傳二進位資料 (bytes)"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_df = summary_df.drop(columns=['基金連結'])
            display_df.to_excel(writer, index=False, header=False, sheet_name='Summary', startrow=1)

            workbook = writer.book
            worksheet = writer.sheets['Summary']
            
            # 使用我們之前定義的樣式邏輯
            ExcelReport._apply_styles(workbook, worksheet, display_df, summary_df)
            ExcelReport._set_columns_width(display_df, worksheet)
            
            worksheet.freeze_panes(1, 0)
        
        return output.getvalue()

    @staticmethod
    def _apply_styles(workbook, worksheet, display_df, original_df):
        base_font = 'Microsoft JhengHei'
        styles = {
            'header': workbook.add_format({'bold': True, 'font_name': base_font, 'bg_color': '#DCE6F1', 'align': 'center', 'valign': 'vcenter', 'border': 1}),
            'text': workbook.add_format({'font_name': base_font, 'text_wrap': True, 'valign': 'top', 'border': 1}),
            'link': workbook.add_format({'font_color': 'blue', 'underline': 1, 'font_name': base_font, 'valign': 'top', 'border': 1}),
            'date': workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': base_font, 'valign': 'top', 'border': 1})
        }

        for col_num, value in enumerate(display_df.columns):
            worksheet.write(0, col_num, value, styles['header'])

        date_cols = [idx for idx, col in enumerate(display_df.columns) if '日期' in str(col)]
        rows, cols = display_df.shape

        for i in range(rows):
            fund_name = display_df.iat[i, 0]
            url = original_df.iloc[i]['基金連結']
            worksheet.write_url(i + 1, 0, url, styles['link'], string=fund_name)

            for j in range(1, cols):
                val = display_df.iat[i, j]
                if j in date_cols and pd.notna(val):
                    if isinstance(val, (str, datetime, pd.Timestamp)):
                         val = pd.to_datetime(val)
                    worksheet.write_datetime(i + 1, j, val, styles['date'])
                elif isinstance(val, (int, float)):
                    worksheet.write_number(i + 1, j, val, styles['text'])
                else:
                    worksheet.write(i + 1, j, str(val), styles['text'])

    @staticmethod
    def _set_columns_width(df, worksheet):
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(lambda x: len(x.encode('utf-8'))).max(),
                len(str(col).encode('utf-8'))
            )
            width = min(max(max_len * 0.9, 10), 50)
            worksheet.set_column(i, i, width)


def main():
    # === 網頁介面設計 ===
    st.title("📊 國泰基金淨值自動分析工具")
    st.markdown("此工具協助您自動抓取國泰基金歷史淨值，計算近一年高低點，並生成 Excel 報表。")

    # 1. 側邊欄：設定基金清單
    with st.sidebar:
        st.header("⚙️ 基金設定")
        default_ids_str = ",\n".join(Config.DEFAULT_FUND_IDS_LIST)
        user_input = st.text_area(
            "請輸入基金代號 (以逗號或換行分隔)", 
            value=default_ids_str, 
            height=300,
            help="你可以隨意新增或刪除這裡的代號"
        )
        
        # 處理使用者輸入
        input_ids = [x.strip() for x in user_input.replace("\n", ",").split(",") if x.strip()]
        st.info(f"目前共選取 {len(input_ids)} 支基金")

    # 2. 主按鈕
    if st.button("🚀 開始分析", type="primary"):
        if not input_ids:
            st.error("請至少輸入一支基金代號！")
            return

        # 3. 執行抓取
        scraper = FundScraper()
        
        # 建立一個進度條物件
        progress_bar = st.progress(0, text="準備開始...")
        
        try:
            # 傳入進度條物件讓 Scraper 更新
            all_data = scraper.fetch_all_funds(input_ids, progress_bar)
            progress_bar.progress(100, text="下載完成！開始分析...")
        except Exception as e:
            st.error(f"發生錯誤：{e}")
            return

        if not all_data:
            st.warning("⚠️ 沒有抓到任何資料，請檢查代號是否正確。")
            return

        # 4. 執行分析
        summary_df = FundAnalyzer.analyze_all(all_data)
        
        # 5. 顯示結果
        st.success("✅ 分析完成！")
        
        # 在網頁上預覽前 10 筆
        st.subheader("📋 分析結果預覽")
        st.dataframe(summary_df.head(10))

        # 6. 下載按鈕
        # 呼叫我們修改過的 ExcelReport，拿到二進位資料
        excel_data = ExcelReport.create_excel_bytes(summary_df)
        
        file_name = f"fund_summary_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        st.download_button(
            label="📥 下載 Excel 完整報表",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()