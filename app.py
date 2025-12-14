import streamlit as st
import requests
import pandas as pd
import urllib3
import logging
import io
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from dateutil.relativedelta import relativedelta  # 新增：處理月份與年份的加減更精確

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
        "00010074", "0074B059", "0012C007", "0012C004", "0012C033", "00120002",
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

    # --- 時間區間選項 ---
    TIME_RANGES = {
        "近1月": relativedelta(months=1),
        "近3月": relativedelta(months=3),
        "近半年": relativedelta(months=6),
        "近1年": relativedelta(years=1),
        "近3年": relativedelta(years=3),
        "近5年": relativedelta(years=5),
        "近10年": relativedelta(years=10),
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

# === 快取資料載入函式 ===
@st.cache_data(ttl=3600, show_spinner="正在自網路下載最新數據...")
def load_data_with_cache(target_markets: Dict[str, str], fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
    all_data = {}
    if target_markets:
        market_scraper = MarketScraper()
        market_data = market_scraper.fetch_all(target_markets)
        all_data.update(market_data)
    if fund_ids:
        fund_scraper = FundScraper()
        fund_data = fund_scraper.fetch_all(fund_ids)
        all_data.update(fund_data)
    return all_data

# === 【優化】 雙軸繪圖函式 (視覺歸一化 + 真實數值軸) ===
def plot_dual_axis_trends(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
    """
    繪製雙Y軸價格走勢比較圖
    特點：
    1. 保留原始價格數值 (Y軸顯示真實股價/淨值)
    2. 視覺上起點重合 (將兩個Y軸的 Range 鎖定在相同的相對比例)
    """
    if not selected_keys:
        st.info("請從上方選單勾選 1~2 項資產進行比較。")
        return

    # 1. 計算篩選的起始日期
    delta = Config.TIME_RANGES.get(time_range_key)
    if not delta:
        delta = relativedelta(years=1)
    
    start_date_limit = pd.to_datetime("today") - delta

    # 2. 準備數據並計算 "全域相對波動範圍"
    plot_data = []
    global_min_ratio = 1.0
    global_max_ratio = 1.0
    
    for key in selected_keys:
        if key in all_data:
            df = all_data[key].copy()
            df = df.sort_values('日期')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[df['日期'] >= start_date_limit]
            
            if not df.empty:
                # 取得起始價格 (作為基數 1)
                start_price = df['NAV'].iloc[0]
                
                # 計算該資產在這段期間的相對波動 (Ratio)
                # 目的只是為了找出大家共同的 "最大/最小漲跌幅範圍"
                min_price = df['NAV'].min()
                max_price = df['NAV'].max()
                
                min_ratio = min_price / start_price
                max_ratio = max_price / start_price
                
                # 更新全域範圍
                if min_ratio < global_min_ratio: global_min_ratio = min_ratio
                if max_ratio > global_max_ratio: global_max_ratio = max_ratio

                # 準備繪圖資訊
                raw_name = df['基金名稱'].iloc[0]
                asset_name = str(raw_name) if raw_name else key
                
                plot_data.append({
                    "data": df,
                    "name": asset_name,
                    "start_price": start_price
                })

    if not plot_data:
        st.warning(f"選取的資產在【{time_range_key}】內無足夠數據可供繪圖。")
        return

    # 3. 為了讓線條不要頂天立地，上下各留 5% 緩衝空間
    range_padding = (global_max_ratio - global_min_ratio) * 0.05
    # 如果波動極小 (例如定存)，給一個預設緩衝
    if range_padding == 0: range_padding = 0.01
    
    final_min_ratio = global_min_ratio - range_padding
    final_max_ratio = global_max_ratio + range_padding

    # 4. 建立 Plotly 雙軸圖表
    fig = go.Figure()

    # --- 第一個資產 (左軸) ---
    d1 = plot_data[0]
    fig.add_trace(go.Scatter(
        x=d1["data"]['日期'], 
        y=d1["data"]['NAV'], 
        name=d1["name"],
        yaxis='y',
        hovertemplate='%{y:,.2f}' # 顯示千分位真實價格
    ))
    
    # 計算左軸的真實價格範圍
    y1_range = [d1["start_price"] * final_min_ratio, d1["start_price"] * final_max_ratio]

    # --- 第二個資產 (右軸，如果有的話) ---
    y2_range = None
    if len(plot_data) > 1:
        d2 = plot_data[1]
        fig.add_trace(go.Scatter(
            x=d2["data"]['日期'], 
            y=d2["data"]['NAV'], 
            name=d2["name"],
            yaxis='y2',
            hovertemplate='%{y:,.2f}'
        ))
        # 計算右軸的真實價格範圍
        y2_range = [d2["start_price"] * final_min_ratio, d2["start_price"] * final_max_ratio]

    # 5. 設定 Layout (關鍵：強制鎖定 Y 軸 Range)
    
    # 共用設定
    fig.update_layout(
        title=f'資產價格走勢比較 ({time_range_key}) - 起點歸一化視角',
        xaxis=dict(title='日期'),
        hovermode='x unified',
        legend=dict(orientation="h", y=1.1)
    )

    # 左軸設定
    fig.update_layout(
        yaxis=dict(
            title=d1["name"],
            title_font=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            range=y1_range, # <--- 關鍵：強制設定範圍
            tickformat=',.2f' # 格式化軸標籤
        )
    )

    # 右軸設定
    if len(plot_data) > 1:
        fig.update_layout(
            yaxis2=dict(
                title=d2["name"],
                title_font=dict(color='#ff7f0e'),
                tickfont=dict(color='#ff7f0e'),
                overlaying='y',
                side='right',
                range=y2_range, # <--- 關鍵：強制設定範圍
                tickformat=',.2f'
            )
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

    if st.button("🚀 開始分析", type="primary"):
        st.session_state['has_run'] = True

    if st.session_state.get('has_run'):
        # 1. 載入資料 (使用快取)
        all_data = load_data_with_cache(target_markets, fund_ids)

        if not all_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return

        # 2. 建立分頁
        tab1, tab2 = st.tabs(["📋 報表總覽", "📈 資產趨勢比較"])

        # === 分頁 1：報表 ===
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

        # === 分頁 2：雙軸圖表 ===
        with tab2:
            st.subheader("資產價格走勢分析")
            st.caption("請選擇 **最多 2 項** 資產進行對照。")
            
            # --- 步驟 1: 選擇時間區間 ---
            time_range = st.radio(
                "選擇時間區間:",
                options=list(Config.TIME_RANGES.keys()),
                index=3, # 預設選 "近1年"
                horizontal=True
            )

            # --- 步驟 2: 建立名稱對照表 ---
            options_map = {}
            for key, df in all_data.items():
                if not df.empty:
                    fund_name = df['基金名稱'].iloc[0]
                    if fund_name != key:
                        display_label = f"{fund_name} ({key})"
                    else:
                        display_label = key
                    options_map[display_label] = key

            # --- 步驟 3: 選擇資產 ---
            selected_labels = st.multiselect(
                "選擇要繪製的資產 (Max 2):",
                options=list(options_map.keys()),
                max_selections=2
            )
            
            selected_keys = [options_map[label] for label in selected_labels]
            
            # --- 步驟 4: 繪圖 ---
            plot_dual_axis_trends(all_data, selected_keys, time_range)

if __name__ == "__main__":
    main()