import streamlit as st
import requests
import pandas as pd
import numpy as np
import urllib3
import logging
import io
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px  # 新增：用於繪製熱力圖
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Tuple
from dateutil.relativedelta import relativedelta

# === 0. 全域環境設定 ===
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="全球市場與基金分析戰情室", layout="wide")

# ==========================================
# 1. 配置與常數 (Configuration)
# ==========================================
class Config:
    """全域配置類別：集中管理所有常數、API 設定與預設清單"""
    
    # --- 國泰基金 API 設定 ---
    API_URL = "https://www.cathaybk.com.tw/cathaybk/service/newwealth/fund/chartservice.asmx/GetFundNavChart"
    BASE_URL = "https://www.cathaybk.com.tw/cathaybk/personal/investment/fund/details/?fundid={}"
    USER_AGENT = "Mozilla/5.0"
    TIMEOUT = 10
    DEFAULT_DATE_FROM = "1900/01/01"
    
    # --- 預設關注的基金代號 ---
    DEFAULT_FUND_IDS_LIST = [
        "00580030", "00400013", "00060004", "00100045", "00010144", "00120001",
        "00040097", "10340003", "10350005", "00060003", "00400029", "00100046",
        "00010145", "00740020", "00120005", "00120018", "00120193", "00120002",
        "00120134", "00100118", "00400156", "00400104", "00040052", "10020058",
        "10110022", "0074B065", "00100058", "00580062", "10310016", "00100063",
        "00560011", "00400072"
    ]

    # --- 全球市場指標 (Yahoo Finance Tickers) ---
    MARKET_TICKERS = {
        # 美股 ETF
        "Vanguard S&P 500 (VOO)": "VOO",
        "Invesco QQQ (QQQ)": "QQQ",
        "Vanguard Total Intl Stock (VXUS)": "VXUS",
        "Vanguard Total World Bond (BNDW)": "BNDW",
        "VanEck Uranium+Nuclear (NLR)": "NLR",
        # 關鍵指數與商品
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

    @staticmethod
    def get_start_date(time_range_key: str) -> datetime:
        """根據時間區間 Key 計算起始日期"""
        delta = Config.TIME_RANGES.get(time_range_key, relativedelta(years=1))
        return datetime.now() - delta

# ==========================================
# 2. 資料獲取層 (Data Scraping Layer)
# ==========================================
class FundScraper:
    """負責抓取國泰基金歷史淨值"""
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

# ==========================================
# 3. 資料處理與分析層 (Data Analysis Layer)
# ==========================================
class FundAnalyzer:
    """負責計算各項指標、報酬率與風險數據"""
    
    @staticmethod
    def analyze_single(df: pd.DataFrame) -> Dict[str, Any]:
        """計算單一基金的基礎統計數據 (用於報表總覽)"""
        df = df.sort_values('日期')
        fund_name = df['基金名稱'].iloc[0]
        url = df['URL'].iloc[0]
        latest_nav = df['NAV'].iloc[-1]
        
        # 歷史極值
        hist_max, hist_min = df['NAV'].max(), df['NAV'].min()
        hist_max_date = df.loc[df['NAV'].idxmax(), '日期']
        hist_min_date = df.loc[df['NAV'].idxmin(), '日期']

        # 近一年數據
        one_year_ago = df['日期'].max() - timedelta(days=365)
        df_1y = df[df['日期'] >= one_year_ago]
        
        if df_1y.empty:
            max_1y, min_1y, max_1y_date, min_1y_date = None, None, None, None
            diff_max_1y_pct, diff_min_1y_pct = None, None
        else:
            max_1y, min_1y = df_1y['NAV'].max(), df_1y['NAV'].min()
            max_1y_date = df_1y.loc[df_1y['NAV'].idxmax(), '日期']
            min_1y_date = df_1y.loc[df_1y['NAV'].idxmin(), '日期']
            
            # 計算與極值的差距百分比
            diff_max_1y_pct = ((latest_nav - max_1y) / max_1y) * 100
            diff_min_1y_pct = ((latest_nav - min_1y) / min_1y) * 100

        return {
            "基金名稱": fund_name,
            "最新價格": latest_nav,
            "最新價格日期": df['日期'].iloc[-1],
            "近一年最高價格": max_1y, "最高價與最新價%": diff_max_1y_pct, "近一年最高價格日期": max_1y_date,
            "近一年最低價格": min_1y, "最低價與最新價%": diff_min_1y_pct, "近一年最低價格日期": min_1y_date,
            "歷史最高價格": hist_max, "歷史最高價格日期": hist_max_date,
            "歷史最低價格": hist_min, "歷史最低價格日期": hist_min_date,
            "基金連結": url
        }

    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame, risk_free_rate: float) -> Dict[str, float]:
        """
        計算進階風險指標：年化標準差、夏普值、最大回撤
        risk_free_rate: 無風險利率 (例如 4.0 代表 4%)
        """
        df = df.sort_values('日期')
        df['pct_change'] = df['NAV'].pct_change()
        returns = df['pct_change'].dropna()
        
        if returns.empty:
            return {"volatility": 0.0, "sharpe": 0.0, "annual_return": 0.0, "mdd": 0.0}

        # 1. 年化標準差 (Volatility)
        volatility = FundAnalyzer._calculate_annualized_volatility(returns)
        
        # 2. 年化報酬率 (CAGR)
        annual_return = FundAnalyzer._calculate_cagr(df)

        # 3. 夏普值 (Sharpe)
        rf_decimal = risk_free_rate / 100.0
        sharpe_ratio = (annual_return - rf_decimal) / volatility if volatility > 0 else 0

        # 4. 最大回撤 (Max Drawdown)
        max_drawdown = FundAnalyzer._calculate_max_drawdown(df['NAV'])

        return {
            "volatility": volatility * 100,
            "sharpe": sharpe_ratio,
            "annual_return": annual_return * 100,
            "mdd": max_drawdown * 100
        }

    # --- 內部輔助計算方法 (Refactored) ---
    @staticmethod
    def _calculate_annualized_volatility(returns: pd.Series) -> float:
        """計算年化標準差"""
        return returns.std() * np.sqrt(252)

    @staticmethod
    def _calculate_cagr(df: pd.DataFrame) -> float:
        """計算年化報酬率 (CAGR)"""
        total_return = (df['NAV'].iloc[-1] / df['NAV'].iloc[0]) - 1
        days = (df['日期'].iloc[-1] - df['日期'].iloc[0]).days
        if days <= 0: return 0.0
        return (1 + total_return) ** (365 / days) - 1

    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """計算最大回撤"""
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.min()

    @staticmethod
    def calculate_correlation_matrix(data_map: Dict[str, pd.DataFrame], selected_keys: List[str], start_date: datetime) -> pd.DataFrame:
        """計算多資產的相關係數矩陣"""
        # 1. 準備合併用的 DataFrame
        merged_df = pd.DataFrame()
        
        for key in selected_keys:
            if key in data_map:
                df = data_map[key].copy()
                df['日期'] = pd.to_datetime(df['日期'])
                # 篩選日期
                df = df[df['日期'] >= start_date]
                if not df.empty:
                    df = df.set_index('日期')
                    # 取出名稱作為欄位名
                    col_name = df['基金名稱'].iloc[0]
                    merged_df[col_name] = df['NAV']
        
        # 2. 計算 pct_change 並計算相關係數
        if merged_df.empty:
            return pd.DataFrame()
            
        # 使用日報酬率來計算相關性才準確，不能用價格直接算
        returns_df = merged_df.pct_change().dropna()
        return returns_df.corr()

    @staticmethod
    def analyze_all(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        summary_list = []
        for df in data_map.values():
            summary_list.append(FundAnalyzer.analyze_single(df))
        return pd.DataFrame(summary_list)


class BacktestEngine:
    """回測計算引擎"""
    
    @staticmethod
    def calculate_lump_sum(df: pd.DataFrame, invest_date: datetime, amount: float):
        """計算單筆投入回報"""
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        start_row = df[df['日期'] >= invest_date].head(1)
        if start_row.empty: return None, "選定日期無有效數據"
        
        start_price = start_row['NAV'].values[0]
        real_start_date = start_row['日期'].dt.date.values[0]
        
        end_price = df['NAV'].iloc[-1]
        end_date = df['日期'].iloc[-1].date()
        
        final_value = (amount / start_price) * end_price
        roi = ((final_value - amount) / amount) * 100
        
        return {
            "type": "單筆投入", "real_start_date": real_start_date, "end_date": end_date,
            "start_price": start_price, "end_price": end_price,
            "invested_capital": amount, "final_value": final_value, "roi": roi
        }, None

    @staticmethod
    def calculate_dca(df: pd.DataFrame, start_date: datetime, monthly_day: int, amount: float):
        """計算定期定額回報"""
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        start_date = pd.to_datetime(start_date)
        records, total_units, total_invested = [], 0, 0
        data_end_date = df['日期'].iloc[-1]
        current_month_first = start_date.replace(day=1)
        
        while current_month_first <= data_end_date:
            try:
                target_date = current_month_first.replace(day=monthly_day)
            except ValueError:
                target_date = (current_month_first + relativedelta(months=1)) - timedelta(days=1)
            
            if target_date >= start_date and target_date <= data_end_date:
                trade_row = df[df['日期'] >= target_date].head(1)
                if not trade_row.empty:
                    price, trade_date = trade_row['NAV'].values[0], trade_row['日期'].dt.date.values[0]
                    if not records or records[-1]['date'] != trade_date:
                        units = amount / price
                        total_units += units
                        total_invested += amount
                        records.append({'date': trade_date, 'price': price, 'units': units, 'cumulative_invested': total_invested})
            
            current_month_first += relativedelta(months=1)
            
        if total_invested == 0: return None, "在此期間內無有效紀錄"
        
        final_value = total_units * df['NAV'].iloc[-1]
        roi = ((final_value - total_invested) / total_invested) * 100
        return {
            "type": "定期定額", "start_date": records[0]['date'], "end_date": data_end_date.date(),
            "total_invested": total_invested, "final_value": final_value, "roi": roi,
            "deduct_count": len(records), "records": pd.DataFrame(records)
        }, None

    @staticmethod
    def generate_quick_summary(df: pd.DataFrame):
        """產生快速回測總表"""
        periods = {
            "近 1 月": relativedelta(months=1), "近 3 月": relativedelta(months=3),
            "近 6 月": relativedelta(months=6), "近 1 年": relativedelta(years=1),
            "近 3 年": relativedelta(years=3), "近 5 年": relativedelta(years=5),
            "近 10 年": relativedelta(years=10)
        }
        results, today = [], datetime.now()
        for name, delta in periods.items():
            start_date = today - delta
            rl, el = BacktestEngine.calculate_lump_sum(df, start_date, 100000)
            rd, ed = BacktestEngine.calculate_dca(df, start_date, 5, 5000)
            results.append({
                "週期": name,
                "單筆報酬率 (%)": f"{rl['roi']:.2f}" if not el else "-",
                "定期定額報酬率 (%)": f"{rd['roi']:.2f}" if not ed else "-"
            })
        return pd.DataFrame(results)

# ==========================================
# 4. 輸出與視覺化層 (Output & Visualization Layer)
# ==========================================
class ExcelReport:
    """負責生成 Excel 報表"""
    @staticmethod
    def create_excel_bytes(summary_df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_disp = summary_df.drop(columns=['基金連結'])
            df_disp.to_excel(writer, index=False, header=False, sheet_name='Summary', startrow=1)
            workbook, worksheet = writer.book, writer.sheets['Summary']
            ExcelReport._apply_styles(workbook, worksheet, df_disp, summary_df)
            ExcelReport._set_columns_width(df_disp, worksheet)
            worksheet.freeze_panes(1, 0)
        return output.getvalue()

    @staticmethod
    def _apply_styles(wb, ws, df, orig):
        fmt = {
            'header': wb.add_format({'bold': True, 'font_name': 'Microsoft JhengHei', 'bg_color': '#DCE6F1', 'align': 'center', 'border': 1}),
            'text': wb.add_format({'font_name': 'Microsoft JhengHei', 'border': 1}),
            'num': wb.add_format({'font_name': 'Microsoft JhengHei', 'border': 1, 'num_format': '#,##0.00'}),
            'link': wb.add_format({'font_color': 'blue', 'underline': 1, 'font_name': 'Microsoft JhengHei', 'border': 1}),
            'date': wb.add_format({'num_format': 'yyyy-mm-dd', 'font_name': 'Microsoft JhengHei', 'border': 1})
        }
        for c, v in enumerate(df.columns): ws.write(0, c, v, fmt['header'])
        date_cols = [i for i, c in enumerate(df.columns) if '日期' in str(c)]
        
        for i in range(len(df)):
            ws.write_url(i+1, 0, orig.iloc[i]['基金連結'], fmt['link'], string=df.iat[i, 0])
            for j in range(1, len(df.columns)):
                v = df.iat[i, j]
                if j in date_cols and pd.notna(v): ws.write_datetime(i+1, j, pd.to_datetime(v), fmt['date'])
                elif isinstance(v, (int, float)): ws.write_number(i+1, j, v, fmt['num'])
                else: ws.write(i+1, j, str(v), fmt['text'])

    @staticmethod
    def _set_columns_width(df, ws):
        for i, col in enumerate(df.columns):
            ml = max(df[col].astype(str).map(lambda x: len(x.encode('utf-8'))).max(), len(str(col).encode('utf-8')))
            ws.set_column(i, i, min(max(ml * 0.8, 10), 50))

class ChartManager:
    """負責繪製 Plotly 圖表"""
    
    @staticmethod
    def _filter_data(all_data, keys, tr_key):
        """共用數據過濾邏輯"""
        start_date = Config.get_start_date(tr_key)
        filtered = {}
        for k in keys:
            if k in all_data:
                df = all_data[k].copy().sort_values('日期')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= start_date]
                if not df.empty:
                    filtered[k] = df
        return filtered

    @staticmethod
    def plot_dual_axis_trends(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
        """繪製雙Y軸價格走勢比較圖"""
        if not selected_keys: return
        
        filtered_data = ChartManager._filter_data(all_data, selected_keys, time_range_key)
        if not filtered_data:
            st.warning("選定區間內無數據")
            return

        plot_dfs = []
        global_min, global_max = 1.0, 1.0
        
        for k, df in filtered_data.items():
            sp = df['NAV'].iloc[0]
            ratios = df['NAV'] / sp
            global_min = min(global_min, ratios.min())
            global_max = max(global_max, ratios.max())
            plot_dfs.append({"data": df, "name": str(df['基金名稱'].iloc[0]), "sp": sp})

        pad = (global_max - global_min) * 0.05
        y_range_min, y_range_max = global_min - pad, global_max + pad

        fig = go.Figure()
        # 第一條線
        d1 = plot_dfs[0]
        fig.add_trace(go.Scatter(x=d1["data"]['日期'], y=d1["data"]['NAV'], name=d1["name"], yaxis='y', hovertemplate='%{y:,.2f}'))
        y1_range = [d1["sp"] * y_range_min, d1["sp"] * y_range_max]
        
        layout_update = {
            'title': f'資產價格走勢比較 ({time_range_key})',
            'xaxis': dict(title='日期'), 'hovermode': 'x unified', 'legend': dict(orientation="h", y=1.1),
            'yaxis': dict(title=d1["name"], range=y1_range, tickformat=',.2f', title_font=dict(color='#1f77b4'), tickfont=dict(color='#1f77b4'))
        }

        # 第二條線 (如果有)
        if len(plot_dfs) > 1:
            d2 = plot_dfs[1]
            fig.add_trace(go.Scatter(x=d2["data"]['日期'], y=d2["data"]['NAV'], name=d2["name"], yaxis='y2', hovertemplate='%{y:,.2f}'))
            y2_range = [d2["sp"] * y_range_min, d2["sp"] * y_range_max]
            layout_update['yaxis2'] = dict(
                title=d2["name"], overlaying='y', side='right', range=y2_range, tickformat=',.2f',
                title_font=dict(color='#ff7f0e'), tickfont=dict(color='#ff7f0e')
            )

        fig.update_layout(**layout_update)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_investment_growth(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
        """繪製 100 萬投資增值圖"""
        if not selected_keys: return
        
        filtered_data = ChartManager._filter_data(all_data, selected_keys, time_range_key)
        if not filtered_data: return

        fig = go.Figure()
        initial_capital = 1_000_000
        
        for k, df in filtered_data.items():
            sp = df['NAV'].iloc[0]
            growth = (df['NAV'] / sp) * initial_capital
            fig.add_trace(go.Scatter(
                x=df['日期'], y=growth, name=str(df['基金名稱'].iloc[0]),
                hovertemplate='%{y:,.0f}'
            ))

        fig.update_layout(
            title=f'100 萬資產增值模擬 ({time_range_key})',
            xaxis=dict(title='日期'),
            yaxis=dict(title='資產總值 (元)', tickformat=',.0f'),
            hovermode='x unified', legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_correlation_heatmap(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
        """【新增】繪製相關性熱力圖"""
        if len(selected_keys) < 2:
            st.info("請至少選擇 2 個標的以顯示相關性矩陣。")
            return

        start_date = Config.get_start_date(time_range_key)
        corr_matrix = FundAnalyzer.calculate_correlation_matrix(all_data, selected_keys, start_date)
        
        if corr_matrix.empty:
            st.warning("選定區間內無共同交易數據，無法計算相關性。")
            return

        fig = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale="RdBu_r", # 紅藍配色 (紅=正相關, 藍=負相關)
            zmin=-1, zmax=1,
            title=f"資產相關性矩陣 ({time_range_key})"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 應用程式邏輯 (Application Logic)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="正在載入數據...")
def load_data_with_cache(target_markets: Dict[str, str], fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """快取資料載入函式"""
    all_data = {}
    if target_markets: all_data.update(MarketScraper().fetch_all(target_markets))
    if fund_ids: all_data.update(FundScraper().fetch_all(fund_ids))
    return all_data

def render_sidebar() -> Tuple[Dict[str, str], List[str]]:
    """渲染側邊欄並回傳使用者的選擇"""
    with st.sidebar:
        st.header("⚙️ 設定面板")
        with st.expander("🌍 全球市場指標", expanded=True):
            selected_market_names = st.multiselect(
                "選擇指標", options=list(Config.MARKET_TICKERS.keys()), default=list(Config.MARKET_TICKERS.keys())
            )
            target_markets = {n: Config.MARKET_TICKERS[n] for n in selected_market_names}
        with st.expander("🏦 國泰基金清單", expanded=True):
            fund_input_str = st.text_area("基金代號", value=",\n".join(Config.DEFAULT_FUND_IDS_LIST), height=300)
            fund_ids = [x.strip() for x in fund_input_str.replace("\n", ",").split(",") if x.strip()]
    return target_markets, fund_ids

def render_tab_overview(all_data: Dict[str, pd.DataFrame]):
    """渲染分頁 1：報表總覽"""
    summary_df = FundAnalyzer.analyze_all(all_data)
    st.success(f"✅ 完成！共分析 {len(summary_df)} 筆標的")
    st.dataframe(summary_df)
    excel_data = ExcelReport.create_excel_bytes(summary_df)
    st.download_button("📥 下載 Excel 報表", excel_data, f"Global_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")

def render_tab_chart(all_data: Dict[str, pd.DataFrame], options_map: Dict[str, str]):
    """渲染分頁 2：趨勢與風險分析"""
    st.subheader("資產價格與風險分析")
    
    # 控制項
    time_range = st.radio("區間:", options=list(Config.TIME_RANGES.keys()), index=3, horizontal=True)
    selected_labels = st.multiselect("選擇資產 (建議 2-5 個):", options=list(options_map.keys()), max_selections=None) # 解除數量限制以便觀看相關性
    selected_keys = [options_map[l] for l in selected_labels]
    
    # 獲取無風險利率 (^TNX)
    rf_rate = 4.0
    tnx_key = "美國 10 年期公債殖利率"
    if tnx_key in all_data and not all_data[tnx_key].empty:
        rf_rate = all_data[tnx_key]['NAV'].iloc[-1]
    
    if selected_keys:
        # --- 1. 風險指標 ---
        st.markdown("##### 📊 風險與報酬指標 (區間年化)")
        
        # 限制指標顯示數量，避免版面過擠
        display_limit = 4
        display_keys = selected_keys[:display_limit]
        cols = st.columns(max(len(display_keys), 1))
        
        filtered_data = ChartManager._filter_data(all_data, display_keys, time_range)
        
        for idx, key in enumerate(display_keys):
            if key in filtered_data:
                df_period = filtered_data[key]
                metrics = FundAnalyzer.calculate_performance_metrics(df_period, rf_rate)
                name = df_period['基金名稱'].iloc[0]
                
                with cols[idx]:
                    st.markdown(f"**{name}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
                    c2.metric("波動度", f"{metrics['volatility']:.1f}%")
                    c3.metric("Max Drawdown", f"{metrics['mdd']:.1f}%", delta_color="inverse")
        
        if len(selected_keys) > display_limit:
            st.caption(f"* 僅顯示前 {display_limit} 筆資產的詳細指標，更多資產請至下方圖表查看。")

        st.caption(f"* 無風險利率採用【美國 10 年期公債殖利率】：{rf_rate:.2f}%")
        st.divider()
        
        # --- 2. 相關性矩陣 (New) ---
        with st.expander("🔗 資產相關性矩陣 (Correlation Heatmap)", expanded=True):
            ChartManager.plot_correlation_heatmap(all_data, selected_keys, time_range)
        
        st.divider()

        # --- 3. 圖表 ---
        # 為了雙軸圖表的可讀性，我們只取前兩個
        plot_keys_dual = selected_keys[:2]
        ChartManager.plot_dual_axis_trends(all_data, plot_keys_dual, time_range)
        if len(selected_keys) > 2:
            st.caption("* 雙軸走勢圖僅顯示前 2 個選定項目，以確保可讀性。")
            
        st.divider()
        ChartManager.plot_investment_growth(all_data, selected_keys, time_range)

def render_tab_backtest(all_data: Dict[str, pd.DataFrame], options_map: Dict[str, str]):
    """渲染分頁 3：投資回測"""
    st.subheader("💰 策略回測")
    # 初始化 Session State
    for key in ['calc_results_lump', 'calc_results_dca']:
        if key not in st.session_state: st.session_state[key] = None
            
    target_label = st.selectbox("標的:", list(options_map.keys()))
    if st.session_state.get('last_target') != target_label:
        st.session_state.update({'last_target': target_label, 'calc_results_lump': None, 'calc_results_dca': None})
    
    target_df = all_data.get(options_map[target_label])
    if target_df is not None:
        st.dataframe(BacktestEngine.generate_quick_summary(target_df), hide_index=True)
        col_lump, col_dca = st.columns(2)
        
        with col_lump:
            st.markdown("### 1️⃣ 單筆投入")
            ld = st.date_input("買入日", value=datetime.now()-relativedelta(years=1), max_value=datetime.now())
            la = st.number_input("投入金額", value=1000000, step=100000)
            if st.button("計算單筆"): 
                st.session_state['calc_results_lump'], _ = BacktestEngine.calculate_lump_sum(target_df, pd.to_datetime(ld), la)
            
            if st.session_state['calc_results_lump']:
                res = st.session_state['calc_results_lump']
                c = "green" if res['roi'] >= 0 else "red"
                st.markdown(f"市值: **{res['final_value']:,.0f}** (ROI: <span style='color:{c}'>{res['roi']:.2f}%</span>)", unsafe_allow_html=True)

        with col_dca:
            st.markdown("### 2️⃣ 定期定額")
            ds = st.date_input("開始日", value=datetime.now()-relativedelta(years=1), max_value=datetime.now())
            dd, da = st.number_input("扣款日", 1, 31, 5), st.number_input("每期金額", value=10000, step=1000)
            if st.button("計算 DCA"): 
                st.session_state['calc_results_dca'], _ = BacktestEngine.calculate_dca(target_df, pd.to_datetime(ds), dd, da)
            
            if st.session_state['calc_results_dca']:
                res = st.session_state['calc_results_dca']
                c = "green" if res['roi'] >= 0 else "red"
                st.markdown(f"市值: **{res['final_value']:,.0f}** (ROI: <span style='color:{c}'>{res['roi']:.2f}%</span>)", unsafe_allow_html=True)
                with st.expander("詳細紀錄"): st.dataframe(res['records'], hide_index=True)

def main():
    st.title("📊 全球市場與基金淨值戰情室")
    target_markets, fund_ids = render_sidebar()

    if st.button("🚀 開始/更新 分析", type="primary"):
        st.session_state['has_run'] = True

    if st.session_state.get('has_run'):
        all_data = load_data_with_cache(target_markets, fund_ids)
        if not all_data: return st.error("❌ 未取得資料")

        options_map = {f"{df['基金名稱'].iloc[0]} ({k})" if df['基金名稱'].iloc[0] != k else k: k for k, df in all_data.items() if not df.empty}
        
        t1, t2, t3 = st.tabs(["📋 報表總覽", "📈 資產趨勢比較", "💰 投資策略回測"])
        with t1: render_tab_overview(all_data)
        with t2: render_tab_chart(all_data, options_map)
        with t3: render_tab_backtest(all_data, options_map)

if __name__ == "__main__":
    main()