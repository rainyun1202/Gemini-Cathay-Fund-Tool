# app.py
import streamlit as st
import pandas as pd
import urllib3
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple

# === 匯入自定義模組 ===
from modules.config import Config
from modules.scraper import FundScraper, MarketScraper
from modules.analyzer import FundAnalyzer, BacktestEngine
from modules.visualizer import ExcelReport, ChartManager

# === 全域設定 ===
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="全球市場與基金分析", layout="wide")

# ==========================================
# 資料獲取與 UI 邏輯
# ==========================================

@st.cache_data(ttl=3600, show_spinner="正在自網路下載最新數據...")
def load_data_with_cache(target_markets: Dict[str, str], fund_ids: List[str]) -> Tuple[Dict, Dict]:
    """
    快取資料載入函式
    Return: (nav_data_map, dividend_data_map)
    """
    nav_data = {}
    div_data = {}
    
    # 1. 下載市場數據 (只有淨值)
    if target_markets:
        market_scraper = MarketScraper()
        market_data = market_scraper.fetch_all(target_markets)
        nav_data.update(market_data)
        
    # 2. 下載基金數據 (淨值 + 配息)
    if fund_ids:
        fund_scraper = FundScraper()
        
        # A. 抓淨值
        fund_navs = fund_scraper.fetch_all_nav(fund_ids)
        nav_data.update(fund_navs)
        
        # B. 抓配息 (新增)
        fund_divs = fund_scraper.fetch_all_dividend(fund_ids)
        div_data.update(fund_divs)
        
    return nav_data, div_data

def render_sidebar() -> Tuple[Dict[str, str], List[str]]:
    with st.sidebar:
        st.header("⚙️ 設定面板")
        with st.expander("🌍 全球市場指標", expanded=True):
            selected_markets = st.multiselect("選擇關注市場指標", options=list(Config.MARKET_TICKERS.keys()), default=list(Config.MARKET_TICKERS.keys()))
            target_markets = {name: Config.MARKET_TICKERS[name] for name in selected_markets}
        with st.expander("🏦 國泰基金清單", expanded=True):
            default_ids = ",\n".join(Config.DEFAULT_FUND_IDS_LIST)
            fund_input = st.text_area("基金代號 (每行一個)", value=default_ids, height=300, help="請輸入基金代號")
            fund_ids = [x.strip() for x in fund_input.replace("\n", ",").split(",") if x.strip()]
    return target_markets, fund_ids

def render_tab_overview(nav_data: Dict, div_data: Dict, full_sort_list: List[Dict]):
    """渲染分頁 1：報表總覽"""
    # 傳入配息 Map
    summary_df = FundAnalyzer.analyze_all(nav_data, div_data, sort_list=full_sort_list)
    
    st.success(f"✅ 完成！共分析 {len(summary_df)} 筆標的")
    st.dataframe(summary_df)

    # 準備完整的配息明細表，以便寫入 Excel Sheet 2
    all_div_list = []
    for fid, df in div_data.items():
        if fid in nav_data:
            # 補上基金名稱方便閱讀
            name = nav_data[fid]['基金名稱'].iloc[0]
            df = df.copy()
            df.insert(0, '基金名稱', name)
            all_div_list.append(df)
            
    if all_div_list:
        all_div_df = pd.concat(all_div_list, ignore_index=True)
    else:
        all_div_df = pd.DataFrame()

    excel_data = ExcelReport.create_excel_bytes(summary_df, all_div_df)
    file_name = f"Global_Market_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    st.download_button("📥 下載完整 Excel 報表", excel_data, file_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def render_tab_chart(all_data: Dict[str, pd.DataFrame], options_map: Dict[str, str]):
    st.subheader("資產價格與風險分析")
    time_range = st.radio("選擇時間區間:", options=list(Config.TIME_RANGES.keys()), index=3, horizontal=True)
    selected_labels = st.multiselect("選擇要繪製的資產 (Max 2):", options=list(options_map.keys()), max_selections=2)
    selected_keys = [options_map[label] for label in selected_labels]
    
    rf_rate_val = 4.0
    tnx_data_key = "美國 10 年期公債殖利率" 
    if tnx_data_key in all_data:
        tnx_df = all_data[tnx_data_key]
        if not tnx_df.empty:
            rf_rate_val = tnx_df['NAV'].iloc[-1]
    
    if selected_keys:
        st.markdown("##### 📊 風險與報酬指標 (年化)")
        cols = st.columns(len(selected_keys))
        delta = Config.TIME_RANGES.get(time_range)
        start_limit = pd.to_datetime("today") - delta
        for idx, key in enumerate(selected_keys):
            if key in all_data:
                df = all_data[key].copy()
                df['日期'] = pd.to_datetime(df['日期'])
                df_period = df[df['日期'] >= start_limit]
                metrics = FundAnalyzer.calculate_performance_metrics(df_period, rf_rate_val)
                fund_name = df['基金名稱'].iloc[0]
                with cols[idx]:
                    st.metric(label=fund_name, value=f"Sharpe: {metrics['sharpe']:.2f}", delta=f"波動度: {metrics['volatility']:.2f}%", delta_color="inverse")
        st.caption(f"* 註：無風險利率採用【美國 10 年期公債殖利率】最新報價：{rf_rate_val:.2f}%")
        st.divider()

    ChartManager.plot_dual_axis_trends(all_data, selected_keys, time_range)
    st.divider()
    ChartManager.plot_investment_growth(all_data, selected_keys, time_range)

def render_tab_backtest(all_data: Dict[str, pd.DataFrame], options_map: Dict[str, str]):
    st.subheader("💰 投資策略回測計算機")
    if 'calc_results_lump' not in st.session_state: st.session_state['calc_results_lump'] = None
    if 'calc_results_dca' not in st.session_state: st.session_state['calc_results_dca'] = None
    
    current_target = st.selectbox("請選擇回測標的:", list(options_map.keys()))
    if 'last_target' not in st.session_state or st.session_state['last_target'] != current_target:
        st.session_state['last_target'] = current_target
        st.session_state['calc_results_lump'] = None
        st.session_state['calc_results_dca'] = None
        
    target_key = options_map[current_target]
    target_df = all_data.get(target_key)

    if target_df is None or target_df.empty:
        st.error("此標的無數據，無法回測")
    else:
        st.markdown("##### ⚡ 歷史報酬率速覽")
        quick_stats_df = BacktestEngine.generate_quick_summary(target_df)
        st.dataframe(quick_stats_df, hide_index=True)
        st.divider()
        col_lump, col_dca = st.columns(2)
        today = datetime.now()
        one_year_ago = today - relativedelta(years=1)
        with col_lump:
            st.markdown("### 1️⃣ 單筆投入 (Lump Sum)")
            lump_date = st.date_input("買入日期", value=one_year_ago, max_value=today)
            lump_amt = st.number_input("投入金額", value=100000, step=10000)
            if st.button("計算單筆報酬"):
                res, err = BacktestEngine.calculate_lump_sum(target_df, pd.to_datetime(lump_date), lump_amt)
                if err: st.error(err)
                else: st.session_state['calc_results_lump'] = res
            if st.session_state['calc_results_lump']:
                res = st.session_state['calc_results_lump']
                color = "green" if res['roi'] >= 0 else "red"
                st.markdown(f"""<div style='background-color:#f0f2f6; padding:15px; border-radius:10px'><h4 style='margin-top:0'>📊 單筆回測結果</h4><ul><li><b>實際買入日</b>: {res['real_start_date']} (淨值: {res['start_price']:.2f})</li><li><b>結算日</b>: {res['end_date']} (淨值: {res['end_price']:.2f})</li><li><b>目前總市值</b>: <b>{res['final_value']:,.0f}</b> 元</li><li><b>投資報酬率</b>: <span style='color:{color};font-size:1.4em'><b>{res['roi']:.2f}%</b></span></li></ul></div>""", unsafe_allow_html=True)
        with col_dca:
            st.markdown("### 2️⃣ 定期定額 (DCA)")
            dca_start = st.date_input("開始扣款日期", value=one_year_ago, max_value=today)
            dca_day = st.number_input("每月扣款日 (1-31)", value=5, min_value=1, max_value=31)
            dca_amt = st.number_input("每期扣款金額", value=5000, step=1000)
            if st.button("計算定期定額"):
                res, err = BacktestEngine.calculate_dca(target_df, pd.to_datetime(dca_start), dca_day, dca_amt)
                if err: st.error(err)
                else: st.session_state['calc_results_dca'] = res
            if st.session_state['calc_results_dca']:
                res = st.session_state['calc_results_dca']
                color = "green" if res['roi'] >= 0 else "red"
                st.markdown(f"""<div style='background-color:#f0f2f6; padding:15px; border-radius:10px'><h4 style='margin-top:0'>📊 定期定額結果</h4><ul><li><b>回測期間</b>: {res['start_date']} ~ {res['end_date']}</li><li><b>總扣款次數</b>: {res['deduct_count']} 次</li><li><b>總投入本金</b>: {res['total_invested']:,} 元</li><li><b>目前總市值</b>: <b>{res['final_value']:,.0f}</b> 元</li><li><b>投資報酬率</b>: <span style='color:{color};font-size:1.4em'><b>{res['roi']:.2f}%</b></span></li></ul></div>""", unsafe_allow_html=True)
                with st.expander("查看詳細扣款紀錄"): st.dataframe(res['records'], hide_index=True)

def main():
    st.title("📊 全球市場與基金淨值戰情室")
    st.markdown("整合 **國泰基金** 與 **全球關鍵市場指標** 的自動化分析工具。")
    target_markets, fund_ids = render_sidebar()
    if st.button("🚀 開始/更新 分析", type="primary"):
        st.session_state['has_run'] = True
    if st.session_state.get('has_run'):
        # 載入 Nav 和 Div 兩種資料
        nav_data, div_data = load_data_with_cache(target_markets, fund_ids)
        if not nav_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return
        
        # 覆寫名稱
        for item in Config.FUND_WATCH_LIST:
            fid = item['id']
            custom_name = item['name']
            if fid in nav_data:
                nav_data[fid]['基金名稱'] = custom_name

        # 建立選項 Map (市場 -> 基金)
        options_map = {}
        processed_keys = set()
        for market_name in Config.MARKET_TICKERS.keys():
            if market_name in nav_data:
                options_map[market_name] = market_name
                processed_keys.add(market_name)
        for item in Config.FUND_WATCH_LIST:
            fid = item['id']
            if fid in nav_data:
                fund_name = nav_data[fid]['基金名稱'].iloc[0]
                display_label = f"{fund_name} ({fid})"
                options_map[display_label] = fid
                processed_keys.add(fid)
        for key, df in nav_data.items():
            if key not in processed_keys and not df.empty:
                fund_name = df['基金名稱'].iloc[0]
                display_label = f"{fund_name} ({key})" if fund_name != key else key
                options_map[display_label] = key

        market_sort_list = [{'id': name, 'name': name} for name in Config.MARKET_TICKERS.keys()]
        full_sort_list = market_sort_list + Config.FUND_WATCH_LIST

        tab1, tab2, tab3 = st.tabs(["📋 報表總覽", "📈 資產趨勢比較", "💰 投資策略回測"])
        with tab1:
            render_tab_overview(nav_data, div_data, full_sort_list)
        with tab2:
            render_tab_chart(nav_data, options_map)
        with tab3:
            render_tab_backtest(nav_data, options_map)

if __name__ == "__main__":
    main()