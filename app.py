# app.py
import streamlit as st
import pandas as pd
import urllib3
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
def load_data_with_cache(target_markets: Dict[str, str], fund_ids: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    快取資料載入函式
    Return: (nav_data_map, dividend_data_map, holdings_data_map)
    """
    nav_data = {}
    div_data = {}
    hold_data = {}
    
    if target_markets:
        market_scraper = MarketScraper()
        market_data = market_scraper.fetch_all(target_markets)
        nav_data.update(market_data)
        
    if fund_ids:
        fund_scraper = FundScraper()
        
        # A. 抓淨值
        fund_navs = fund_scraper.fetch_all_nav(fund_ids)
        nav_data.update(fund_navs)
        
        # B. 抓配息
        fund_divs = fund_scraper.fetch_all_dividend(fund_ids)
        div_data.update(fund_divs)
        
        # C. 抓取成分股
        fund_holds = fund_scraper.fetch_all_holdings(fund_ids)
        hold_data.update(fund_holds)
        
    return nav_data, div_data, hold_data

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
    summary_df = FundAnalyzer.analyze_all(nav_data, div_data, sort_list=full_sort_list)
    st.success(f"✅ 完成！共分析 {len(summary_df)} 筆標的")
    st.dataframe(summary_df)

    all_div_list = []
    for fid, df in div_data.items():
        if fid in nav_data:
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

def render_tab_holdings(hold_data: Dict, options_map: Dict[str, str]):
    st.subheader("🔍 基金主要10大成分股")
    
    valid_options = {k: v for k, v in options_map.items() if v in hold_data}
    if not valid_options:
        st.warning("目前選擇的標的無成分股資料可供檢視。")
        return

    current_target = st.selectbox("請選擇要查看的基金:", list(valid_options.keys()), key="holdings_select")
    target_key = valid_options[current_target]
    
    fund_hold_info = hold_data.get(target_key)
    if not fund_hold_info or fund_hold_info.get("data").empty:
        st.info("此基金目前無成分股資料。")
        return
    
    date_str = fund_hold_info.get("date", "未知")
    df_hold = fund_hold_info.get("data")
    
    st.markdown(f"#### {current_target}")
    st.caption(f"📅 資料發布日期: **{date_str}**")
    
    st.dataframe(df_hold, hide_index=True)
    
    excel_data = ExcelReport.create_single_excel_bytes(df_hold, sheet_name="成分股")
    file_name = f"{target_key}_Holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"
    st.download_button("📥 下載原始成分股表 (Excel)", excel_data, file_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.markdown("#### ⚡ 即時成分股技術狀態分析")
    st.info("💡 系統將自動清洗不規則名稱，並透過 Yahoo Finance 模糊搜尋對應的 Ticker，計算近一年高點與回撤。")
    
    if st.button("🚀 載入成分股即時報價", type="primary"):
        with st.spinner("正在透過 Yahoo Finance 模糊搜尋並獲取報價... 大約需要 3~5 秒鐘。"):
            name_col = df_hold.columns[0]
            for col in df_hold.columns:
                if any(kw in str(col) for kw in ['名稱', '標的', '股票', '成分']):
                    name_col = col
                    break
            
            companies = df_hold[name_col].tolist()
            market_scraper = MarketScraper()
            results = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(market_scraper.fetch_stock_stats, comp): comp for comp in companies}
                for future in as_completed(futures):
                    comp = futures[future]
                    try:
                        stats = future.result()
                        stats['成分股名稱'] = comp
                        results.append(stats)
                    except Exception:
                        results.append({"成分股名稱": comp, "Ticker": "解析失敗", "最新價格": None, "近一年最高": None, "高點0.618": None, "距高點回撤(%)": None})
            
            df_stats = pd.DataFrame(results)
            df_stats = df_stats.set_index('成分股名稱').reindex(companies).reset_index()
            
            cols = ['成分股名稱', 'Ticker', '最新價格', '近一年最高', '高點0.618', '距高點回撤(%)']
            df_stats = df_stats[cols]
            
            numeric_cols = ['最新價格', '近一年最高', '高點0.618', '距高點回撤(%)']
            for c in numeric_cols:
                df_stats[c] = pd.to_numeric(df_stats[c], errors='coerce')
            
            st.dataframe(
                df_stats.style.format({
                    "最新價格": "{:,.2f}",
                    "近一年最高": "{:,.2f}",
                    "高點0.618": "{:,.2f}",
                    "距高點回撤(%)": "{:,.2f}%"
                }, na_rep="-").background_gradient(subset=["距高點回撤(%)"], cmap="RdYlGn", vmin=-50, vmax=0),
                hide_index=True
            )

# ==========================================
# 新增：第五分頁 (均線策略與全歷史資料)
# ==========================================
def render_tab_ma_strategy(nav_data: Dict, options_map: Dict[str, str]):
    st.subheader("📈 均線避險策略分析 (20MA)")
    st.info("💡 策略邏輯：當淨值站在 20MA 之上時持有；跌破 20MA 則空手換回現金。此策略旨在規避市場大幅修正。")

    current_target = st.selectbox("請選擇分析標的:", list(options_map.keys()), key="ma_select")
    target_key = options_map[current_target]
    df_nav = nav_data.get(target_key)

    if df_nav is not None and not df_nav.empty:
        # 1. 執行策略回測
        res = BacktestEngine.calculate_ma_strategy(df_nav)
        df_strategy = res['df']
        stats = res['stats']

        # 2. 顯示成效指標
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("策略總報酬", f"{stats['均線策略總報酬 (%)']:.1f}%", 
                      delta=f"{stats['均線策略總報酬 (%)'] - stats['買入持有總報酬 (%)']:.1f}% (vs B&H)")
        with col2:
            st.metric("買入持有報酬", f"{stats['買入持有總報酬 (%)']:.1f}%")
        with col3:
            st.metric("策略最大回撤 (MDD)", f"{stats['均線策略 MDD (%)']:.1f}%")
        with col4:
            st.metric("買入持有 MDD", f"{stats['買入持有 MDD (%)']:.1f}%")

        # 3. 顯示圖表
        ChartManager.plot_nav_with_ma(df_strategy, current_target)
        ChartManager.plot_strategy_comparison(df_strategy)

        # 4. 成立以來淨值資料保存與檢視
        with st.expander("📂 檢視成立以來完整淨值紀錄"):
            st.write(f"共計 {len(df_nav)} 筆交易日資料")
            csv = df_nav.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載完整歷史資料 (CSV)",
                data=csv,
                file_name=f"{target_key}_history.csv",
                mime='text/csv',
            )
            st.dataframe(df_nav.sort_values('日期', ascending=False), use_container_width=True)
    else:
        st.error("此標的無效，無法進行均線分析。")

def main():
    st.title("📊 全球市場與基金淨值戰情室")
    st.markdown("整合 **國泰基金** 與 **全球關鍵市場指標** 的自動化分析工具。")
    
    target_markets, fund_ids = render_sidebar()
    
    if st.button("🚀 開始/更新 分析", type="primary"):
        st.session_state['has_run'] = True
        
    if st.session_state.get('has_run'):
        nav_data, div_data, hold_data = load_data_with_cache(target_markets, fund_ids)
        if not nav_data:
            st.error("❌ 未取得任何資料，請檢查網路或代號。")
            return
        
        for item in Config.FUND_WATCH_LIST:
            fid = item['id']
            custom_name = item['name']
            if fid in nav_data:
                nav_data[fid]['基金名稱'] = custom_name

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

        # === 五個分頁 ===
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 報表總覽", "📈 資產趨勢比較", "💰 投資策略回測", "🔍 10大成分股", "🛡️ 均線策略分析"])
        
        with tab1:
            render_tab_overview(nav_data, div_data, full_sort_list)
        with tab2:
            render_tab_chart(nav_data, options_map)
        with tab3:
            render_tab_backtest(nav_data, options_map)
        with tab4:
            render_tab_holdings(hold_data, options_map)
        with tab5:
            render_tab_ma_strategy(nav_data, options_map)

if __name__ == "__main__":
    main()