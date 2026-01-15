# modules/analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional

class FundAnalyzer:
    """負責計算各項指標與報酬率"""
    
    @staticmethod
    def analyze_single(df_nav: pd.DataFrame, df_div: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        整合淨值與配息分析
        df_nav: 淨值 DataFrame
        df_div: 配息 DataFrame (可選)
        """
        # --- 1. 淨值分析 (原有邏輯) ---
        df_nav = df_nav.sort_values('日期')
        fund_name = df_nav['基金名稱'].iloc[0]
        url = df_nav['URL'].iloc[0]
        latest = df_nav.iloc[-1]
        latest_nav = latest['NAV']
        
        hist_max_idx = df_nav['NAV'].idxmax()
        hist_min_idx = df_nav['NAV'].idxmin()

        one_year_ago = df_nav['日期'].max() - timedelta(days=365)
        df_1y = df_nav[df_nav['日期'] >= one_year_ago]
        
        if df_1y.empty:
            max_1y, min_1y = None, None
            diff_max_1y_pct, diff_min_1y_pct = None, None
            max_1y_date, min_1y_date = None, None
        else:
            max_1y_idx = df_1y['NAV'].idxmax()
            min_1y_idx = df_1y['NAV'].idxmin()
            max_1y = df_1y.loc[max_1y_idx, 'NAV']
            max_1y_date = df_1y.loc[max_1y_idx, '日期']
            min_1y = df_1y.loc[min_1y_idx, 'NAV']
            min_1y_date = df_1y.loc[min_1y_idx, '日期']
            diff_max_1y_pct = ((latest_nav - max_1y) / max_1y) * 100
            diff_min_1y_pct = ((latest_nav - min_1y) / min_1y) * 100

        # --- 2. 配息分析 (新增邏輯) ---
        div_stats = {
            "近一年配息率總和(%)": None,
            "最近一次配息日": None,
            "最近一次配息金額": None,
            "最近一次當期配息率(%)": None
        }

        if df_div is not None and not df_div.empty:
            # 確保按日期降序 (最新的在上面)
            # 因為原始資料可能是字串日期，先轉 datetime 排序較保險
            df_div['sort_date'] = pd.to_datetime(df_div['配息基準日'])
            df_div = df_div.sort_values('sort_date', ascending=False)
            
            # 取得最近一筆
            latest_div = df_div.iloc[0]
            div_stats["最近一次配息日"] = latest_div["配息基準日"]
            div_stats["最近一次配息金額"] = latest_div["每單位配息金額"]
            div_stats["最近一次當期配息率(%)"] = latest_div["當期配息率(%)"]

            # 計算近一年總和 (取最近 12 筆資料)
            # 若基金為月配，12筆約等於一年；若為季配，則會加總到過去3年，
            # 但這裡需求是"最近12次"或"一年"，為了避免季配被高估，我們嚴格過濾日期
            
            cutoff_date = datetime.now() - relativedelta(years=1)
            df_last_year = df_div[df_div['sort_date'] >= cutoff_date]
            
            if not df_last_year.empty:
                 div_stats["近一年配息率總和(%)"] = df_last_year["當期配息率(%)"].sum()
            else:
                # 如果一年內沒配息 (例如剛成立，或年配還沒到)，則顯示 0
                div_stats["近一年配息率總和(%)"] = 0.0

        # --- 3. 合併結果 ---
        result = {
            "基金名稱": fund_name,
            "最新價格": latest_nav,
            "最新價格日期": latest['日期'],
            "近一年最高價格": max_1y,
            "最高價與最新價%": diff_max_1y_pct,
            "近一年最高價格日期": max_1y_date,
            "近一年最低價格": min_1y,
            "最低價與最新價%": diff_min_1y_pct,
            "近一年最低價格日期": min_1y_date,
            "歷史最高價格": df_nav.loc[hist_max_idx, 'NAV'],
            "歷史最高價格日期": df_nav.loc[hist_max_idx, '日期'],
            "歷史最低價格": df_nav.loc[hist_min_idx, 'NAV'],
            "歷史最低價格日期": df_nav.loc[hist_min_idx, '日期'],
            "基金連結": url
        }
        result.update(div_stats) # 加入配息數據
        return result

    @staticmethod
    def analyze_all(
        nav_map: Dict[str, pd.DataFrame], 
        div_map: Dict[str, pd.DataFrame], 
        sort_list: List[Dict] = None
    ) -> pd.DataFrame:
        """
        分析所有資料 (整合淨值與配息)
        """
        summary_list = []
        processed_keys = set()
        
        # 1. 優先處理 sort_list
        if sort_list:
            for item in sort_list:
                fid = item['id']
                if fid in nav_map:
                    # 取出對應的配息表 (如果有的話)
                    df_div = div_map.get(fid)
                    summary_list.append(FundAnalyzer.analyze_single(nav_map[fid], df_div))
                    processed_keys.add(fid)
        
        # 2. 處理剩餘項目
        for key, df_nav in nav_map.items():
            if key not in processed_keys:
                df_div = div_map.get(key)
                summary_list.append(FundAnalyzer.analyze_single(df_nav, df_div))
                
        return pd.DataFrame(summary_list)

    # ... (calculate_performance_metrics 等其他方法保持不變，可省略或照舊) ...
    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame, risk_free_rate: float) -> Dict[str, float]:
        df = df.sort_values('日期')
        df['pct_change'] = df['NAV'].pct_change()
        returns = df['pct_change'].dropna()
        if returns.empty: return {"volatility": 0.0, "sharpe": 0.0, "annual_return": 0.0}
        volatility = returns.std() * np.sqrt(252)
        total_return = (df['NAV'].iloc[-1] / df['NAV'].iloc[0]) - 1
        days = (df['日期'].iloc[-1] - df['日期'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
        rf_decimal = risk_free_rate / 100.0
        sharpe_ratio = (annual_return - rf_decimal) / volatility if volatility > 0 else 0.0
        return {"volatility": volatility * 100, "sharpe": sharpe_ratio, "annual_return": annual_return * 100}

class BacktestEngine:
    # ... (保持原有的回測引擎程式碼不變) ...
    @staticmethod
    def calculate_lump_sum(df: pd.DataFrame, invest_date: datetime, amount: float):
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        start_row = df[df['日期'] >= invest_date].head(1)
        if start_row.empty: return None, "選定日期無有效數據 (可能過晚)"
        start_price = start_row['NAV'].values[0]
        real_start_date = start_row['日期'].dt.date.values[0]
        end_price = df['NAV'].iloc[-1]
        end_date = df['日期'].iloc[-1].date()
        units = amount / start_price
        final_value = units * end_price
        roi = ((final_value - amount) / amount) * 100
        return {
            "type": "單筆投入", "real_start_date": real_start_date, "end_date": end_date,
            "start_price": start_price, "end_price": end_price, "invested_capital": amount,
            "final_value": final_value, "roi": roi
        }, None

    @staticmethod
    def calculate_dca(df: pd.DataFrame, start_date: datetime, monthly_day: int, amount: float):
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        start_date = pd.to_datetime(start_date)
        records = []
        total_units = 0; total_invested = 0
        data_end_date = df['日期'].iloc[-1]
        current_month_first = start_date.replace(day=1)
        while current_month_first <= data_end_date:
            try:
                target_date = current_month_first.replace(day=monthly_day)
            except ValueError:
                next_month = current_month_first + relativedelta(months=1)
                target_date = next_month - timedelta(days=1)
            if target_date >= start_date and target_date <= data_end_date:
                trade_row = df[df['日期'] >= target_date].head(1)
                if not trade_row.empty:
                    price = trade_row['NAV'].values[0]
                    trade_date = trade_row['日期'].dt.date.values[0]
                    if not records or records[-1]['date'] != trade_date:
                        units = amount / price
                        total_units += units
                        total_invested += amount
                        records.append({'date': trade_date, 'price': price, 'units': units, 'cumulative_invested': total_invested})
            current_month_first += relativedelta(months=1)
        if total_invested == 0: return None, "在此期間內無有效扣款紀錄"
        final_price = df['NAV'].iloc[-1]
        final_value = total_units * final_price
        roi = ((final_value - total_invested) / total_invested) * 100
        return {
            "type": "定期定額", "start_date": records[0]['date'], "end_date": data_end_date.date(),
            "total_invested": total_invested, "final_value": final_value, "roi": roi,
            "deduct_count": len(records), "records": pd.DataFrame(records)
        }, None

    @staticmethod
    def generate_quick_summary(df: pd.DataFrame):
        periods = { "近 1 月": relativedelta(months=1), "近 3 月": relativedelta(months=3), "近 6 月": relativedelta(months=6), "近 1 年": relativedelta(years=1), "近 3 年": relativedelta(years=3), "近 5 年": relativedelta(years=5), "近 10 年": relativedelta(years=10) }
        results = []
        today = datetime.now()
        for name, delta in periods.items():
            start_date = today - delta
            res_lump, err_lump = BacktestEngine.calculate_lump_sum(df, start_date, 100000)
            roi_lump = res_lump['roi'] if not err_lump else None
            res_dca, err_dca = BacktestEngine.calculate_dca(df, start_date, 5, 5000)
            roi_dca = res_dca['roi'] if not err_dca else None
            results.append({ "週期": name, "單筆報酬率 (%)": f"{roi_lump:.2f}" if roi_lump is not None else "-", "定期定額報酬率 (%)": f"{roi_dca:.2f}" if roi_dca is not None else "-" })
        return pd.DataFrame(results)