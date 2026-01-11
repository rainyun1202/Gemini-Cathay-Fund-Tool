# modules/analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List

class FundAnalyzer:
    """負責計算各項指標與報酬率"""
    @staticmethod
    def analyze_single(df: pd.DataFrame) -> Dict[str, Any]:
        df = df.sort_values('日期')
        fund_name = df['基金名稱'].iloc[0]
        url = df['URL'].iloc[0]
        latest = df.iloc[-1]
        latest_nav = latest['NAV']
        
        hist_max_idx = df['NAV'].idxmax()
        hist_min_idx = df['NAV'].idxmin()

        one_year_ago = df['日期'].max() - timedelta(days=365)
        df_1y = df[df['日期'] >= one_year_ago]
        
        if df_1y.empty:
            max_1y, min_1y, max_1y_date, min_1y_date = None, None, None, None
            diff_max_1y_pct, diff_min_1y_pct = None, None
        else:
            max_1y_idx = df_1y['NAV'].idxmax()
            min_1y_idx = df_1y['NAV'].idxmin()
            
            max_1y = df_1y.loc[max_1y_idx, 'NAV']
            max_1y_date = df_1y.loc[max_1y_idx, '日期']
            
            min_1y = df_1y.loc[min_1y_idx, 'NAV']
            min_1y_date = df_1y.loc[min_1y_idx, '日期']

            diff_max_1y_pct = ((latest_nav - max_1y) / max_1y) * 100
            diff_min_1y_pct = ((latest_nav - min_1y) / min_1y) * 100

        return {
            "基金名稱": fund_name,
            "最新價格": latest_nav,
            "最新價格日期": latest['日期'],
            "近一年最高價格": max_1y,
            "最高價與最新價%": diff_max_1y_pct,
            "近一年最高價格日期": max_1y_date,
            "近一年最低價格": min_1y,
            "最低價與最新價%": diff_min_1y_pct,
            "近一年最低價格日期": min_1y_date,
            "歷史最高價格": df.loc[hist_max_idx, 'NAV'],
            "歷史最高價格日期": df.loc[hist_max_idx, '日期'],
            "歷史最低價格": df.loc[hist_min_idx, 'NAV'],
            "歷史最低價格日期": df.loc[hist_min_idx, '日期'],
            "基金連結": url
        }

    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame, risk_free_rate: float) -> Dict[str, float]:
        df = df.sort_values('日期')
        df['pct_change'] = df['NAV'].pct_change()
        returns = df['pct_change'].dropna()
        
        if returns.empty:
            return {"volatility": 0.0, "sharpe": 0.0, "annual_return": 0.0}

        volatility = returns.std() * np.sqrt(252)
        
        total_return = (df['NAV'].iloc[-1] / df['NAV'].iloc[0]) - 1
        days = (df['日期'].iloc[-1] - df['日期'].iloc[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0.0

        rf_decimal = risk_free_rate / 100.0
        if volatility > 0:
            sharpe_ratio = (annual_return - rf_decimal) / volatility
        else:
            sharpe_ratio = 0.0

        return {
            "volatility": volatility * 100,
            "sharpe": sharpe_ratio,
            "annual_return": annual_return * 100
        }

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
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        start_row = df[df['日期'] >= invest_date].head(1)
        
        if start_row.empty:
            return None, "選定日期無有效數據 (可能過晚)"
            
        start_price = start_row['NAV'].values[0]
        real_start_date = start_row['日期'].dt.date.values[0]
        
        end_price = df['NAV'].iloc[-1]
        end_date = df['日期'].iloc[-1].date()
        
        units = amount / start_price
        final_value = units * end_price
        roi = ((final_value - amount) / amount) * 100
        
        return {
            "type": "單筆投入",
            "real_start_date": real_start_date,
            "end_date": end_date,
            "start_price": start_price,
            "end_price": end_price,
            "invested_capital": amount,
            "final_value": final_value,
            "roi": roi
        }, None

    @staticmethod
    def calculate_dca(df: pd.DataFrame, start_date: datetime, monthly_day: int, amount: float):
        df = df.sort_values('日期').reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        
        start_date = pd.to_datetime(start_date)
        records = []
        total_units = 0
        total_invested = 0
        
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
                        records.append({
                            'date': trade_date,
                            'price': price,
                            'units': units,
                            'cumulative_invested': total_invested
                        })
            
            current_month_first += relativedelta(months=1)
            
        if total_invested == 0:
            return None, "在此期間內無有效扣款紀錄"

        final_price = df['NAV'].iloc[-1]
        final_value = total_units * final_price
        roi = ((final_value - total_invested) / total_invested) * 100
        
        return {
            "type": "定期定額",
            "start_date": records[0]['date'],
            "end_date": data_end_date.date(),
            "total_invested": total_invested,
            "final_value": final_value,
            "roi": roi,
            "deduct_count": len(records),
            "records": pd.DataFrame(records)
        }, None

    @staticmethod
    def generate_quick_summary(df: pd.DataFrame):
        periods = {
            "近 1 月": relativedelta(months=1),
            "近 3 月": relativedelta(months=3),
            "近 6 月": relativedelta(months=6),
            "近 1 年": relativedelta(years=1),
            "近 3 年": relativedelta(years=3),
            "近 5 年": relativedelta(years=5),
            "近 10 年": relativedelta(years=10),
        }
        
        results = []
        today = datetime.now()
        
        for name, delta in periods.items():
            start_date = today - delta
            res_lump, err_lump = BacktestEngine.calculate_lump_sum(df, start_date, 100000)
            roi_lump = res_lump['roi'] if not err_lump else None
            
            res_dca, err_dca = BacktestEngine.calculate_dca(df, start_date, 5, 5000)
            roi_dca = res_dca['roi'] if not err_dca else None
            
            results.append({
                "週期": name,
                "單筆報酬率 (%)": f"{roi_lump:.2f}" if roi_lump is not None else "-",
                "定期定額報酬率 (%)": f"{roi_dca:.2f}" if roi_dca is not None else "-"
            })
            
        return pd.DataFrame(results)