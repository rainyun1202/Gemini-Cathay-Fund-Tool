# modules/visualizer.py
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional
from .config import Config

class ExcelReport:
    """負責生成 Excel 報表 (含多 Sheet)"""
    @staticmethod
    def create_excel_bytes(summary_df: pd.DataFrame, all_div_df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # --- Sheet 1: 總覽 Summary ---
            display_df = summary_df.drop(columns=['基金連結'])
            display_df.to_excel(writer, index=False, header=False, sheet_name='總覽 Summary', startrow=1)
            ws_summary = writer.sheets['總覽 Summary']
            ExcelReport._apply_styles(workbook, ws_summary, display_df, summary_df)
            ExcelReport._set_columns_width(display_df, ws_summary)

            # --- Sheet 2: 配息明細 Dividend Details ---
            if not all_div_df.empty:
                # 重新整理欄位順序
                cols = ['基金名稱', '配息基準日', '除息日', '每單位配息金額', '當期配息率(%)', '原始配息率字串']
                # 只取存在的欄位
                valid_cols = [c for c in cols if c in all_div_df.columns]
                div_output = all_div_df[valid_cols]
                
                div_output.to_excel(writer, index=False, header=False, sheet_name='配息明細 Details', startrow=1)
                ws_div = writer.sheets['配息明細 Details']
                ExcelReport._apply_styles_simple(workbook, ws_div, div_output)
                ExcelReport._set_columns_width(div_output, ws_div)
            
            ws_summary.freeze_panes(1, 0)
        return output.getvalue()

    @staticmethod
    def _apply_styles(workbook, worksheet, display_df, original_df):
        """主表樣式 (含超連結)"""
        base_font = 'Microsoft JhengHei'
        header_fmt = workbook.add_format({'bold': True, 'font_name': base_font, 'bg_color': '#DCE6F1', 'align': 'center', 'valign': 'vcenter', 'border': 1})
        text_fmt = workbook.add_format({'font_name': base_font, 'valign': 'top', 'border': 1})
        num_fmt = workbook.add_format({'font_name': base_font, 'valign': 'top', 'border': 1, 'num_format': '#,##0.00'}) 
        link_fmt = workbook.add_format({'font_color': 'blue', 'underline': 1, 'font_name': base_font, 'valign': 'top', 'border': 1})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': base_font, 'valign': 'top', 'border': 1})

        # 寫入 Header
        for col, val in enumerate(display_df.columns):
            worksheet.write(0, col, val, header_fmt)

        date_cols = [i for i, c in enumerate(display_df.columns) if '日期' in str(c) or 'Date' in str(c) or '日' in str(c)]
        
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
                    # === 修正重點：檢查是否為 NaN ===
                    # 雖然 NaN 也是 float，但 xlsxwriter write_number 不支援，必須改寫為文字 "-"
                    if pd.isna(val):
                        worksheet.write(i+1, j, "-", text_fmt)
                    else:
                        worksheet.write_number(i+1, j, val, num_fmt)
                else:
                    worksheet.write(i+1, j, str(val) if pd.notna(val) else "-", text_fmt)

    @staticmethod
    def _apply_styles_simple(workbook, worksheet, df):
        """明細表樣式 (不含超連結)"""
        base_font = 'Microsoft JhengHei'
        header_fmt = workbook.add_format({'bold': True, 'font_name': base_font, 'bg_color': '#EBF1DE', 'align': 'center', 'border': 1})
        text_fmt = workbook.add_format({'font_name': base_font, 'border': 1})
        
        for col, val in enumerate(df.columns):
            worksheet.write(0, col, val, header_fmt)
            
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iat[i, j]
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

class ChartManager:
    # ... (Plotly 繪圖部分與之前相同，保持不變) ...
    @staticmethod
    def plot_dual_axis_trends(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
        if not selected_keys:
            st.info("請從上方選單勾選 1~2 項資產進行比較。")
            return
        delta = Config.TIME_RANGES.get(time_range_key)
        if not delta: delta = relativedelta(years=1)
        start_date_limit = pd.to_datetime("today") - delta
        plot_data = []
        global_min_ratio = 1.0; global_max_ratio = 1.0
        for key in selected_keys:
            if key in all_data:
                df = all_data[key].copy()
                df = df.sort_values('日期')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= start_date_limit]
                if not df.empty:
                    start_price = df['NAV'].iloc[0]
                    min_price = df['NAV'].min(); max_price = df['NAV'].max()
                    min_ratio = min_price / start_price; max_ratio = max_price / start_price
                    if min_ratio < global_min_ratio: global_min_ratio = min_ratio
                    if max_ratio > global_max_ratio: global_max_ratio = max_ratio
                    raw_name = df['基金名稱'].iloc[0]
                    asset_name = str(raw_name) if raw_name else key
                    plot_data.append({ "data": df, "name": asset_name, "start_price": start_price })
        if not plot_data:
            st.warning(f"選取的資產在【{time_range_key}】內無足夠數據可供繪圖。")
            return
        range_padding = (global_max_ratio - global_min_ratio) * 0.05
        if range_padding == 0: range_padding = 0.01
        final_min_ratio = global_min_ratio - range_padding; final_max_ratio = global_max_ratio + range_padding
        fig = go.Figure()
        d1 = plot_data[0]
        fig.add_trace(go.Scatter(x=d1["data"]['日期'], y=d1["data"]['NAV'], name=d1["name"], yaxis='y', hovertemplate='%{y:,.2f}'))
        y1_range = [d1["start_price"] * final_min_ratio, d1["start_price"] * final_max_ratio]
        y2_range = None
        if len(plot_data) > 1:
            d2 = plot_data[1]
            fig.add_trace(go.Scatter(x=d2["data"]['日期'], y=d2["data"]['NAV'], name=d2["name"], yaxis='y2', hovertemplate='%{y:,.2f}'))
            y2_range = [d2["start_price"] * final_min_ratio, d2["start_price"] * final_max_ratio]
        fig.update_layout(title=f'資產價格走勢比較 ({time_range_key}) - 起點歸一化視角', xaxis=dict(title='日期'), hovermode='x unified', legend=dict(orientation="h", y=1.1))
        fig.update_layout(yaxis=dict(title=d1["name"], title_font=dict(color='#1f77b4'), tickfont=dict(color='#1f77b4'), range=y1_range, tickformat=',.2f'))
        if len(plot_data) > 1:
            fig.update_layout(yaxis2=dict(title=d2["name"], title_font=dict(color='#ff7f0e'), tickfont=dict(color='#ff7f0e'), overlaying='y', side='right', range=y2_range, tickformat=',.2f'))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_investment_growth(all_data: Dict[str, pd.DataFrame], selected_keys: List[str], time_range_key: str):
        if not selected_keys: return
        delta = Config.TIME_RANGES.get(time_range_key)
        if not delta: delta = relativedelta(years=1)
        start_date_limit = pd.to_datetime("today") - delta
        plot_data = []
        global_min_val = 1_000_000; global_max_val = 1_000_000
        initial_capital = 1_000_000
        for key in selected_keys:
            if key in all_data:
                df = all_data[key].copy()
                df = df.sort_values('日期')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= start_date_limit]
                if not df.empty:
                    start_price = df['NAV'].iloc[0]
                    df['Growth'] = (df['NAV'] / start_price) * initial_capital
                    min_val = df['Growth'].min(); max_val = df['Growth'].max()
                    if min_val < global_min_val: global_min_val = min_val
                    if max_val > global_max_val: global_max_val = max_val
                    raw_name = df['基金名稱'].iloc[0]
                    asset_name = str(raw_name) if raw_name else key
                    plot_data.append({ "data": df, "name": asset_name })
        if not plot_data: return
        val_range = global_max_val - global_min_val
        padding = val_range * 0.05
        if padding == 0: padding = 10000
        y_range = [global_min_val - padding, global_max_val + padding]
        fig = go.Figure()
        for item in plot_data:
            fig.add_trace(go.Scatter(x=item["data"]['日期'], y=item["data"]['Growth'], name=item["name"], hovertemplate='%{y:,.0f}'))
        fig.update_layout(title=f'100 萬資產增值模擬 ({time_range_key})', xaxis=dict(title='日期'), yaxis=dict(title='資產總值 (TWD/USD 依標的計價)', tickformat=',.0f', range=y_range), hovermode='x unified', legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)