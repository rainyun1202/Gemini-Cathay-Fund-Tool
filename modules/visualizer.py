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
            
            # ==========================================
            # Sheet 1: 總覽 Summary (自定義欄位排序)
            # ==========================================
            
            # 定義您想要的欄位順序
            # 邏輯：基本資料 -> 近一年高低 -> 【配息資訊】 -> 歷史高低
            desired_order = [
                '基金名稱',
                '最新價格', '最新價格日期',
                '近一年最高價格', '最高價與最新價%', '近一年最高價格日期',
                '近一年最低價格', '最低價與最新價%', '近一年最低價格日期',
                # --- 將配息欄位移到這裡 ---
                '近一年配息率總和(%)', '最近一次配息日', '最近一次配息金額', '最近一次當期配息率(%)',
                # --- 歷史欄位放到最後 ---
                '歷史最高價格', '歷史最高價格日期',
                '歷史最低價格', '歷史最低價格日期'
            ]
            
            # 1. 確保 DataFrame 包含這些欄位 (避免有些欄位沒產生導致報錯)
            # 先取出目前 df 有的所有欄位，交集比對
            existing_cols = list(summary_df.columns)
            final_cols = []
            
            # 加入指定的排序欄位
            for col in desired_order:
                if col in existing_cols:
                    final_cols.append(col)
            
            # 加入剩下沒在指定清單中的欄位 (以防萬一有漏掉的)
            for col in existing_cols:
                if col not in final_cols and col != '基金連結': # 連結不直接顯示
                    final_cols.append(col)

            # 2. 重新排序 DataFrame
            display_df = summary_df[final_cols]
            
            # 3. 寫入 Excel
            display_df.to_excel(writer, index=False, header=False, sheet_name='總覽 Summary', startrow=1)
            ws_summary = writer.sheets['總覽 Summary']
            ExcelReport._apply_styles(workbook, ws_summary, display_df, summary_df)
            ExcelReport._set_columns_width(display_df, ws_summary)

            # ==========================================
            # Sheet 2: 配息明細 Details (只留最近一年)
            # ==========================================
            if not all_div_df.empty:
                # 1. 複製一份避免影響原資料
                div_output = all_div_df.copy()
                
                # 2. 日期過濾邏輯：只保留「配息基準日」在最近一年內的資料
                # 為了過濾，先暫時轉成 datetime
                div_output['temp_date'] = pd.to_datetime(div_output['配息基準日'], errors='coerce')
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=1)
                
                # 執行篩選
                div_output = div_output[div_output['temp_date'] >= cutoff_date]
                
                # 3. 欄位整理
                target_cols = ['基金名稱', '配息基準日', '除息日', '每單位配息金額', '當期配息率(%)', '原始配息率字串']
                valid_cols = [c for c in target_cols if c in div_output.columns]
                div_output = div_output[valid_cols]
                
                # 4. 寫入 Excel
                if not div_output.empty:
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
            # 注意：因為 display_df 已經重新排序，我們需要用「基金名稱」回去 original_df 找連結
            # 這裡做一個簡單的 map 查找，確保連結正確
            fund_name = display_df.iat[i, 0]
            
            # 在原始資料中找到對應的 URL (假設基金名稱是唯一的，如果不唯一可能會有風險，但在這個應用場景通常沒問題)
            # 安全起見，我們還是用 original_df 的 index i，前提是 display_df 只是換欄位順序，沒有換列順序
            # 只要上面的 display_df = summary_df[final_cols] 沒有 sort_values，index 對應就是正確的
            if '基金連結' in original_df.columns:
                url = original_df.iloc[i]['基金連結']
                worksheet.write_url(i+1, 0, url, link_fmt, string=str(name))
            else:
                worksheet.write(i+1, 0, str(name), text_fmt)

            for j in range(1, len(display_df.columns)):
                val = display_df.iat[i, j]
                
                if j in date_cols and pd.notna(val):
                    if isinstance(val, (str, datetime, pd.Timestamp)): val = pd.to_datetime(val)
                    worksheet.write_datetime(i+1, j, val, date_fmt)
                elif isinstance(val, (int, float)):
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
    # ... (ChartManager 內容保持不變，直接沿用即可) ...
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