# modules/scraper.py
import requests
import pandas as pd
import yfinance as yf
import logging
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from .config import Config

logger = logging.getLogger(__name__)

class FundScraper:
    """負責抓取國泰基金歷史淨值、配息紀錄與成分股"""
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
            logger.error(f"基金淨值 {fund_id} 失敗: {e}")
            return None

    def fetch_dividend(self, fund_id: str) -> Optional[pd.DataFrame]:
        target_url = f"https://www.cathaybk.com.tw/cathaybk/personal/investment/fund/details/?fundid={fund_id}#tab6"
        try:
            resp = self.session.get(target_url, timeout=Config.TIMEOUT)
            resp.raise_for_status()
            df = self._parse_dividend_html(resp.text)
            if df.empty: return None
            df['fund_id'] = fund_id
            return df
        except Exception as e:
            logger.error(f"基金配息 {fund_id} 失敗: {e}")
            return None

    def fetch_holdings(self, fund_id: str) -> Dict[str, Any]:
        target_url = f"https://www.cathaybk.com.tw/cathaybk/personal/investment/fund/details/?fundid={fund_id}#tab6"
        try:
            resp = self.session.get(target_url, timeout=Config.TIMEOUT)
            resp.raise_for_status()
            result = self._parse_holdings_html(resp.text)
            return result
        except Exception as e:
            logger.error(f"基金成分股 {fund_id} 失敗: {e}")
            return {"date": "", "data": pd.DataFrame()}

    def _parse_dividend_html(self, html_str: str) -> pd.DataFrame:
        soup = BeautifulSoup(html_str, 'html.parser')
        target_header = soup.find('h3', string=lambda text: text and '近期配息紀錄' in text)
        if not target_header: return pd.DataFrame()

        header_container = target_header.find_parent('div', class_='cubinvest-l-header')
        if header_container:
            table_container = header_container.find_next_sibling('div', class_='cubinvest-l-topHeadTable')
        else:
            table_container = target_header.find_next('div', class_='cubinvest-l-topHeadTable')

        if not table_container: return pd.DataFrame()
        tbody = table_container.find('div', class_='cubinvest-l-topHeadTable__tbody')
        if not tbody: return pd.DataFrame()

        rows = tbody.find_all('div', class_='cubinvest-l-topHeadTable__tr')
        data_list = []
        for row in rows:
            cells = row.find_all('div', class_='cubinvest-l-topHeadTable__td')
            if len(cells) >= 4:
                try:
                    rate_str = cells[3].text.strip().replace('%', '')
                    rate_val = float(rate_str) if rate_str else 0.0
                    amt_str = cells[2].text.strip()
                    amt_val = float(amt_str) if amt_str else 0.0
                    data_list.append({
                        "配息基準日": cells[0].text.strip(),
                        "除息日": cells[1].text.strip(),
                        "每單位配息金額": amt_val,
                        "當期配息率(%)": rate_val, 
                        "原始配息率字串": cells[3].text.strip()
                    })
                except ValueError:
                    continue 
        return pd.DataFrame(data_list)

    def _parse_holdings_html(self, html_str: str) -> Dict[str, Any]:
        """【更新】解析成分股 HTML，精準抓取日期與 DataFrame"""
        soup = BeautifulSoup(html_str, 'html.parser')
        
        # === 1. 修正日期抓取邏輯：使用精確的 ID 定位 ===
        date_text = "未知"
        date_div = soup.find('div', id='layout_0_main_0_div_Rpt_Tab05_TopHolding_Date')
        
        if date_div:
            raw_date_text = date_div.text.strip()
            # 利用正則表達式把「資料日期 :」等無用字眼過濾掉，只留下真正的日期字串 (如 "2026年01月")
            date_match = re.search(r'資料日期\s*[:：]?\s*(.+)', raw_date_text)
            if date_match:
                date_text = date_match.group(1).strip()
            else:
                # 萬一格式變了，至少保留原始文字
                date_text = raw_date_text

        # 2. 鎖定標題 "主要10大成分股" (或包含此字眼的標籤)
        target_header = soup.find('h3', string=lambda text: text and ('主要10大成分股' in text or '主要持股' in text))
        if not target_header: return {"date": date_text, "data": pd.DataFrame()}

        # 3. 找表格容器
        header_container = target_header.find_parent('div', class_='cubinvest-l-header')
        if header_container:
            table_container = header_container.find_next_sibling('div', class_='cubinvest-l-topHeadTable')
        else:
            table_container = target_header.find_next('div', class_='cubinvest-l-topHeadTable')

        if not table_container: return {"date": date_text, "data": pd.DataFrame()}

        # 4. 提取表頭 (Thead)
        thead = table_container.find('div', class_='cubinvest-l-topHeadTable__thead')
        col_names = []
        if thead:
            th_cells = thead.find_all('div', class_='cubinvest-l-topHeadTable__th')
            col_names = [th.text.strip() for th in th_cells]

        # 5. 提取資料 (Tbody)
        tbody = table_container.find('div', class_='cubinvest-l-topHeadTable__tbody')
        if not tbody: return {"date": date_text, "data": pd.DataFrame()}

        rows = tbody.find_all('div', class_='cubinvest-l-topHeadTable__tr')
        data_list = []
        for row in rows:
            cells = row.find_all('div', class_='cubinvest-l-topHeadTable__td')
            row_data = [cell.text.strip() for cell in cells]
            if row_data:
                data_list.append(row_data)

        df = pd.DataFrame(data_list, columns=col_names if col_names else None)
        return {"date": date_text, "data": df}

    def fetch_all_nav(self, fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
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

    def fetch_all_dividend(self, fund_ids: List[str]) -> Dict[str, pd.DataFrame]:
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(self.fetch_dividend, fid): fid for fid in fund_ids}
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    df = future.result()
                    if df is not None: results[fid] = df
                except Exception: pass
        return results

    def fetch_all_holdings(self, fund_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {executor.submit(self.fetch_holdings, fid): fid for fid in fund_ids}
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    res = future.result()
                    if not res["data"].empty: 
                        results[fid] = res
                except Exception: pass
        return results

class MarketScraper:
    """負責抓取 Yahoo Finance 市場數據與成分股即時狀態"""
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
    # ⚡ 智慧解析引擎：處理不規則成分股名稱
    # ==========================================
    @staticmethod
    def clean_company_name(name: str) -> str:
        """清洗髒資料，移除產業別與註解"""
        name = str(name).strip()
        # 1. 切除逗號後面的純中文 (如 "Alphabet, Inc.,通訊服務" -> "Alphabet, Inc.")
        name = re.sub(r',[^\x00-\x7F]+$', '', name)
        # 2. 切除全形逗號後面的內容 (如 "台積電，半導體")
        name = re.sub(r'，.*$', '', name)
        # 3. 切除括號內容 (如 "巴里克黃金(加拿大 原物料)" -> "巴里克黃金")
        name = re.sub(r'\(.*?\)|（.*?）', '', name)
        # 4. 移除常見中文公司後綴干擾
        name = re.sub(r'股份|有限公司|公司|控股|集團', '', name).strip()
        return name

    @staticmethod
    def search_symbol(company_name: str) -> Optional[str]:
        """利用多階段清洗與搜尋引擎 API 解析 Ticker"""
        original_name = str(company_name).strip()
        
        # [階段 0]：優先看 Config 是否有強制指定的字典 (安全網)
        if hasattr(Config, 'CUSTOM_TICKER_MAP'):
            for key_name, ticker in Config.CUSTOM_TICKER_MAP.items():
                if key_name in original_name:
                    return ticker

        # [階段 1]：字串清洗
        clean_name = MarketScraper.clean_company_name(original_name)
        if not clean_name:
            return None

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        # [階段 2]：Yahoo Finance 官方 Search API (最快，適合英文或標準中文)
        search_candidates = [clean_name]
        
        # 嘗試抽出純英文部分 (如 "巴里克黃金 BARRICK" -> 抽出 "BARRICK")
        eng_parts = re.findall(r'[A-Za-z\.\-\& ]+', original_name)
        if eng_parts:
            longest_eng = max(eng_parts, key=len).strip()
            if len(longest_eng) > 3:
                search_candidates.append(longest_eng)

        for candidate in search_candidates:
            if not candidate: continue
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(candidate)}&quotesCount=1"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                data = res.json()
                quotes = data.get('quotes', [])
                if quotes and 'symbol' in quotes[0]:
                    return quotes[0]['symbol']
            except Exception:
                pass

        # [階段 3]：搜尋引擎 Web 爬蟲 (最強模糊比對，解決「越南外貿銀行」這類難題)
        # 使用 DuckDuckGo Lite 版，免 API Key 且不怕被擋
        query = f'"{clean_name}" stock quote site:finance.yahoo.com/quote'
        ddg_url = "https://lite.duckduckgo.com/lite/"
        try:
            res = requests.post(ddg_url, headers=headers, data={"q": query}, timeout=5)
            # 從搜尋結果的 HTML 文本中，直接找出 Yahoo Finance 的網址特徵
            match = re.search(r'finance\.yahoo\.com/quote/([A-Za-z0-9\.\-\=]+)', res.text)
            if match:
                return match.group(1)
        except Exception as e:
            logger.debug(f"Search Engine Error for {clean_name}: {e}")

        return None

    def fetch_stock_stats(self, company_name: str) -> Dict[str, Any]:
        """抓取並計算單一股票的近一年數據"""
        symbol = self.search_symbol(company_name)
        base_result = {"Ticker": symbol, "最新價格": None, "近一年最高": None, "高點0.618": None, "距高點回撤(%)": None}
        if not symbol:
            return base_result
        
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if hist.empty:
                return base_result
            
            latest_price = hist['Close'].iloc[-1]
            high_52w = hist['High'].max()
            fib_0618 = high_52w * 0.618
            drawdown = ((latest_price - high_52w) / high_52w) * 100
            
            return {
                "Ticker": symbol,
                "最新價格": latest_price,
                "近一年最高": high_52w,
                "高點0.618": fib_0618,
                "距高點回撤(%)": drawdown
            }
        except Exception as e:
            logger.warning(f"抓取 {symbol} 數據失敗: {e}")
            return base_result