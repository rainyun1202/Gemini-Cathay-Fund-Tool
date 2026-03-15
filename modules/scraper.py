# modules/scraper.py
import requests
import pandas as pd
import yfinance as yf
import logging
import re  # <--- 新增正則表達式，用於抓取日期
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
        """抓取淨值"""
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
        """抓取配息紀錄"""
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
        """【新增】抓取前十大成分股"""
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
        """解析配息 HTML"""
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
        """【新增】解析成分股 HTML，回傳日期與 DataFrame"""
        soup = BeautifulSoup(html_str, 'html.parser')
        
        # 1. 鎖定標題 "主要10大成分股" (或包含此字眼的標籤)
        target_header = soup.find('h3', string=lambda text: text and ('主要10大成分股' in text or '主要持股' in text))
        if not target_header: return {"date": "", "data": pd.DataFrame()}

        # 2. 抓取資料日期 (通常在標題旁邊的括號內)
        date_text = ""
        header_parent = target_header.parent
        if header_parent:
            date_match = re.search(r'資料日期\s*([0-9/]+)', header_parent.text)
            if date_match:
                date_text = date_match.group(1)

        # 3. 找表格容器
        header_container = target_header.find_parent('div', class_='cubinvest-l-header')
        if header_container:
            table_container = header_container.find_next_sibling('div', class_='cubinvest-l-topHeadTable')
        else:
            table_container = target_header.find_next('div', class_='cubinvest-l-topHeadTable')

        if not table_container: return {"date": date_text, "data": pd.DataFrame()}

        # 4. 提取表頭 (Thead) 讓資料適應不同資產類型的欄位
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
        """【新增】批次抓取成分股"""
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