# modules/config.py
from dateutil.relativedelta import relativedelta

class Config:
    """全域配置類別：集中管理所有常數與設定"""
    
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
        "00010145", "0074B060", "00120005", "00120018", "00120193", "00120002",
        "00120134", "00100118", "00400156", "00400104", "00040052", "10020058",
        "10110022", "0074B065", "00100058", "00580062", "10310016", "00100063",
        "00560011", "00400072"
    ]

    # --- 全球市場指標 ---
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