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
    
    # --- 關注的基金清單 (包含自定義名稱與排序) ---
    # 參考自選 Excel 檔案順序排列，並指定名稱
    FUND_WATCH_LIST = [
        {'id': '00120001', 'name': '聯博國際科技基金(美元累積)'},
        {'id': '00120002', 'name': '聯博美國成長基金(美元累積)'},
        {'id': '00400013', 'name': '貝萊德世界科技基金(美元累積)'},
        {'id': '00400156', 'name': '貝萊德全球智慧數據股票入息基金(美元累積)'},
        {'id': '00580051', 'name': '摩根士丹利美國增長基金(美元累積)'},
        {'id': '00010144', 'name': '摩根美國科技基金(美元累積)'},
        {'id': '00060004', 'name': '富蘭克林高科技基金(美元年配)'},
        {'id': '00100045', 'name': '富蘭克林科技基金(美元累積)'},
        {'id': '00040097', 'name': '安聯AI人工智慧基金(美元累積)'},
        {'id': '10350005', 'name': '國泰大中華基金(台幣累積)'},
        {'id': '00100058', 'name': '富蘭克林大中華基金(美元累積)'},
        {'id': '0074B065', 'name': '高盛大中華股票基金(美元累積)'},
        {'id': '10340003', 'name': '安聯台灣科技基金(台幣累積)'},
        {'id': '00040052', 'name': '安聯中國股票基金(美元年配)'},
        {'id': '00400104', 'name': '貝萊德中國基金(美元累積)'},
        {'id': '00010145', 'name': '摩根中國基金(美元累積)'},
        {'id': '10020041', 'name': '摩根中國A股基金(台幣累積)'},
        {'id': '10020058', 'name': '摩根中國A股基金(美元累積)'},
        {'id': '10110022', 'name': '兆豐中國A股基金(美元累積)'},
        {'id': '00060003', 'name': '富蘭克林黃金基金(美元年配)'},
        {'id': '00400072', 'name': '貝萊德世界黃金基金(美元累積)'},
        {'id': '00120005', 'name': '聯博美國收益基金(美元累積)'},
        {'id': '00120134', 'name': '聯博全球多元收益基金(美元累積)'},
        {'id': '00120018', 'name': '聯博全球非投資等級債券基金(美元累積)'},
        {'id': '00120193', 'name': '聯博優化短期非投資等級債券基金(美元累積)'},
        {'id': '00100118', 'name': '富蘭克林穩定月收益基金(美元累積)'},
        {'id': '0074B060', 'name': '高盛投資級公司債基金(美元累積)'},
        {'id': '00560011', 'name': '法巴乾淨能源股票基金(美元累積)'},
        {'id': '00400029', 'name': '貝萊德世界能源基金(美元累積)'},
        {'id': '00100063', 'name': '富蘭克林天然資源基金(美元累積)'},
        {'id': '00100046', 'name': '富蘭克林生技領航基金(美元累積)'},
        {'id': '10310016', 'name': '中國信託越南機會基金(美元累積)'},
        {'id': '00580061', 'name': '摩根士丹利新興領先股票基金(美元累積)'},
    ]

    # 自動產生舊版相容的 ID 列表 (給爬蟲使用)
    DEFAULT_FUND_IDS_LIST = [item['id'] for item in FUND_WATCH_LIST]

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