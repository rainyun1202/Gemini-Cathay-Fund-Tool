# test_cathay_scraper.py
import requests
import pandas as pd
import logging
import urllib3
import sys
import os

# === 設定環境 ===
current_path = os.getcwd()
if current_path not in sys.path:
    sys.path.append(current_path)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 測試參數 ===
TEST_FUND_ID = "00120001" 

def test_scraper_full_functionality():
    """
    測試 FundScraper 的完整功能 (淨值 + 配息)
    """
    print(f"\n🚀 [測試] 正在呼叫 FundScraper 進行整合測試...")
    
    try:
        from modules.scraper import FundScraper
        scraper = FundScraper()
        
        # 1. 測試淨值
        print(f"⏳ (1/2) 抓取淨值 (Fund ID: {TEST_FUND_ID})...")
        df_nav = scraper.fetch_nav(TEST_FUND_ID)
        if df_nav is not None and not df_nav.empty:
            print(f"✅ 淨值抓取成功！大小: {df_nav.shape}")
            print(df_nav.head(2))
        else:
            print("⚠️ 淨值回傳為空")

        # 2. 測試配息 (新增)
        print(f"\n⏳ (2/2) 抓取配息紀錄 (Fund ID: {TEST_FUND_ID})...")
        df_div = scraper.fetch_dividend(TEST_FUND_ID)
        
        if df_div is not None and not df_div.empty:
            print(f"✅ 配息抓取成功！大小: {df_div.shape}")
            print("📊 配息資料預覽:")
            print(df_div.head(3))
            
            # 驗證數值轉換
            if '當期配息率(%)' in df_div.columns:
                print(f"🧐 檢查數值型態: {df_div['當期配息率(%)'].dtype}")
        else:
            print("⚠️ 配息紀錄為空 (可能是累積型基金或無資料)")
            
    except ImportError:
        print("❌ 無法匯入 modules，請確保檔案位置正確")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    test_scraper_full_functionality()