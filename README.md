# 📊 全球市場與基金淨值戰情室 (Global Market & Fund Dashboard)

這是一個基於 Python 與 Streamlit 開發的自動化金融分析儀表板。
專為投資人設計，能夠即時整合 **全球關鍵市場指標 (Yahoo Finance)** 與 **指定國泰基金 (Cathay API)** 的歷史淨值數據，
提供一站式的淨值追蹤、趨勢比較、風險分析與投資策略回測。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Data Source](https://img.shields.io/badge/Data-Yahoo_Finance_%26_Cathay-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🚀 主要功能 (Key Features)

### 1. 📋 市場全覽與報表自動化 (Overview & Reporting)
* **跨平台數據整合**：同時抓取美股 ETF (VOO, QQQ)、加密貨幣 (BTC)、原物料 (黃金, 原油)、總經指標 (VIX, 10年期公債殖利率) 與指定基金淨值。
* **自動化摘要**：即時計算最新淨值、近一年高低點及其乖離率。
* **Excel 報表輸出**：一鍵匯出包含完整格式與超連結的 Excel 分析報告 (`.xlsx`)。

### 2. 📈 進階趨勢與風險分析 (Advanced Charting & Risk Analysis)
* **雙軸趨勢比較**：支援同時繪製兩種不同單位的資產（例如：台幣計價基金 vs. 美元指數），並自動進行起點歸一化 (Normalized) 比較。
* **資產增值模擬**：模擬 100 萬本金在不同標的下的歷史資產增長曲線。
* **風險指標計算**：自動計算所選區間的 **年化夏普值 (Sharpe Ratio)** 與 **年化波動度 (Volatility)**，並以無風險利率 (美國 10 年期公債) 為基準。

### 3. 💰 投資策略回測引擎 (Backtesting Engine)
* **單筆投入 (Lump Sum)**：計算在特定日期單筆買入後的持有報酬率與現值。
* **定期定額 (DCA)**：模擬每月固定日期扣款的投資績效，包含總投入成本、累積單位數與最終報酬率。
* **歷史績效速覽**：自動生成近 1 月至近 10 年的快速報酬率對照表。

---

## 📂 專案架構 (Project Structure)

```text
Gemini-Cathay-Fund-Tool/
│
├── app.py                  # 【主程式】Streamlit 入口與 UI 邏輯控制
├── requirements.txt        # 套件依賴清單
├── README.md               # 專案說明文件
│
└── modules/                # 【核心模組】
    ├── __init__.py
    ├── config.py           # 【配置層】基金清單、市場代號、全域常數設定
    ├── scraper.py          # 【資料層】負責爬取 API 與 Yahoo Finance 數據
    ├── analyzer.py         # 【邏輯層】負責指標計算、回測演算法
    └── visualizer.py       # 【表現層】負責 Excel 樣式生成與 Plotly 圖表繪製
```

## 🛠️ 技術棧 (Tech Stack)

* **核心語言**: Python 3.9+
* **網頁框架**: [Streamlit](https://streamlit.io/)
* **數據來源**:
    * `yfinance` (Yahoo Finance API)
    * `requests` (Web Scraping)
* **數據處理**: Pandas, NumPy
* **報表生成**: XlsxWriter

## 📦 如何在本地端執行 (Local Installation)

1.  **Clone 此專案**
    ```bash
    git clone [https://github.com/rainyun1202/Gemini-Cathay-Fund-Tool.git](https://github.com/rainyun1202/Gemini-Cathay-Fund-Tool.git)
    cd Gemini-Cathay-Fund-Tool
    ```

2.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```
    *(註：requirements.txt 已包含 `yfinance`, `streamlit`, `pandas`, `xlsxwriter` 等必要套件)*

3.  **啟動應用程式**
    ```bash
    streamlit run app.py
    ```
    執行後，瀏覽器將自動開啟並顯示戰情室介面。

## ☁️ 雲端部署 (Deployment)

本專案已優化並支援直接部署於 **Streamlit Community Cloud**，且無需設定任何 API Key：

1.  將此專案 Push 到您的 GitHub Repository。
2.  登入 [Streamlit Community Cloud](https://streamlit.io/cloud)。
3.  選擇 `New app` -> `Use existing repo`。
4.  選擇您的 Repository 與 Branch (`main`)。
5.  設定 Main file path 為 `app.py`。
6.  點擊 **Deploy** 即可完成部署。

## ⚠️ 免責聲明 (Disclaimer)

* 本工具僅供程式開發學習、學術研究與個人輔助使用。

* 國泰基金數據來源為公開網頁；市場數據由 Yahoo Finance 提供，可能會有延遲。

* 本工具不提供任何投資建議，使用者應自行評估市場風險。

Created with ❤️ by Rain