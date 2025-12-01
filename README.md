# 📊 國泰基金淨值自動化分析工具 (Cathay Fund Tool)

這是一個基於 Python 與 Streamlit 構建的即時金融數據分析工具。本專案能夠自動抓取國泰投顧的基金歷史淨值，進行關鍵數據分析（如近一年高低點、歷史極值），並自動生成包含專業格式與超連結的 Excel 報表供使用者下載。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🚀 功能特色

* **自動化爬蟲 (Automated Scraping)**：採用多執行緒 (Multi-threading) 技術併發抓取數據，大幅縮短等待時間。
* **即時數據分析 (Real-time Analysis)**：
    * 自動計算最新淨值與日期。
    * 歷史最高/最低價格與日期偵測。
    * 近一年 (52週) 最高/最低價格波動區間分析。
* **專業報表輸出 (Professional Reporting)**：一鍵生成 `.xlsx` 報表，內建自動欄寬調整、條件格式化與基金連結跳轉功能。
* **互動式網頁介面 (Interactive UI)**：使用者可透過側邊欄 (Sidebar) 自定義需要追蹤的基金代號清單。

## 🛠️ 技術棧 (Tech Stack)

* **核心語言**: Python 3.9+
* **網頁框架**: [Streamlit](https://streamlit.io/)
* **數據處理**: Pandas, NumPy
* **網路請求**: Requests, Urllib3
* **報表生成**: XlsxWriter

## 📦 如何在本地端執行 (Local Installation)

如果您希望在自己的電腦上運行此專案，請依照以下步驟操作：

1.  **Clone 此專案**
    ```bash
    git clone [https://github.com/rainyun1202/Gemini-Cathay-Fund-Tool.git](https://github.com/rainyun1202/Gemini-Cathay-Fund-Tool.git)
    cd Gemini-Cathay-Fund-Tool
    ```

2.  **安裝依賴套件**
    建議建立一個虛擬環境 (Virtual Environment) 後執行：
    ```bash
    pip install -r requirements.txt
    ```

3.  **啟動應用程式**
    ```bash
    streamlit run app.py
    ```
    執行後，瀏覽器將自動開啟並顯示應用程式介面 (通常位於 `http://localhost:8501`)。

## ☁️ 雲端部署 (Deployment)

本專案已優化並支援直接部署於 **Streamlit Community Cloud**：

1.  將此專案 Fork 或 Push 到您的 GitHub Repository。
2.  登入 [Streamlit Community Cloud](https://streamlit.io/cloud)。
3.  選擇 `New app` -> `Use existing repo`。
4.  選擇您的 Repository (`rainyun1202/Gemini-Cathay-Fund-Tool`) 與 Branch (`main`)。
5.  設定 Main file path 為 `app.py`。
6.  點擊 **Deploy** 即可完成部署。

## 📂 專案結構

```text
Gemini-Cathay-Fund-Tool/
├── app.py              # 主應用程式邏輯 (Streamlit)
├── requirements.txt    # Python 依賴清單
├── README.md           # 專案說明文件
└── .gitignore          # Git 忽略設定
```

## ⚠️ 免責聲明 (Disclaimer)

本工具僅供程式開發學習、學術研究與個人輔助使用。抓取之數據來源為公開網頁，數據準確性以來源網站為準。本工具不提供任何投資建議，使用者應自行評估風險。

Created with ❤️ by Rain