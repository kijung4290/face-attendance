# 🎯 Face Attendance - 얼굴인식 출석체크 시스템

이 프로젝트는 얼굴인식을 통해 출석을 기록하는 시스템입니다.
각 버전별로 다른 목적과 기능을 가지고 있으므로, 원하는 폴더로 들어가서 사용하세요.

---

## 📂 프로젝트 구조 및 사용 방법

### 1️⃣ `v1_basic_local/` (로컬 전용 버전)
> **인터넷 연결 없이 PC에서 독립적으로 사용하고 싶을 때 추천합니다.**
- **주요 기능**: 얼굴 등록, 출석 확인 (로컬 DB)
- **실행**: `python main.py`
- **특징**: 가장 가볍고 단순함

### 2️⃣ `v2_google_sheet_pc/` (구글 시트 연동 PC 버전)
> **PC에서 키오스크처럼 사용하며 Google Sheets에 자동 기록하고 싶을 때 추천합니다.** (👑 메인 버전)
- **주요 기능**: **Modern UI**, 실시간 시침, 구글 시트 연동
- **준비물**: `client_secret.json` (Google Cloud 서비스 계정 키 파일) 필요
- **실행**: `python main.py`

### 3️⃣ `v3_web_app/` (웹 배포용 버전)
> **PC에 설치하지 않고 브라우저로 접속하거나, 서버에 배포하고 싶을 때 추천합니다.**
- **주요 기능**: 웹캠 연동, 어디서든 접속 가능
- **실행**: `streamlit run streamlit_app.py`
- **배포**: Render 등에 바로 배포 가능 (Streamlit 기반)

---

## 🚀 시작하기

1. 원하는 폴더로 이동합니다.
   ```bash
   cd v2_google_sheet_pc
   ```
2. 필요한 라이브러리를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 프로그램을 실행합니다.
   ```bash
   python main.py
   ```
