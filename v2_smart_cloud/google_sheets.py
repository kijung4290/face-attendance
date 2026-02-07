# -*- coding: utf-8 -*-
"""
Google Sheets 연동 모듈
출석 기록을 Google Spreadsheets에 실시간으로 동기화합니다.
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import threading

class GoogleSheetsManager:
    """Google Sheets 연동 관리 클래스"""
    
    def __init__(self, key_file='client_secret.json', sheet_name='출석부'):
        """
        초기화
        Args:
            key_file (str): Google Cloud 서비스 계정 키 파일 경로
            sheet_name (str): 구글 스프레드시트 이름
        """
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        self.key_file = key_file
        self.sheet_name = sheet_name
        self.client = None
        self.sheet = None
        self.is_connected = False
        self.is_connecting = False
        
        # 비동기 연결 시도
        self.connect_async()
        
    def connect_async(self):
        """비동기로 구글 시트 연결"""
        if self.is_connecting:
            return
            
        thread = threading.Thread(target=self._connect, daemon=True)
        thread.start()
        
    def _connect(self):
        """실제 연결 로직"""
        if not os.path.exists(self.key_file):
            print("⚠ Google Sheets: 키 파일을 찾을 수 없습니다. (client_secret.json 필요)")
            self.is_connected = False
            return

        try:
            self.is_connecting = True
            print("Google Sheets 연결 시도 중...")
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.key_file, self.scope)
            self.client = gspread.authorize(creds)
            
            # 시트 열기 (없으면 예외 발생)
            try:
                # 1. 시트 이름으로 열기 시도
                spreadsheet = self.client.open(self.sheet_name)
                self.sheet = spreadsheet.sheet1
                print(f"✓ Google Sheets '{self.sheet_name}' 연결 성공")
                
            except gspread.SpreadsheetNotFound:
                print(f"⚠ 시트 '{self.sheet_name}'를 찾을 수 없습니다. 새로 생성합니다.")
                try:
                    spreadsheet = self.client.create(self.sheet_name)
                    spreadsheet.share(creds.service_account_email, perm_type='user', role='owner')
                    self.sheet = spreadsheet.sheet1
                    
                    # 헤더 추가
                    self.sheet.append_row(["날짜", "시간", "이름", "부서", "상태", "비고"])
                    print(f"✓ 새 시트 '{self.sheet_name}' 생성 완료")
                    
                except Exception as e:
                    print(f"❌ 시트 생성 실패: {e}")
                    self.is_connected = False
                    self.is_connecting = False
                    return
            
            self.is_connected = True
            
            # 헤더 확인 (비어있으면 추가)
            if not self.sheet.get_all_values():
                self.sheet.append_row(["날짜", "시간", "이름", "부서", "상태", "비고"])
                
        except Exception as e:
            print(f"❌ Google Sheets 연결 실패: {e}")
            self.is_connected = False
            
        finally:
            self.is_connecting = False

    def add_record(self, name, department, status, note=""):
        """
        출석 기록 추가 (비동기 처리)
        """
        if not self.is_connected:
            return False
            
        # 별도 스레드에서 시트 업데이트 (UI 블로킹 방지)
        thread = threading.Thread(
            target=self._append_row,
            args=(name, department, status, note),
            daemon=True
        )
        thread.start()
        return True

    def _append_row(self, name, department, status, note):
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")     # 2024-02-07
            time_str = now.strftime("%H:%M:%S")     # 14:30:00
            
            row = [date_str, time_str, name, department, status, note]
            self.sheet.append_row(row)
            print(f"✓ 시트 기록 완료: {name} ({status})")
            
        except Exception as e:
            print(f"⚠ 시트 기록 실패: {e}")
            # 재연결 시도
            self.is_connected = False
            self.connect_async()
