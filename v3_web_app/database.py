# -*- coding: utf-8 -*-
"""
얼굴 데이터와 출석 기록을 관리하는 데이터베이스 모듈
SQLite를 사용하여 로컬에 데이터 저장
"""

import sqlite3
import json
import os
from datetime import datetime, date
from typing import Optional, List, Tuple
import numpy as np


class DatabaseManager:
    """SQLite 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """데이터베이스 연결"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """필요한 테이블 생성"""
        # 사용자 테이블
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                face_encoding TEXT NOT NULL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 출석 기록 테이블
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                check_out_time TIMESTAMP,
                date DATE NOT NULL,
                status TEXT DEFAULT 'present',
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)
            )
        ''')
        
        self.conn.commit()
    
    def add_user(self, name: str, face_encoding: np.ndarray, 
                 department: str = "", photo_path: str = "") -> int:
        """
        새로운 사용자 등록
        
        Args:
            name: 사용자 이름
            face_encoding: 얼굴 인코딩 데이터 (numpy array)
            department: 소속 부서
            photo_path: 사진 파일 경로
            
        Returns:
            생성된 사용자 ID
        """
        encoding_json = json.dumps(face_encoding.tolist())
        
        self.cursor.execute('''
            INSERT INTO users (name, department, face_encoding, photo_path)
            VALUES (?, ?, ?, ?)
        ''', (name, department, encoding_json, photo_path))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_all_users(self) -> List[Tuple]:
        """모든 사용자 조회"""
        self.cursor.execute('''
            SELECT id, name, department, face_encoding, photo_path, created_at
            FROM users
            ORDER BY name
        ''')
        return self.cursor.fetchall()
    
    def get_user_by_id(self, user_id: int) -> Optional[Tuple]:
        """ID로 사용자 조회"""
        self.cursor.execute('''
            SELECT id, name, department, face_encoding, photo_path
            FROM users WHERE id = ?
        ''', (user_id,))
        return self.cursor.fetchone()
    
    def get_all_face_encodings(self) -> List[Tuple[int, str, np.ndarray]]:
        """모든 사용자의 얼굴 인코딩 데이터 조회"""
        self.cursor.execute('''
            SELECT id, name, face_encoding FROM users
        ''')
        
        results = []
        for row in self.cursor.fetchall():
            user_id, name, encoding_json = row
            encoding = np.array(json.loads(encoding_json))
            results.append((user_id, name, encoding))
        
        return results
    
    def delete_user(self, user_id: int) -> bool:
        """사용자 삭제"""
        try:
            # 출석 기록도 함께 삭제
            self.cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
            self.cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"사용자 삭제 오류: {e}")
            return False
    
    def record_attendance(self, user_id: int, 
                          check_type: str = "in") -> Tuple[bool, str]:
        """
        출석 기록
        
        Args:
            user_id: 사용자 ID
            check_type: "in" (출근) 또는 "out" (퇴근)
            
        Returns:
            (성공 여부, 메시지)
        """
        today = date.today()
        now = datetime.now()
        
        # 오늘 출석 기록 확인
        self.cursor.execute('''
            SELECT id, check_in_time, check_out_time 
            FROM attendance 
            WHERE user_id = ? AND date = ?
        ''', (user_id, today))
        
        existing = self.cursor.fetchone()
        
        if check_type == "in":
            if existing:
                return False, "이미 오늘 출석 체크를 하셨습니다."
            
            self.cursor.execute('''
                INSERT INTO attendance (user_id, date, check_in_time)
                VALUES (?, ?, ?)
            ''', (user_id, today, now))
            self.conn.commit()
            return True, f"출근 체크 완료! ({now.strftime('%H:%M:%S')})"
        
        else:  # check_out
            if not existing:
                return False, "먼저 출근 체크를 해주세요."
            
            if existing[2]:  # check_out_time이 이미 있음
                return False, "이미 퇴근 체크를 하셨습니다."
            
            self.cursor.execute('''
                UPDATE attendance 
                SET check_out_time = ?
                WHERE id = ?
            ''', (now, existing[0]))
            self.conn.commit()
            return True, f"퇴근 체크 완료! ({now.strftime('%H:%M:%S')})"
    
    def get_attendance_by_date(self, target_date: date) -> List[Tuple]:
        """특정 날짜의 출석 기록 조회"""
        self.cursor.execute('''
            SELECT u.name, u.department, a.check_in_time, a.check_out_time, a.status
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.date = ?
            ORDER BY a.check_in_time
        ''', (target_date,))
        return self.cursor.fetchall()
    
    def get_user_attendance_history(self, user_id: int, 
                                     limit: int = 30) -> List[Tuple]:
        """특정 사용자의 출석 이력 조회"""
        self.cursor.execute('''
            SELECT date, check_in_time, check_out_time, status
            FROM attendance
            WHERE user_id = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (user_id, limit))
        return self.cursor.fetchall()
    
    def get_today_attendance_count(self) -> int:
        """오늘 출석한 인원 수"""
        today = date.today()
        self.cursor.execute('''
            SELECT COUNT(*) FROM attendance WHERE date = ?
        ''', (today,))
        return self.cursor.fetchone()[0]
    
    def get_total_user_count(self) -> int:
        """전체 등록 인원 수"""
        self.cursor.execute('SELECT COUNT(*) FROM users')
        return self.cursor.fetchone()[0]
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()


# 테스트 코드
if __name__ == "__main__":
    db = DatabaseManager("test_attendance.db")
    
    # 더미 데이터로 테스트
    dummy_encoding = np.random.rand(128)
    user_id = db.add_user("홍길동", dummy_encoding, "개발팀")
    print(f"사용자 등록 완료: ID = {user_id}")
    
    # 출석 체크
    success, msg = db.record_attendance(user_id, "in")
    print(msg)
    
    # 퇴근 체크
    success, msg = db.record_attendance(user_id, "out")
    print(msg)
    
    # 오늘 출석 현황
    today_records = db.get_attendance_by_date(date.today())
    print(f"오늘 출석 현황: {today_records}")
    
    db.close()
    
    # 테스트 DB 삭제
    os.remove("test_attendance.db")
    print("테스트 완료!")
