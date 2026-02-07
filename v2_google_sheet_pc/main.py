# -*- coding: utf-8 -*-
"""
얼굴인식 출석체크 프로그램 - Modern UI
CustomTkinter 기반의 현대적인 디자인 적용
Google Sheets 연동 포함
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import threading
from datetime import datetime, date
import time
from typing import Optional, List
import numpy as np

# 사용자 모듈
from database import DatabaseManager
from face_recognition_module import FaceRecognitionModule, CameraManager, FaceInfo
from google_sheets import GoogleSheetsManager

# CustomTkinter 설정
ctk.set_appearance_mode("Dark")  # 모드 설정: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # 테마 설정: "blue" (standard), "green", "dark-blue"

class ModernFaceApp(ctk.CTk):
    """모던한 디자인의 얼굴인식 출석체크 앱"""
    
    def __init__(self):
        super().__init__()
        
        # 1. 메인 윈도우 설정
        self.title("FacePass - 스마트 출석체크")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # 2. 아이콘 설정 (생략 가능)
        # self.iconbitmap("icon.ico")
        
        # 3. 데이터 및 모듈 초기화
        self._init_modules()
        
        # 4. 상태 변수
        self.is_running = False
        self.current_mode = "attendance"  # "attendance" or "register"
        self.captured_frame = None
        self.recognition_cooldown = {}
        self.last_faces = []
        self.recent_logs = []  # 최근 출석 로그 저장용
        
        # 5. UI 구성
        self._create_layout()
        self._setup_camera_loop()
        
        # 6. 종료 이벤트
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _init_modules(self):
        """모듈 초기화"""
        print("시스템 초기화 중...")
        self.db = DatabaseManager()
        self.face_module = FaceRecognitionModule(tolerance=0.45, det_size=(320, 320))
        self.camera = CameraManager()
        self.sheets_manager = GoogleSheetsManager()  # 구글 시트 매니저
        
        # 얼굴 데이터 로드
        self._load_face_data()
        
    def _create_layout(self):
        """전체 레이아웃 생성"""
        # 그리드 설정 (2열 구조: 왼쪽 카메라, 오른쪽 정보 패널)
        self.grid_columnconfigure(0, weight=3)  # 카메라 영역 (넓게)
        self.grid_columnconfigure(1, weight=1)  # 사이드 패널 (좁게)
        self.grid_rowconfigure(0, weight=1)
        
        # === 왼쪽: 카메라 영역 ===
        self.left_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # 헤더 (타이틀 + 시계)
        self._create_header(self.left_frame)
        
        # 카메라 뷰 (메인)
        self.camera_frame = ctk.CTkFrame(self.left_frame, corner_radius=15, fg_color="#1a1a1a")
        self.camera_frame.pack(fill="both", expand=True, pady=10)
        
        # 카메라 캔버스 (Tkinter Canvas 사용 - 고성능 렌더링)
        self.camera_canvas = tk.Canvas(
            self.camera_frame,
            bg="#1a1a1a",
            highlightthickness=0,
            bd=0
        )
        self.camera_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        # 오버레이 메시지 (인식 성공 시 표시)
        self.overlay_label = ctk.CTkLabel(
            self.camera_frame,
            text="",
            font=("Pretendard", 24, "bold"),
            fg_color="transparent",
            text_color="#00E676"  # 밝은 초록색
        )
        self.overlay_label.place(relx=0.5, rely=0.9, anchor="center")
        
        # === 오른쪽: 사이드 정보 패널 ===
        self.right_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_propagate(False)
        
        self._create_side_panel(self.right_frame)
        
    def _create_header(self, parent):
        """상단 헤더"""
        header = ctk.CTkFrame(parent, fg_color="transparent", height=60)
        header.pack(fill="x", pady=(0, 10))
        
        # 로고/타이틀
        title = ctk.CTkLabel(
            header, 
            text="FacePass", 
            font=("Roboto", 28, "bold"),
            text_color="#4facfe"
        )
        title.pack(side="left")
        
        subtitle = ctk.CTkLabel(
            header,
            text="AI Attendance System",
            font=("Roboto", 14),
            text_color="gray"
        )
        subtitle.pack(side="left", padx=10, pady=(10, 0))
        
        # 디지털 시계
        self.time_label = ctk.CTkLabel(
            header,
            text="00:00:00",
            font=("Roboto Mono", 24),
            text_color="#ffffff"
        )
        self.time_label.pack(side="right")
        self._update_clock()
        
    def _create_side_panel(self, parent):
        """우측 사이드 패널 구성"""
        # 1. 탭 뷰 (출석현황 / 관리자)
        self.tab_view = ctk.CTkTabview(parent, fg_color="transparent")
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_attendance = self.tab_view.add("출석 현황")
        self.tab_admin = self.tab_view.add("관리자 모드")
        
        # === 탭 1: 출석 현황 ===
        # 최근 출석 리스트 (스크롤 가능)
        self.log_scroll = ctk.CTkScrollableFrame(self.tab_attendance, label_text="실시간 로그")
        self.log_scroll.pack(fill="both", expand=True, pady=10)
        
        # 통계 요약
        self.stats_frame = ctk.CTkFrame(self.tab_attendance, height=100)
        self.stats_frame.pack(fill="x", pady=10)
        
        self.count_label = ctk.CTkLabel(
            self.stats_frame,
            text="오늘 출석: 0명",
            font=("Pretendard", 16, "bold")
        )
        self.count_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # === 탭 2: 관리자 모드 ===
        self._init_admin_tab(self.tab_admin)

    def _init_admin_tab(self, parent):
        """관리자 탭 초기화"""
        ctk.CTkLabel(parent, text="신규 사용자 등록", font=("Pretendard", 16, "bold")).pack(pady=10)
        
        self.entry_name = ctk.CTkEntry(parent, placeholder_text="이름 입력")
        self.entry_name.pack(fill="x", pady=5)
        
        self.entry_dept = ctk.CTkEntry(parent, placeholder_text="부서 입력")
        self.entry_dept.pack(fill="x", pady=5)
        
        # 카메라 캡처 버튼
        self.btn_capture = ctk.CTkButton(
            parent,
            text="📸 얼굴 촬영 및 등록",
            command=self._capture_and_register,
            fg_color="#00C853",
            hover_color="#00E676",
            height=40
        )
        self.btn_capture.pack(fill="x", pady=20)
        
        # 등록된 사용자 관리
        ctk.CTkLabel(parent, text="사용자 관리", font=("Pretendard", 16, "bold")).pack(pady=(20, 10))
        
        self.user_list = tk.Listbox(parent, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=0)
        self.user_list.pack(fill="both", expand=True, pady=5)
        self._refresh_user_list()
        
        # 삭제 버튼
        ctk.CTkButton(
            parent,
            text="선택 삭제",
            command=self._delete_user,
            fg_color="#D32F2F",
            hover_color="#E53935"
        ).pack(fill="x", pady=5)

    def _update_clock(self):
        """시계 업데이트"""
        now = datetime.now()
        self.time_label.configure(text=now.strftime("%H:%M:%S"))
        # 날짜 표시도 업데이트 가능
        self.after(1000, self._update_clock)
        
    def _setup_camera_loop(self):
        """카메라 루프 시작"""
        if self.camera.start():
            self.is_running = True
            self._update_camera_frame()
        else:
            messagebox.showerror("오류", "카메라를 시작할 수 없습니다.")
            
    def _update_camera_frame(self):
        """카메라 프레임 업데이트 및 얼굴 인식"""
        if not self.is_running:
            return
            
        frame = self.camera.read_frame()
        if frame is not None:
            self.captured_frame = frame.copy()
            
            # 얼굴 인식 수행 (비동기로 하면 더 좋지만 일단 간단히)
            # 성능을 위해 매 프레임 하지 않고 간격 조절
            # (FaceRecognitionModule 내부 최적화 활용)
            
            faces = []
            # 탭에 따라 모드 결정
            current_tab = self.tab_view.get()
            
            if current_tab == "관리자 모드":
                # 등록 모드: 얼굴 감지만 (박스 그리기용)
                faces = self.face_module.detect_faces(frame)
            else:
                # 출석 모드: 얼굴 인식
                # 약 3~4 프레임마다 인식 수행 (메인 스레드 부하 분산)
                if int(time.time() * 10) % 3 == 0:
                    self.last_faces = self.face_module.recognize_faces(frame)
                    # 인식된 얼굴 처리
                    for face in self.last_faces:
                        if face.name != "Unknown":
                            self._process_attendance(face)
                            
                faces = self.last_faces
            
            # 박스 그리기
            frame = self.face_module.draw_face_boxes(frame, faces)
            
            # 화면 표시
            self._display_frame(frame)
            
        self.after(30, self._update_camera_frame)
        
    def _display_frame(self, frame):
        """OpenCV 프레임을 Canvas에 표시"""
        # 화면 크기에 맞게 리사이즈 (Canvas 크기 기준)
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        
        # 캔버스가 아직 생성되지 않았거나 너무 작으면 기본 크기 사용
        if canvas_width < 100 or canvas_height < 100:
            canvas_width = 800
            canvas_height = 600
            
        # 비율 유지 리사이즈 계산
        h, w = frame.shape[:2]
        if w > 0 and h > 0:
            scale = min(canvas_width/w, canvas_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            if new_w > 0 and new_h > 0:
                frame = cv2.resize(frame, (new_w, new_h))
                
                # BGR -> RGB 변환
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                # 캔버스 중앙에 이미지 표시
                self.camera_canvas.create_image(
                    canvas_width//2, canvas_height//2,
                    image=img_tk, anchor="center"
                )
                self.camera_canvas.image = img_tk  # 참조 유지 (GC 방지)
        
    def _process_attendance(self, face: FaceInfo):
        """출석 처리 로직"""
        user_id = face.user_id
        current_time = time.time()
        
        # 쿨다운 체크 (10초)
        if user_id in self.recognition_cooldown:
            if current_time - self.recognition_cooldown[user_id] < 10:
                return
                
        self.recognition_cooldown[user_id] = current_time
        
        # DB 기록
        success, msg = self.db.record_attendance(user_id, "in")
        
        if success:
            # 1. UI 피드백 (오버레이)
            self._show_overlay_message(f"환영합니다, {face.name}님!")
            
            # 2. 로그 추가
            self._add_log_item(face.name, datetime.now().strftime("%H:%M:%S"))
            
            # 3. 구글 시트 업로드 (비동기)
            # 부서 정보는 DB에서 가져와야 함 (여기선 간단히 face 객체에 없으면 조회)
            user_info = self.db.get_user_by_id(user_id) # (id, name, dept, ...)
            dept = user_info[2] if user_info else ""
            
            self.sheets_manager.add_record(face.name, dept, "출근")
            
            # 4. 통계 업데이트
            self._update_stats()
            
    def _show_overlay_message(self, text):
        """화면 중앙 오버레이 메시지 표시"""
        self.overlay_label.configure(text=text)
        # 3초 후 사라짐
        self.after(3000, lambda: self.overlay_label.configure(text=""))
        
    def _add_log_item(self, name, time_str):
        """출석 로그 UI에 카드 추가"""
        card = ctk.CTkFrame(self.log_scroll, fg_color="#2b2b2b", corner_radius=10)
        card.pack(fill="x", pady=5, padx=5)
        
        ctk.CTkLabel(card, text=name, font=("bold", 14)).pack(side="left", padx=10, pady=10)
        ctk.CTkLabel(card, text=time_str, text_color="gray").pack(side="right", padx=10)
        
        # 최대 개수 유지 (20개)
        if len(self.log_scroll.winfo_children()) > 20:
            self.log_scroll.winfo_children()[0].destroy()
            
    def _load_face_data(self):
        """얼굴 데이터 로딩"""
        data = self.db.get_all_face_encodings()
        self.face_module.load_known_faces(data)
        
    def _capture_and_register(self):
        """얼굴 등록"""
        name = self.entry_name.get().strip()
        dept = self.entry_dept.get().strip()
        
        if not name:
            messagebox.showwarning("경고", "이름을 입력해주세요.")
            return
            
        if self.captured_frame is None:
            return
            
        # 얼굴 인코딩
        encoding = self.face_module.encode_face_from_frame(self.captured_frame)
        if encoding is None:
            messagebox.showerror("실패", "얼굴을 찾을 수 없습니다. 정면을 봐주세요.")
            return
            
        # 사진 저장
        photo_dir = "user_photos"
        os.makedirs(photo_dir, exist_ok=True)
        filename = f"{name}_{int(time.time())}.jpg"
        path = os.path.join(photo_dir, filename)
        cv2.imwrite(path, self.captured_frame)
        
        # DB 저장
        self.db.add_user(name, encoding, dept, path)
        self._load_face_data()
        self._refresh_user_list()
        
        messagebox.showinfo("성공", f"{name}님 등록 완료!")
        self.entry_name.delete(0, "end")
        self.entry_dept.delete(0, "end")
        
    def _refresh_user_list(self):
        """사용자 목록 UI 갱신"""
        self.user_list.delete(0, "end")
        users = self.db.get_all_users()
        for u in users:
            self.user_list.insert("end", f"{u[1]} ({u[2]})")

    def _delete_user(self):
        """사용자 삭제"""
        selection = self.user_list.curselection()
        if not selection:
            return
            
        idx = selection[0]
        users = self.db.get_all_users()
        target = users[idx]
        
        if messagebox.askyesno("삭제", f"{target[1]}님을 삭제하시겠습니까?"):
            self.db.delete_user(target[0])
            self._load_face_data()
            self._refresh_user_list()

    def _update_stats(self):
        """통계 업데이트"""
        count = self.db.get_today_attendance_count()
        self.count_label.configure(text=f"오늘 출석: {count}명")

    def _on_closing(self):
        """종료 처리"""
        self.is_running = False
        if self.camera:
            self.camera.stop()
        self.destroy()

if __name__ == "__main__":
    app = ModernFaceApp()
    app.mainloop()
