# -*- coding: utf-8 -*-
"""
얼굴인식 출석체크 프로그램 - 로컬 DB 버전
초기 Tkinter 기반 그래픽 인터페이스
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import threading
from datetime import datetime, date
from typing import Optional
import numpy as np

from database import DatabaseManager
from face_recognition_module import FaceRecognitionModule, CameraManager, FaceInfo


class FaceAttendanceApp:
    """얼굴인식 출석체크 메인 애플리케이션 (v1)"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎯 얼굴인식 출석체크 시스템 (Local)")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)  # 최소 창 크기 설정
        self.root.configure(bg='#1a1a2e')
        
        # 모듈 초기화
        self.db = DatabaseManager()
        self.face_module = FaceRecognitionModule(tolerance=0.5, det_size=(320, 320))
        self.camera = CameraManager()
        
        # 얼굴 데이터 로드
        self._load_face_data()
        
        # 상태 변수
        self.is_running = False
        self.current_mode = "attendance"  # "attendance" or "register"
        self.register_name = ""
        self.register_department = ""
        self.captured_frame = None
        self.recognition_cooldown = {}  # 중복 인식 방지
        
        # 프레임 처리 최적화
        self.frame_count = 0
        self.process_every_n_frames = 3  # 3프레임마다 얼굴 인식 수행
        self.last_faces = []  # 마지막 인식 결과 캐싱
        self.recognition_scale = 0.5
        
        # 사진 저장 폴더
        self.photos_dir = "user_photos"
        os.makedirs(self.photos_dir, exist_ok=True)
        
        # UI 구성
        self._setup_styles()
        self._create_ui()
        
        # 창 닫기 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_styles(self):
        """커스텀 스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 버튼 스타일
        style.configure('Primary.TButton',
                       background='#4361ee',
                       foreground='white',
                       font=('맑은 고딕', 12, 'bold'),
                       padding=(20, 10))
        
        style.configure('Success.TButton',
                       background='#2ecc71',
                       foreground='white',
                       font=('맑은 고딕', 12, 'bold'),
                       padding=(20, 10))
        
        style.configure('Danger.TButton',
                       background='#e74c3c',
                       foreground='white',
                       font=('맑은 고딕', 12, 'bold'),
                       padding=(20, 10))
        
        style.configure('Info.TLabel',
                       background='#1a1a2e',
                       foreground='#ffffff',
                       font=('맑은 고딕', 11))
    
    def _create_ui(self):
        """UI 생성"""
        # 메인 컨테이너
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 헤더
        self._create_header(main_container)
        
        # 콘텐츠 영역 (좌: 카메라, 우: 패널)
        content_frame = tk.Frame(main_container, bg='#1a1a2e')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 오른쪽: 컨트롤 패널 (먼저 배치해야 잘림 방지)
        self._create_control_panel(content_frame)
        
        # 왼쪽: 카메라 뷰
        self._create_camera_view(content_frame)
        
        # 하단: 출석 현황
        self._create_attendance_panel(main_container)
    
    def _create_header(self, parent):
        """헤더 생성"""
        header_frame = tk.Frame(parent, bg='#16213e', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # 타이틀
        title_label = tk.Label(
            header_frame,
            text="🎯 얼굴인식 출석체크 시스템",
            font=('맑은 고딕', 24, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)
        
        # 현재 시간 표시
        self.time_label = tk.Label(
            header_frame,
            text="",
            font=('맑은 고딕', 16),
            bg='#16213e',
            fg='#4cc9f0'
        )
        self.time_label.pack(side=tk.RIGHT, padx=30, pady=20)
        self._update_time()
        
        # 통계 표시
        self.stats_label = tk.Label(
            header_frame,
            text="",
            font=('맑은 고딕', 12),
            bg='#16213e',
            fg='#b8c5d6'
        )
        self.stats_label.pack(side=tk.RIGHT, padx=20, pady=20)
        self._update_stats()
    
    def _create_camera_view(self, parent):
        """카메라 뷰 생성"""
        camera_frame = tk.Frame(parent, bg='#0f0f23', relief=tk.RAISED)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 카메라 레이블
        camera_title = tk.Label(
            camera_frame,
            text="📷 카메라 화면",
            font=('맑은 고딕', 14, 'bold'),
            bg='#0f0f23',
            fg='#ffffff'
        )
        camera_title.pack(pady=10)
        
        # 카메라 캔버스 (Canvas 사용하여 정확한 픽셀 크기 지정)
        self.camera_canvas = tk.Canvas(
            camera_frame,
            bg='#000000',
            width=640,
            height=480,
            highlightthickness=0
        )
        self.camera_canvas.pack(padx=20, pady=10)
        
        # 카메라 이미지 표시용 레이블 (Canvas 위에)
        self.camera_label = tk.Label(
            self.camera_canvas,
            bg='#000000'
        )
        self.camera_label.place(x=0, y=0, width=640, height=480)
        
        # 상태 메시지
        self.status_label = tk.Label(
            camera_frame,
            text="카메라가 꺼져 있습니다",
            font=('맑은 고딕', 12),
            bg='#0f0f23',
            fg='#f8961e'
        )
        self.status_label.pack(pady=10)
        
        # 인식 결과 메시지
        self.result_label = tk.Label(
            camera_frame,
            text="",
            font=('맑은 고딕', 16, 'bold'),
            bg='#0f0f23',
            fg='#2ecc71'
        )
        self.result_label.pack(pady=5)
    
    def _create_control_panel(self, parent):
        """컨트롤 패널 생성"""
        control_frame = tk.Frame(parent, bg='#16213e', width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # 모드 선택
        mode_label = tk.Label(
            control_frame,
            text="🔧 모드 선택",
            font=('맑은 고딕', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        mode_label.pack(pady=(20, 10))
        
        self.mode_var = tk.StringVar(value="attendance")
        
        attendance_radio = tk.Radiobutton(
            control_frame,
            text="📋 출석 체크 모드",
            variable=self.mode_var,
            value="attendance",
            font=('맑은 고딕', 11),
            bg='#16213e',
            fg='#ffffff',
            selectcolor='#4361ee',
            activebackground='#16213e',
            activeforeground='#ffffff',
            command=self._on_mode_change
        )
        attendance_radio.pack(anchor=tk.W, padx=30, pady=5)
        
        register_radio = tk.Radiobutton(
            control_frame,
            text="➕ 얼굴 등록 모드",
            variable=self.mode_var,
            value="register",
            font=('맑은 고딕', 11),
            bg='#16213e',
            fg='#ffffff',
            selectcolor='#4361ee',
            activebackground='#16213e',
            activeforeground='#ffffff',
            command=self._on_mode_change
        )
        register_radio.pack(anchor=tk.W, padx=30, pady=5)
        
        # 구분선
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, padx=20, pady=20)
        
        # 카메라 제어 버튼
        camera_label = tk.Label(
            control_frame,
            text="📹 카메라 제어",
            font=('맑은 고딕', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        camera_label.pack(pady=(0, 10))
        
        self.camera_btn = tk.Button(
            control_frame,
            text="▶ 카메라 시작",
            font=('맑은 고딕', 12, 'bold'),
            bg='#4361ee',
            fg='white',
            activebackground='#3a56d4',
            activeforeground='white',
            width=20,
            height=2,
            command=self._toggle_camera
        )
        self.camera_btn.pack(pady=10)
        
        # 등록 모드 전용 컨트롤 (항상 이 위치에 배치)
        self.register_frame = tk.Frame(control_frame, bg='#16213e')
        # 처음에는 숨김 상태로 시작 (나중에 pack할 때 before 옵션 사용)
        
        # 등록 프레임 위치 마커 (이 위젯 앞에 등록 폼이 배치됨)
        self.register_marker = tk.Frame(control_frame, height=0, bg='#16213e')
        self.register_marker.pack(fill=tk.X)
        
        tk.Label(
            self.register_frame,
            text="이름:",
            font=('맑은 고딕', 11),
            bg='#16213e',
            fg='#ffffff'
        ).pack(anchor=tk.W)
        
        self.name_entry = tk.Entry(
            self.register_frame,
            font=('맑은 고딕', 12),
            width=25
        )
        self.name_entry.pack(pady=5)
        
        tk.Label(
            self.register_frame,
            text="부서:",
            font=('맑은 고딕', 11),
            bg='#16213e',
            fg='#ffffff'
        ).pack(anchor=tk.W, pady=(10, 0))
        
        self.dept_entry = tk.Entry(
            self.register_frame,
            font=('맑은 고딕', 12),
            width=25
        )
        self.dept_entry.pack(pady=5)
        
        self.capture_btn = tk.Button(
            self.register_frame,
            text="📸 얼굴 캡처 및 등록",
            font=('맑은 고딕', 12, 'bold'),
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            activeforeground='white',
            width=20,
            height=2,
            command=self._capture_and_register
        )
        self.capture_btn.pack(pady=10)
        
        # 초기에는 등록 프레임 숨김 (register_marker 앞에 pack됨)
        
        # 구분선
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # 사용자 관리
        manage_label = tk.Label(
            control_frame,
            text="👥 등록된 사용자",
            font=('맑은 고딕', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        manage_label.pack(pady=(0, 10))
        
        # 사용자 리스트박스
        list_frame = tk.Frame(control_frame, bg='#16213e')
        list_frame.pack(fill=tk.X, padx=20)
        
        self.user_listbox = tk.Listbox(
            list_frame,
            font=('맑은 고딕', 10),
            height=4,
            width=30,
            bg='#0f0f23',
            fg='#ffffff',
            selectbackground='#4361ee'
        )
        self.user_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.user_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.user_listbox.yview)
        
        self._refresh_user_list()
        
        # 삭제 버튼
        delete_btn = tk.Button(
            control_frame,
            text="🗑 선택 사용자 삭제",
            font=('맑은 고딕', 11),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            width=20,
            command=self._delete_selected_user
        )
        delete_btn.pack(pady=10)
    
    def _create_attendance_panel(self, parent):
        """출석 현황 패널 생성"""
        attendance_frame = tk.Frame(parent, bg='#16213e', height=200)
        attendance_frame.pack(fill=tk.X, pady=(10, 0))
        attendance_frame.pack_propagate(False)
        
        # 헤더
        header = tk.Frame(attendance_frame, bg='#16213e')
        header.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            header,
            text="📊 오늘의 출석 현황",
            font=('맑은 고딕', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side=tk.LEFT)
        
        refresh_btn = tk.Button(
            header,
            text="🔄 새로고침",
            font=('맑은 고딕', 10),
            bg='#4cc9f0',
            fg='white',
            command=self._refresh_attendance
        )
        refresh_btn.pack(side=tk.RIGHT)
        
        # 출석 테이블
        columns = ('name', 'department', 'check_in', 'check_out', 'status')
        
        self.attendance_tree = ttk.Treeview(
            attendance_frame,
            columns=columns,
            show='headings',
            height=5
        )
        
        self.attendance_tree.heading('name', text='이름')
        self.attendance_tree.heading('department', text='부서')
        self.attendance_tree.heading('check_in', text='출근 시간')
        self.attendance_tree.heading('check_out', text='퇴근 시간')
        self.attendance_tree.heading('status', text='상태')
        
        self.attendance_tree.column('name', width=150, anchor='center')
        self.attendance_tree.column('department', width=150, anchor='center')
        self.attendance_tree.column('check_in', width=150, anchor='center')
        self.attendance_tree.column('check_out', width=150, anchor='center')
        self.attendance_tree.column('status', width=100, anchor='center')
        
        self.attendance_tree.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self._refresh_attendance()
    
    def _load_face_data(self):
        """데이터베이스에서 얼굴 데이터 로드"""
        face_data = self.db.get_all_face_encodings()
        self.face_module.load_known_faces(face_data)
    
    def _update_time(self):
        """시간 업데이트"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self._update_time)
    
    def _update_stats(self):
        """통계 업데이트"""
        total_users = self.db.get_total_user_count()
        today_attendance = self.db.get_today_attendance_count()
        self.stats_label.config(text=f"등록: {total_users}명 | 오늘 출석: {today_attendance}명")
    
    def _on_mode_change(self):
        """모드 변경 시 처리"""
        mode = self.mode_var.get()
        self.current_mode = mode
        
        if mode == "register":
            # 마커 앞에 등록 폼 배치 (위치 고정)
            self.register_frame.pack(fill=tk.X, padx=20, pady=10, before=self.register_marker)
            self.status_label.config(text="등록 모드: 이름과 부서를 입력 후 얼굴을 캡처하세요")
        else:
            self.register_frame.pack_forget()
            self.status_label.config(text="출석 체크 모드: 카메라에 얼굴을 보여주세요")
    
    def _toggle_camera(self):
        """카메라 시작/정지 토글"""
        if self.is_running:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """카메라 시작"""
        if self.camera.start():
            self.is_running = True
            self.camera_btn.config(text="⏹ 카메라 정지", bg='#e74c3c')
            self.status_label.config(text="카메라 작동 중...", fg='#2ecc71')
            self._update_camera()
        else:
            messagebox.showerror("오류", "카메라를 시작할 수 없습니다.")
    
    def _stop_camera(self):
        """카메라 정지"""
        self.is_running = False
        self.camera.stop()
        self.camera_btn.config(text="▶ 카메라 시작", bg='#4361ee')
        self.status_label.config(text="카메라가 꺼져 있습니다", fg='#f8961e')
        
        # 카메라 레이블 초기화
        self.camera_label.config(image='')
    
    def _update_camera(self):
        """카메라 프레임 업데이트"""
        if not self.is_running:
            return
        
        try:
            frame = self.camera.read_frame()
            
            if frame is not None:
                self.captured_frame = frame.copy()
                self.frame_count += 1
                
                # N 프레임마다 얼굴 인식 수행 (성능 최적화)
                if self.frame_count % self.process_every_n_frames == 0:
                    # 얼굴 인식용 축소 프레임 생성 (속도 향상)
                    small_frame = cv2.resize(frame, None, 
                                            fx=self.recognition_scale, 
                                            fy=self.recognition_scale,
                                            interpolation=cv2.INTER_AREA)
                    
                    if self.current_mode == "attendance":
                        # 출석 체크 모드: 얼굴 인식 (축소 프레임 사용)
                        small_faces = self.face_module.recognize_faces(small_frame)
                        
                        # 좌표를 원본 크기로 스케일 업
                        self.last_faces = []
                        scale_factor = 1.0 / self.recognition_scale
                        for face in small_faces:
                            top, right, bottom, left = face.location
                            scaled_location = (
                                int(top * scale_factor),
                                int(right * scale_factor),
                                int(bottom * scale_factor),
                                int(left * scale_factor)
                            )
                            from face_recognition_module import FaceInfo
                            self.last_faces.append(FaceInfo(
                                location=scaled_location,
                                encoding=face.encoding,
                                name=face.name,
                                user_id=face.user_id,
                                confidence=face.confidence,
                                detection_score=face.detection_score
                            ))
                            
                            if face.user_id is not None and face.name != "Unknown":
                                self._process_attendance(face)
                    else:
                        # 등록 모드: 얼굴 감지만 (축소 프레임 사용)
                        small_faces = self.face_module.detect_faces(small_frame)
                        
                        # 좌표를 원본 크기로 스케일 업
                        self.last_faces = []
                        scale_factor = 1.0 / self.recognition_scale
                        for face in small_faces:
                            top, right, bottom, left = face.location
                            scaled_location = (
                                int(top * scale_factor),
                                int(right * scale_factor),
                                int(bottom * scale_factor),
                                int(left * scale_factor)
                            )
                            from face_recognition_module import FaceInfo
                            self.last_faces.append(FaceInfo(
                                location=scaled_location,
                                encoding=face.encoding,
                                name=face.name,
                                user_id=face.user_id,
                                confidence=face.confidence,
                                detection_score=face.detection_score
                            ))
                
                # 캐싱된 얼굴 결과로 박스 그리기
                if self.last_faces:
                    if self.current_mode == "attendance":
                        frame = self.face_module.draw_face_boxes(frame, self.last_faces)
                    else:
                        frame = self.face_module.draw_face_boxes(frame, self.last_faces, show_confidence=False)
                
                # tkinter용 이미지로 변환 (PIL 직접 변환으로 최적화)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
        except Exception as e:
            print(f"⚠ 카메라 업데이트 오류: {e}")
        
        # 다음 프레임 예약 (약 30fps)
        self.root.after(33, self._update_camera)
    
    def _process_attendance(self, face: FaceInfo):
        """출석 처리"""
        import time
        
        current_time = time.time()
        user_id = face.user_id
        
        # 5초 쿨다운 (같은 사람 중복 인식 방지)
        if user_id in self.recognition_cooldown:
            if current_time - self.recognition_cooldown[user_id] < 5:
                return
        
        self.recognition_cooldown[user_id] = current_time
        
        # 출석 기록
        success, message = self.db.record_attendance(user_id, "in")
        
        if success:
            self.result_label.config(
                text=f"✅ {face.name}님 {message}",
                fg='#2ecc71'
            )
            self._update_stats()
            self._refresh_attendance()
        else:
            self.result_label.config(
                text=f"ℹ️ {face.name}님: {message}",
                fg='#f8961e'
            )
        
        # 3초 후 메시지 지우기
        self.root.after(3000, lambda: self.result_label.config(text=""))
    
    def _capture_and_register(self):
        """얼굴 캡처 및 등록"""
        name = self.name_entry.get().strip()
        department = self.dept_entry.get().strip()
        
        if not name:
            messagebox.showwarning("경고", "이름을 입력해주세요.")
            return
        
        if self.captured_frame is None:
            messagebox.showwarning("경고", "카메라를 먼저 시작해주세요.")
            return
        
        # 얼굴 인코딩 추출
        encoding = self.face_module.encode_face_from_frame(self.captured_frame)
        
        if encoding is None:
            messagebox.showwarning("경고", "얼굴을 인식할 수 없습니다.\n카메라에 얼굴이 잘 보이도록 해주세요.")
            return
        
        # 사진 저장
        photo_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join(self.photos_dir, photo_filename)
        cv2.imwrite(photo_path, self.captured_frame)
        
        # 데이터베이스에 저장
        try:
            user_id = self.db.add_user(name, encoding, department, photo_path)
            
            # 얼굴 데이터 리로드
            self._load_face_data()
            
            # UI 업데이트
            self._refresh_user_list()
            self._update_stats()
            
            # 입력 필드 초기화
            self.name_entry.delete(0, tk.END)
            self.dept_entry.delete(0, tk.END)
            
            messagebox.showinfo("성공", f"{name}님의 얼굴이 등록되었습니다!")
            self.result_label.config(
                text=f"✅ {name}님 등록 완료!",
                fg='#2ecc71'
            )
            
        except Exception as e:
            messagebox.showerror("오류", f"등록 중 오류가 발생했습니다: {e}")
    
    def _refresh_user_list(self):
        """사용자 목록 새로고침"""
        self.user_listbox.delete(0, tk.END)
        
        users = self.db.get_all_users()
        for user in users:
            user_id, name, department, _, _, created_at = user
            display = f"{name} ({department or '부서없음'})"
            self.user_listbox.insert(tk.END, display)
    
    def _delete_selected_user(self):
        """선택된 사용자 삭제"""
        selection = self.user_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("경고", "삭제할 사용자를 선택해주세요.")
            return
        
        users = self.db.get_all_users()
        selected_user = users[selection[0]]
        user_id, name = selected_user[0], selected_user[1]
        
        if messagebox.askyesno("확인", f"{name}님을 정말 삭제하시겠습니까?\n출석 기록도 함께 삭제됩니다."):
            if self.db.delete_user(user_id):
                self._load_face_data()
                self._refresh_user_list()
                self._update_stats()
                self._refresh_attendance()
                messagebox.showinfo("성공", f"{name}님이 삭제되었습니다.")
            else:
                messagebox.showerror("오류", "삭제 중 오류가 발생했습니다.")
    
    def _refresh_attendance(self):
        """출석 현황 새로고침"""
        # 기존 데이터 삭제
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # 오늘 출석 기록 조회
        records = self.db.get_attendance_by_date(date.today())
        
        for record in records:
            name, department, check_in, check_out, status = record
            
            # 시간 포맷팅
            check_in_str = check_in.split()[1] if check_in else "-"
            check_out_str = check_out.split()[1] if check_out else "-"
            
            # 상태 한글화
            status_kr = "출근" if status == "present" else status
            if check_out:
                status_kr = "정상 근무"
            
            self.attendance_tree.insert('', tk.END, values=(
                name,
                department or "-",
                check_in_str,
                check_out_str,
                status_kr
            ))
    
    def _on_closing(self):
        """창 닫기 처리"""
        self._stop_camera()
        self.db.close()
        self.root.destroy()


def main():
    """메인 함수"""
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
