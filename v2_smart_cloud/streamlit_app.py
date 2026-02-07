# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image

# 기존 모듈 임포트
from face_recognition_module import FaceRecognitionModule, FaceInfo
from google_sheets import GoogleSheetsManager
from database import DatabaseManager

# 페이지 설정
st.set_page_config(
    page_title="FacePass Cloud",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 세션 상태 초기화
if 'face_module' not in st.session_state:
    with st.spinner('AI 모델 로딩 중... (처음엔 좀 걸려요 😅)'):
        # Render 무료 서버 성능 고려하여 가장 가벼운 설정
        st.session_state.face_module = FaceRecognitionModule(
            tolerance=0.45, 
            det_size=(320, 320),
            ctx_id=-1  # CPU 모드 강제
        )
        
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()
    
if 'sheets' not in st.session_state:
    st.session_state.sheets = GoogleSheetsManager()
    
# 얼굴 데이터 로드 (매번 최신 상태 유지)
if 'data_loaded' not in st.session_state:
    data = st.session_state.db.get_all_face_encodings()
    st.session_state.face_module.load_known_faces(data)
    st.session_state.data_loaded = True

# --- 메인 UI ---

st.title("🎯 FacePass Cloud Attendance")
st.markdown("웹캠으로 얼굴을 인증하고 출석을 체크하세요!")

# 탭 구성
tab1, tab2 = st.tabs(["📷 출석 체크", "⚙️ 관리자 모드"])

# 1. 출석 체크 탭
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("카메라 인증")
        
        # 1-1. 카메라 입력 받기
        img_file_buffer = st.camera_input("얼굴을 정면으로 보여주세요", key="camera")
        
        if img_file_buffer is not None:
            # 이미지 변환 (Streamlit -> OpenCV)
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # 얼굴 인식 수행
            with st.spinner("얼굴 분석 중..."):
                faces = st.session_state.face_module.recognize_faces(cv2_img)
                
                # 결과 표시용 이미지 복사
                result_img = cv2_img.copy()
                result_img = st.session_state.face_module.draw_face_boxes(result_img, faces)
                
                # BGR -> RGB 변환
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)
                
                # 인식 결과 처리
                unknown_faces = 0
                recognized_faces = []
                
                for face in faces:
                    if face.name != "Unknown":
                        recognized_faces.append(face.name)
                        
                        # 출석 처리 (쿨다운 없이 즉시 처리 - 웹 특성상)
                        user_id = face.user_id
                        
                        # DB 기록
                        success, msg = st.session_state.db.record_attendance(user_id, "in")
                        
                        # 시트 기록
                        user_info = st.session_state.db.get_user_by_id(user_id)
                        dept = user_info[2] if user_info else ""
                        st.session_state.sheets.add_record(face.name, dept, "출근(웹)")
                        
                    else:
                        unknown_faces += 1
                
                # 메시지 표시
                if recognized_faces:
                    st.success(f"✅ 환영합니다! {', '.join(recognized_faces)}님 출석 완료되었습니다.")
                    st.balloons()
                elif unknown_faces > 0:
                    st.warning("⚠️ 얼굴은 감지되었으나 등록되지 않은 사용자입니다.")
                else:
                    st.error("❌ 얼굴을 찾을 수 없습니다. 정면을 바라봐주세요.")

    with col2:
        st.subheader("📊 실시간 현황")
        
        # 구글 시트 링크 (있으면)
        st.markdown(f"[Google Sheet 보기](https://docs.google.com/spreadsheets/d/your-sheet-id)")
        
        # 최근 출석자 표시 (DB 기준)
        today_count = st.session_state.db.get_today_attendance_count()
        st.metric("오늘 출석 인원", f"{today_count}명")
        
        limit = 10
        st.caption(f"최근 {limit}명 (로컬 기록)")
        records = st.session_state.db.get_attendance_by_date(datetime.now().date())
        
        if records:
            for rec in records[-limit:]: # 뒤에서부터 (최신순 아님, 보통 DB는 입력순)
                st.info(f"{rec[0]} ({rec[2].split()[1] if rec[2] else '?'})")
        else:
            st.text("아직 출석 기록이 없습니다.")

# 2. 관리자 모드 탭
with tab2:
    st.subheader("관리자 로그인")
    password = st.text_input("비밀번호", type="password")
    
    if password == "1234":  # 임시 비밀번호
        st.success("로그인 성공")
        
        st.divider()
        st.subheader("➕ 신규 사용자 등록")
        
        with st.form("register_form"):
            new_name = st.text_input("이름")
            new_dept = st.text_input("부서")
            uploaded_file = st.file_uploader("사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])
            
            submit_btn = st.form_submit_button("등록하기")
            
            if submit_btn and new_name and uploaded_file:
                # 파일 처리
                bytes_data = uploaded_file.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # 얼굴 추출
                embedding = st.session_state.face_module.encode_face_from_frame(cv2_img)
                
                if embedding is not None:
                    # 저장
                    save_path = f"user_photos/{new_name}_{int(time.time())}.jpg"
                    if not os.path.exists("user_photos"):
                        os.makedirs("user_photos")
                        
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # DB 저장
                    st.session_state.db.add_user(new_name, embedding, new_dept, save_path)
                    
                    # 메모리 갱신
                    data = st.session_state.db.get_all_face_encodings()
                    st.session_state.face_module.load_known_faces(data)
                    
                    st.success(f"{new_name}님 등록이 완료되었습니다!")
                    st.rerun()  # 화면 갱신
                else:
                    st.error("사진에서 얼굴을 찾을 수 없습니다.")
                    
        st.divider()
        st.subheader("👥 사용자 목록")
        users = st.session_state.db.get_all_users()
        for u in users:
            col_a, col_b, col_c = st.columns([2, 2, 1])
            col_a.text(u[1]) # 이름
            col_b.text(u[2]) # 부서
            if col_c.button("삭제", key=f"del_{u[0]}"):
                st.session_state.db.delete_user(u[0])
                st.rerun()
                
    elif password:
        st.error("비밀번호가 틀렸습니다.")

# 하단 푸터
st.markdown("---")
st.caption("Powered by InsightFace & Streamlit | Render Deployment Ready")
