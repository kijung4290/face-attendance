# -*- coding: utf-8 -*-
"""
얼굴 인식 모듈
InsightFace 라이브러리를 사용하여 얼굴 감지 및 인식 처리
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

# InsightFace 임포트
from insightface.app import FaceAnalysis


@dataclass
class FaceInfo:
    """얼굴 정보 데이터 클래스"""
    location: Tuple[int, int, int, int]  # (top, right, bottom, left) 또는 (x1, y1, x2, y2)
    encoding: Optional[np.ndarray] = None
    name: str = "Unknown"
    user_id: Optional[int] = None
    confidence: float = 0.0
    detection_score: float = 0.0


class FaceRecognitionModule:
    """InsightFace 기반 얼굴 인식 처리 클래스"""
    
    def __init__(self, tolerance: float = 0.5, ctx_id: int = 0, det_size: tuple = (320, 320)):
        """
        Args:
            tolerance: 얼굴 매칭 허용 거리 (낮을수록 엄격)
            ctx_id: GPU ID (-1 for CPU, 0+ for GPU)
            det_size: 감지 해상도 (작을수록 빠름, 기본값 320x320으로 속도 개선)
        """
        self.tolerance = tolerance
        self.ctx_id = ctx_id
        self.det_size = det_size
        
        # InsightFace 앱 초기화
        print("InsightFace 모델 로딩 중...")
        self.app = FaceAnalysis(
            name='buffalo_s',  # 경량 모델 사용 (속도 개선)
            providers=['CPUExecutionProvider']  # CPU 사용
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("✓ InsightFace 모델 로딩 완료!")
        
        # 등록된 얼굴 데이터
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.known_face_ids: List[int] = []
    
    def load_known_faces(self, face_data: List[Tuple[int, str, np.ndarray]]):
        """
        알려진 얼굴 데이터 로드 (벡터화된 검색을 위해 행렬로 저장)
        
        Args:
            face_data: [(user_id, name, encoding), ...] 형태의 리스트
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        for user_id, name, encoding in face_data:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            self.known_face_ids.append(user_id)
        
        # 벡터화된 유사도 계산을 위한 행렬 준비
        if self.known_face_encodings:
            self.known_embeddings_matrix = np.array(self.known_face_encodings)
            # 정규화 (코사인 유사도 계산 가속화)
            norms = np.linalg.norm(self.known_embeddings_matrix, axis=1, keepdims=True)
            self.known_embeddings_normalized = self.known_embeddings_matrix / norms
        else:
            self.known_embeddings_matrix = None
            self.known_embeddings_normalized = None
        
        print(f"✓ {len(face_data)}명의 얼굴 데이터 로드 완료")
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 얼굴 임베딩 간의 코사인 유사도 계산
        
        Returns:
            유사도 (-1 ~ 1, 높을수록 유사)
        """
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2)
    
    def _compute_similarities_batch(self, embedding: np.ndarray) -> np.ndarray:
        """
        하나의 임베딩과 모든 등록된 얼굴 간의 유사도를 벡터화하여 한번에 계산
        
        Returns:
            모든 등록된 얼굴과의 유사도 배열
        """
        if self.known_embeddings_normalized is None:
            return np.array([])
        
        # 입력 임베딩 정규화
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        # 벡터화된 내적 계산 (매우 빠름)
        similarities = np.dot(self.known_embeddings_normalized, embedding_norm)
        return similarities
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceInfo]:
        """
        프레임에서 얼굴 감지
        
        Args:
            frame: BGR 형식의 이미지 프레임
            
        Returns:
            감지된 얼굴 정보 리스트
        """
        # InsightFace로 얼굴 감지
        faces = self.app.get(frame)
        
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # (top, right, bottom, left) 형식으로 변환
            location = (y1, x2, y2, x1)
            
            result.append(FaceInfo(
                location=location,
                detection_score=float(face.det_score) if hasattr(face, 'det_score') else 0.0
            ))
        
        return result
    
    def recognize_faces(self, frame: np.ndarray) -> List[FaceInfo]:
        """
        프레임에서 얼굴 인식 (감지 + 식별)
        
        Args:
            frame: BGR 형식의 이미지 프레임
            
        Returns:
            인식된 얼굴 정보 리스트 (이름, 신뢰도 포함)
        """
        # InsightFace로 얼굴 분석
        faces = self.app.get(frame)
        
        result = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            location = (y1, x2, y2, x1)  # (top, right, bottom, left)
            
            # 얼굴 임베딩 추출
            embedding = face.embedding if hasattr(face, 'embedding') else None
            
            name = "Unknown"
            user_id = None
            confidence = 0.0
            
            if embedding is not None and len(self.known_face_encodings) > 0:
                # 벡터화된 유사도 계산 (모든 등록 얼굴과 한번에 비교)
                similarities = self._compute_similarities_batch(embedding)
                
                # 가장 유사한 얼굴 찾기
                best_match_index = np.argmax(similarities)
                best_similarity = similarities[best_match_index]
                
                # 유사도 임계값 (0.5 이상이면 동일인으로 판단)
                if best_similarity >= self.tolerance:
                    name = self.known_face_names[best_match_index]
                    user_id = self.known_face_ids[best_match_index]
                    confidence = float(best_similarity)
            
            result.append(FaceInfo(
                location=location,
                encoding=embedding,
                name=name,
                user_id=user_id,
                confidence=confidence,
                detection_score=float(face.det_score) if hasattr(face, 'det_score') else 0.0
            ))
        
        return result
    
    def encode_face_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        이미지 파일에서 얼굴 임베딩 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            얼굴 임베딩 또는 None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠ 이미지를 읽을 수 없습니다: {image_path}")
                return None
            
            faces = self.app.get(image)
            
            if len(faces) == 0:
                print("⚠ 이미지에서 얼굴을 찾을 수 없습니다.")
                return None
            elif len(faces) > 1:
                print("⚠ 이미지에 여러 얼굴이 감지되었습니다. 첫 번째 얼굴을 사용합니다.")
            
            return faces[0].embedding
            
        except Exception as e:
            print(f"❌ 이미지 처리 오류: {e}")
            return None
    
    def encode_face_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 얼굴 임베딩 추출
        
        Args:
            frame: BGR 형식의 이미지 프레임
            
        Returns:
            얼굴 임베딩 또는 None
        """
        faces = self.app.get(frame)
        
        if len(faces) == 0:
            return None
        
        return faces[0].embedding
    
    def draw_face_boxes(self, frame: np.ndarray, faces: List[FaceInfo],
                        show_confidence: bool = True) -> np.ndarray:
        """
        프레임에 얼굴 박스와 이름 그리기
        
        Args:
            frame: 원본 프레임
            faces: 얼굴 정보 리스트
            show_confidence: 신뢰도 표시 여부
            
        Returns:
            박스가 그려진 프레임
        """
        result = frame.copy()
        
        for face in faces:
            top, right, bottom, left = face.location
            
            # 색상 결정 (인식됨: 초록, 미인식: 빨강)
            if face.name != "Unknown":
                color = (0, 255, 0)  # 초록색 (BGR)
            else:
                color = (0, 0, 255)  # 빨간색 (BGR)
            
            # 얼굴 박스 그리기
            cv2.rectangle(result, (left, top), (right, bottom), color, 2)
            
            # 이름 표시 영역
            label_height = 35
            cv2.rectangle(result, (left, bottom), (right, bottom + label_height), color, cv2.FILLED)
            
            # 이름 텍스트
            display_text = face.name
            if show_confidence and face.confidence > 0:
                display_text += f" ({face.confidence:.0%})"
            
            # 텍스트 그리기
            cv2.putText(result, display_text, (left + 6, bottom + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return result


class CameraManager:
    """카메라 관리 클래스"""
    
    def __init__(self, camera_index: int = 0):
        """
        Args:
            camera_index: 카메라 인덱스 (기본값 0)
        """
        self.camera_index = camera_index
        self.cap = None
    
    def start(self) -> bool:
        """카메라 시작"""
        import time
        
        # Windows에서는 DirectShow 백엔드 사용
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # 카메라 초기화 대기
        time.sleep(0.5)
        
        if not self.cap.isOpened():
            # DirectShow 실패 시 기본 백엔드로 재시도
            print(f"⚠ DirectShow 실패, 기본 백엔드로 재시도...")
            self.cap = cv2.VideoCapture(self.camera_index)
            time.sleep(0.5)
            
            if not self.cap.isOpened():
                print(f"❌ 카메라 {self.camera_index}를 열 수 없습니다.")
                return False
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 첫 프레임 읽기 시도 (카메라 웜업)
        for _ in range(5):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)
        
        print(f"✓ 카메라 {self.camera_index} 시작됨")
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """프레임 읽기"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        return frame
    
    def stop(self):
        """카메라 정지"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("카메라 정지됨")
    
    def is_opened(self) -> bool:
        """카메라 상태 확인"""
        return self.cap is not None and self.cap.isOpened()


# 테스트 코드
if __name__ == "__main__":
    print("InsightFace 기반 얼굴 인식 모듈 테스트")
    
    camera = CameraManager()
    face_module = FaceRecognitionModule(tolerance=0.5)
    
    if camera.start():
        print("카메라가 시작되었습니다. 'q'를 눌러 종료하세요.")
        
        while True:
            frame = camera.read_frame()
            if frame is None:
                continue
            
            # 얼굴 감지
            faces = face_module.detect_faces(frame)
            
            # 박스 그리기
            result = face_module.draw_face_boxes(frame, faces, show_confidence=False)
            
            # 정보 표시
            cv2.putText(result, f"Detected: {len(faces)} faces", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Face Detection Test", result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.stop()
        cv2.destroyAllWindows()
