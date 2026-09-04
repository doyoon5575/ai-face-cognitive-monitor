"""
카메라 디바이스 진단 스크립트: 연결된 카메라 ID(0, 1, 2...) 탐색 및 테스트
"""
import cv2

def scan_cameras():
    print("=== 카메라 디바이스 진단 시작 ===")
    available_cameras = []
    
    for cam_id in range(4):
        for backend_name, backend in [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("DEFAULT", cv2.CAP_ANY)]:
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w, _ = frame.shape
                    print(f"[O] 카메라 ID {cam_id} ({backend_name}) 작동 성공! 해상도: {w}x{h}")
                    available_cameras.append((cam_id, backend_name, (w, h)))
                    cap.release()
                    break
                else:
                    print(f"[-] 카메라 ID {cam_id} ({backend_name}) 열렸으나 프레임 읽기 실패 (ret={ret})")
                cap.release()
            else:
                pass
                
    print("\n=== 사용 가능한 카메라 목록 ===")
    if available_cameras:
        for cam_id, backend_name, res in available_cameras:
            print(f" -> 카메라 ID {cam_id}: {backend_name}, 해상도 {res[0]}x{res[1]}")
    else:
        print("[!] 사용 가능한 웹캠을 찾지 못했습니다. Windows 카메라 개인정보 설정에서 앱 권한을 확인해주세요.")
    return available_cameras

if __name__ == "__main__":
    scan_cameras()
