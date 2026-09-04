"""
원클릭 AI 다중 생체신호 모니터링 시스템 & 외부 접속 터널 런처
- Streamlit 대시보드(senior_app.py)를 로컬 포트 8501로 자동 실행
- Cloudflare 터널(cloudflared.exe)을 자동 연동하여 외부 접속용 보안 HTTPS 링크 즉시 발급
- 터미널에 접속 링크를 크고 선명하게 표시하고 tunnel_url.txt 파일로 저장
"""

import os
import sys
import time
import subprocess
import re
import socket
from pathlib import Path

# UTF-8 콘솔 인코딩 보장
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED_PATH = BASE_DIR / "cloudflared.exe"
LOG_PATH = BASE_DIR / "tunnel.log"
URL_FILE_PATH = BASE_DIR / "tunnel_url.txt"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def main():
    print("=" * 70, flush=True)
    print(" 🧠 AI 다중 생체신호(안면·음성) 컨디션 & 인지 모니터 - 외부 접속 런처", flush=True)
    print("=" * 70, flush=True)

    # 1. Cloudflare 실행 파일 확인
    if not CLOUDFLARED_PATH.exists():
        print(f"❌ cloudflared.exe를 찾을 수 없습니다: {CLOUDFLARED_PATH}", flush=True)
        return

    # 기존 로그 파일 정리
    if LOG_PATH.exists():
        try:
            LOG_PATH.unlink()
        except Exception:
            pass

    # 2. Streamlit 서버 시작
    streamlit_proc = None
    if not is_port_in_use(8501):
        print("🚀 [1/2] 로컬 Streamlit 서버 가동 중...", flush=True)
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(BASE_DIR / "senior_app.py"),
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        streamlit_proc = subprocess.Popen(
            streamlit_cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(25):
            if is_port_in_use(8501):
                break
            time.sleep(0.4)
        print("✅ [1/2] Streamlit 서버 정상 가동 완료 (http://localhost:8501)", flush=True)
    else:
        print("ℹ️ [1/2] 이미 포트 8501에서 서버가 실행 중입니다.", flush=True)

    # 3. Cloudflare 터널 시작
    print("🌐 [2/2] Cloudflare 보안 터널 생성 및 외부 접속 주소 발급 중...", flush=True)
    tunnel_cmd = [
        str(CLOUDFLARED_PATH),
        "tunnel",
        "--url", "http://localhost:8501",
        "--logfile", str(LOG_PATH)
    ]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # tunnel.log를 모니터링하여 URL 캡처
    start_time = time.time()
    while time.time() - start_time < 30:
        if LOG_PATH.exists():
            try:
                with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    match = url_pattern.search(content)
                    if match:
                        tunnel_url = match.group(0)
                        break
            except Exception:
                pass
        
        if tunnel_proc.poll() is not None:
            print("⚠️ cloudflared 프로세스가 조기 종료되었습니다.", flush=True)
            break
        time.sleep(0.5)

    if tunnel_url:
        # 텍스트 파일로도 저장
        with open(URL_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(tunnel_url.strip())

        print("\n" + "=" * 70, flush=True)
        print("  🎉 외부 접속 터널이 성공적으로 활성화되었습니다!", flush=True)
        print("=" * 70, flush=True)
        print(f"\n  📱 [외부 / 스마트폰 접속 링크] (전 세계 어디서나):", flush=True)
        print(f"     👉  {tunnel_url}  👈", flush=True)
        print(f"\n  💻 [내 컴퓨터(로컬) 접속 링크]:", flush=True)
        print(f"     👉  http://localhost:8501", flush=True)
        print("\n" + "-" * 70, flush=True)
        print("  💡 스마트폰 브라우저나 외부 컴퓨터에서 위 [외부 접속 링크]를 누르면", flush=True)
        print("     실시간 안면인식 & 생체 바이오마커 화면을 그대로 확인하실 수 있습니다.", flush=True)
        print("  ⚠️ 주의: 이 창을 닫거나 컴퓨터를 끄면 외부 접속이 중단됩니다.", flush=True)
        print("=" * 70 + "\n", flush=True)
        print("프로그램을 종료하려면 이 창을 닫거나 Ctrl + C 를 누르세요...\n", flush=True)
    else:
        print("\n❌ 외부 접속 주소 발급에 실패했습니다.", flush=True)
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                print("최근 로그:\n", f.read()[-500:])

    try:
        while True:
            if tunnel_proc.poll() is not None:
                print("⚠️ 터널 프로세스가 종료되었습니다.", flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 시스템을 정상 종료하는 중입니다...", flush=True)
    finally:
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
        if streamlit_proc and streamlit_proc.poll() is None:
            streamlit_proc.terminate()
        print("👋 모든 프로세스가 안전하게 종료되었습니다.", flush=True)

if __name__ == "__main__":
    main()
