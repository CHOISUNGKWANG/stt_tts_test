import os
import streamlit as st
import requests
import numpy as np
from openai import AzureOpenAI
from scipy.io.wavfile import write
import io
import tempfile
from datetime import datetime

# ============================================
# Streamlit 페이지 설정
# ============================================
st.set_page_config(
    page_title="🎙️ Voice Converter",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CSS 스타일링
# ============================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .record-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .record-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stop-btn {
        background: #ff6b6b;
        color: white;
    }
    
    .title-section {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .title-section h1 {
        font-size: 48px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .title-section p {
        color: #666;
        font-size: 16px;
    }
    
    .status-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 14px;
    }
    
    .result-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e9ecef;
    }
    
    .section-label {
        font-size: 14px;
        font-weight: 600;
        color: #667eea;
        margin-top: 25px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .text-display {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
        color: #333;
        margin: 10px 0;
        min-height: 50px;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 세션 상태 초기화
# ============================================
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = None
if "tts_file" not in st.session_state:
    st.session_state.tts_file = None

# ============================================
# Azure OpenAI 클라이언트 설정
# ============================================
@st.cache_resource
def get_stt_client():
    return AzureOpenAI(
        api_key=st.secrets.get("stt_apikey"),
        api_version="2024-06-01",
        azure_endpoint=st.secrets.get("stt_endpoint")
    )

# ============================================
# STT 함수
# ============================================
def speech_to_text(audio_bytes, sample_rate=16000):
    """음성을 텍스트로 변환"""
    try:
        client = get_stt_client()
        
        # 바이너리 데이터를 WAV 파일로 변환
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            write(tmp_file.name, sample_rate, audio_bytes)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper",
                language='ko'
            )
        
        os.remove(tmp_file_path)
        return result.text
    
    except Exception as e:
        st.error(f"STT 에러: {e}")
        return None

# ============================================
# TTS 함수
# ============================================
def text_to_speech(text, voice="nova"):
    """텍스트를 음성으로 변환"""
    try:
        headers = {
            'Content-Type': 'application/json',
            'api-key': st.secrets.get("tts_apikey")
        }
        
        payload = {
            "model": "gpt-4o-mini-tts",
            "input": text,
            "voice": voice
        }
        
        response = requests.post(
            st.secrets.get("tts_url"),
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"TTS API 호출 실패: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"TTS 에러: {e}")
        return None

# ============================================
# 메인 UI
# ============================================
def main():
    # 제목 섹션
    st.markdown("""
    <div class="title-section">
        <h1>🎙️ Voice Converter</h1>
        <p>음성을 인식하고 다른 목소리로 변환해보세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 컬럼 레이아웃
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-label">📝 음성 입력</div>', unsafe_allow_html=True)
        
        # 음성 녹음
        audio_data = st.audio_input("마이크 버튼을 클릭하여 녹음하세요", label_visibility="collapsed")
        
        if audio_data is not None:
            st.session_state.audio_data = audio_data
            st.success("✓ 음성이 캡처되었습니다")
            
            # STT 버튼
            if st.button("🔄 음성을 텍스트로 변환", key="stt_btn", use_container_width=True):
                with st.spinner("음성을 텍스트로 변환 중..."):
                    # 오디오 데이터를 numpy 배열로 변환
                    audio_array = np.frombuffer(audio_data.getvalue(), dtype=np.int16)
                    
                    # STT 실행
                    text = speech_to_text(audio_array)
                    
                    if text:
                        st.session_state.recognized_text = text
                        st.success("✓ STT 변환 완료!")
    
    with col2:
        st.markdown('<div class="section-label">🔊 음성 출력</div>', unsafe_allow_html=True)
        
        if st.session_state.recognized_text:
            # 인식된 텍스트 표시
            st.markdown("""
            <div class="result-box">
                <div style="color: #666; font-size: 12px; margin-bottom: 8px;">인식된 텍스트</div>
                <div class="text-display">
            """ + st.session_state.recognized_text + """
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 음성 커스터마이징 옵션
            st.markdown('<div class="section-label">⚙️ 설정</div>', unsafe_allow_html=True)
            voice_option = st.selectbox(
                "음성 선택",
                ["nova", "echo", "alloy", "shimmer", "fable"],
                index=0,
                label_visibility="collapsed"
            )
            
            # TTS 버튼
            if st.button("🎵 텍스트를 음성으로 변환", key="tts_btn", use_container_width=True):
                with st.spinner("음성으로 변환 중..."):
                    audio_content = text_to_speech(st.session_state.recognized_text, voice=voice_option)
                    
                    if audio_content:
                        st.session_state.tts_file = audio_content
                        st.success("✓ TTS 변환 완료!")
        
        else:
            st.info("📝 왼쪽에서 음성을 입력하면 여기에 결과가 표시됩니다")
    
    # 음성 재생 섹션
    if st.session_state.tts_file:
        st.markdown("---")
        st.markdown('<div class="section-label">▶️ 결과 음성 재생</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.audio(st.session_state.tts_file, format="audio/mp3")
        with col2:
            # 다운로드 버튼
            st.download_button(
                label="⬇️ 다운로드",
                data=st.session_state.tts_file,
                file_name=f"voice_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
    
    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.audio_data = None
        st.session_state.recognized_text = None
        st.session_state.tts_file = None
        st.rerun()

if __name__ == "__main__":
    main()
