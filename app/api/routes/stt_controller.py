from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech_v2
import asyncio
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
import json
import queue
import threading
from collections import deque

router = APIRouter()


# 환경 변수 로드 및 검증
credentials_path = os.path.join("./app/stt-kidok-447304-64973ec15a3e.json")
# credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not credentials_path or not os.path.exists(credentials_path):
    raise ValueError("Invalid GOOGLE_APPLICATION_CREDENTIALS path")

# 서비스 계정 JSON에서 project_id 직접 읽기
with open(credentials_path) as f:
    project_id = json.load(f)["project_id"]

class STTHandler:
    def __init__(self):
        # 서비스 계정 인증 정보 로드
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = speech_v2.SpeechClient(credentials=credentials)
        
        # 큐 초기화
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_streaming = False
        
        # Recognizer 경로 설정
        self.recognizer = f"projects/{project_id}/locations/global/recognizers/_"
        # print(f"Using recognizer path: {self.recognizer}")  # 디버깅용
        
        self._setup_streaming_config()

    def _setup_streaming_config(self):
        """V2 API 설정 최적화 - 불필요한 기능 제거"""
        self.config = speech_v2.RecognitionConfig(
            explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1
            ),
            language_codes=["ko-KR"],
            model="latest_long",
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True  # 기본 문장부호만 활성화
            )
        )
        
        self.streaming_config = speech_v2.StreamingRecognitionConfig(
            config=self.config,
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True  # 실시간 결과 활성화
            )
        )

    def _create_streaming_request(self):
        """스트리밍 요청 생성기"""
        first_request = speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=self.streaming_config
        )
        yield first_request
        print("Initial config request sent")

        while self.is_streaming:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:
                    break
                yield speech_v2.StreamingRecognizeRequest(audio=audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in request generator: {e}")
                break

    def start_recognition(self):
        """음성 인식 시작"""
        try:
            print(f"Starting recognition with config: {self.config}")
            responses = self.client.streaming_recognize(
                requests=self._create_streaming_request()
            )
            
            for response in responses:
                if not response.results:
                    continue
                
                for result in response.results:
                    if result.alternatives:
                        self.response_queue.put({
                            "text": result.alternatives[0].transcript,
                            "is_final": result.is_final,
                            "confidence": result.alternatives[0].confidence
                        })
                        print(f"Received transcript: {result.alternatives[0].transcript}")
        except Exception as e:
            print(f"Recognition error: {e}")
            import traceback
            print(traceback.format_exc())
            self.response_queue.put({"error": str(e)})
        finally:
            self.is_streaming = False

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    stt_handler = STTHandler()
    stt_handler.is_streaming = True
    final_text = []
    latest_text = ""
    websocket_closed = False  # WebSocket 상태 추적
    
    silence_start = None  # Silence 시작 시간
    SILENCE_THRESHOLD_SECONDS = 3  # 침묵 유지 시간 기준 (5초)
    VOLUME_WINDOW_SIZE = 10  # 최근 10개의 오디오 데이터 볼륨 평균 계산
    
    # 최근 볼륨 데이터를 저장할 슬라이딩 윈도우 큐
    volume_window = deque(maxlen=VOLUME_WINDOW_SIZE)

    # 초기 배경 소음을 측정하기 위한 변수
    background_noise_levels = []
    dynamic_volume_threshold = None  # 동적으로 계산된 볼륨 기준값
    INITIAL_BACKGROUND_NOISE_SECONDS = 2  # 초기 배경 소음 측정 시간
    VOLUME_MARGIN = 5  # 배경 소음 대비 말소리 감지 기준 여유 값

    
    async def measure_background_noise():
        """초기 배경 소음을 측정하여 동적 볼륨 기준값 설정"""
        nonlocal dynamic_volume_threshold
        print("Measuring background noise...")
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < INITIAL_BACKGROUND_NOISE_SECONDS:
            try:
                data = await websocket.receive_json()
                audio_data = bytes(data["audio_data"])
                average_volume = sum(audio_data) / len(audio_data) if len(audio_data) > 0 else 0
                background_noise_levels.append(average_volume)
            except Exception as e:
                print(f"Error measuring background noise: {e}")
                break
        if background_noise_levels:
            average_noise = sum(background_noise_levels) / len(background_noise_levels)
            dynamic_volume_threshold = average_noise + VOLUME_MARGIN
            print(f"Background noise measured: {average_noise}, Threshold set to: {dynamic_volume_threshold}")
        else:
            dynamic_volume_threshold = VOLUME_MARGIN  # 기본값 설정
            print("Failed to measure background noise. Using default threshold.")

    async def silence_timer():
        """침묵 감지 타이머"""
        nonlocal silence_start, websocket_closed
        while stt_handler.is_streaming:
            await asyncio.sleep(0.1)  # 100ms 간격으로 체크
            if silence_start:
                elapsed_time = asyncio.get_event_loop().time() - silence_start
                if elapsed_time > SILENCE_THRESHOLD_SECONDS:
                    print("Silence detected for 5 seconds. Stopping recording.")
                    if not websocket_closed:
                        await websocket.send_json({
                            "message": "마이크 볼륨과 상태를 확인해주세요.",
                            "is_final": True,
                            "is_complete": True
                        })
                        await websocket.close()
                        websocket_closed = True
                    stt_handler.is_streaming = False
                    break


    # 음성 인식 쓰레드 시작
    recognition_thread = threading.Thread(target=stt_handler.start_recognition)
    recognition_thread.start()
    
    # 배경 소음 측정
    await measure_background_noise()

    # 침묵 감지 타이머 시작
    asyncio.create_task(silence_timer())

    try:
        while stt_handler.is_streaming:
            try:
                data = await websocket.receive_json()
                
                if data.get("isFinal") or data.get("silenceDetected"):
                    print("Received end signal, current final_text:", final_text)
                    await asyncio.sleep(0.5)
                    
                    complete_text = " ".join(final_text).strip()
                    if not complete_text and latest_text:
                        complete_text = latest_text
                    
                    if not websocket_closed:  # WebSocket이 열려있을 때만 전송
                        print("Sending complete text:", complete_text)
                        final_response = {
                            "text": complete_text,
                            "is_final": True,
                            "is_complete": True
                        }
                        await websocket.send_json(final_response)
                        await websocket.close()
                        websocket_closed = True
                    break

                audio_data = bytes(data["audio_data"])

                # 볼륨 계산 및 슬라이딩 윈도우 업데이트
                average_volume = sum(audio_data) / len(audio_data) if len(audio_data) > 0 else 0
                volume_window.append(average_volume)

                # 최근 볼륨 평균 계산
                sliding_average = sum(volume_window) / len(volume_window)
                print(f"Sliding average volume: {sliding_average}")

                # 말소리 감지 여부 판단
                if sliding_average > (dynamic_volume_threshold or VOLUME_MARGIN):
                    print(sliding_average)
                    print((dynamic_volume_threshold or VOLUME_MARGIN))
                    silence_start = None  # 말소리 감지 시 타이머 리셋
                else:
                    if silence_start is None:
                        print('침묵 시작')
                        silence_start = asyncio.get_event_loop().time()  # 침묵 시작 시간 기록



                stt_handler.audio_queue.put(audio_data)

                try:
                    while True:
                        response = stt_handler.response_queue.get_nowait()
                        if response.get("is_final"):
                            final_text.append(response["text"].strip())
                        else:
                            latest_text = response["text"].strip()
                        if not websocket_closed:  # WebSocket이 열려있을 때만 전송
                            await websocket.send_json(response)
                except queue.Empty:
                    pass

                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    finally:
        # 연결이 아직 열려있고 최종 텍스트가 전송되지 않았을 때만 시도
        if not websocket_closed and not final_text and latest_text:
            try:
                final_response = {
                    "text": latest_text,
                    "is_final": True,
                    "is_complete": True
                }
                await websocket.send_json(final_response)
                await websocket.close()
            except:
                pass
        
        stt_handler.is_streaming = False
        stt_handler.audio_queue.put(None)
        recognition_thread.join()
        print("WebSocket closed")
