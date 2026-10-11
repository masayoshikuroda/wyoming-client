import sys
import asyncio
import argparse
import logging
import numpy as np
import sounddevice as sd
import webrtcvad
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)

class WyomingSTTClient:
    """Wyoming プロトコル経由で Whisper 等の STT サーバーへ音声を送り、
    VAD(発話検知)により自動で録音開始/終了を行うクライアント
    """

    def __init__(
        self,
        host: str = "docker.local",
        port: int = 10300,
        rate: int = 16000,
        channels: int = 1,
        vad_mode: int = 3,  # 0(感度低/ノイズに強い) 〜 3(感度高/厳格)
        silence_duration: float = 1.0,  # 発話終了とみなす無音時間(秒)
    ):
        self.host = host
        self.port = port
        self.rate = rate
        self.channels = channels
        self.silence_duration = silence_duration

        # webrtcvad の初期化
        self.vad = webrtcvad.Vad(vad_mode)

        # webrtcvad は 10ms, 20ms, または 30ms のフレーム単位のみ対応
        # 16kHz / 16bit(2bytes) / 1ch の場合: 30ms = 480 サンプル = 960 バイト
        self.frame_duration_ms = 30
        self.frame_bytes = int(self.rate * (self.frame_duration_ms / 1000.0) * 2)

    async def transcribe(self, timeout: float = 10.0) -> str:
        """マイク入力を監視し、発話開始〜終了（無音検出）を自動判定して STT 認識結果を返す"""
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()
        raw_buffer = bytearray()

        # マイク入力コールバック（30ms 単位のフレームに分割してキューに投げる）
        def input_callback(indata, frames, time, status):
            nonlocal raw_buffer
            if status and status.input_overflow:
                logger.error("マイク入力バッファオーバーフローが発生しました。")

            raw_buffer.extend(indata.tobytes())

            # 30ms (960 バイト) ごとに切り出してキューに送る
            while len(raw_buffer) >= self.frame_bytes:
                frame = bytes(raw_buffer[: self.frame_bytes])
                del raw_buffer[: self.frame_bytes]
                loop.call_soon_threadsafe(audio_queue.put_nowait, frame)

        transcript_text = ""

        # マイクストリーム準備
        stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            callback=input_callback,
            blocksize=self.frame_bytes // 2,
        )

        async with AsyncTcpClient(self.host, self.port) as client:
            logger.info(f"[{self.host}:{self.port}] Wyoming STT サーバーに接続しました。")
            logger.info("マイク待機中... 話し始めてください。")

            # 1. 音声ストリーム開始イベント (AudioStart)
            await client.write_event(
                AudioStart(
                    rate=self.rate,
                    width=2,
                    channels=self.channels,
                ).event()
            )

            is_speaking = False
            silence_counter = 0
            max_silence_frames = int(
                self.silence_duration / (self.frame_duration_ms / 1000.0)
            )

            # Pre-roll バッファ（発話開始直前の巻き戻し音声を保持して語頭切れを防ぐ）
            pre_roll_buffer = []
            max_pre_roll = 10  # 約 300ms 分

            with stream:
                start_time = asyncio.get_event_loop().time()

                while True:
                    # タイムアウトチェック
                    if (
                        not is_speaking
                        and (asyncio.get_event_loop().time() - start_time) > timeout
                    ):
                        logger.error("発話が検知されないためタイムアウトしました。")
                        break

                    try:
                        frame = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    # VAD 判定 (True: 有声, False: 無音)
                    is_speech = self.vad.is_speech(frame, self.rate)

                    if not is_speaking:
                        # --- 発話開始前 ---
                        pre_roll_buffer.append(frame)
                        if len(pre_roll_buffer) > max_pre_roll:
                            pre_roll_buffer.pop(0)

                        if is_speech:
                            is_speaking = True
                            logger.info(">>> 発話を検知しました（録音・送信開始）")

                            # 語頭切れ防止のため Pre-roll バッファを先に送信
                            for pre_frame in pre_roll_buffer:
                                await client.write_event(
                                    AudioChunk(
                                        rate=self.rate,
                                        width=2,
                                        channels=self.channels,
                                        audio=pre_frame,
                                    ).event()
                                )
                    else:
                        # --- 発話中 ---
                        # 音声チャンクの送信
                        await client.write_event(
                            AudioChunk(
                                rate=self.rate,
                                width=2,
                                channels=self.channels,
                                audio=frame,
                            ).event()
                        )

                        if is_speech:
                            silence_counter = 0
                        else:
                            silence_counter += 1

                        # 一定時間無音が続いたら発話終了とみなす
                        if silence_counter >= max_silence_frames:
                            logger.info("<<< 無音を検出しました（発話終了）")
                            break

            # 2. 録音終了イベント (AudioStop) の送信
            await client.write_event(AudioStop().event())

            if is_speaking:
                logger.info("STT 認識結果を待っています...")
                # 3. 認識結果 (Transcript) 受信
                while True:
                    event = await client.read_event()
                    if event is None:
                        break

                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        transcript_text = transcript.text
                        logger.info(f"認識結果: {transcript_text}")
                        break

        return transcript_text


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=10300)
    args = parser.parse_args()

    client = WyomingSTTClient(args.host, args.port)
    return await client.transcribe()

if __name__ == '__main__':
    asyncio.run(main())
