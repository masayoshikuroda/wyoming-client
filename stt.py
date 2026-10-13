import sys
import asyncio
import argparse
import logging
import wave
from collections.abc import AsyncIterator
from typing import Protocol

import sounddevice as sd
import webrtcvad
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)


class AudioSource(Protocol):
    """Wyoming STT に送信する音声チャンクを提供するソースのインターフェイス。"""

    rate: int
    channels: int
    width: int

    def stream(self, timeout: float) -> AsyncIterator[bytes]:
        """PCM 音声をチャンク単位で返す。"""
        ...


class MicrophoneAudioSource:
    """マイク入力を VAD で区切って音声チャンクとして提供する。"""

    def __init__(
        self,
        rate: int,
        channels: int,
        vad: webrtcvad.Vad,
        silence_duration: float,
        frame_duration_ms: int,
    ):
        self.rate = rate
        self.channels = channels
        self.width = 2
        self.vad = vad
        self.silence_duration = silence_duration
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(rate * frame_duration_ms / 1000) * channels * self.width

    async def stream(self, timeout: float) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        raw_buffer = bytearray()

        def input_callback(indata, frames, time, status):
            nonlocal raw_buffer
            if status and status.input_overflow:
                logger.error("マイク入力バッファオーバーフローが発生しました。")

            raw_buffer.extend(indata.tobytes())
            while len(raw_buffer) >= self.frame_bytes:
                frame = bytes(raw_buffer[: self.frame_bytes])
                del raw_buffer[: self.frame_bytes]
                loop.call_soon_threadsafe(audio_queue.put_nowait, frame)

        stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            callback=input_callback,
            blocksize=self.frame_bytes // self.width,
        )

        is_speaking = False
        silence_counter = 0
        max_silence_frames = int(
            self.silence_duration / (self.frame_duration_ms / 1000.0)
        )
        pre_roll_buffer: list[bytes] = []
        max_pre_roll = 10

        with stream:
            start_time = loop.time()
            while True:
                if not is_speaking and loop.time() - start_time > timeout:
                    logger.error("発話が検知されないためタイムアウトしました。")
                    break

                try:
                    frame = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                is_speech = self.vad.is_speech(frame, self.rate)
                if not is_speaking:
                    pre_roll_buffer.append(frame)
                    if len(pre_roll_buffer) > max_pre_roll:
                        pre_roll_buffer.pop(0)

                    if is_speech:
                        is_speaking = True
                        logger.info(">>> 発話を検知しました（録音・送信開始）")
                        for pre_frame in pre_roll_buffer:
                            yield pre_frame
                else:
                    yield frame
                    if is_speech:
                        silence_counter = 0
                    else:
                        silence_counter += 1

                    if silence_counter >= max_silence_frames:
                        logger.info("<<< 無音を検出しました（発話終了）")
                        break


class WavAudioSource:
    """非圧縮 PCM WAV ファイルを音声チャンクとして提供する。"""

    def __init__(
        self,
        wav_path: str,
        rate: int,
        channels: int,
        frame_duration_ms: int,
    ):
        self.wav_path = wav_path
        self.rate = rate
        self.channels = channels
        self.width = 2
        self.frame_duration_ms = frame_duration_ms

    async def stream(self, timeout: float) -> AsyncIterator[bytes]:
        with wave.open(self.wav_path, "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("WAV file must use uncompressed PCM.")

            rate = wav_file.getframerate()
            if rate != self.rate:
                raise ValueError(
                    f"WAV sample rate must be {self.rate} Hz, got {rate} Hz."
                )

            channels = wav_file.getnchannels()
            if channels != self.channels:
                raise ValueError(
                    f"WAV file must have {self.channels} channels, got {channels}."
                )

            width = wav_file.getsampwidth()
            if width != self.width:
                raise ValueError(
                    f"WAV sample width must be {self.width} bytes, got {width} bytes."
                )

            frame_count = wav_file.getnframes()
            if rate <= 0 or channels <= 0 or frame_count <= 0:
                raise ValueError("WAV file must contain valid audio data.")

            audio_data = wav_file.readframes(frame_count)

        chunk_bytes = (
            int(self.rate * self.frame_duration_ms / 1000)
            * self.channels
            * self.width
        )
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        sent_frames = 0
        for offset in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[offset : offset + chunk_bytes]
            yield chunk
            sent_frames += len(chunk) // (self.channels * self.width)
            delay = start_time + sent_frames / self.rate - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)


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
        self.frame_bytes = int(self.rate * (self.frame_duration_ms / 1000.0) * self.channels * 2)

    def microphone_source(self) -> MicrophoneAudioSource:
        """クライアント設定に合ったマイク音声ソースを生成する。"""
        return MicrophoneAudioSource(
            rate=self.rate,
            channels=self.channels,
            vad=self.vad,
            silence_duration=self.silence_duration,
            frame_duration_ms=self.frame_duration_ms,
        )

    def wav_source(self, wav_path: str) -> WavAudioSource:
        """クライアント設定に合った WAV 音声ソースを生成する。"""
        return WavAudioSource(
            wav_path=wav_path,
            rate=self.rate,
            channels=self.channels,
            frame_duration_ms=self.frame_duration_ms,
        )

    async def transcribe_wav(self, wav_path: str) -> str:
        """WAV 音声ソースを使って音声認識結果を返す。"""
        return await self.transcribe(source=self.wav_source(wav_path))

    async def transcribe(self, timeout: float = 10.0, source: AudioSource | None = None) -> str:
        """音声ソースから音声を送り、Wyoming STT の認識結果を返す。"""
        if source is None:
            source = self.microphone_source()
        transcript_text = ""
        sent_audio = False

        async with AsyncTcpClient(self.host, self.port) as client:
            logger.info(f"[{self.host}:{self.port}] Wyoming STT サーバーに接続しました。")
            if isinstance(source, MicrophoneAudioSource):
                logger.info("マイク待機中... 話し始めてください。")
            elif isinstance(source, WavAudioSource):
                logger.info(
                    f"WAV 送信開始: {source.wav_path} "
                    f"(Rate={source.rate}Hz, Channels={source.channels}, "
                    f"Width={source.width}bytes)"
                )

            await client.write_event(AudioStart(rate=source.rate, width=source.width, channels=source.channels).event())
            logger.debug("sent: audio-start")
            try:
                async for chunk in source.stream(timeout):
                    sent_audio = True
                    await client.write_event(AudioChunk(rate=source.rate, width=source.width, channels=source.channels, audio=chunk).event())
                    logger.debug("sent: audio-chunk")
            finally:
                await client.write_event(AudioStop().event())
                logger.debug("sent: audio-stop")

            if sent_audio:
                logger.info("STT 認識結果を待っています...")
                while True:
                    event = await client.read_event()
                    if event is None:
                        logger.info("サーバーから切断されました。")
                        break

                    logger.debug("recv: " + event.type)
                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        transcript_text = transcript.text
                        logger.info(f"認識結果: {transcript_text}")
                        break

        return transcript_text


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=10300)
    parser.add_argument("--wav", type=str)
    parser.add_argument("--source", choices=("mic", "wav"))
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    client = WyomingSTTClient(args.host, args.port)
    source_type = args.source or ("wav" if args.wav else "mic")
    if source_type == "wav":
        if not args.wav:
            parser.error("--source wav requires --wav")
        source = client.wav_source(args.wav)
    else:
        if args.wav:
            parser.error("--wav cannot be used with --source mic")
        source = client.microphone_source()
    return await client.transcribe(source=source)


if __name__ == '__main__':
    asyncio.run(main())
