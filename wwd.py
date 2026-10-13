import sys
import asyncio
import argparse
import logging
import numpy as np
import sounddevice as sd
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.wake import Detect

logger = logging.getLogger(__name__)

class WyomingWWDClient:
    """wyoming プロトコル経由で openWakeWord 等の WWD サーバーへ音声をストリーミングし、
    ウェイクワード検知(detection)を待機するクライアント
    """

    def __init__(
        self,
        host: str = "docker.local",
        port: int = 10400,
        rate: int = 16000,
        channels: int = 1,
        wake_words: list[str] | None = None,  # 検知したいモデル名リスト (例: ["ok_nabu", "hey_jarvis"])
    ):
        self.host = host
        self.port = port
        self.rate = rate
        self.channels = channels
        self.wake_words = wake_words or []
        self._is_listening = False

    async def detect(self, timeout: float | None = None) -> str | None:
        """マイク音声を送り続け、ウェイクワードが検知されたらその名前(name)を返す"""
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def input_callback(indata, frames, time, status):
            if status and status.input_overflow:
                print("マイク入力バッファオーバーフローが発生しました。")
            if self._is_listening:
                pcm_bytes = indata.tobytes()
                loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_bytes)

        detected_name = None

        stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            callback=input_callback,
            blocksize=1024,
        )

        async with AsyncTcpClient(self.host, self.port) as client:
            logger.info(f"[{self.host}:{self.port}] Wyoming WWD サーバーに接続しました。")

            # 1. 検出開始リクエスト (Detect) の送信 ★ここを追加★
            # サーバーに対して検知対象のモデル名を伝える (空リストの場合はサーバー既定の全モデル)
            await client.write_event(Detect(names=self.wake_words).event())
            logger.debug("sent: detect")

            # 2. 音声ストリーム開始イベント (AudioStart) の送信
            await client.write_event(AudioStart(rate=self.rate, width=2, channels=self.channels).event())
            logger.debug("sent: audio-start")

            self._is_listening = True

            async def send_audio():
                while self._is_listening:
                    try:
                        pcm_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        await client.write_event(AudioChunk(rate=self.rate, width=2, channels=self.channels, audio=pcm_bytes).event())
                        logger.debug("sent: audio-chunk")
                    except asyncio.TimeoutError:
                        continue

            send_task = asyncio.create_task(send_audio())

            logger.info("ウェイクワード待機中...")
            start_time = loop.time()

            with stream:
                while self._is_listening:
                    if timeout and (loop.time() - start_time) > timeout:
                        logger.info("ウェイクワード検知タイムアウト。")
                        break

                    try:
                        event = await asyncio.wait_for(client.read_event(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if event is None:
                        logger.info("サーバーから切断されました。")
                        break

                    # 3. サーバーからの detection イベント検知
                    logger.debug("recv: " + event.type)
                    if event.type == "detection":
                        data = event.data or {}
                        detected_name = data.get("name", "unknown")
                        logger.info(f"ウェイクワード検知: '{detected_name}'")
                        break

            # クリーンアップ
            self._is_listening = False
            await send_task
            await client.write_event(AudioStop().event())
            logger.debug("sent: audio-stop")
        return detected_name

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    type=str,   default='localhost')
    parser.add_argument("--port",    type=int,   default=10300)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    client = WyomingWWDClient(args.host, args.port)
    return await client.detect(timeout=args.timeout)

if __name__ == '__main__':
    asyncio.run(main())