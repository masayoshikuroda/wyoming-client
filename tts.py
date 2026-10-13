import sys
import asyncio
import argparse
import logging
import numpy as np
import sounddevice as sd
from wyoming.audio import AudioChunk
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

logger = logging.getLogger(__name__)

class WyomingTTSClient:
    """Wyoming プロトコル経由で Piper 等の TTS サーバーにテキストを送り、音声を再生するクライアント"""

    def __init__(self, host: str = "docker.local", port: int = 10200):
        self.host = host
        self.port = port
        self._stream = None
        self._audio_queue = asyncio.Queue()
        self._remainder = np.array([], dtype=np.int16)

    def _audio_callback(self, outdata, frames, time, status):
        """sounddevice から呼ばれるバッファ書き込みコールバック"""
        needed = frames
        output_chunks = []

        # 1. 前回の端数（あまり）があれば先頭に詰める
        if len(self._remainder) > 0:
            if len(self._remainder) >= needed:
                outdata[:] = self._remainder[:needed].reshape(-1, outdata.shape[1])
                self._remainder = self._remainder[needed:]
                return
            else:
                output_chunks.append(self._remainder)
                needed -= len(self._remainder)
                self._remainder = np.array([], dtype=self._remainder.dtype)

        # 2. キューから必要フレーム分を取り出して結合
        while needed > 0:
            try:
                chunk = self._audio_queue.get_nowait()
                chunk_flat = chunk.flatten()
                if len(chunk_flat) <= needed:
                    output_chunks.append(chunk_flat)
                    needed -= len(chunk_flat)
                else:
                    output_chunks.append(chunk_flat[:needed])
                    self._remainder = chunk_flat[needed:]  # 端数を保持
                    needed = 0
            except asyncio.QueueEmpty:
                break

        # 3. outdata に書き込み（不足分は無音で埋める）
        if output_chunks:
            full_data = np.concatenate(output_chunks)
            if len(full_data) < frames:
                pad = np.zeros(frames - len(full_data), dtype=full_data.dtype)
                full_data = np.concatenate([full_data, pad])
            outdata[:] = full_data.reshape(-1, outdata.shape[1])
        else:
            outdata.fill(0)

    async def speak(self, text: str) -> None:
        """指定したテキストを合成し、スピーカーから再生する"""
        # 状態リセット
        self._remainder = np.array([], dtype=np.int16)
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

        async with AsyncTcpClient(self.host, self.port) as client:
            logger.info(f"[{self.host}:{self.port}] Wyoming TTS サーバーに接続しました。")

            # 音声合成リクエスト送信
            logger.info(f"音声合成リクエスト: {text}")
            await client.write_event(Synthesize(text=text).event())
            logger.debug("sent: synthesize")

            while True:
                event = await client.read_event()
                if event is None:
                    logger.info("サーバーから切断されました。")
                    break

                logger.debug("recv: " + event.type)
                if event.type == "audio-start":
                    logger.info(f"音声受信開始")
                elif event.type == "audio-chunk":
                    logger.debug(f"イベント受信: {event.type}")
                    chunk = AudioChunk.from_event(event)
                    # 初回チャンク受信時に OutputStream を動的生成
                    if self._stream is None:
                        logger.info( f"音声フォーマット検出: Rate={chunk.rate}Hz Channels={chunk.channels}, Width={chunk.width}bytes")
                        dtype = "int16" if chunk.width == 2 else "int8"

                        self._stream = sd.OutputStream(
                            samplerate=chunk.rate,
                            channels=chunk.channels,
                            dtype=dtype,
                            callback=self._audio_callback,
                        )
                        self._stream.start()

                    pcm_data = np.frombuffer(
                        chunk.audio, dtype=np.int16 if chunk.width == 2 else np.int8
                    )
                    pcm_data = pcm_data.reshape(-1, chunk.channels)
                    await self._audio_queue.put(pcm_data)

                elif event.type == "audio-stop":
                    logger.info(f"音声受信完了")
                    # キューと端数バッファが完全に消費されるまで待機
                    while not self._audio_queue.empty() or len(self._remainder) > 0:
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.3)  # 最終吐き出し用の余韻
                    break

        # ストリームのクリーンアップ
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=10200)
    parser.add_argument("--text", type=str, default='こんにちは、私はWyomingクライアントです。')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    client = WyomingTTSClient(args.host, args.port)
    await client.speak(args.text)

if __name__ == '__main__':
    asyncio.run(main())
