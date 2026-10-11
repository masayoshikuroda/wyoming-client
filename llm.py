import sys
import asyncio
import argparse
import logging
from wyoming.asr import Transcript
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)

class WyomingLLMClient:
    """wyoming の handled イベントを直接検知して会話テキストを取り出すクライアント"""

    def __init__(self, host: str = "docker.local", port: int = 10500):
        self.host = host
        self.port = port

    async def chat(self, text: str, timeout: float = 15.0) -> str:
        """テキストメッセージを直接送信し、handled イベントから応答テキストを取得する"""
        response_text = ""

        async with AsyncTcpClient(self.host, self.port) as client:
            logger.info(f"[{self.host}:{self.port}] Wyoming サーバーに接続しました。")

            # 1. 直接 Transcript (テキスト) イベントを送信
            logger.info(f"送信メッセージ: '{text}'")
            await client.write_event(Transcript(text=text).event())

            # 2. レスポンスの受信ループ
            start_time = asyncio.get_event_loop().time()

            while True:
                if (asyncio.get_event_loop().time() - start_time) > timeout:
                    logger.error("LLM/Intent レスポンスタイムアウト")
                    break

                event = await client.read_event()
                if event is None:
                    logger.error("サーバーから切断されました。")
                    break

                logger.debug(f"受信イベント型: {event.type}")
                data = event.data or {}

                # 3. handled イベントの解析 ★ここがポイント★
                if event.type == "handled":
                    # handled イベント内にテキストが含まれている場合
                    response_text = (
                        data.get("text")
                        or data.get("speech")
                        or data.get("response", {})
                        .get("speech", {})
                        .get("plain", {})
                        .get("speech", "")
                    )
                    
                    if response_text:
                        logger.info(f"受信メッセージ {response_text}")
                    else:
                        logger.warn(f"{event.type }  イベント受信:（テキスト応答なし）。")
                    break

                # 4. intent-response や synthesize 等の互換用フォールバック
                elif event.type in ("intent-response", "synthesize"):
                    speech_data = (
                        data.get("response", {})
                        .get("speech", {})
                        .get("plain", {})
                        .get("speech", "")
                    )
                    response_text = speech_data or data.get("text", "")
                    if response_text:
                        logger.info(f"受信メッセージ: {response_text}")
                        break

        return response_text

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=10200)
    parser.add_argument("--text", type=str, default='こんばんは、月がきれいですね。')
    args = parser.parse_args()

    client = WyomingLLMClient(args.host, args.port)
    return await client.chat(args.text)

if __name__ == '__main__':
    asyncio.run(main())

