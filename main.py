import sys
import asyncio
import argparse
import logging
from stt import WyomingSTTClient
from tts import WyomingTTSClient
from llm import WyomingLLMClient
from wwd import WyomingWWDClient

logger = logging.getLogger(__name__)

async def detect(wwd: WyomingWWDClient, timeout: float):
    detected = await wwd.detect(timeout=timeout)
    if detected:
        logger.info(f"ウェイクワード検知成功!: {detected}")
        return True
    else:
        logger.info("ウェイクワード検知失敗!")
        return False
    
async def conversation(stt: WyomingSTTClient, tts: WyomingTTSClient, llm: WyomingLLMClient):
    text = await stt.transcribe();
    if (len(text) > 0):
        logger.info(f"音声認識成功: テキスト: '{text}'")
    else:
        logger.info(f"音声認識失敗")
        return False

    text = await llm.chat(text)
    if (len(text) > 0):
        logger.info(f"LLM応答成功")
    else:
        logger.info(f"LLM応答失敗:")

    await tts.speak(text)
    logger.info(f"音声応答完了")

    return True

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--tts-port", type=int, default=10200)
    parser.add_argument("--stt-port", type=int, default=10300)
    parser.add_argument("--wwd-port", type=int, default=10400)
    parser.add_argument("--llm-port", type=int, default=10500)
    parser.add_argument("--wwd-timeout", type=float, default=60.0)
    parser.add_argument("--stt-silence-duration", type=float, default=3.0)
    args = parser.parse_args()

    tts = WyomingTTSClient(args.host, args.tts_port)
    stt = WyomingSTTClient(args.host, args.stt_port, silence_duration=args.stt_silence_duration)
    wwd = WyomingWWDClient(args.host, args.wwd_port)
    llm = WyomingLLMClient(args.host, args.llm_port)

    while True:
        if await detect(wwd, 60.0):
            for i in range(100):
                if not await conversation(stt, tts, llm):
                    break;
            await tts.speak('またね')

if __name__ == "__main__":
    asyncio.run(main())