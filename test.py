import pytest
import asyncio

from tts import WyomingTTSClient
from stt import WyomingSTTClient
from wwd import WyomingWWDClient
from llm import WyomingLLMClient

HOST = 'docker.local'

@pytest.mark.parametrize("port", [
    (10200),
    (10201),
    (10202),
])
@pytest.mark.asyncio
async def test_tts(port):
    tts = WyomingTTSClient(HOST, port)
    await tts.speak('こんにちは')


@pytest.mark.parametrize("port", [
    (10300),
    (10301),
    (10302),
])
@pytest.mark.asyncio
async def test_stt(port):
    stt = WyomingSTTClient(HOST, port)
    text = await stt.transcribe()
    print(text)
    assert len(text) > 0

@pytest.mark.parametrize("port", [
    (10400),
])
@pytest.mark.asyncio
async def test_wwd(port):
    wwd = WyomingWWDClient(HOST, port)
    detected = await wwd.detect(timeout=5.0)
    print(detected)

@pytest.mark.parametrize("port", [
    (10500),
    (10501),
    (10502),
])
@pytest.mark.asyncio
async def test_llm(port):
    llm = WyomingLLMClient(HOST, port)
    text = await llm.chat('こんにちは')
    print(text)
    assert len(text) > 0
