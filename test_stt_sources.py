import wave
from types import SimpleNamespace

import pytest

import stt


def write_wav(path, rate=16000, channels=1, width=2, frames=960):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\0" * frames * channels * width)


@pytest.mark.asyncio
async def test_wav_source_streams_pcm_in_frames(tmp_path):
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, frames=960)
    source = stt.WavAudioSource(str(wav_path), 16000, 1, 30)

    chunks = [chunk async for chunk in source.stream(timeout=1)]

    assert [len(chunk) for chunk in chunks] == [960, 960]
    assert b"".join(chunks) == b"\0" * 1920


@pytest.mark.asyncio
async def test_wav_source_rejects_sample_rate_mismatch(tmp_path):
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, rate=8000)
    source = stt.WavAudioSource(str(wav_path), 16000, 1, 30)

    with pytest.raises(ValueError, match="sample rate must be 16000 Hz"):
        _ = [chunk async for chunk in source.stream(timeout=1)]


@pytest.mark.asyncio
async def test_transcribe_sends_audio_from_given_source(monkeypatch):
    class Source:
        rate = 16000
        channels = 1
        width = 2

        async def stream(self, timeout):
            assert timeout == 2
            yield b"\x01\x02"

    class FakeClient:
        def __init__(self):
            self.events = []
            self.read_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def write_event(self, event):
            self.events.append(event)

        async def read_event(self):
            self.read_count += 1
            if self.read_count == 1:
                return SimpleNamespace(type="transcript")
            return None

    class FakeTranscript:
        @staticmethod
        def is_type(event_type):
            return event_type == "transcript"

        @staticmethod
        def from_event(event):
            return SimpleNamespace(text="recognized speech")

    fake_client = FakeClient()
    monkeypatch.setattr(stt, "AsyncTcpClient", lambda host, port: fake_client)
    monkeypatch.setattr(stt, "Transcript", FakeTranscript)
    client = stt.WyomingSTTClient()

    result = await client.transcribe(timeout=2, source=Source())

    assert result == "recognized speech"
    assert [event.type for event in fake_client.events] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]
