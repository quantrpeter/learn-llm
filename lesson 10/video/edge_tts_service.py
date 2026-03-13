"""Custom Edge TTS speech service for manim-voiceover."""

import asyncio
from pathlib import Path

import edge_tts
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService


class EdgeTTSService(SpeechService):
    """Speech service using Microsoft Edge TTS (free, no API key needed)."""

    def __init__(self, voice="en-US-AriaNeural", rate="+0%", **kwargs):
        SpeechService.__init__(self, **kwargs)
        self.voice = voice
        self.rate = rate

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        input_data = {
            "input_text": input_text,
            "service": "edge_tts",
            "voice": self.voice,
            "rate": self.rate,
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path

        output_file = str(Path(cache_dir) / audio_path)

        communicate = edge_tts.Communicate(input_text, self.voice, rate=self.rate)
        asyncio.run(communicate.save(output_file))

        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }
