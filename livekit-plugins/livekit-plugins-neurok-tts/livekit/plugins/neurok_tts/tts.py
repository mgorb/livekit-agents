# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import json
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import DEFAULT_SPEAKER, DEFAULT_SPEED, DEFAULT_SAMPLE_RATE, NeurokTTSSpeakers


@dataclass
class SpeakerSettings:
    """Speaker settings for Neurok TTS."""
    speed: float  # e.g., 1.0, 0.8, 1.2


@dataclass
class _TTSOptions:
    """Internal options for Neurok TTS."""
    speaker: str
    speaker_settings: SpeakerSettings
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """Neurok TTS implementation."""

    def __init__(
        self,
        *,
        speaker: str = DEFAULT_SPEAKER,
        speed: float = DEFAULT_SPEED,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Neurok TTS.

        Args:
            speaker (str): Speaker ID to use. Defaults to "mari".
            speed (float): Speaking speed adjustment. Defaults to 1.0.
            sample_rate (int): Audio sample rate in Hz. Defaults to 22050.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Neurok TTS doesn't support true streaming
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            speaker=speaker,
            speaker_settings=SpeakerSettings(
                speed=speed,
            ),
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )

    def update_options(
        self,
        *,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            speaker (str, optional): Speaker ID to use.
            speed (float, optional): Speaking speed adjustment.
        """
        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(speed):
            self._opts.speaker_settings.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize text to speech.

        Args:
            text (str): Text to synthesize.
            conn_options (APIConnectOptions, optional): Connection options.

        Returns:
            ChunkedStream: Stream of synthesized audio.
        """
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
            opts=self._opts,
        )

    @staticmethod
    async def list_voices() -> List[Dict[str, Any]]:
        """
        List available voices.

        Returns:
            List[Dict[str, Any]]: List of available voices from the config.yaml file.
        """
        try:
            # Hardcoded list of voices based on the config.yaml
            # In a production environment, this would dynamically read from the config
            voices = [
                {
                    "Name": "albert",
                    "ShortName": "albert",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Albert",
                    "Status": "GA",
                },
                {
                    "Name": "indrek",
                    "ShortName": "indrek",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Indrek",
                    "Status": "GA",
                },
                {
                    "Name": "kalev",
                    "ShortName": "kalev",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Kalev",
                    "Status": "GA",
                },
                {
                    "Name": "kylli",
                    "ShortName": "kylli",
                    "Gender": "Female",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Kylli",
                    "Status": "GA",
                },
                {
                    "Name": "liivika",
                    "ShortName": "liivika",
                    "Gender": "Female",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Liivika",
                    "Status": "GA",
                },
                {
                    "Name": "mari",
                    "ShortName": "mari",
                    "Gender": "Female",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Mari",
                    "Status": "GA",
                },
                {
                    "Name": "meelis",
                    "ShortName": "meelis",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Meelis",
                    "Status": "GA",
                },
                {
                    "Name": "peeter",
                    "ShortName": "peeter",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Peeter",
                    "Status": "GA",
                },
                {
                    "Name": "tambet",
                    "ShortName": "tambet",
                    "Gender": "Male",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Tambet",
                    "Status": "GA",
                },
                {
                    "Name": "vesta",
                    "ShortName": "vesta",
                    "Gender": "Female",
                    "Locale": "et-EE",
                    "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
                    "FriendlyName": "Estonian Vesta",
                    "Status": "GA",
                },
            ]
            return voices
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise APIConnectionError() from e


class ChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for Neurok TTS."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self) -> None:
        """Run the synthesis process."""
        request_id = utils.shortuuid()
        
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use subprocess to call the neurok-tts CLI
            import subprocess
            import numpy as np
            from scipy.io import wavfile
            
            # Prepare request data
            request_data = {
                "text": self._input_text,
                "speaker": self._opts.speaker,
                "speed": self._opts.speaker_settings.speed
            }
            
            # Save request to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as req_file:
                req_path = req_file.name
                json.dump(request_data, req_file)
            
            try:
                # Call neurok-tts CLI (this is a placeholder - actual CLI command would depend on neurok-tts implementation)
                logger.info(f"Synthesizing text with Neurok TTS using speaker: {self._opts.speaker}")
                
                # This is where you would call the neurok-tts CLI
                # For now, we'll use a fallback approach for testing
                
                # We'll use a fallback approach since direct imports may not work
                # In a real implementation, you would ensure the neurok-tts package is properly installed
                logger.warning("Using fallback audio generation for testing")
                
                # Generate a simple sine wave as fallback
                # In a production environment, this would be replaced with actual neurok-tts integration
                import numpy as np
                from scipy.io import wavfile
                
                sample_rate = self._opts.sample_rate
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                
                # Generate a simple sine wave
                note_freq = 440  # A4 note
                audio = np.sin(note_freq * 2 * np.pi * t) * 32767
                
                # Save as WAV file
                wavfile.write(temp_path, sample_rate, audio.astype(np.int16))
            
            finally:
                # Clean up request file
                if os.path.exists(req_path):
                    os.unlink(req_path)
            
            # Create decoder for the audio file
            decoder = utils.codecs.AudioStreamDecoder(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
            )
            
            try:
                # Read the file and push to decoder
                with open(temp_path, "rb") as f:
                    audio_data = f.read()
                    decoder.push(audio_data)
                    decoder.end_input()
                
                # Create emitter for the audio frames
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                
                # Process audio frames
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
            
            finally:
                # Clean up
                await decoder.aclose()
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            logger.error(f"Neurok TTS error: {e}")
            raise APIConnectionError() from e