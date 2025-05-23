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
import os
import ssl
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import edge-tts at function level to avoid import errors
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
from .models import EdgeTTSLanguages, Gender

# Default values
DEFAULT_VOICE = "et-EE-AnuNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_SAMPLE_RATE = 24000


@dataclass
class VoiceSettings:
    """Voice settings for Edge TTS."""
    rate: str  # e.g., "+0%", "+10%", "-10%"
    volume: str  # e.g., "+0%", "+10%", "-10%"
    pitch: str  # e.g., "+0Hz", "+10Hz", "-10Hz"


@dataclass
class _TTSOptions:
    """Internal options for Edge TTS."""
    voice: str
    voice_settings: VoiceSettings
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """Microsoft Edge TTS implementation."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        rate: str = DEFAULT_RATE,
        volume: str = DEFAULT_VOLUME,
        pitch: str = DEFAULT_PITCH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Edge TTS.

        Args:
            voice (str): Voice ID to use. Defaults to "et-EE-AnuNeural".
            rate (str): Speaking rate adjustment. Defaults to "+0%".
            volume (str): Volume adjustment. Defaults to "+0%".
            pitch (str): Pitch adjustment. Defaults to "+0Hz".
            sample_rate (int): Audio sample rate in Hz. Defaults to 24000.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Edge TTS doesn't support true streaming
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            voice=voice,
            voice_settings=VoiceSettings(
                rate=rate,
                volume=volume,
                pitch=pitch,
            ),
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        rate: NotGivenOr[str] = NOT_GIVEN,
        volume: NotGivenOr[str] = NOT_GIVEN,
        pitch: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice (str, optional): Voice ID to use.
            rate (str, optional): Speaking rate adjustment.
            volume (str, optional): Volume adjustment.
            pitch (str, optional): Pitch adjustment.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(rate):
            self._opts.voice_settings.rate = rate
        if is_given(volume):
            self._opts.voice_settings.volume = volume
        if is_given(pitch):
            self._opts.voice_settings.pitch = pitch

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        import inspect
        frame = inspect.stack()[1]  # Caller is at stack index 1
        filename = frame.filename
        lineno = frame.lineno
        funcname = frame.function
        print(f"Called from {funcname} in {filename}:{lineno}")
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
            List[Dict[str, Any]]: List of available voices.
        """
        try:
            # Import here to avoid import errors
            try:
                import edge_tts
            except ImportError:
                raise ImportError(
                    "edge-tts package is not installed. Please install it with 'pip install edge-tts'"
                )
            
            # Get available voices using the correct API
            try:
                # Try with VoicesManager if available
                voices_manager = await edge_tts.VoicesManager.create()
                return voices_manager.voices
            except AttributeError:
                # Fallback to direct list_voices if available
                return await edge_tts.list_voices()
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise APIConnectionError() from e


class ChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for Edge TTS."""

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
        
        # Store original SSL context creation function
        original_create_default_context = ssl.create_default_context
        
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Import here to avoid import errors
            try:
                import edge_tts
            except ImportError:
                raise ImportError(
                    "edge-tts package is not installed. Please install it with 'pip install edge-tts'"
                )
            
            # Check for custom CA bundle from environment variable
            ca_bundle_path = os.environ.get('REQUESTS_CA_BUNDLE')
            ssl_verification_disabled = False
            
            # Define a patched SSL context creation function
            def patched_create_default_context(*args, **kwargs):
                nonlocal ca_bundle_path, ssl_verification_disabled
                
                if ca_bundle_path and os.path.exists(ca_bundle_path):
                    # Use the custom CA bundle
                    # logger.info(f"Using custom CA bundle from REQUESTS_CA_BUNDLE: {ca_bundle_path}")
                    try:
                        # Check if cafile is already in kwargs
                        if 'cafile' not in kwargs:
                            kwargs['cafile'] = ca_bundle_path
                        context = original_create_default_context(*args, **kwargs)
                        return context
                    except Exception as e:
                        logger.warning(f"Failed to create SSL context with custom CA bundle: {e}")
                
                # If we get here, either:
                # 1. REQUESTS_CA_BUNDLE is not set
                # 2. The file doesn't exist
                # 3. Creating context with the CA bundle failed
                # In all cases, disable SSL verification
                logger.warning("Disabling SSL certificate verification for Edge TTS")
                warnings.warn(
                    "SSL certificate verification has been disabled for Edge TTS. "
                    "This is a security risk and should only be used for testing. "
                    "Set REQUESTS_CA_BUNDLE to a valid CA bundle file for proper verification.",
                    UserWarning
                )
                
                context = original_create_default_context(*args, **kwargs)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                ssl_verification_disabled = True
                return context
            
            # Apply the monkey patch
            ssl.create_default_context = patched_create_default_context
            #logger.info("Applied SSL context patch")
            
            # Configure the voice using the correct API
            try:
                logger.info(f"Attempting to connect to Edge TTS service with voice: {self._opts.voice} and text {self._input_text}")
                communicate = edge_tts.Communicate(
                    text=self._input_text,
                    voice=self._opts.voice,
                    rate=self._opts.voice_settings.rate,
                    volume=self._opts.voice_settings.volume,
                    pitch=self._opts.voice_settings.pitch,
                )
            except AttributeError:
                # Fallback for different versions of edge-tts
                logger.warning("Using alternative edge-tts API")
                from edge_tts import Communicate
                communicate = Communicate(
                    text=self._input_text,
                    voice=self._opts.voice,
                    rate=self._opts.voice_settings.rate,
                    volume=self._opts.voice_settings.volume,
                    pitch=self._opts.voice_settings.pitch,
                )
            
            # Generate audio and save to temporary file
            try:
                logger.info("Attempting to save audio to temporary file")
                await communicate.save(temp_path)
                logger.info("Successfully saved audio to temporary file")
                
                if ssl_verification_disabled:
                    logger.warning(
                        "Audio was generated successfully, but SSL verification was disabled. "
                        "This is a security risk. Consider setting REQUESTS_CA_BUNDLE to a valid CA bundle file."
                    )
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")
                
                # Check if this is an SSL error and provide more detailed information
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    logger.error("SSL certificate verification error detected despite mitigation attempts")
                    
                    # Try to get more information about the SSL configuration
                    logger.info(f"SSL default verify paths: {ssl.get_default_verify_paths()}")
                    
                    # Provide guidance on fixing the issue
                    logger.error(
                        "To fix this SSL certificate issue, you can either:\n"
                        "1. Set the REQUESTS_CA_BUNDLE environment variable to point to a valid CA bundle file\n"
                        "   Example: export REQUESTS_CA_BUNDLE=/path/to/cacert.pem\n"
                        "2. Update your system's CA certificates\n"
                        "   On macOS: Install/update certificates via the Keychain Access app\n"
                        "   On Linux: Update ca-certificates package\n"
                        "   On Windows: Update certificates via Windows Update"
                    )
                    
                    # Check if we can access the host directly
                    import socket
                    try:
                        logger.info("Attempting to resolve speech.platform.bing.com")
                        ip = socket.gethostbyname("speech.platform.bing.com")
                        logger.info(f"Resolved to IP: {ip}")
                    except Exception as dns_error:
                        logger.error(f"DNS resolution failed: {dns_error}")
                
                # Restore original SSL context creation function
                ssl.create_default_context = original_create_default_context
                
            
            # Restore original SSL context creation function
            ssl.create_default_context = original_create_default_context
            
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
            # Restore original SSL context creation function
            ssl.create_default_context = original_create_default_context
            raise APITimeoutError() from e
        except Exception as e:
            # Restore original SSL context creation function
            ssl.create_default_context = original_create_default_context
            logger.error(f"Edge TTS error: {e}")
            raise APIConnectionError() from e