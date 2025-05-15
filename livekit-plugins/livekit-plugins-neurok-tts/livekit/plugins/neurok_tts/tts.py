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
import re
import tempfile
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from scipy.io import wavfile
from nltk import sent_tokenize

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
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
from .models import DEFAULT_SAMPLE_RATE, DEFAULT_SPEAKER, DEFAULT_SPEED

# Import neurok-tts components
try:
    import tensorflow as tf
    from TransformerTTS.model.models import ForwardTransformer
    from tts_worker.vocoding.predictors import HiFiGANPredictor
    from tts_worker.config import ModelConfig, Speaker
    from tts_worker.utils import clean, split_sentence
    NEUROK_AVAILABLE = True
except ImportError:
    logger.warning("Neurok TTS dependencies not available. Using fallback implementation.")
    NEUROK_AVAILABLE = False


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
    model_config_path: str = ""
    model_name: str = "neurok-tts"


class NeurokSynthesizer:
    """Adapter for the Neurok TTS Synthesizer."""
    
    def __init__(self, model_config_path: str, model_name: str):
        """
        Initialize the Neurok TTS Synthesizer.
        
        Args:
            model_config_path: Path to the model configuration file
            model_name: Name of the model to use
        """
        if not NEUROK_AVAILABLE:
            raise ImportError("Neurok TTS dependencies are not available")
        
        # Load model configuration
        try:
            # In a real implementation, this would use read_model_config
            # For now, we'll create a minimal configuration
            speakers = {
                "albert": Speaker(speaker_id=0, vocoder="hifigan"),
                "indrek": Speaker(speaker_id=1, vocoder="hifigan"),
                "kalev": Speaker(speaker_id=2, vocoder="hifigan"),
                "kylli": Speaker(speaker_id=3, vocoder="hifigan"),
                "liivika": Speaker(speaker_id=4, vocoder="hifigan"),
                "mari": Speaker(speaker_id=5, vocoder="hifigan"),
                "meelis": Speaker(speaker_id=6, vocoder="hifigan"),
                "peeter": Speaker(speaker_id=7, vocoder="hifigan"),
                "tambet": Speaker(speaker_id=8, vocoder="hifigan"),
                "vesta": Speaker(speaker_id=9, vocoder="hifigan"),
            }
            
            # This is a placeholder - in a real implementation, paths would be read from config
            model_path = os.path.join(os.path.dirname(model_config_path), "models", model_name)
            vocoder_path = os.path.join(os.path.dirname(model_config_path), "models", "hifigan")
            
            self.model_config = ModelConfig(
                model_name=model_name,
                model_path=model_path,
                frontend="est",
                speakers=speakers,
                vocoders={"hifigan": vocoder_path}
            )
            
            # Initialize the model
            self.model = ForwardTransformer.load_model(self.model_config.model_path)
            self.vocoders = {}
            
            for speaker in self.model_config.speakers.values():
                if speaker.vocoder not in self.vocoders:
                    self.vocoders[speaker.vocoder] = HiFiGANPredictor.from_folder(
                        self.model_config.vocoders[speaker.vocoder]
                    )
            
            self.speakers = self.model_config.speakers
            self.frontend = self.model_config.frontend
            
            self.sampling_rate = self.model.config['sampling_rate']
            self.hop_length = self.model.config['hop_length']
            self.win_length = self.model.config['win_length']
            
            self.silence = np.zeros(self.sampling_rate // 2 - (self.sampling_rate // 2) % self.hop_length,
                                    dtype=np.int16)  # ~0.5 sec
            self.silence_len = self.silence.shape[0] // self.hop_length
            
            self.gst_len = self.model.text_pipeline.tokenizer.zfill
            
            self.max_input_length = self.model.config['encoder_max_position_encoding'] - self.gst_len
            self.last_input_len = 0
            
            logger.info("Neurok TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neurok TTS: {e}")
            raise
    
    def synthesize(self, text: str, speaker: str, speed: float = 1.0) -> bytes:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID to use
            speed: Speaking speed adjustment
            
        Returns:
            Audio data as bytes
        """
        if not NEUROK_AVAILABLE:
            raise ImportError("Neurok TTS dependencies are not available")
        
        try:
            waveforms = []
            vocoder = self.vocoders[self.speakers[speaker].vocoder]
            
            # The quotation marks need to be unified, otherwise sentence tokenization won't work
            sentences = sent_tokenize(re.sub(r'[«»"„]', r'"', text), 'estonian')
            
            durations = []
            normalized_text = ""
            
            for i, sentence in enumerate(sentences):
                normalized_text += " "
                
                logger.debug(f"Original sentence {i} ({len(sentence)} chars): {sentence}")
                normalized_sentence = clean(sentence, self.model.config['alphabet'], frontend=self.frontend)
                logger.debug(f"Cleaned sentence {i} ({len(normalized_sentence)} chars): {normalized_sentence}")
                
                while True:
                    try:
                        sent_durations = []
                        if len(normalized_sentence) > self.max_input_length:
                            inputs = split_sentence(normalized_sentence, max_len=self.max_input_length)
                            logger.debug(f'Sentence split into {len(inputs)} parts: '
                                        f'{[x[:10] + " ... " + x[-10:] for x in inputs]}')
                        else:
                            inputs = [normalized_sentence]
                        
                        for input_sentence in inputs:
                            self.last_input_len = len(input_sentence)
                            
                            tts_out = self.model.predict(input_sentence,
                                                        speed_regulator=speed,
                                                        speaker_id=self.speakers[speaker].speaker_id)
                            mel_spec = tts_out['mel'].numpy().T
                            sent_durations += np.rint(
                                tts_out['duration'].numpy().squeeze()
                            ).astype(int)[self.gst_len:].tolist()
                            
                            logger.debug(f"Predicted mel-spectrogram dimensions: {mel_spec.shape}")
                            
                            if mel_spec.size:  # don't send empty mel-spectrograms to vocoder
                                waveform = vocoder([mel_spec])[0]
                                waveforms.append(waveform)
                        normalized_text += ''.join(inputs)
                        durations += sent_durations
                        break
                    except tf.errors.ResourceExhaustedError:
                        logger.warning(
                            f"Synthesis failed with max input length {self.max_input_length}, "
                            f"reducing max length to {int(self.last_input_len * 0.9)} and tying again...")
                        self.max_input_length = int(self.last_input_len * 0.9)
                waveforms.append(self.silence)
                durations.append(self.silence_len)
            
            waveforms.append(self.silence)
            durations.append(self.silence_len)
            normalized_text += " "
            
            waveform = np.concatenate(waveforms)
            
            out = io.BytesIO()
            wavfile.write(out, self.sampling_rate, waveform.astype(np.int16))
            out.seek(0)
            
            return out.read()
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise


class TTS(tts.TTS):
    """Neurok TTS implementation."""

    def __init__(
        self,
        *,
        speaker: str = DEFAULT_SPEAKER,
        speed: float = DEFAULT_SPEED,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        model_config_path: str = "",
        model_name: str = "neurok-tts",
    ) -> None:
        """
        Create a new instance of Neurok TTS.

        Args:
            speaker (str): Speaker ID to use. Defaults to "mari".
            speed (float): Speaking speed adjustment. Defaults to 1.0.
            sample_rate (int): Audio sample rate in Hz. Defaults to 22050.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            model_config_path (str): Path to the model configuration file.
            model_name (str): Name of the model to use.
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
            model_config_path=model_config_path,
            model_name=model_name,
        )
        
        self._synthesizer = None
        if NEUROK_AVAILABLE and model_config_path:
            try:
                self._synthesizer = NeurokSynthesizer(model_config_path, model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Neurok synthesizer: {e}")

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
            synthesizer=self._synthesizer,  # type: ignore
        )

    @staticmethod
    async def list_voices() -> list[dict[str, Any]]:
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
        synthesizer: NeurokSynthesizer | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._synthesizer = synthesizer

    async def _run(self) -> None:
        """Run the synthesis process."""
        request_id = utils.shortuuid()
        
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Try to use the Neurok synthesizer if available
            if NEUROK_AVAILABLE and self._synthesizer:
                try:
                    logger.info(f"Synthesizing text with Neurok TTS using speaker: {self._opts.speaker}")
                    audio_data = self._synthesizer.synthesize(
                        text=self._input_text,
                        speaker=self._opts.speaker,
                        speed=self._opts.speaker_settings.speed
                    )
                    
                    # Write the audio data to the temporary file
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                        
                except Exception as e:
                    logger.error(f"Neurok synthesis failed, falling back to placeholder: {e}")
                    self._fallback_synthesis(temp_path)
            else:
                # Use fallback synthesis if Neurok is not available
                logger.warning("Neurok TTS not available, using fallback audio generation")
                self._fallback_synthesis(temp_path)
            
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
    
    def _fallback_synthesis(self, output_path: str) -> None:
        """Generate a fallback audio file when Neurok TTS is not available."""
        import numpy as np
        from scipy.io import wavfile
        
        sample_rate = self._opts.sample_rate
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate a simple sine wave
        note_freq = 440  # A4 note
        audio = np.sin(note_freq * 2 * np.pi * t) * 32767
        
        # Save as WAV file
        wavfile.write(output_path, sample_rate, audio.astype(np.int16))