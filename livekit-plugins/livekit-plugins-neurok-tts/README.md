# LiveKit Neurok TTS Plugin

This plugin integrates the Neurok TTS (Text-to-Speech) system with LiveKit Agents, providing high-quality Estonian speech synthesis.

## Features

- Support for multiple Estonian voices
- Adjustable speech speed
- High-quality neural text-to-speech synthesis

## Installation

```bash
# Basic installation
pip install livekit-plugins-neurok-tts

# With full Neurok TTS dependencies
pip install livekit-plugins-neurok-tts[neurok]
```

## Requirements

- Python 3.9 or higher
- LiveKit Agents 0.7.0 or higher

### Dependencies

The plugin requires the following main dependencies:
- tensorflow (≥2.10.0)
- tf-keras (≥2.15.0)
- torch (≥2.0.0)
- torchaudio (≥2.0.0)
- numpy (≥1.20.0)
- scipy (≥1.7.0)
- nltk (≥3.9.0)
- ruamel.yaml (≥0.16.6)
- pydantic (≥2.0.0)
- pydantic-settings (≥2.0.0)
- pyyaml (≥6.0.0)
- matplotlib (≥3.2.2)
- webrtcvad (≥2.0.10)
- pyworld (≥0.3.4)
- p_tqdm (≥1.4.0)
- pika (≥1.3.0) - for RabbitMQ integration

### Full Neurok TTS Support

For full Neurok TTS functionality, you'll need:

1. The optional neurok dependencies: `pip install livekit-plugins-neurok-tts[neurok]`
   - This includes:
     - estnltk (≥1.7.0)
     - phonemizer (≥3.0.0)
     - librosa (≥0.10.0)
     - soundfile (≥0.12.0)
     - tqdm (≥4.60.0)
     - numba (≥0.53.0)
2. The Neurok TTS system installed and configured
3. Access to the Neurok TTS models and vocoders

## Usage

```python
from livekit.plugins.neurok_tts import TTS

# Initialize the TTS engine with default settings (fallback mode)
tts = TTS()

# Or with full Neurok TTS support
tts = TTS(
    speaker="mari",       # Choose from available voices
    speed=1.0,            # Adjust speech speed (1.0 is normal)
    model_config_path="/path/to/neurok-tts/config/config.yaml",
    model_name="multispeaker"
)

# Update options after initialization
tts.update_options(
    speaker="albert",
    speed=1.2,
)

# Use with LiveKit Agents
from livekit.agents.voice import Agent

agent = Agent(
    tts=tts,
    # other agent parameters...
)
```

## Available Voices

The following Estonian voices are available:

- albert
- indrek
- kalev
- kylli
- liivika
- mari (default)
- meelis
- peeter
- tambet
- vesta

## Configuration

When using the full Neurok TTS functionality, you need to provide:

1. `model_config_path`: Path to the Neurok TTS configuration file (e.g., 'neurok-tts/config/config.yaml')
2. `model_name`: Name of the model to use (e.g., 'multispeaker')

If these are not provided or if the required dependencies are not installed, the plugin will fall back to a simple sine wave generator for testing purposes.

### Environment Variables

You can also configure the plugin using environment variables:

- `NEUROK_TTS_CONFIG`: Path to the Neurok TTS configuration file
- `NEUROK_TTS_MODEL`: Model name to use

## License

Apache 2.0