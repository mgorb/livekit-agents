# LiveKit Neurok TTS Plugin

This plugin integrates the Neurok TTS (Text-to-Speech) system with LiveKit Agents, providing high-quality Estonian speech synthesis.

## Features

- Support for multiple Estonian voices
- Adjustable speech speed
- High-quality neural text-to-speech synthesis

## Installation

```bash
pip install livekit-plugins-neurok-tts
```

## Requirements

- Python 3.8 or higher
- LiveKit Agents 0.7.0 or higher
- Neurok TTS system installed and configured

## Usage

```python
from livekit.plugins.neurok_tts import TTS

# Initialize the TTS engine with default settings
tts = TTS()

# Or with custom settings
tts = TTS(
    speaker="mari",  # Choose from available voices
    speed=1.0,       # Adjust speech speed (1.0 is normal)
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

The plugin looks for the following environment variables:

- `NEUROK_TTS_CONFIG`: Path to the Neurok TTS configuration file (default: 'neurok-tts/config/config.yaml')
- `NEUROK_TTS_MODEL`: Model name to use (default: 'multispeaker')

## License

Apache 2.0