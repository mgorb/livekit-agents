from typing import Literal

# Voice gender types
Gender = Literal["Male", "Female"]

# Estonian speakers available in neurok-tts
NeurokTTSSpeakers = Literal[
    "albert",
    "indrek",
    "kalev",
    "kylli",
    "liivika",
    "mari",
    "meelis",
    "peeter",
    "tambet",
    "vesta",
]

# Default configuration values
DEFAULT_SPEAKER = "mari"
DEFAULT_SPEED = 1.0
DEFAULT_SAMPLE_RATE = 22050  # Based on neurok-tts default