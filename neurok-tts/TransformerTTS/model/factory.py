from typing import Tuple
from pathlib import Path

import ruamel.yaml

from model.models import ForwardTransformer, Aligner
from tf_keras.utils import get_file


def tts_ljspeech(step='95000') -> Tuple[ForwardTransformer, dict]:
    model_name = f'bdf06b9_ljspeech_step_{step}.zip'
    remote_dir = 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/api_weights/bdf06b9_ljspeech/'
    custom_model_path = get_file(model_name,
                                 remote_dir + model_name,
                                 extract=True,
                                 archive_format='zip',
                                 cache_subdir='TransformerTTS_models')
    custom_model_path = Path(custom_model_path).with_suffix('')  # remove extension
    return ForwardTransformer.load_model(custom_model_path.as_posix())


def tts_custom(config_path: str, weights_path: str) -> Tuple[ForwardTransformer, dict]:
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'rb') as session_yaml:
        config = yaml.load(session_yaml)
    model = ForwardTransformer.from_config(config)
    model.build_model_weights()
    model.load_weights(weights_path)
    return model, config


def aligner_custom(config_path: str, weights_path: str) -> Tuple[Aligner, dict]:
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'rb') as session_yaml:
        config = yaml.load(session_yaml)
    model = Aligner.from_config(config)
    model.build_model_weights()
    model.load_weights(weights_path)
    return model, config
