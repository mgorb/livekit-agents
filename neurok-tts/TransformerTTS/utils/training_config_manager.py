import subprocess
import shutil
from pathlib import Path
from enum import Enum
from typing import Optional

import numpy as np
import ruamel.yaml
from tf_keras.optimizers import Adam


class TTSMode(str, Enum):
    DATA = "data"
    ALIGNER = "aligner"
    EXTRACT = "extract"
    TTS = "tts"
    PREDICT = "predict"
    WEIGHTS = "weights"


class TrainingConfigManager:
    def __init__(self,
                 mode: TTSMode = TTSMode.PREDICT,
                 config_path: str = "config/training_config.yaml",
                 seed: Optional[int] = None,
                 wav_directory: str = "",
                 metadata_path: str = "",
                 save_directory: str = "",
                 train_data_directory: str = "",
                 mel_directory: str = "mels",
                 pitch_directory: str = "pitch",
                 duration_directory: str = "duration",
                 character_pitch_directory: str = "char_pitch",
                 check_git_hash: bool = True,
                 **_):
        if mode in ['aligner', 'extract']:
            self.model_kind = 'aligner'
        else:
            self.model_kind = 'tts'

        self.yaml = ruamel.yaml.YAML()
        self.config = self._load_config(config_path)

        self.git_hash = self._get_git_hash() if check_git_hash else False

        # create paths
        self.wav_directory = Path(wav_directory)
        self.metadata_path = Path(metadata_path)

        self.data_dir = Path(train_data_directory)
        self.train_metadata_path = self.data_dir / "train_metadata.txt"
        self.valid_metadata_path = self.data_dir / "valid_metadata.txt"
        self.phonemized_metadata_path = self.data_dir / "phonemized_metadata.txt"

        self.mel_dir = self.data_dir / mel_directory
        self.pitch_dir = self.data_dir / pitch_directory
        self.duration_dir = self.data_dir / duration_directory
        self.pitch_per_char = self.data_dir / character_pitch_directory

        self.base_dir = Path(save_directory) / self.model_kind
        self.log_dir = self.base_dir / 'logs'
        self.weights_dir = self.base_dir / 'weights'

        # training parameters
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        if self.model_kind == 'aligner':
            self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
            self.stop_scaling = self.config.get('stop_loss_scaling', 1.)

        self.seed = seed

    def _load_config(self, config_path):
        config_path = Path(config_path)
        all_config = {}
        with open(str(config_path), 'rb') as session_yaml:
            session_config = self.yaml.load(session_yaml)
        if 'automatic' in session_config and session_config['automatic']:
            return session_config
        else:
            for key in ['dataset', 'training_data_settings', 'audio_settings', 'text_settings',
                        f'{self.model_kind}_settings']:
                all_config.update(session_config[key])
            return all_config

    @staticmethod
    def _get_git_hash():
        try:
            return subprocess.check_output(['git', 'describe', '--always']).strip().decode()
        except Exception as e:
            print(f'WARNING: could not retrieve git hash. {e}')

    def _check_hash(self):
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            if self.config['git_hash'] != git_hash:
                print(
                    f"WARNING: git hash mismatch. Current: {git_hash}. Training config hash: {self.config['git_hash']}")
        except Exception as e:
            print(f'WARNING: could not check git hash. {e}')

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * ' '
        print(tab + '-', key_name, ':', values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(dictionary[key], key_name=key, level=recursion_level)

    def print_config(self):
        print('\nCONFIGURATION', self.model_kind)
        self._print_dictionary(self.config)

    def update_config(self):
        self.config['git_hash'] = self.git_hash
        self.config['automatic'] = True

    def get_model(self):
        from model.models import Aligner, ForwardTransformer
        if self.git_hash:
            self._check_hash()
        if self.model_kind == 'aligner':
            return Aligner.from_config(self.config, max_r=self.max_r)
        else:
            return ForwardTransformer.from_config(self.config)

    def compile_model(self, model, beta_1=0.9, beta_2=0.98):
        import tensorflow as tf
        optimizer = Adam(self.learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=1e-9)
        if self.model_kind == 'aligner':
            model.compile_model(stop_scaling=self.stop_scaling, optimizer=optimizer)
        else:
            model.compile_model(optimizer=optimizer)

    def dump_config(self):
        self.update_config()
        with open(self.base_dir / f"config.yaml", 'w') as model_yaml:
            self.yaml.dump(self.config, model_yaml)

    def create_remove_dirs(self, clear_dir=False, clear_logs=False, clear_weights=False):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True)
        self.pitch_dir.mkdir(exist_ok=True)
        self.pitch_per_char.mkdir(exist_ok=True)
        self.mel_dir.mkdir(exist_ok=True)
        self.duration_dir.mkdir(exist_ok=True)
        if clear_dir:
            delete = input(f'Delete {self.log_dir} AND {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f'Delete {self.log_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f'Delete {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)

    def load_model(self, checkpoint_path: str = None, verbose=True):
        import tensorflow as tf
        from utils.scheduling import reduction_schedule
        model = self.get_model()
        self.compile_model(model)
        ckpt = tf.train.Checkpoint(net=model)

        if checkpoint_path:
            ckpt.restore(checkpoint_path)
            if verbose:
                print(f'restored weights from {checkpoint_path} at step {model.step}')
        else:
            manager = tf.train.CheckpointManager(ckpt, self.weights_dir / "latest",
                                                 max_to_keep=None)
            if manager.latest_checkpoint is None:
                print(f'WARNING: could not find weights file. Trying to load from \n {manager.directory}')
                print('Edit config to point at the right log directory.')
            ckpt.restore(manager.latest_checkpoint)
            if verbose:
                print(f'restored weights from {manager.latest_checkpoint} at step {model.step}')
        if self.model_kind == 'aligner':
            reduction_factor = reduction_schedule(model.step, self.config['reduction_factor_schedule'])
            model.set_constants(reduction_factor=reduction_factor)
        return model
