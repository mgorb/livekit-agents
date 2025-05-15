from argparse import ArgumentParser
from .training_config_manager import TTSMode


def tts_argparser(mode: TTSMode):
    parser = ArgumentParser()

    parser.add_argument('--config', dest='config_path', type=str,
                        help="Path to the configuration file")
    parser.add_argument('--seed', type=int)

    config_group = parser.add_argument_group("Options to override values in the session config file")

    config_group.add_argument('--wav-directory', type=str, default="",
                              help="A directory with .wav files. All file paths in metadata are relative to this path.")
    config_group.add_argument('--metadata-path', type=str, default="",
                              help="A path to a metadata file that specifies wav file paths, transcriptions and "
                                   "optionally the speaker's identity.")
    config_group.add_argument('--save-directory', type=str, default="",
                              help="A path where the models and logs are saved or laoded from, will be created if it "
                                   "doesn't exist.")
    config_group.add_argument('--train-data-directory', type=str, default="",
                              help="The root where preprocessed training data is stored "
                                   "(metadata files, mes, pitch contours, durations, etc.)")

    data_group = parser.add_argument_group(
        "Options to override specific data directory paths."
        "If a relative path is given, it will be relative to the training data directory."
    )
    data_group.add_argument('--mel-directory', type=str, default="mels")
    data_group.add_argument('--pitch-directory', type=str, default="pitch")
    data_group.add_argument('--duration-directory', type=str, default="duration")
    data_group.add_argument('--character-pitch-directory', type=str, default="char_pitch")

    if mode == TTSMode.ALIGNER:
        parser.add_argument('--test-files', nargs='+', default=["test_files/aligner_test_sentences.txt"],
                            help="A list of text files with one sentence per line to generate test samples with."
                                 "Predictions are stored in tensorboard logs.")
    if mode == TTSMode.TTS:
        parser.add_argument('--test-files', nargs='+', default=["test_files/test_sentences.txt"],
                            help="A list of text files with one sentence per line to generate test samples with."
                                 "Predictions are stored in tensorboard logs.")

    if mode in [TTSMode.ALIGNER, TTSMode.TTS]:
        parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                            help="deletes everything under this config's folder.")
        parser.add_argument('--reset_logs', dest='clear_logs', action='store_true',
                            help="deletes logs under this config's folder.")
        parser.add_argument('--reset_weights', dest='clear_weights', action='store_true',
                            help="deletes weights under this config's folder.")

    elif mode == TTSMode.DATA:
        parser.add_argument('--skip-phonemes', action='store_true')
        parser.add_argument('--skip-mels', action='store_true')

    elif mode == TTSMode.EXTRACT:
        parser.add_argument('--best', dest='best', action='store_true',
                            help='Use best head instead of weighted average of heads.')
        parser.add_argument('--autoregressive_weights', type=str,
                            help='Explicit path to autoregressive model weights.')
        parser.add_argument('--skip_char_pitch', dest='skip_char_pitch', action='store_true')
        parser.add_argument('--skip_durations', dest='skip_durations', action='store_true')

    elif mode == TTSMode.PREDICT:
        parser.add_argument('--path', type=str)
        parser.add_argument('--text', type=str)
        parser.add_argument('--file', type=str)
        parser.add_argument('--outdir', type=str)
        parser.add_argument('--store_mel', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--single', action='store_true')
        parser.add_argument('--speaker-id', default=0, type=int)

    elif mode == TTSMode.WEIGHTS:
        parser.add_argument('--checkpoint-path', type=str)
        parser.add_argument('--target-dir', type=str)

    return parser
