from pathlib import Path

import numpy as np

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser
from model.factory import tts_ljspeech
from data.audio import Audio
from model.models import ForwardTransformer

MODE = TTSMode("predict")

if __name__ == '__main__':
    parser = tts_argparser(MODE)
    args = parser.parse_args()

    fname = None
    text = None

    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()

    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')

    if args.path is not None:
        print(f'Loading model from {args.path}')
        model = ForwardTransformer.load_model(args.path)
    else:
        print(f'Trying to load the latest checkpoint from model from {args.save_directory}')
        if args.config_path is None:
            args.config_path = str(Path(args.save_directory) / 'tts' / 'config.yaml')
        config_manager = TrainingConfigManager(mode=MODE, **vars(args))
        model = config_manager.load_model()

    file_name = f"{fname}_{model.config['step']}"
    outdir = outdir / 'outputs' / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    output_path = (outdir / file_name).with_suffix('.wav')

    audio = Audio.from_config(model.config)
    print(f'Output wav under {output_path.parent}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons, args.speaker_id)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: "{phons}"')
            print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, speaker_id=args.speaker_id, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        wav = audio.reconstruct_waveform(mel)
        wavs.append(wav)
        if args.store_mel:
            np.save(str((outdir / (file_name + f'_{i}')).with_suffix('.mel')), out['mel'].numpy())
        if args.single:
            audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
    audio.save_wav(np.concatenate(wavs), output_path)
