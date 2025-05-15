from pathlib import Path

from utils.training_config_manager import TrainingConfigManager, TTSMode
from utils.argparser import tts_argparser

MODE = TTSMode.WEIGHTS

if __name__ == '__main__':
    parser = tts_argparser(MODE)
    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = str(Path(args.save_directory) / 'tts' / 'config.yaml')

    config = TrainingConfigManager(mode=MODE, **vars(args))

    model = config.load_model(checkpoint_path=args.checkpoint_path)  # None defaults to latest

    if args.target_dir is None:
        args.target_dir = config.base_dir / 'weights' / f'weights_step_{model.step}'

    model.save_model(args.target_dir)

    print('Done.')
    print(f'Model weights saved under {args.target_dir}')
