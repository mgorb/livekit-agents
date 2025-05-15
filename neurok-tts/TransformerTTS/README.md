<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/transformer_logo.png" width="400"/>
    <br>
</p>

<h2 align="center">
<p>Transformer-based Text-to-Speech in TensorFlow 2</p>
</h2>


Implementation of a non-autoregressive Transformer-based neural network for Text-to-Speech (TTS).

This is repository is managed by [TartuNLP](https://tartunlp.ai), and it is a fork of the implementation
by [Axel Springer](https://github.com/as-ideas/TransformerTTS). Our contributions compared to the original repository
are:

- Support for grapheme-based synthesis
- Multi-speaker synthesis
- Pretrained models for Estonian
- Open source TTS applications:
    - [Kratt application](https://koodivaramu.eesti.ee/tartunlp/text-to-speech)
    - [API](https://github.com/TartuNLP/text-to-speech-api)
      \+ [worker](https://github.com/TartuNLP/text-to-speech-worker) combo.
- Numerous minor changes to streamline training and make the repository easier to use with new datasets.

This code is based, among others, on the following papers:

- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://fastpitch.github.io/)

The models are compatible with the pre-trained vocoders:

- [MelGAN](https://github.com/seungwonpark/melgan)
- [HiFiGAN](https://github.com/jik876/hifi-gan)

For quick inference with these vocoders, checkout the Vocoding branch.

Being non-autoregressive, this Transformer model is:

- Robust: No repeats and failed attention modes for challenging sentences.
- Fast: With no autoregression, predictions take a fraction of the time.
- Controllable: It is possible to control the speed and pitch of the generated utterance.

## 🔈 Samples

Samples from the original implementation can be found [here](https://tartunlp.github.io/TransformerTTS/original).
Estonian and multispeaker samples can be found [here](https://tartunlp.github.io/TransformerTTS/).

## Updates

- 06/22: Multi-speaker synthesis (TartuNLP)
- 05/22: Merged updates from the original repository (TartuNLP)
- 06/21: Grapheme-based synthesis and Estonian models (TartuNLP)
- 06/20: Added normalisation and pre-trained models compatible with the
  faster [MelGAN](https://github.com/seungwonpark/melgan) vocoder.
- 11/20: Added pitch prediction. Autoregressive model is now specialized as an Aligner and Forward is now the only TTS
  model. Changed models architectures. Discontinued WaveRNN support. Improved duration extraction with Dijkstra
  algorithm.
- 03/20: Vocoding branch.

## 📖 Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
    - [Aligner](#train-aligner-model)
    - [TTS](#train-tts-model)
- [Prediction](#prediction)
- [Model Weights](#model-weights)

## Installation

Make sure you have:

* Python >= 3.6

Install espeak as phonemizer backend (for macOS use brew) if you plan to use phonemized inputs:

```
sudo apt-get install espeak
```

Then install the rest with pip:

```
pip install -r requirements.txt
```

Or use the `environment.yml` file to set up a Conda environment.

## Dataset

You can directly use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) to create the training dataset.

#### Configuration

* If training on LJSpeech, or if unsure, simply use ```config/training_config.yaml``` to
  create [MelGAN](https://github.com/seungwonpark/melgan) or [HiFiGAN](https://github.com/jik876/hifi-gan) compatible
  models
    * swap the content of ```data_config_wavernn.yaml``` in ```config/training_config.yaml``` to create models
      compatible with [WaveRNN](https://github.com/fatchord/WaveRNN)
* **EDIT PATHS**: in `config/training_config.yaml` edit the paths to point at your dataset and log folders **OR** use
  the command line flags to override them. Information about configuration flags can be seen with the `-h` flag of each
  script.

#### Custom dataset

Prepare a folder containing your metadata and wav files, for instance

```
|- dataset_folder/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```

if `metadata.csv` has the following format
``` wav_file_name|transcription ``` or ``` wav_file_name|transcription|speaker_id ```
you can use the ljspeech preprocessor in ```data/metadata_readers.py```, otherwise add your own under the same file.

Make sure that:

- the metadata reader function name is the same as ```metadata_reader``` field in ```training_config.yaml```.
- the metadata file (can be anything) is specified under ```metadata_path``` in ```training_config.yaml```
- for multispeaker training, review the `multispeaker` and `n_speakers` values.
- to disable phonemization, edit the `text_settings` section of the configuration file.

## Training

Change the ```--config``` argument based on the configuration of your choice.

### Train Aligner Model

#### Create training dataset

```bash
python create_training_data.py --config config/training_config.yaml
```

This will populate the training data directory.

#### Training

```bash
python train_aligner.py --config config/training_config.yaml
```

### Train TTS Model

#### Compute alignment dataset

First use the aligner model to create the durations dataset

```bash
python extract_durations.py --config config/training_config.yaml
```

this will add the duration as well as the char-wise pitch folders to the training data directory.

#### Training

```bash
python train_tts.py --config config/training_config.yaml
```

#### Training & Model configuration

- Training and model settings can be configured in `training_config.yaml`

#### Resume or restart training

- To resume training simply use the same configuration files
- To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (
  both) or `--reset_logs`, `--reset_weights`

#### Monitor training

```bash
tensorboard --logdir /logs/directory/
```

![Tensorboard Demo](https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/tboard_demo.gif)

## Prediction

### From the latest training checkpoint

```commandline
python predict_tts.py --text "Please, say something." --save-directory /path/to/save_dir/
```

### With model weights

From command line with

```commandline
python predict_tts.py --text "Please, say something." -p /path/to/weights_dir/
```

Or in a python script

```python
from model.models import ForwardTransformer
from data.audio import Audio

model = ForwardTransformer.load_model('/path/to/weights_dir/')
audio = Audio.from_config(model.config)
out = model.predict('Please, say something.', speaker_id=0)

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
```

## Model Weights

Newer models are added to the [Releases](https://github.com/TartuNLP/TransformerTTS/releases) of this repository.

Access the original pre-trained models with the API call.

Old weights

| Model URL                                                                                                                                                              | Commit  | Vocoder Commit |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| [ljspeech_tts_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ljspeech_weights_tts.zip)                                                             | 0cd7d33 | aca5990        |
| [ljspeech_melgan_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_melgan_forward_transformer.zip)                    | 1c1cb03 | aca5990        |
| [ljspeech_melgan_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_melgan_autoregressive_transformer.zip)   | 1c1cb03 | aca5990        |
| [ljspeech_wavernn_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_wavernn_forward_transformer.zip)                  | 1c1cb03 | 3595219        |
| [ljspeech_wavernn_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_wavernn_autoregressive_transformer.zip) | 1c1cb03 | 3595219        |
| [ljspeech_wavernn_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_forward_transformer.zip)                          | d9ccee6 | 3595219        |
| [ljspeech_wavernn_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_autoregressive_transformer.zip)         | d9ccee6 | 3595219        |
| [ljspeech_wavernn_autoregressive_model_v1](https://github.com/as-ideas/tts_model_outputs/tree/master/ljspeech_transformertts)                                          | 2f3a1b5 | 3595219        |

## Maintainers

* [TartuNLP](https://tartunlp.ai) - the NLP research group at the University of Tartu.

## Special thanks

[Francesco Cardinale](https://github.com/cfrancesco) from Axel Springer for the original implementation.

[MelGAN](https://github.com/seungwonpark/melgan) and [WaveRNN](https://github.com/fatchord/WaveRNN): data normalization
and samples' vocoders are from these repos.

[Erogol](https://github.com/erogol) and the Mozilla TTS team for the lively exchange on the topic.

## Copyright

See [LICENSE](LICENSE) for details.
