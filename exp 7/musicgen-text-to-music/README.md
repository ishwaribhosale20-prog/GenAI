# MusicGen Text-to-Music Lab

## Aim and objectives

Use Meta's pretrained `facebook/musicgen-small` text-to-music foundation model to generate short music clips from natural-language descriptions. Listen to the generated audio, save it as a WAV file, and compare how prompts about genre, instruments, mood, and tempo influence the result.

This project is **inference only**: it does not train a model and does not require collecting a dataset.

## Theory: foundation models and MusicGen

A foundation model is trained in advance on a large collection of data, then reused for tasks described by a user. MusicGen is Meta's Transformer-based text-to-music model. It converts a text description into audio-code sequences and decodes those sequences into a waveform. The project loads the pretrained `facebook/musicgen-small` weights from Hugging Face and generates audio conditioned on the prompt.

The prompt is the model's natural-language conditioning. Specific descriptions such as “gentle acoustic guitar, slow tempo, warm and peaceful” communicate more musical attributes than a one-word prompt. Generation is probabilistic, so repeated runs and different prompts/settings can produce different audio. Prompt wording guides results; it is not a guarantee that every instrument or style will be rendered exactly.

## Requirements and compatibility

The supported, recommended environment is **Python 3.10–3.12**. Dependencies use Hugging Face Transformers `4.44.2`, which includes `MusicgenForConditionalGeneration` and `AutoProcessor`. PyTorch is not forced to one CUDA build by this project: use the PyTorch build appropriate to your machine/runtime. Do not install AudioCraft into this project; it is a separate implementation with additional version and system dependencies.

`facebook/musicgen-small` generates up to about 30 seconds per sequence. The default is 8 seconds. The model weights are several gigabytes including loading overhead, so a GPU is recommended; CPU inference may be slow.

## Local installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

For macOS/Linux, create and activate the environment with `python3 -m venv .venv` and `. .venv/bin/activate`. If the installed PyTorch package does not match the local CUDA runtime, install the appropriate PyTorch/Torchvision build from the official PyTorch installation selector, then install this project's other requirements.

The first generation downloads model files from Hugging Face and may take several minutes. An internet connection is needed for the initial model download; subsequent runs can use the local model cache.

## Google Colab with GPU

1. Open `musicgen_colab.ipynb`.
2. Select **Runtime > Change runtime type > T4 GPU** or another available GPU.
3. Run the setup and dependency cells in order.
4. The first model load downloads pretrained weights. No dataset upload or training step is involved.
5. Generate a sample, listen in the notebook, save/download the WAV, then try other prompts.

The notebook also has an optional Drive-mount cell for persistent output and a final optional Gradio launch cell.

## Run the Gradio app

From this directory after installing requirements:

```bash
python app.py
```

The app loads the model lazily on the first generation request and reuses it for following requests. It provides a prompt textbox, five example prompts, duration control (1–30 seconds), generation status, audio playback, and a downloadable WAV. The **Compare prompts** tab accepts up to five prompts (one per line) and generates an audio player next to each prompt. The first request can be slower because it downloads and loads model weights.

To expose the app in a Colab session, the notebook's optional launch command uses `--share`; share links are accessible to others, so enable them only when intended.

## Example prompts

- Peaceful instrumental piano with soft flute and a relaxing atmosphere.
- Energetic electronic music with a fast tempo and powerful synthesizers.
- Indian classical instrumental music featuring sitar and tabla.
- Cinematic orchestral music with emotional strings and dramatic percussion.
- Gentle acoustic guitar music for studying.

## Generate, play, and save with Python

```python
from IPython.display import Audio, display
from music_generator import MusicGenerator

music = MusicGenerator()
result = music.generate(
    "Gentle acoustic guitar music for studying.",
    duration_seconds=8,
    output_dir="outputs",
    filename="musicgen_output.wav",
)
print(result.prompt, result.duration_seconds, result.sampling_rate, result.wav_path)
display(Audio(result.waveform, rate=result.sampling_rate))
```

The returned waveform is a mono NumPy float array. `MusicGenerator.save_wav` converts it to signed 16-bit PCM and writes a WAV header at the pretrained model's sampling rate (32 kHz for this model). In Colab, download it with:

```python
from google.colab import files
files.download(str(result.wav_path))
```

To save in Google Drive, mount Drive and pass a destination such as `/content/drive/MyDrive/musicgen_output.wav` as `output_dir` with `filename="musicgen_output.wav"`.

## Code overview

- `config.py` — the default model name, duration limits, output folder, guidance setting, and example prompts.
- `music_generator.py` — `MusicGenerator.load()` lazily loads and caches the processor/model; `generate()` validates the prompt and duration, performs inference, converts the model result into a waveform, and saves a WAV; `save_wav()` writes mono PCM audio at the supplied sample rate.
- `app.py` — builds the Gradio UI; `generate_music()` connects the Generate button to inference and playback/download; `compare_music_prompts()` generates up to five prompt examples side by side.
- `musicgen_colab.ipynb` — ordered Colab setup, model load, generation, playback, WAV save/download, prompt experiments, optional Drive, and optional Gradio app.
- `tests/test_music_generator.py` — checks prompt/duration validation, waveform conversion, and the WAV sampling rate/header without downloading the model.

## Limitations and troubleshooting

- **No GPU or out-of-memory:** use a GPU runtime, reduce duration, restart the runtime to release memory, or select CPU (slower). The app reports CUDA memory failures as a clear error.
- **Model download/load error:** check network access to `huggingface.co`, available disk/cache space, and the `facebook/musicgen-small` model ID. Run from Python 3.10–3.12 and install the pinned Transformers dependency. A fresh runtime may help after incompatible package changes.
- **Slow generation:** model inference takes time, scales with requested duration, and the first call also downloads weights. Keep the duration short for a practical demo.
- **Creative differences:** output quality and prompt adherence vary. MusicGen may not precisely follow all instruments, genre, or structure words.
- **Output length:** requested duration is approximated through the model's audio-token frame rate. Actual WAV length can differ slightly; the app reports measured duration and sample rate.
- **No training or dataset:** generated examples come from pretrained weights. They are not evidence that this project trained MusicGen or learned from a user dataset.

## Practical demonstration / viva

1. State that the system uses an already-trained Transformer foundation model and performs inference only.
2. Enter a prompt and explain the influence of genre, instrument, mood, and tempo details.
3. Set a short duration and generate; observe the loading/generation status and then play the resulting clip.
4. Download the WAV and verify that it is 32 kHz PCM audio.
5. Use the comparison tab to generate different genres or moods and listen for differences.
6. Explain that the text is conditioning, the Transformer predicts audio tokens, and the audio decoder reconstructs a waveform.
7. Mention GPU memory, generation time, and imperfect prompt adherence as limitations.

## Conclusion (practical record)

The practical demonstrates pretrained text-to-music generation using Meta's MusicGen-small model. A natural-language prompt conditions Transformer inference; the generated waveform is played in the interface and saved as a standard WAV file. Different descriptions can be compared without collecting data or training a model, while output quality, duration, and prompt adherence depend on model behavior and available compute.

## Verification

Run local checks without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

This verifies duration/prompt validation, waveform conversion, and WAV metadata. A real generation check requires downloading the pretrained model and sufficient memory; it is separate from these quick unit tests.
