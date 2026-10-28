# Text-to-Music Generation with MusicGen

## Aim

Generate short, original music samples from natural-language descriptions using
Meta's pretrained `facebook/musicgen-small` foundation model. Play each sample
and save it as a downloadable WAV file. This practical performs **inference
only**: it does not train a model or collect a dataset.

## Objectives

- Understand how a pretrained generative foundation model uses a text prompt.
- Generate and compare music with different instruments, genres, moods, and tempos.
- Listen to the result in a Gradio app or a Google Colab notebook.
- Save a correctly sampled, 16-bit PCM WAV file.

## Theory

A **foundation model** is a large model trained in advance on broad data and
then reused for many tasks. Here, the pretrained model is used as-is: a prompt
conditions its output, and inference runs the model to generate audio. No
weights are updated during this practical.

**MusicGen** is Meta's autoregressive Transformer-based text-to-music model. In
the Transformers implementation, a text encoder represents the prompt, a
Transformer predicts discrete audio tokens, and an audio decoder turns those
tokens into a waveform. The `facebook/musicgen-small` checkpoint has about 300
million parameters. Its model card describes a 32 kHz audio tokenizer and a
token rate of about 50 tokens per second. The app uses this rate to relate the
duration control to the number of generated tokens.

A **prompt** describes the sound the model should attempt to create. Useful
details include genre, instruments, mood, tempo, and context. Sampling and
other generation settings also affect the result, so the same prompt can
produce different compositions.

## Project files

- `app.py` — Gradio interface, playback, download, and status messages.
- `music_generator.py` — model loading, inference, input checks, and WAV writing.
- `config.py` — model identifier, duration limits, and example prompts.
- `musicgen_colab.ipynb` — ordered notebook workflow, including a Gradio option.
- `requirements.txt` — compatible Python package constraints.
- `test_music_generator.py` — quick tests for validation, stubbed inference, and
  WAV output; no model download is needed for these tests.

## Recommended environment: Google Colab

This project uses Hugging Face Transformers rather than AudioCraft, avoiding
AudioCraft's additional version and system-library constraints. Transformers
4.46.3 is pinned because its documented MusicGen API is used here. PyTorch is
left platform-selected so Colab can keep its CUDA-enabled build.

1. Open `musicgen_colab.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the cells from top to bottom. The notebook checks the runtime, installs
   the pinned Transformers/Gradio versions, downloads the pretrained model,
   generates and plays audio, writes WAV files, and offers download controls.
4. The model files are large. Allow time for the first download; later runs can
   reuse Colab's model cache.

The notebook warns and continues on CPU if CUDA is unavailable. CPU generation
works but can be much slower. Use a GPU runtime for the in-class demonstration.

## Run the Gradio application locally

Use Python 3.10–3.12 for the pinned Transformers/tokenizer stack. This
workspace's Python 3.14 and CPU-only PyTorch do not provide enough free memory
to load the checkpoint; use Colab or configure a supported environment with
suitable RAM/GPU memory.

1. Install a PyTorch build matching your operating system and CUDA version by
   following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
   On Colab, keep its preinstalled PyTorch.
2. In a terminal opened in this project folder, install the remaining
   requirements:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If your Python command is not `py -3.12`, use the command for your installed
   supported interpreter. On a GPU machine, install the correct PyTorch wheel
   before the requirements so pip can reuse it.
3. Start the app:

   ```powershell
   python app.py
   ```

4. Open the local Gradio URL printed in the terminal. Choose an example or
   type a prompt, select a duration from 1 to 20 seconds, then press
   **Generate Music**. Listen with the audio player or download the WAV.
5. Files are saved under `outputs/` with a timestamped `musicgen_output_...wav`
   name. The model is loaded only on the first request and reused by subsequent
   generations.

The first generation requires internet access to fetch model files from the
Hugging Face Hub. Generated files are local; they are not automatically copied
to Google Drive. To keep an output in Drive, mount Drive in Colab and copy the
WAV file to the desired Drive folder.

## Example prompts and expected results

Use the examples in the interface or notebook:

1. `Peaceful instrumental piano with soft flute and a relaxing atmosphere.`
2. `Energetic electronic music with a fast tempo and powerful synthesizers.`
3. `Indian classical instrumental music featuring sitar and tabla.`
4. `Cinematic orchestral music with emotional strings and dramatic percussion.`
5. `Gentle acoustic guitar music for studying.`

The expected output is a short, generated music clip that reflects aspects of
the prompt. It is not guaranteed to reproduce every instrument or style
perfectly. Compare clips by changing one prompt detail at a time while keeping
the duration fixed.

## How the code works

- `MusicGenerator.generate()` checks the prompt and duration, tokenizes the text,
  then calls the pretrained model inside PyTorch inference mode. The prompt is
  never used to train or fine-tune weights.
- The generator selects CUDA when available and otherwise uses CPU. It loads
  the processor and model once per `MusicGenerator` instance.
- The model returns audio as a tensor shaped like
  `(batch, channels, samples)`. The code verifies that it is non-empty, changes
  it into the sample layout required for WAV, clips values to the valid audio
  range, and writes 16-bit PCM using the model's configured sample rate.
- The Gradio audio component plays the generated WAV path. The file component
  provides the same file for download.
- The Colab notebook includes the same inference steps in separate, ordered
  cells so each practical stage can be demonstrated and explained.

## Optional output to Google Drive

After generating a file in Colab, mount Drive and copy it to a folder you
choose:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount("/content/drive")
drive_folder = Path("/content/drive/MyDrive/MusicGen")
drive_folder.mkdir(parents=True, exist_ok=True)
shutil.copy2("musicgen_output.wav", drive_folder / "musicgen_output.wav")
```

## Tests and verification

Run the fast tests from this folder:

```powershell
python -m unittest -v
```

These tests check empty prompts, duration limits, a stubbed inference call, and
mono/stereo WAV headers without downloading the model. A live model run must
also generate a non-empty tensor, verify its sampling rate/shape, play it, and
save it. The notebook contains the steps to perform those checks in Colab.

## Limitations and troubleshooting

- **Model download fails:** Check internet access and retry. The first run must
  fetch the model from `facebook/musicgen-small`; check available disk space.
- **Out of memory:** Use a Colab GPU runtime, restart the runtime, reduce
  duration, or close other memory-heavy applications. The maximum in this
  practical is 20 seconds.
- **CUDA unavailable:** The app falls back to CPU and states this in its status.
  Generation will be slower.
- **Missing dependencies:** Install PyTorch for your platform, then run
  `python -m pip install -r requirements.txt`.
- **No sound or an unsupported audio file:** Confirm the file ends in `.wav`
  and is opened by a WAV-compatible player. The project checks the WAV header,
  sample rate, channel count, and non-empty frame count before returning it.
- **Prompt does not match exactly:** Text-to-music generation is probabilistic.
  Try a more specific prompt or generate another sample.
- **Usage rights:** The model weights are distributed under the
  [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
  Check the model card and license terms before redistributing outputs or using
  the model commercially.

## Viva demonstration

1. Explain that MusicGen is a pretrained Transformer and that this practical
   performs inference, not training.
2. Show the prompt and identify the instrument, mood, genre, and tempo words.
3. Run generation and explain that the duration controls the generated token
   count; the GPU is selected when available.
4. Play the audio, then download and open the WAV file.
5. Change one part of the prompt and compare the second result.
6. State the configured sample rate (normally 32,000 Hz), generated duration,
   and why results vary between sampling runs.

## Conclusion

This practical demonstrates how a pretrained generative foundation model can
create music from natural language without collecting a dataset or training a
model. The generated composition can be auditioned and saved as a WAV file for
further study.
