# Text-to-Video Generation Using a Pretrained Model

## Aim

Generate, preview, and download a short MP4 from a natural-language description
using a pretrained text-to-video diffusion model. This is inference only: the
project collects no dataset and does not train or fine-tune a model.

## Objectives

- Explain how a video diffusion model uses text to synthesize a sequence of frames.
- Experiment with prompts and reproducible generation settings.
- Export real model-generated frames into a playable MP4.
- Run the same practical in a Gradio app or Google Colab.

## Selected model and license

The project uses [`cerspense/zeroscope_v2_576w`](https://huggingface.co/cerspense/zeroscope_v2_576w).
Hugging Face currently identifies it as a public, ungated Diffusers
text-to-video model. Its model card reports 576 x 320 output, training at
24 frames, and about 7.9 GB of VRAM for a 30-frame render. The card's example
uses `TextToVideoSDPipeline`, a DPM-Solver scheduler, and CPU offload.

**License notice:** the model metadata and model card specify
**Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC-4.0)**. It is
available without an API key, but it is **not an unrestricted open-source
license**. This project is intended for non-commercial educational use. Read
the license and verify that your intended use is permitted before distributing
outputs or using the model commercially. The model repository does not contain
a separate `LICENSE` file; the license declaration is in its model-card
metadata.

The Apache-2.0-licensed
[`zai-org/CogVideoX-2b`](https://huggingface.co/zai-org/CogVideoX-2b) is a
public alternative for projects that require a permissive model license, but it
is a different, newer Diffusers pipeline with larger hardware and dependency
requirements. It is not silently substituted here: ZeroScope is the requested
starting model and its published memory estimate is a closer fit for a Colab
T4-class GPU.

## How text-to-video diffusion works

A diffusion model starts with noisy visual data and repeatedly denoises it
while conditioning on a text embedding. A video model generates a connected
sequence rather than treating every frame as an unrelated picture. Its
spatiotemporal layers help represent both image content and motion. The prompt
embedding guides the scene, objects, action, and visual appearance.

### Temporal consistency

Temporal consistency means that subjects, objects, and scene details remain
recognizable as the frames change. Motion can still flicker, deform, or drift:
the model is trained on a finite set of clips and is not guaranteed to produce
physically correct or perfectly smooth video. Clear prompts with one simple
action are often easier to follow than prompts containing many simultaneous
events.

### Prompt engineering

Try specifying:

1. **Scene/environment:** lake, beach, forest, or city.
2. **Subject:** a boat, butterfly, vehicle, or robot.
3. **Action/motion:** sailing, flying, walking, or waves rolling.
4. **Camera:** wide shot, close-up, or slow camera movement.
5. **Lighting/mood:** sunrise, neon lights, warm sunset, or peaceful.
6. **Style:** cinematic, realistic, or animated.

Example: “A small boat sailing across a calm lake at sunrise, gentle water
movement, cinematic wide shot.” Results vary with prompt wording, seed, model,
and inference settings.

## Project structure

```text
text-to-video-generation/
├── app.py
├── config.py
├── video_generator.py
├── requirements.txt
├── README.md
├── text_to_video_colab.ipynb
├── tests/
│   └── test_video_generator.py
└── outputs/                  # created automatically after generation
```

## Hardware and software

- Python 3.10–3.12 is the intended environment. This workspace was inspected
  with Python 3.12.10.
- A CUDA-capable NVIDIA GPU with approximately 8 GB or more of VRAM is the
  minimum practical target; 12–16 GB is more comfortable. The model card reports
  7.9 GB for 30 frames, so memory headroom matters.
- CPU offload and attention slicing are enabled when CUDA is available. These
  can reduce VRAM use but may make generation slower. CPU-only inference is
  supported by the pipeline code but will be very slow and can require
  substantial system RAM.
- Downloading model weights requires an internet connection and roughly 14 GB
  of repository storage. First load can take several minutes; generation can
  also take several minutes.
- CUDA use requires a compatible NVIDIA driver and a PyTorch build for the
  installed CUDA environment.

## Windows installation and launch

Open PowerShell in this project folder. Create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch appropriate to your hardware. For CUDA, use the command
recommended by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/);
for example, the CUDA 12.4 wheel command (only if supported by your driver):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

For a CPU-only environment, install the CPU wheel instead:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining pinned dependencies and start the interface:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal. On Windows, if script
activation is blocked, use `.\.venv\Scripts\python.exe` in place of `python`
instead of changing the system execution policy.

## Google Colab GPU setup

1. Open `text_to_video_colab.ipynb` in Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
3. Run notebook cells from top to bottom.
4. Wait for the initial model download; it is large.
5. Edit the prompt and settings, then run the generation cell.
6. Preview the returned MP4 in the notebook and run the download cell.

The notebook uses Colab's preinstalled CUDA-enabled PyTorch rather than
replacing it with a generic wheel. It also includes an optional Google Drive
mount-and-copy cell. The notebook is self-contained; the local app is not
required for the Colab workflow.

## Generate, preview, and download

In Gradio, type a prompt or select one of the example prompts, choose settings,
and press **Generate video**. The actual generated MP4 appears in the preview
and download controls. The file is also saved under `outputs/` with a safe
prompt-based name and timestamp. A session history table records the prompt,
duration, FPS, settings, and filename.

In a notebook, run the prompt and settings cells, generate the frames, and
encode the MP4. The notebook confirms the output exists and is non-empty before
displaying or downloading it.

The current implementation supports 8–32 frames, 1–24 FPS, 10–50 inference
steps, guidance scale 1–15, and a seed of `-1` (random) or an unsigned
32-bit integer. Video duration is `number of frames / FPS`. These are practical
UI bounds; the model was trained at 24 frames and 576 x 320. Increasing frames
raises memory and compute costs. Increasing inference steps generally takes
longer; guidance changes prompt adherence and can affect visual quality.

## Example prompts

- A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot.
- A butterfly flying through a colorful garden, flowers moving gently in the breeze.
- A futuristic city at night with flying vehicles, glowing neon lights, and slow camera movement.
- Waves gently hitting a beach at sunset, warm lighting and a peaceful atmosphere.
- A small robot walking through a green forest, cinematic lighting and smooth motion.

## Important functions (beginner guide)

- `GenerationSettings.validate()` checks that each requested value is inside
  the supported range before model work begins.
- `validate_prompt()` rejects empty or excessively long prompts.
- `VideoGenerator._load_pipeline()` downloads and loads the model only once,
  selects the available compute device, and configures memory-saving options.
- `VideoGenerator.generate()` runs inference, reports progress, and sends the
  generated frames to the MP4 encoder.
- `save_video()` encodes RGB frames with the H.264 codec and verifies the MP4
  exists and is non-empty.
- `app.py` connects these functions to prompt controls, settings, progress,
  video preview, a download link, and a session history table.

## Optional: save to Google Drive

The notebook includes an optional cell to mount Drive and copy the verified MP4.
In Colab, the copy command is:

```python
from google.colab import drive
drive.mount("/content/drive")
shutil.copy2(video_path, "/content/drive/MyDrive/generated_video.mp4")
```

## Troubleshooting

- **No GPU / CUDA unavailable:** choose a GPU runtime in Colab, then rerun the
  hardware-check cell. Integrated graphics do not provide CUDA.
- **Insufficient VRAM / CUDA out of memory:** reduce frame count and inference
  steps, close other GPU tasks, and use a 12–16 GB GPU. The app enables CPU
  offload and attention slicing on CUDA, but that cannot guarantee success on
  every GPU.
- **Download or model-load error:** check internet access, disk space, and RAM;
  retry after the Hugging Face model repository is reachable.
- **MP4 encoding error:** reinstall the pinned `imageio` and `imageio-ffmpeg`
  dependencies and check that the output folder is writable.
- **Slow generation:** this model is computationally demanding. Start with 16
  frames and 15 inference steps, then increase settings for later experiments.
- **Prompt does not match:** simplify the description to one subject and one
  visible action; compare seeds using the same settings.
- **License question:** ZeroScope is CC-BY-NC-4.0. Do not assume commercial use
  is allowed; review the linked model card and license terms.

## Tests and verification status

Run the local unit tests using:

```powershell
py -3.12 -m unittest discover -s tests -v
```

These tests cover prompt/settings validation and encode real sample frames to
MP4. A model-load and inference smoke test requires downloading the checkpoint
and access to a sufficiently capable CUDA GPU.

For this inspected workspace: Python 3.12.10 is present, but no project
dependencies are installed; no NVIDIA CUDA device is detected (Intel integrated
graphics only). Therefore actual ZeroScope loading, model inference, GPU memory
validation, and generated-video preview/download could not be run locally.
They must be validated in the Colab GPU notebook; no placeholder video is used
or represented as a successful model output.

## Conclusion (practical record)

This practical demonstrates inference from a pretrained text-to-video
foundation model. A natural-language prompt and generation settings condition
the model to produce a sequence of frames, which is encoded as a verified MP4
for playback and download. The experiment illustrates prompt engineering,
temporal consistency, reproducibility, GPU memory constraints, and responsible
use of a model under its non-commercial license. No dataset or model training
is needed.

## Viva questions

1. **What is inference?** Using learned model weights to produce an output,
   without updating those weights.
2. **Why does a video model need temporal information?** To model how visual
   content changes between frames and help preserve motion and identity.
3. **What does the random seed do?** It initializes the random noise so a run
   can be repeated more consistently with the same settings and software.
4. **How do inference steps affect generation?** More denoising steps usually
   increase compute time and may improve details, but do not guarantee a better
   result.
5. **Why is video more memory-intensive than an image?** The model processes
   many frames and their spatial/temporal features together.
6. **What is the model's license?** CC-BY-NC-4.0, which includes
   non-commercial restrictions; verify the terms for the intended use.
