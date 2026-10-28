# GenAI Lab Portal

A dependency-free local dashboard for the existing GenAI Lab experiments. The
portal uses Python's standard library and launches each experiment in its own
process, from its original project directory, with its existing virtual
environment when one is present.

## Launch on Windows

From `D:\gen_ai` in PowerShell:

```powershell
py -3.14 .\portal\portal.py
```

Open <http://127.0.0.1:8760>. Keep the terminal open while using the portal;
press `Ctrl+C` to stop the portal and any experiment processes started from it.
No portal package installation is required.

The portal has exactly ten experiment cards: Experiments 2–10, with 5A and 5B
as separate cards. Experiment 1 was omitted from the portal at the user's
selection; its existing dashboard and experiment code remain unchanged.

The portal checks each project's actual virtual-environment paths and verifies
the app's required top-level modules without importing model binaries. It
prefers a compatible environment with the required modules; otherwise it may
use the installed Python 3.12 launcher if that runtime already has those
modules. It never creates an environment or installs packages. Cards
distinguish a missing project, missing/incompatible environment, missing
dependencies, and launch failure. A failed experiment is isolated from the
portal and other cards; startup output is retained in `portal\logs`.

## Experiments 6–10 environment setup

Separate Python 3.12 virtual environments have been created in each project's
`.venv` folder. They contain only Python and pip; no experiment packages or
model weights were installed or downloaded. The portal selects these local
environments and reports missing dependencies rather than silently using an
incompatible interpreter. Each project's existing `requirements.txt` matches
the app/training imports, so the manifests were not replaced.

Run the following from the matching experiment folder after reviewing the
package downloads. Use the official PyTorch selector to choose a CPU or CUDA
wheel that matches the machine before installing the remaining requirements.
This machine currently has 15.5 GB RAM, about 85 GB free on D:, and Intel
integrated graphics (no CUDA GPU detected).

| Experiment | Python / entry point | Data, model, or hardware requirement |
|---|---|---|
| 6 | Python 3.12; Flask `app.py` | CelebA is not bundled; generation also needs training and a checkpoint. |
| 7 | Python 3.12; Gradio `app.py --checkpoint outputs/latest.pt` | A local image folder or ZIP is required for training; no trained checkpoint is present. |
| 8 | Python 3.12; Gradio `app.py` | The old parent `exp 8\.venv` is Python 3.14.7 and remains untouched. `facebook/musicgen-small` is downloaded on first generation (budget roughly 1–2 GB for weights/cache); a GPU is recommended, while CPU inference is slow. |
| 9 | Python 3.12; Streamlit `app.py` | First model use downloads Qwen2.5-0.5B (roughly 1–2 GB) and the all-MiniLM embedding model (under 100 MB). Windows Application Control previously blocked a Pandas compiled binary; if that recurs, use only an IT-approved runtime or supported OS. |
| 10 | Python 3.12; Gradio `app.py` | ZeroScope weights require about 14 GB of download/storage and an NVIDIA GPU with roughly 8 GB VRAM is the practical minimum. This machine does not meet that GPU requirement. |

Example setup and launch commands (do not run installation commands until the
download and hardware requirements are acceptable):

```powershell
# Experiment 6
Set-Location 'D:\gen_ai\exp 6\dcgan_celeba'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 8767

# Experiment 7
Set-Location 'D:\gen_ai\exp 7'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --dataset 'D:\path\to\faces.zip' --epochs 100 --batch-size 16 --output-dir outputs
.\.venv\Scripts\python.exe app.py --checkpoint outputs/latest.pt

# Experiment 8
Set-Location 'D:\gen_ai\exp 8\musicgen-text-to-music'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py

# Experiment 9
Set-Location 'D:\gen_ai\exp 9\personalized-ai-assistant'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py

# Experiment 10
Set-Location 'D:\gen_ai\exp 10'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

For Experiments 8 and 10, choose/install the appropriate PyTorch wheel before
the remaining requirements as their READMEs describe. Experiment 9's Pandas
binary restriction must not be bypassed; preflight checks intentionally do not
import compiled model or data-science binaries.

## Runtime notes

- Experiment 1's existing dashboard is not included in this portal's ten-card
  selection. Its known Windows Application Control restriction on the Pandas
  compiled module remains unresolved; the portal does not attempt to bypass it.
- Experiment 7 needs a trained `outputs\latest.pt` checkpoint before its
  existing Gradio application can start.
- Experiment 8 has a shared `exp 8\.venv`, but it uses Python 3.14.7 while its
  README targets Python 3.10–3.12; it is not selected as a compatible runtime.
- Experiment 6's Flask page can be opened, but generating faces requires a
  trained checkpoint.
- Experiment 4 requires Hugging Face and Supabase credentials. Experiments
  using pretrained models may require internet access, a first-run model
  download, and suitable memory or GPU hardware.
- Apps use separate local ports in the 8762-8771 range. If a port is already in
  use, stop the other process or resolve the conflict before launching.
