"""Local, dependency-free launcher for the GenAI Lab experiments."""

from __future__ import annotations

import atexit
import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTAL_DIR = Path(__file__).resolve().parent
PORTAL_HOST = "127.0.0.1"
PORTAL_PORT = 8760
PYTHON_LAUNCHER = shutil.which("py")
PORTAL_TOKEN = secrets.token_urlsafe(32)
PROCESS_LOCK = threading.Lock()
PROCESSES: dict[str, dict[str, Any]] = {}
RUNTIME_CACHE_LOCK = threading.Lock()
RUNTIME_CACHE: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, Any]]] = {}
PORTAL_HTML = (PORTAL_DIR / "index.html").read_text(encoding="utf-8")

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "exp2",
        "number": "02",
        "label": "EXPERIMENT 02",
        "title": "TensorFlow & Python Operations for GenAI Foundations",
        "description": "Explore Python, NumPy, and TensorFlow tensor operations through reusable lab modules.",
        "framework": "Streamlit",
        "technologies": ["TensorFlow", "NumPy", "Python"],
        "icon": "cpu",
        "accent": "violet",
        "folder": PROJECT_ROOT / "exp 2" / "tensor_genai_lab",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe")],
        "required_modules": ["streamlit", "tensorflow", "numpy"],
        "port": 8762,
        "kind": "streamlit",
        "setup": "The project has its own .venv. If dependencies are missing, install this experiment's requirements in that environment.",
    },
    {
        "id": "exp3",
        "number": "03",
        "label": "EXPERIMENT 03",
        "title": "Text Generation Question–Answer System",
        "description": "Ask questions with Google's FLAN-T5 model and see transformer text generation in action.",
        "framework": "Gradio",
        "technologies": ["PyTorch", "Transformers", "FLAN-T5"],
        "icon": "message",
        "accent": "cyan",
        "folder": PROJECT_ROOT / "exp 3" / "text_generation_qa",
        "entry": "app.py",
        "python_envs": [
            Path(".venv/Scripts/python.exe"),
            Path("../.venv/Scripts/python.exe"),
        ],
        "required_modules": ["gradio", "torch", "transformers", "sentencepiece", "accelerate"],
        "port": 8763,
        "kind": "gradio",
        "setup": "The application-specific text_generation_qa\\.venv contains its Gradio, PyTorch, Transformers, SentencePiece, and Accelerate dependencies. First run may download google/flan-t5-base and needs internet access and several GB of storage.",
    },
    {
        "id": "exp4",
        "number": "04",
        "label": "EXPERIMENT 04",
        "title": "Hugging Face AI Image Generator",
        "description": "Create images from text prompts through the Hugging Face inference service.",
        "framework": "Streamlit",
        "technologies": ["Hugging Face", "FLUX.1-schnell", "Supabase"],
        "icon": "image",
        "accent": "pink",
        "folder": PROJECT_ROOT / "exp 4",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe")],
        "required_modules": ["streamlit", "huggingface_hub", "supabase", "PIL"],
        "port": 8764,
        "kind": "streamlit",
        "setup": "The project has its own .venv. Image generation requires configured HF_TOKEN and Supabase credentials plus internet access.",
    },
    {
        "id": "exp5a",
        "number": "05A",
        "label": "EXPERIMENT 05A",
        "title": "Conditional VAE MNIST Digit Generator",
        "description": "Generate a handwritten digit on demand with a trained conditional variational autoencoder.",
        "framework": "Flask",
        "technologies": ["TensorFlow", "Keras", "MNIST"],
        "icon": "spark",
        "accent": "amber",
        "folder": PROJECT_ROOT / "exp 5 A",
        "entry": "app.py",
        "python_envs": [Path(".venv312/Scripts/python.exe"), Path("venv/Scripts/python.exe")],
        "required_modules": ["flask", "tensorflow", "numpy"],
        "port": 8765,
        "kind": "flask",
        "setup": "The project has an existing Python environment and trained decoder. Flask and TensorFlow must import successfully in that environment.",
    },
    {
        "id": "exp5b",
        "number": "05B",
        "label": "EXPERIMENT 05B",
        "title": "MNIST VAE Latent Space Explorer",
        "description": "Explore a trained MNIST VAE latent space and generate digits from selected latent coordinates.",
        "framework": "Flask",
        "technologies": ["TensorFlow", "Keras", "Latent Space"],
        "icon": "orbit",
        "accent": "orange",
        "folder": PROJECT_ROOT / "exp 5 B",
        "entry": "app.py",
        "python_envs": [Path(".venv312/Scripts/python.exe")],
        "required_modules": ["flask", "tensorflow", "numpy", "PIL"],
        "port": 8766,
        "kind": "flask",
        "setup": "The project has an existing Python environment, decoder, and latent-point data. Flask and TensorFlow must import successfully.",
    },
    {
        "id": "exp6",
        "number": "06",
        "label": "EXPERIMENT 06",
        "title": "DCGAN Face Generation on CelebA",
        "description": "Train a compact convolutional GAN and generate synthetic 64 × 64 face images.",
        "framework": "Flask",
        "technologies": ["PyTorch", "DCGAN", "CelebA"],
        "icon": "face",
        "accent": "green",
        "folder": PROJECT_ROOT / "exp 6" / "dcgan_celeba",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe"), Path("venv/Scripts/python.exe")],
        "required_modules": ["flask", "torch", "numpy", "matplotlib", "PIL"],
        "port": 8767,
        "kind": "flask",
        "setup": "Python 3.12 environment: from exp 6\\dcgan_celeba run `py -3.12 -m venv .venv`, then `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`. Launch via the portal or run `.\\.venv\\Scripts\\python.exe -m flask --app app run --host 127.0.0.1 --port 8767`. Face generation needs the CelebA dataset and a trained checkpoint; neither is included.",
        "model_note": "No trained checkpoint is currently present; generation becomes available after training.",
    },
    {
        "id": "exp7",
        "number": "07",
        "label": "EXPERIMENT 07",
        "title": "Educational StyleGAN Face Lab",
        "description": "Explore mapping, style mixing, layer controls, and noise in a compact educational StyleGAN.",
        "framework": "Gradio",
        "technologies": ["PyTorch", "StyleGAN", "Style Mixing"],
        "icon": "layers",
        "accent": "indigo",
        "folder": PROJECT_ROOT / "exp 7",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe"), Path("venv/Scripts/python.exe")],
        "required_modules": ["gradio", "torch", "numpy", "matplotlib", "PIL"],
        "min_python": (3, 10),
        "port": 8768,
        "kind": "gradio",
        "checkpoint": Path("outputs/latest.pt"),
        "model_note": "No trained outputs/latest.pt checkpoint is present; train the model before launching.",
        "setup": "Python 3.12 environment: from exp 7 run `py -3.12 -m venv .venv`, then `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`. Train with `.\\.venv\\Scripts\\python.exe train.py --dataset <image-folder-or-zip> --epochs 100 --batch-size 16 --output-dir outputs`; then launch with `.\\.venv\\Scripts\\python.exe app.py --checkpoint outputs/latest.pt` or use the portal. The dataset is not bundled and training is compute-intensive.",
    },
    {
        "id": "exp8",
        "number": "08",
        "label": "EXPERIMENT 08",
        "title": "Text-to-Music Generation with MusicGen",
        "description": "Turn natural-language prompts into short music clips with Meta's pretrained MusicGen model.",
        "framework": "Gradio",
        "technologies": ["PyTorch", "Transformers", "MusicGen"],
        "icon": "music",
        "accent": "teal",
        "folder": PROJECT_ROOT / "exp 8" / "musicgen-text-to-music",
        "entry": "app.py",
        "python_envs": [
            Path(".venv/Scripts/python.exe"),
            Path("venv/Scripts/python.exe"),
            Path("../.venv/Scripts/python.exe"),
        ],
        "required_modules": ["gradio", "torch", "numpy", "transformers", "huggingface_hub"],
        "min_python": (3, 10),
        "max_python": (3, 12),
        "port": 8769,
        "kind": "gradio",
        "setup": "Use a project-local Python 3.12 environment (the existing parent exp 8\\.venv is Python 3.14.7 and is incompatible): from exp 8\\musicgen-text-to-music run `py -3.12 -m venv .venv`, then `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` after selecting an appropriate PyTorch wheel. Launch with `.\\.venv\\Scripts\\python.exe app.py` or use the portal. MusicGen small is about 1.5 GB of model weights (FP32); allow additional cache and generation memory. The app does not download weights until generation is requested; a GPU is recommended.",
    },
    {
        "id": "exp9",
        "number": "09",
        "label": "EXPERIMENT 09",
        "title": "Personalized AI Assistant",
        "description": "Compare a base language model, document-grounded RAG, and optional LoRA fine-tuning.",
        "framework": "Streamlit",
        "technologies": ["LLM", "RAG", "FAISS", "LoRA"],
        "icon": "brain",
        "accent": "blue",
        "folder": PROJECT_ROOT / "exp 9" / "personalized-ai-assistant",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe"), Path("venv/Scripts/python.exe")],
        "required_modules": [
            "streamlit", "transformers", "torch", "sentence_transformers",
            "faiss", "pypdf", "docx", "accelerate", "peft", "trl", "numpy", "pandas",
        ],
        "port": 8770,
        "kind": "streamlit",
        "setup": "Python 3.12 environment: from exp 9\\personalized-ai-assistant run `py -3.12 -m venv .venv`, then `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` and launch with `.\\.venv\\Scripts\\python.exe -m streamlit run app.py` or use the portal. First model use downloads Qwen2.5-0.5B (roughly 1–2 GB) and all-MiniLM-L6-v2 (under 100 MB). Windows Application Control has previously blocked a Pandas compiled binary; if that recurs, stop and use an IT-approved runtime or another supported OS—do not change security policy.",
    },
    {
        "id": "exp10",
        "number": "10",
        "label": "EXPERIMENT 10",
        "title": "Text-to-Video Generation with a Pretrained Model",
        "description": "Generate a short video from a text prompt with the ZeroScope diffusion pipeline.",
        "framework": "Gradio",
        "technologies": ["PyTorch", "Diffusers", "ZeroScope"],
        "icon": "video",
        "accent": "violet",
        "folder": PROJECT_ROOT / "exp 10",
        "entry": "app.py",
        "python_envs": [Path(".venv/Scripts/python.exe"), Path("venv/Scripts/python.exe")],
        "required_modules": [
            "gradio", "torch", "diffusers", "transformers", "accelerate",
            "huggingface_hub", "numpy", "PIL", "imageio",
        ],
        "min_python": (3, 10),
        "max_python": (3, 12),
        "port": 8771,
        "kind": "gradio",
        "setup": "Python 3.12 environment: from exp 10 run `py -3.12 -m venv .venv`, then `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` after selecting a suitable PyTorch wheel. Launch with `.\\.venv\\Scripts\\python.exe app.py` or use the portal. The ZeroScope model download is about 14 GB; plan for about 20 GB free disk and an NVIDIA GPU with roughly 8 GB VRAM or more. Model weights download only when generation is requested.",
    },
]
EXPERIMENT_BY_ID = {item["id"]: item for item in EXPERIMENTS}


def _check_runtime(command: list[str], modules: list[str]) -> dict[str, Any]:
    key = (" ".join(command), tuple(modules))
    now = time.monotonic()
    with RUNTIME_CACHE_LOCK:
        cached = RUNTIME_CACHE.get(key)
        if cached and now - cached[0] < 30:
            return cached[1]

    script = (
        "import importlib.util,json,sys; "
        "modules=json.loads(sys.argv[1]); "
        "missing=[name for name in modules if importlib.util.find_spec(name) is None]; "
        "print(json.dumps({'version':sys.version.split()[0],'missing':missing}))"
    )
    try:
        result = subprocess.run(
            command + ["-c", script, json.dumps(modules)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        info = {"error": f"Could not inspect this Python runtime: {exc}"}
    else:
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()
            info = {
                "error": (
                    detail[-1]
                    if detail
                    else f"Python runtime check exited with code {result.returncode}."
                )
            }
        else:
            try:
                info = json.loads(result.stdout)
            except json.JSONDecodeError:
                info = {"error": "Python runtime returned an invalid dependency-check result."}

    with RUNTIME_CACHE_LOCK:
        RUNTIME_CACHE[key] = (time.monotonic(), info)
    return info


def _windows_policy_blocked(message: str) -> bool:
    lowered = message.lower()
    return (
        "application control policy has blocked" in lowered
        or "code integrity" in lowered
        or "0xc0000428" in lowered
    )


def _runtime_info(experiment: dict[str, Any]) -> dict[str, Any]:
    modules = experiment.get("required_modules", [])
    candidates: list[dict[str, Any]] = []
    min_python = experiment.get("min_python")
    max_python = experiment.get("max_python")
    for relative_path in experiment["python_envs"]:
        executable = experiment["folder"] / relative_path
        if not executable.is_file():
            continue
        command = [str(executable)]
        result = _check_runtime(command, modules)
        candidate = {
            "command": command,
            "runtime": f"Project environment · {relative_path.as_posix()}",
            "environment_found": True,
            **result,
        }
        version = result.get("version")
        if version and (min_python or max_python):
            version_pair = tuple(int(part) for part in version.split(".")[:2])
            if min_python and version_pair < tuple(min_python):
                candidate["compatibility_error"] = (
                    f"Python {version} is incompatible; this project requires "
                    f"Python {min_python[0]}.{min_python[1]} or newer."
                )
            elif max_python and version_pair > tuple(max_python):
                candidate["compatibility_error"] = (
                    f"Python {version} is incompatible; this project supports "
                    f"Python {max_python[0]}.{max_python[1]} or earlier."
                )
        candidates.append(candidate)

    ready = [
        candidate
        for candidate in candidates
        if not candidate.get("error")
        and not candidate.get("compatibility_error")
        and not candidate.get("missing")
    ]
    if ready:
        result = dict(ready[0])
        result["project_environment_found"] = True
        return result

    fallback_info: dict[str, Any]
    if PYTHON_LAUNCHER:
        fallback = [PYTHON_LAUNCHER, "-3.12"]
        fallback_info = {
            "command": fallback,
            "runtime": "System Python 3.12",
            "environment_found": False,
            **_check_runtime(fallback, modules),
        }
    elif sys.version_info[:2] <= (3, 12):
        fallback = [sys.executable]
        fallback_info = {
            "command": fallback,
            "runtime": f"Portal Python {sys.version_info.major}.{sys.version_info.minor}",
            "environment_found": False,
            **_check_runtime(fallback, modules),
        }
    else:
        fallback_info = {
            "command": None,
            "runtime": "No compatible system Python 3.12 launcher",
            "environment_found": False,
            "error": (
                "No project environment is available and the portal's Python "
                f"{sys.version_info.major}.{sys.version_info.minor} is outside "
                "the supported 3.10–3.12 fallback range."
            ),
        }

    project_environment_note = "; ".join(
        f"{candidate['runtime']}: "
        f"{candidate.get('compatibility_error') or candidate.get('error') or 'required app dependencies are missing (' + ', '.join(candidate.get('missing', [])) + ')'}"
        for candidate in candidates
    )
    fallback_info["project_environment_found"] = bool(candidates)
    if project_environment_note:
        fallback_info["project_environment_note"] = project_environment_note
    if not fallback_info.get("error") and not fallback_info.get("missing"):
        return fallback_info

    compatible_candidates = [
        candidate for candidate in candidates if not candidate.get("compatibility_error")
    ]
    if compatible_candidates:
        selected = min(
            compatible_candidates,
            key=lambda candidate: len(candidate.get("missing", []))
            if not candidate.get("error")
            else len(modules) + 1,
        )
        alternatives = [candidate for candidate in candidates if candidate is not selected]
        if alternatives:
            selected["project_environment_note"] = "; ".join(
                f"{candidate['runtime']}: "
                f"{candidate.get('compatibility_error') or candidate.get('error') or 'required app dependencies are missing (' + ', '.join(candidate.get('missing', [])) + ')'}"
                for candidate in alternatives
            )
        selected["project_environment_found"] = bool(candidates)
        return selected
    return fallback_info


def _command_for(experiment: dict[str, Any]) -> tuple[list[str], str]:
    runtime_info = _runtime_info(experiment)
    if runtime_info.get("error"):
        raise RuntimeError(runtime_info["error"])
    if runtime_info.get("missing"):
        missing = ", ".join(runtime_info["missing"])
        raise RuntimeError(
            f"Missing dependencies in {runtime_info['runtime']}: {missing}. "
            "Install this experiment's requirements in that runtime, or select "
            "a compatible existing environment."
        )
    python = runtime_info["command"]
    runtime = runtime_info["runtime"]
    entry = experiment["folder"] / experiment["entry"]
    if not entry.is_file():
        raise RuntimeError(f"Experiment entry point is missing: {entry}")

    kind = experiment["kind"]
    if kind == "streamlit":
        command = python + [
            "-m",
            "streamlit",
            "run",
            str(entry),
            "--server.address",
            PORTAL_HOST,
            "--server.port",
            str(experiment["port"]),
            "--browser.gatherUsageStats",
            "false",
        ]
    elif kind == "flask":
        command = python + [
            "-m",
            "flask",
            "--app",
            "app",
            "run",
            "--host",
            PORTAL_HOST,
            "--port",
            str(experiment["port"]),
            "--no-reload",
        ]
    elif kind == "gradio":
        command = python + [str(entry)]
        if experiment.get("checkpoint"):
            command.extend(["--checkpoint", str(experiment["folder"] / experiment["checkpoint"])])
    else:
        raise RuntimeError(f"Unsupported experiment framework: {kind}")
    return command, runtime


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection((PORTAL_HOST, port), timeout=0.15):
            return True
    except OSError:
        return False


def _status_for(experiment: dict[str, Any]) -> dict[str, Any]:
    project_found = experiment["folder"].is_dir() and (
        experiment["folder"] / experiment["entry"]
    ).is_file()
    item = {
        "id": experiment["id"],
        "title": experiment["title"],
        "number": experiment["number"],
        "label": experiment["label"],
        "description": experiment["description"],
        "framework": experiment["framework"],
        "technologies": experiment["technologies"],
        "icon": experiment["icon"],
        "accent": experiment["accent"],
        "port": experiment["port"],
        "url": f"http://{PORTAL_HOST}:{experiment['port']}",
        "setup": experiment["setup"],
        "model_note": experiment.get("model_note"),
        "blocked": experiment.get("blocked"),
        "project_found": project_found,
        "state": "blocked" if experiment.get("blocked") else "ready",
        "status_message": experiment.get("blocked") or experiment["setup"],
        "runtime": None,
    }

    if experiment.get("blocked"):
        return item

    if not project_found:
        item["state"] = "project_missing"
        item["status_message"] = (
            f"Project folder or entry point not found: "
            f"{experiment['folder']}\\{experiment['entry']}"
        )
        return item

    runtime_info = _runtime_info(experiment)
    item["runtime"] = runtime_info["runtime"]
    item["environment_found"] = runtime_info["environment_found"]
    item["project_environment_found"] = runtime_info.get(
        "project_environment_found", runtime_info["environment_found"]
    )
    item["project_environment_note"] = runtime_info.get("project_environment_note")
    project_environment_note = (
        item["project_environment_note"].rstrip(".")
        if item["project_environment_note"]
        else None
    )
    if runtime_info.get("error"):
        if _windows_policy_blocked(runtime_info["error"]):
            item["state"] = "blocked"
            item["status_message"] = (
                "Windows Application Control blocked a required runtime binary: "
                f"{runtime_info['error']} Use an IT-approved runtime; do not change "
                "security policy."
            )
        else:
            item["state"] = "environment_missing"
            item["status_message"] = runtime_info["error"]
        return item

    if runtime_info.get("missing"):
        item["missing_dependencies"] = runtime_info["missing"]
        item["state"] = "dependencies_missing"
        missing = ", ".join(runtime_info["missing"])
        if runtime_info["environment_found"]:
            item["status_message"] = (
                f"Project environment found, but dependencies are missing: {missing}."
            )
        elif item["project_environment_found"]:
            item["status_message"] = (
                f"Project environment found but cannot provide a compatible runtime. "
                f"{runtime_info['runtime']} is available, but dependencies are missing: {missing}."
            )
        else:
            item["status_message"] = (
                f"No project environment found. {runtime_info['runtime']} is available, "
                f"but dependencies are missing: {missing}."
            )
        if project_environment_note:
            item["status_message"] += (
                f" Existing project environment note: {project_environment_note}."
            )
        return item

    if runtime_info["environment_found"]:
        item["status_message"] = (
            f"Project environment and required app dependencies detected "
            f"({runtime_info.get('version', 'Python version unknown')})."
        )
    else:
        item["environment_state"] = "missing"
        if item["project_environment_found"]:
            item["status_message"] = (
                f"Project environment found but unavailable; using {runtime_info['runtime']} "
                f"({runtime_info.get('version', 'Python version unknown')}). "
                "Required app dependencies are detected."
            )
        else:
            item["status_message"] = (
                f"No project environment found; using {runtime_info['runtime']} "
                f"({runtime_info.get('version', 'Python version unknown')}). "
                "Required app dependencies are detected."
            )
    if project_environment_note:
        item["status_message"] += (
            f" Existing project environment note: {project_environment_note}."
        )

    checkpoint = experiment.get("checkpoint")
    if checkpoint and not (experiment["folder"] / checkpoint).is_file():
        item["state"] = "setup"
        item["status_message"] = (
            f"Checkpoint missing: {checkpoint.as_posix()}. Train a model first; "
            "see this experiment's README for the training command."
        )
        return item

    with PROCESS_LOCK:
        process_info = PROCESSES.get(experiment["id"])
        if process_info:
            process = process_info["process"]
            return_code = process.poll()
            if return_code is None:
                item["runtime"] = process_info["runtime"]
                if _port_is_open(experiment["port"]):
                    item["state"] = "running"
                    item["status_message"] = "Experiment is responding on its local port."
                else:
                    item["state"] = "starting"
                    item["status_message"] = "Starting the experiment in its own process…"
                item["can_stop"] = True
                return item
            log_path = process_info["log_path"]
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file()
                else ""
            )
            policy_line = next(
                (
                    line.strip()
                    for line in log_text.splitlines()
                    if _windows_policy_blocked(line)
                ),
                None,
            )
            missing_module = next(
                (
                    line.strip()
                    for line in log_text.splitlines()
                    if "ModuleNotFoundError:" in line or "No module named" in line
                ),
                None,
            )
            if policy_line:
                item["state"] = "blocked"
                item["status_message"] = (
                    f"Windows security policy blocked a required binary: {policy_line}"
                )
            elif missing_module:
                item["state"] = "dependencies_missing"
                item["status_message"] = (
                    f"Required package import failed: {missing_module} "
                    f"See portal\\logs\\{experiment['id']}.log."
                )
            else:
                item["state"] = "launch_failed"
                detail = next(
                    (
                        line.strip()
                        for line in reversed(log_text.splitlines())
                        if line.strip() and not line.startswith("[")
                    ),
                    "No additional process output was captured.",
                )
                item["status_message"] = (
                    f"Launch failed: {detail} See portal\\logs\\{experiment['id']}.log."
                )
            item["exit_code"] = return_code
            PROCESSES.pop(experiment["id"], None)
            return item

    return item


def _all_statuses() -> list[dict[str, Any]]:
    return [_status_for(experiment) for experiment in EXPERIMENTS]


def _start_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = EXPERIMENT_BY_ID[experiment_id]
    if experiment.get("blocked"):
        raise RuntimeError(experiment["blocked"])
    checkpoint = experiment.get("checkpoint")
    if checkpoint and not (experiment["folder"] / checkpoint).is_file():
        raise RuntimeError(
            f"Checkpoint missing: {checkpoint.as_posix()}. Train a model first; "
            "see this experiment's README for instructions."
        )
    if not experiment["folder"].is_dir():
        raise RuntimeError(f"Experiment folder is missing: {experiment['folder']}")

    with PROCESS_LOCK:
        current = PROCESSES.get(experiment_id)
        if current and current["process"].poll() is None:
            return {"message": "This experiment is already starting or running."}

        if _port_is_open(experiment["port"]):
            raise RuntimeError(
                f"Port {experiment['port']} is already in use. Close the other app "
                "or choose a different port before launching this experiment."
            )

        command, runtime = _command_for(experiment)
        log_dir = PORTAL_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{experiment_id}.log"
        environment = os.environ.copy()
        environment.update(
            {
                "STREAMLIT_SERVER_HEADLESS": "true",
                "GRADIO_SERVER_NAME": PORTAL_HOST,
                "GRADIO_SERVER_PORT": str(experiment["port"]),
                "GRADIO_ANALYTICS_ENABLED": "False",
                "FLASK_DEBUG": "0",
            }
        )
        if experiment["kind"] == "flask":
            # Flask's CLI sets the port; disable the legacy script's debug reloader.
            environment["FLASK_DEBUG"] = "0"

        with log_path.open("ab") as log_file:
            log_file.write(
                (
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Starting {experiment['title']} ({runtime})\n"
                ).encode("utf-8")
            )
            process = subprocess.Popen(
                command,
                cwd=experiment["folder"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
            )

        PROCESSES[experiment_id] = {
            "process": process,
            "runtime": runtime,
            "log_path": log_path,
        }
    return {"message": "Launch requested in a separate process.", "runtime": runtime}


def _stop_experiment(experiment_id: str) -> dict[str, str]:
    with PROCESS_LOCK:
        process_info = PROCESSES.get(experiment_id)
        if not process_info:
            return {"message": "No portal-launched process is running for this experiment."}
        process = process_info["process"]
        if process.poll() is not None:
            PROCESSES.pop(experiment_id, None)
            return {"message": "The experiment process has already stopped."}
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    with PROCESS_LOCK:
        PROCESSES.pop(experiment_id, None)
    return {"message": "Experiment process stopped."}


def _stop_all_experiments() -> None:
    with PROCESS_LOCK:
        active = list(PROCESSES.items())
    for experiment_id, process_info in active:
        process = process_info["process"]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=4)
        with PROCESS_LOCK:
            PROCESSES.pop(experiment_id, None)


atexit.register(_stop_all_experiments)


class PortalHandler(BaseHTTPRequestHandler):
    server_version = "GenAILabPortal/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[portal] {self.address_string()} - {format_string % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            page = PORTAL_HTML.replace("__PORTAL_TOKEN__", PORTAL_TOKEN)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/experiments":
            self._send(
                200,
                json.dumps(_all_statuses()).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        expected_hosts = {f"{PORTAL_HOST}:{PORTAL_PORT}", f"localhost:{PORTAL_PORT}"}
        if (
            self.headers.get("X-Portal-Token") != PORTAL_TOKEN
            or host not in expected_hosts
            or origin not in {f"http://{value}" for value in expected_hosts}
        ):
            self._send(403, b"Forbidden", "text/plain; charset=utf-8")
            return

        route = urlparse(self.path).path.strip("/").split("/")
        if len(route) != 4 or route[:2] != ["api", "experiments"]:
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        experiment_id, action = route[2], route[3]
        if experiment_id not in EXPERIMENT_BY_ID or action not in {"start", "stop"}:
            self._send(404, b"Unknown experiment action", "text/plain; charset=utf-8")
            return

        try:
            result = (
                _start_experiment(experiment_id)
                if action == "start"
                else _stop_experiment(experiment_id)
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self._send(
                409,
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        self._send(
            200,
            json.dumps(result).encode("utf-8"),
            "application/json; charset=utf-8",
        )


def main() -> None:
    server = ThreadingHTTPServer((PORTAL_HOST, PORTAL_PORT), PortalHandler)
    server.daemon_threads = True
    print("GenAI Lab Portal is ready.")
    print(f"Open http://{PORTAL_HOST}:{PORTAL_PORT} in your browser.")
    print("Use Ctrl+C to stop the portal and its launched experiment processes.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GenAI Lab Portal…")
    finally:
        server.server_close()
        _stop_all_experiments()


if __name__ == "__main__":
    main()
