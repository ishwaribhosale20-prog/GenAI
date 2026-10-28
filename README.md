# GenAI Lab Experiments

A collection of practical experiments exploring **Generative Artificial Intelligence (GenAI)**, deep learning, foundation models, image generation, and large language models using Python.

This repository contains lab experiments, implementation code, and a portal for exploring different GenAI concepts.

## Objectives

- Understand the fundamentals of Generative AI and foundation models.
- Explore tensor operations and deep learning frameworks.
- Implement text generation, image generation, and other generative AI applications.
- Learn about GANs, diffusion models, and StyleGAN.
- Explore Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning techniques such as LoRA and QLoRA.
- Develop practical experience with Python and AI/ML libraries.

## Experiments

The repository is organized into separate folders for individual experiments.

| Experiment | Topic |
|---|---|
| Experiment 1 | Dataset analysis and preprocessing |
| Experiment 2 | Tensor operations using TensorFlow |
| Experiment 3 | Text generation and question answering using FLAN-T5 |
| Experiment 4 | Text-to-image generation using Stable Diffusion XL (SDXL) |
| Experiment 5A | Refer to the experiment folder for details |
| Experiment 5B | Refer to the experiment folder for details |
| Experiment 6 | Image generation using DCGAN |
| Experiment 7 | StyleGAN and style-based image generation |
| Experiment 8 | Text-to-music generation using MusicGen |
| Experiment 9 | Personalized AI assistant using an LLM, RAG, and LoRA/QLoRA |
| Experiment 10 | Text-to-video generation |

*Note: Refer to each experiment's source code and documentation for its exact scope and requirements.*

## Technologies Used

- **Language:** Python
- **Deep Learning:** TensorFlow, PyTorch
- **AI Models:** Hugging Face Transformers, FLAN-T5, Stable Diffusion XL, DCGAN, StyleGAN, MusicGen
- **Generative AI Concepts:** Foundation Models, Diffusion Models, GANs, LLMs, RAG, LoRA, QLoRA
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib and other experiment-specific libraries
- **Web Interface:** Python-based portal; see the `portal` folder for implementation details

## Repository Structure

```text
GenAI/
├── portal/
├── exp1/
├── exp 2/
├── exp 3/
├── exp 4/
├── exp 5 A/
├── exp 5 B/
├── exp 6/
├── exp 7/
├── exp 8/
├── exp 9/
├── exp 10/
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ishwaribhosale20-prog/GenAI.git
cd GenAI
```

### 2. Check the requirements

Open the folder for the experiment you want to run and check its README, requirements file, or other dependency documentation.

### 3. Create a virtual environment

On Windows, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If the selected experiment contains a `requirements.txt` file, install its dependencies:

```powershell
python -m pip install -r "PATH_TO_REQUIREMENTS.txt"
```

Replace the placeholder with the actual path to the experiment's requirements file.

### 5. Run an experiment

Follow the instructions in the selected experiment folder. Different experiments may require different dependencies, model downloads, hardware, or API credentials.

## Important Notes

- Some experiments may require a GPU and significant memory.
- Large pretrained models may need to be downloaded separately.
- Configure API keys locally using environment variables or the relevant secrets configuration.
- Never commit API keys, passwords, tokens, or other sensitive information.
- Review each model's license and usage terms before using or distributing it.
- Dependency requirements and execution steps may differ between experiments.

## Learning Outcomes

By working through these experiments, learners can gain practical exposure to:

- Generative AI and foundation models
- Tensor operations and deep learning
- Natural language processing
- Text-to-image and text-to-video generation
- GANs and style-based image synthesis
- Text-to-music generation
- LLM applications, RAG, and parameter-efficient fine-tuning

## Author

**Ishwari Bhosale**

GitHub: [@ishwaribhosale20-prog](https://github.com/ishwaribhosale20-prog)

---

*Created for learning, experimentation, and practical exploration of Generative AI.*
