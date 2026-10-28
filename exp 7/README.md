# Educational StyleGAN Face Lab (CelebA, 64 x 64)

## Aim and objectives

Build a small, understandable, StyleGAN-inspired PyTorch model that learns to generate face images from CelebA or a user-provided image collection. The practical demonstrates:

1. Mapping a random latent vector from **Z space** into **W space**.
2. Applying learned styles at multiple synthesis-network depths.
3. Approximate coarse-, middle-, and fine-layer controls.
4. Mixing the styles of two latent identities at a selectable layer.
5. Adding independently seeded stochastic noise in synthesis layers.

This is an educational implementation, not NVIDIA's original StyleGAN. Its architecture and training objective are intentionally compact; photorealistic results are not promised, especially after only a few epochs.

## Theory

### Z and W spaces; mapping network

Each sample begins with `z ~ N(0, I)` in **Z space**. `MappingNetwork` normalizes `z` and passes it through a small multilayer perceptron to produce **w** in W space. W is an intermediate representation that the generator uses to control feature statistics. This mapping is intended to make style control more flexible; a small trained model does not guarantee that every W direction has a clean human-interpretable meaning.

The synthesis network uses a **W+** tensor: one mapped style vector is repeated for every styled convolution. The UI and `mix_styles` can then replace selected layer styles independently. This project does not include the original StyleGAN's learned affine transforms for convolutional weight modulation/demodulation; it applies a simplified Adaptive Instance Normalization (AdaIN) scale and bias at each styled convolution.

### Synthesis, layer controls, and style mixing

The generator starts from a learned 4 x 4 constant and grows through 8, 16, 32, and 64 pixel stages. Each stage uses two convolutions conditioned by its W+ entries. In the demonstration, indices 0-2 are grouped as coarse, 3-5 as middle, and 6-8 as fine. These are **approximate depth groupings**, not guaranteed semantic controls: their visual effect depends on the learned weights and training time.

Style mixing combines W+ from identities A and B: layers before the chosen cutoff use A, while the cutoff layer and later layers use B. Early layers tend to affect broad structure and later layers tend to affect detail in well-trained StyleGAN models, but that behavior is not guaranteed for this simplified network.

### Noise injection

Each styled convolution adds a one-channel spatial noise map multiplied by learned per-channel strengths. During training, fresh noise is sampled; for a repeatable demonstration, the UI supplies a seed. With the same latent and noise seed, output is reproducible. Changing the seed changes stochastic details; setting strength to zero disables their contribution.

### Discriminator and training

The discriminator is a compact CNN that returns one real/fake logit per 64 x 64 image. Training uses the non-saturating logistic GAN objective implemented with `BCEWithLogitsLoss` and Adam. The model does not use the original StyleGAN's progressive growing, path-length regularization, R1 regularization, minibatch standard deviation, or full-resolution training recipes. Loss values are useful diagnostics, not proof of visual quality.

## Dataset preparation

Supported inputs:

- **CelebA directory**: a torchvision CelebA root directory. The dataset must already be available unless `--download-celeba` is given.
- **Image directory**: JPG, JPEG, PNG, WEBP, and BMP files are discovered recursively.
- **Image ZIP**: a ZIP containing supported image files, including images in subfolders. The archive is read directly; it is not extracted.

Preprocessing uses `Resize(about 1.12 x resolution)`, `CenterCrop(64)`, conversion to RGB, `ToTensor`, and normalization with mean/std 0.5, mapping pixels from `[0, 1]` to `[-1, 1]`. Center crop removes excess borders while maintaining a square model input. Images are returned as `[3, 64, 64]` tensors and batched by a PyTorch `DataLoader`.

The project does not bundle CelebA images. Follow the CelebA dataset terms and provide its files through the supported local/Colab route.

## Installation and execution

Python 3.10 or newer is recommended. In a local environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and activate `.venv/bin/activate`. Install a PyTorch/Torchvision build appropriate for your machine if the default pip build does not match your CUDA setup. Google Colab users can run the included notebook `stylegan_colab.ipynb` after selecting **Runtime > Change runtime type > T4 GPU** (or another available GPU).

Run a tiny one-step model/training/checkpoint smoke test without a dataset:

```bash
python train.py --smoke-test --output-dir outputs/smoke
```

The smoke test intentionally trains on synthetic random images, **not CelebA**. To run that same tiny test on a real dataset, pass `--dataset <path>` too.

Train on an image folder or image ZIP:

```bash
python train.py --dataset data/faces --epochs 100 --batch-size 16 --output-dir outputs
python train.py --dataset data/faces.zip --epochs 100 --batch-size 8 --output-dir outputs
```

Train/download torchvision CelebA (dataset files may be large and upstream download access can vary):

```bash
python train.py --dataset data/celeba --download-celeba --epochs 100 --batch-size 16
```

The default device is CUDA when available, otherwise CPU. For a T4 GPU, begin with batch size 8 or 16; reduce it if CUDA reports out-of-memory. Use `--num-workers 2` by default; on Windows or notebook environments where worker startup is an issue, use `--num-workers 0`. Use `--steps N` to cap optimizer iterations. Training a useful face generator takes substantial time and many updates; short runs generally yield noisy, low-quality images.

Training writes:

- `outputs/latest.pt`: generator, discriminator, both optimizer states, epoch, configuration, and loss history.
- `outputs/losses.json`: per-epoch generator and discriminator loss values.
- `outputs/dataset_samples.png`: a preview grid of preprocessed training images.
- `outputs/samples/epoch_XXXX.png`: sample grids at configured intervals.

Resume from the most recent checkpoint:

```bash
python train.py --dataset data/faces.zip --epochs 100 --resume outputs/latest.pt
```

`--epochs` is the target total epoch count. To save directly to Google Drive in Colab, run:

```bash
python train.py --dataset /content/drive/MyDrive/faces.zip --epochs 100 --save-to-drive
```

The script mounts Drive and, unless `--output-dir` is specified, saves under `MyDrive/stylegan-celeba/outputs`. Alternatively mount Drive in the notebook and pass `--output-dir /content/drive/MyDrive/stylegan-celeba/outputs`.

## Interactive Gradio demonstration

Install the dependencies, train at least one checkpoint, then launch:

```bash
python app.py --checkpoint outputs/latest.pt
```

The optional `--share` flag creates a public Gradio link; use it only when you intend to expose the demo. The tabs exercise:

1. **Generate Faces** — batched random latents and noise strength.
2. **Latent Space** — Z-to-W dimensions/statistics and an image from that latent.
3. **Coarse-to-Fine Control** — scales three groups of W+ styles.
4. **Style Mixing** — shows identity A, identity B, and a selectable-cutoff mix.
5. **Noise Injection** — same latent with two reproducible noise seeds.
6. **Training Progress** — reads saved losses and the latest sample grid.

All controls are connected to the loaded generator or its saved training artifacts. If history/sample files were moved away from the checkpoint's output directory, the progress tab reports that they are unavailable.

### Colab notebook steps

The included notebook expects a ZIP archive of this project to be uploaded first. It installs the requirements, mounts Drive, uploads an image dataset ZIP, runs a one-step smoke test, and provides a separate full-training cell. Select a GPU runtime before training. The smoke test is deliberately separate from CelebA training.

## Major files and functions

- `config.py` — `TrainConfig` keeps model/training settings in one place and validates the supported 64 x 64 configuration.
- `dataset.py` — `face_transform`, `ImageFolderDataset`, `ZipImageDataset`, `load_dataset`, and `make_dataloader` implement input discovery and preprocessing.
- `models/mapping_network.py` — `MappingNetwork.forward` normalizes Z and maps it to W.
- `models/generator.py` — `Generator.styles_from_z` makes W+ styles; `Generator.forward` synthesizes an image with optional seeded noise; `mix_styles` selects A/B layer styles.
- `models/discriminator.py` — `Discriminator.forward` returns real/fake logits.
- `train.py` — `train` implements the GAN updates, regular sample grids, loss history, checkpoint/resume, and smoke mode; `main` parses CLI options and optional Drive mounting.
- `utils/checkpoint.py` — `save_checkpoint` and `load_checkpoint` persist and restore model/optimizer/training state.
- `utils/visualization.py` — `tensor_to_pil` converts normalized tensors; `save_image_grid` writes training samples.
- `app.py` — `load_generator` restores a trained model and `build_demo` wires the Gradio tabs to generation and inspection controls.
- `tests/test_project.py` — verifies model output, mapping, mixing, noise determinism, ZIP preprocessing/DataLoader batching, and checkpoint round-tripping.
- `stylegan_colab.ipynb` — ordered Colab cells for project upload, installation, Drive, dataset upload, smoke test, and full training.

## Expected outputs and practical viva

**Expected outputs:** 64 x 64 RGB images with tensor values in `[-1, 1]` before display conversion; printed epoch losses; saved sample grids and a resumable checkpoint. Early samples can look like colored noise or indistinct blobs. Do not claim that a short run produced realistic faces.

Suggested demonstration and viva points:

1. Show a real image batch and explain resize, center crop, RGB conversion, and `[-1, 1]` normalization.
2. Compare the 128-element Z and W vectors shown by the latent tab. Explain that W is a learned mapped representation, not another real image.
3. Change one coarse/middle/fine slider while holding the latent seed fixed. Describe any change as approximate because this compact model may not learn semantic controls.
4. Mix two identity seeds and move the cutoff. Point out that exact identity/detail separation depends on training.
5. Keep a latent fixed and change the noise seed. Explain stochastic detail versus the style-controlled signal.
6. Show G/D losses, a saved sample grid, and the contents of a checkpoint; optionally resume and compare epochs.
7. Explain why a T4 is useful, why batch size may need reduction, and why a short smoke test is not evidence of CelebA training quality.

## Pretrained-model alternative and limitations

If training time is limited, a separately obtained official pretrained StyleGAN model can be used for **inference-only** demonstrations with its own compatible official runtime. That demonstrates sampling from existing weights; it is not this project's model trained on CelebA, and pretrained weights are not included here.

This implementation is deliberately small and fixed to 64 x 64 RGB images. It omits several original StyleGAN mechanisms and regularizers, uses direct AdaIN rather than convolutional weight modulation/demodulation, and provides only approximate layer grouping. Training data, compute, random seed, and optimization all affect output quality. A brief run is for pipeline verification, not high-quality face synthesis.

## Conclusion (practical record)

The practical implements an educational StyleGAN-style generator and discriminator in PyTorch and demonstrates latent mapping, layer-wise style control, style mixing, and noise injection. CelebA images are preprocessed to normalized 64 x 64 tensors and used for adversarial training. The modular design makes the flow from dataset to generator, discriminator, saved outputs, and interactive inference observable, while the simplified architecture and limited training budget are stated explicitly.

## Verification

Run the focused CPU tests:

```bash
python -m unittest discover -s tests -v
```

Run the training/checkpoint smoke test:

```bash
python train.py --smoke-test --output-dir outputs/smoke
```

Tests use a tiny model and a generated ZIP image, so they need no CelebA download or GPU. Full CelebA training and Gradio GPU performance require the user's dataset and runtime.
