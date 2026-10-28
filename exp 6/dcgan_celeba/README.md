# DCGAN Face Generation on CelebA

This project provides a beginner-friendly PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the CelebA face dataset. The code generates synthetic human faces using a trained generator and includes an optional Flask interface for easy online demos.

## Project structure

```text
dcgan_celeba/
├── train.py
├── generate.py
├── model.py
├── dataset.py
├── utils.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── generated/
├── outputs/
│   ├── samples/
│   └── plots/
├── checkpoints/
├── notebooks/
│   └── dcgan_celeba_colab.ipynb
└── data/
    └── celeba/
```

## Environment setup

Use Python 3.12 or 3.13 if available. The code also works well with Python 3.11+ and PyTorch 2.x.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation

Place the CelebA aligned images in a folder such as:

```text
./data/celeba/
```

The folder should contain image files such as `000001.jpg`, `000002.jpg`, and so on. If your extracted dataset contains a subfolder like `img_align_celeba`, point to that directory instead.

You can also upload the dataset to Google Drive or Google Colab and set `--dataset-dir` to the downloaded path.

For an optional Kaggle download, set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment and pass the dataset slug explicitly:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
python train.py --dataset-dir ./data/celeba --kaggle-dataset your-account/celeba-dataset
```

## Training

Train the DCGAN on your local machine or in Google Colab:

```bash
python train.py --dataset-dir ./data/celeba --epochs 50 --batch-size 128
```

Helpful optional flags:

- `--batch-size 64` when GPU memory is limited
- `--epochs 25` for a quick smoke test
- `--checkpoint-every 5` for periodic saves
- `--seed 42` for reproducible runs

The script saves:

- `outputs/samples/epoch_XXX.png`
- `outputs/training_history.json`
- `outputs/plots/loss_history.png`
- `checkpoints/generator.pth`
- `checkpoints/discriminator.pth`
- `checkpoints/dcgan_final_checkpoint.pth`

## Generating new faces

After training, generate synthetic images:

```bash
python generate.py --checkpoint ./checkpoints/dcgan_final_checkpoint.pth --count 9 --output-dir ./outputs/generated
```

The script creates new synthetic outputs from random latent vectors. It can generate one image or a batch of images depending on the value of `--count`.

## Optional Flask web app

To launch the local web demo:

```bash
python app.py
```

Then open the browser at:

```text
http://localhost:5000/
```

The app will show a button to generate new face images from the trained generator. If no checkpoint exists yet, it displays a clear error message instead of pretending the model is ready.

## Notes

- This is an educational implementation and is meant to be easy to follow.
- GAN training can be unstable and losses may fluctuate.
- Model quality depends on training time, dataset size, and hardware availability.
- The output is always generated directly by the trained generator and never loaded from a static sample gallery.

## License

This project is intended for educational and research use. Please review the CelebA dataset licence before use in a commercial setting.
