# light-curve-image-cnn

Classification of variable stars by encoding their light curves (brightness time
series) as 2D histogram images and classifying them with a convolutional neural
network. Published in Astronomy & Astrophysics (2024):
[Monsalves et al., *Application of Convolutional Neural Networks to time domain
astrophysics: 2D image analysis of OGLE light curves*](https://arxiv.org/abs/2408.11960).

## The idea

Irregularly sampled time series are awkward inputs for standard deep-learning
architectures. Instead of feeding the raw sequence to a recurrent model, each
light curve is folded on its period and converted into a 2D histogram image
(phase vs. magnitude). Classification then becomes an image-recognition problem,
which CNNs solve fast and robustly.

## Results

- Trained on OGLE light curves across the main variable-star classes
  (eclipsing binaries, Cepheids, RR Lyrae, long-period variables, and more).
- **Throughput:** classifies half a million objects in about 3 minutes on a
  single GPU, engineered to scale toward the ~10 million alerts per night
  expected from upcoming survey telescopes.
- Hyperparameters tuned with Ray Tune (random search over the CNN architecture,
  see `hyperparam_tuning/`).

## Repository contents

| Path | Description |
|---|---|
| `CNN_2dhist_function.py` | Core pipeline: light curve → 2D histogram image → CNN classification |
| `How to use the CNN.ipynb` | Worked example applying the trained model to new light curves |
| `Paper_code.ipynb` | Full analysis notebook behind the paper |
| `hyperparam_tuning/` | Ray Tune random-search runs for the CNN architecture |
| `UseCNN/Lc_Data` | Example light-curve data |
| `OGLE-BLG-ECL-001923.dat` | Sample OGLE light curve |
| `2dhistogramCNN.yml`, `alerce_environment.yml` | Conda environments |

## Quickstart

```bash
conda env create -f 2dhistogramCNN.yml
conda activate 2dhistogramCNN
jupyter lab "How to use the CNN.ipynb"
```

## Citation

```bibtex
@article{Monsalves2024,
  author  = {Monsalves, N. and Jaque Arancibia, M. and Bayo, A. and S\'anchez-S\'aez, P. and others},
  title   = {Application of Convolutional Neural Networks to time domain astrophysics. 2D image analysis of OGLE light curves},
  journal = {Astronomy \& Astrophysics},
  year    = {2024},
  eprint  = {2408.11960},
  archivePrefix = {arXiv}
}
```
