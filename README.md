## TemperFlow <img src="https://statr.me/images/sticker-temperflow.png" alt="LBFGS++" height="150px" align="right" />

This repository stores the code files for the article [Efficient Multimodal Sampling via Tempered Distribution Flow](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2198059) by Yixuan Qiu and Xiao Wang.

### Workflow

We provide two implementations of the TemperFlow algorithm,
one using the PyTorch framework (in the `torch` folder),
and the other using the TensorFlow framework (in the `tf` folder).

The workflow to reproduce the results and plots in the article is as follows:

1. Download the following two model files into the `tf/pretrained` folder.
    - `face-gmmvae-generator.npz`: https://1drv.ms/u/s!ArsORq8a24WmoHtrued5dY6APVtH?e=noYKZr
    - `face-gmmvae-flow.npz`: https://1drv.ms/u/s!ArsORq8a24WmoHwOx8nlOoXpXYEW?e=7aQ6ks

2. Install the CUDA and cuDNN environments if you use GPU for computing. An installation guide can be found at https://www.tensorflow.org/install/pip.

3. Install the PyTorch and TensorFlow frameworks, following the installation guides at https://pytorch.org/get-started/locally/ and https://www.tensorflow.org/install/pip.

4. Run the main script as follows:
    ```bash
    sh run_experiments.sh
    ```
    This will call individual scripts under the `torch` and `tf` directories. When the script finishes, two new folders, both named `model`, will be created under the `torch` and `tf` directories, respectively. They will contain model data that are further used to create tables and plots.

5. When Step 4 finishes, the two `model` folders would already contain image files for Figures 7, 9, 10, S2, S3, S4, and S11.

6. The individual R scripts in the `visualization` directory provide the code to generate other figures and tables, based on the model data created in previous steps. For example, the script `fig1-figs1-kl-sampler.R` produces Figure 1 and S1, and `tabs1-benchmark.R` outputs the numbers in Table S1.

### Software environment

GPU computing:

- CUDA 11.7
- cuDNN 8.6.0

Python:

- Python 3.10.8
- Numpy 1.23.5
- SciPy 1.9.3
- Pandas 1.5.2
- POT 0.8.2
- Matplotlib 3.6.2
- Seaborn 0.12.1
- PyTorch 1.13.1
- Pyro 1.8.3
- TensorFlow 2.10.0
- TensorFlow Probability 0.18

R:

- R 4.2.0
- reticulate 1.25
- readr 2.1.2
- jsonlite 1.8.0
- tibble 3.1.8
- reshape2 1.4.4
- dplyr 1.0.9
- ggplot2 3.3.6
- GGally 2.1.2
- gridExtra 2.3
- rgl 0.108.3.2
- showtext 0.9.5
- transport 0.12.2
- kernlab 0.9.31
- nor1mix 1.3.0
- copula 1.1.0
