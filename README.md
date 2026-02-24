# Van Gogh Computer Vision

## Abstract

This project focuses on applying modern computer vision and deep learning techniques to **distinguish authentic Vincent van Gogh artworks from imitations**. Using convolutional and transformer-based architectures, the code in this repository shows our work investigating how architectural choices, regularization, and data-handling strategies influence model stability and generalization in art authentication tasks.

Publications by the firm Art Recognition AG (AR) of Adiswil, Switzerland, initially motivated this project. Our [Project background](technical-discussion/background.md) section fully documents these origins, but in short, the Swiss firm has published multiple pieces on using computer vision (CV) to authenticate paintings by van Gogh, and LMI Group has an interest in this subject area. We therefore reconstructed the method best-described in the Swiss firm's 2024 paper, "Art Authentication with Vision Transformers," by Schaerf et al., in order to understand the published work.

Our work contributes to reproducing and validating AI-based authentication methods, documenting both successes and challenges encountered during research. The repository includes:

* deployment-ready, trained models
* experimental notebooks
* reproducible scripts, and
* technical discussions.

By providing open resources and detailed documentation, this work aims to promote transparency in AI-driven art attribution, an aspect of this emergent field that demands greater attention.

If you want to get started using the code while reading our narrative and documentation, feel free to go to the final chapter of this book, on [Reproducibility](./reproducibility/copy-this-code.md).

The purpose of this front page is to provide an overview of the key elements of the project motivation, the architectures used and models developed, a brief summary of our findings, and an outline of the experiments we ran. Finally, we provide a map of the repository structure for the more technically-oriented reader.

For those interested in a more global view on the questions raised by this project and the potential utility of AI for art authentication, LMI Group has published in tandem with this repository [a white paper on the subject](https://www.lmigroupintl.com/pdf/potential-role-ai-art-authentication).

***

## Table of Contents

In the accompanying documentation, we dive deep into the above subjects and others, in order to give as full an account as possible of how we tested and re-tested the underlying methods. Our goal is to strike the right balance in making this repository both relatively accessible to the general public and sufficiently rigorous for the trained practitioner.

The documentation is broken into the following sections:

* [**Project background**](technical-discussion/background.md)\
  Overview of the 2025 reimplementation, with context from Schaerf et al. (2024) and their methodology using vision transformers for art authentication.
* [**Data & Data Processing**](technical-discussion/data-and-data-processing.md)\
  Details on dataset curation, provenance validation, preprocessing choices, and challenges in ensuring data quality and reproducibility.
* [**AI Models: Training and Results**](technical-discussion/ai-models-training-and-results.md)\
  A walk-through of training protocols for EfficientNet-B5 and Swin-Tiny, showing how stabilization and regularization improved reliability, with extended experiments including _Elimar_ painting predictions.
* [**Discussion and Conclusion**](technical-discussion/discussion-and-conclusion.md)\
  Analysis of findings, comparison to prior work, limitations, and implications for reproducibility in art authentication research.
* [**Copy this code**](reproducibility/copy-this-code.md)\
  Instructions for downloading and adapting the codebase, with notes on dataset access and documentation practices.
* [**Run this code yourself**](reproducibility/run-this-code-yourself.md)\
  Step-by-step guide to reproducing training runs, setting up datasets, and running experiments locally.

***

## Motivation

This project builds on the methodology described in _“Art Authentication with Vision Transformers”_ (Schaerf et al., 2024), authored by researchers associated with the Swiss firm AR. The original work applied Swin Transformer (tiny) and EfficientNet models to van Gogh authentication, generating controversial findings—including a claim made in _Wired_ Magazine that _Elimar_ had been mis-attributed to Van Gogh with a probability of 97%.

Here, we **replicate and extend the pipeline** described in the 2024 paper with a focus on:

* documenting dataset provenance, preprocessing, and evaluation steps
* testing patch-based training strategies, and
* demonstrating how model behavior shifts with changes in data and model optimization.

Our experiments have demonstrated the **sensitivity of these CV methods to dataset variations, overfitting, and undocumented training choices**. This highlights the importance of transparency and reproducibility in AI-based art authentication.

The implications of this are laid out in our [results](technical-discussion/ai-models-training-and-results.md) and [discussion](technical-discussion/discussion-and-conclusion.md) sections. But as the next section of our introduction shows, our auditable reconstruction of prior work suggests that these CV methods are highly sensitive to training data — and in the context of fine art, where training data is uneven, it cannot yet reliably be applied to the high-stakes work of art authentication.

***

## Models, Data, & Findings

In this project, we document our reconstruction of the experiment described by Schaef et al.'s paper, using two CV architectures, and provide links to our underlying code as well as to the finished models. We also extensively discuss our validation of our methods and the outcomes of our experiment—we will summarize these here.

### Architectures

We discuss our experiments with two architectures: Swin-Tiny and EfficientNet. For each of these, you will find two subcategories of model in both architectures:

* Early experiments showing systemic errors: "overfit" and "unstable"
* Mature experiments in which we carefully corrected these errors: "regularized" and "stable"

#### Swin Transformer Models

* Early/Overfit: Exhibited classic overfitting—training loss decreased while validation loss increased.
* Mature/Regularized: Careful hyperparameter tuning and regularization reduced overfitting and improved the reliability of the model’s confidence estimates overall.

#### EfficientNet Models

* Early/Unstable: Initial experiments showed high variance in validation loss, indicating instability.
* Mature/Stable: Applying stabilization techniques improved consistency and overall performance.

### Data

Following best practices for transparency and reproducibility, we include as much of our training data as possible and document it as thoroughly as possible.

* All datasets are stored in [`data/`](https://github.com/lmigroupintl/VanGoghComputerVision/tree/main/data).
* Each run corresponds to a dataset tagged with its experiment configuration.
* Run 4: This was a special case. As a check against our van Gogh models, we trained a model on paintings by Cézanne. Here, Cézanne was treated as authentic, while van Gogh and other artists were treated as imitation.

### Findings

#### Statistical Validation

* Statistical tests (see [notebooks/statistical tests/](https://github.com/lmigroupintl/VanGoghComputerVision/tree/main/notebooks/statistical%20tests)) indicate that the regularized Swin and stable EfficientNet models show statistically significant improvements over their earlier versions.

#### _Elimar_ Predictions

* **Overfit Swin & Unstable EffNet**: Initially predicted _Elimar_ as imitation with high confidence.
* **Improved Models**: Predictions remained imitation but with relatively lower confidence, indicating more calibrated behavior.
* **Larger Datasets (Runs #2, #3, #5, #6, & #7)**: With additional training data, improved models classified _Elimar_ as authentic, with moderate to high confidence depending on image quality.

***

## Experiments

In addition to the documentation provided here, we present our experiments with in-line documentation in notebook format.

* **Exploration notebooks**: Initial dataset analysis
* **Experiment notebooks**: Swin and EfficientNet training runs using the run 1, 2, and 3 datasets from [`data/`](https://github.com/lmigroupintl/VanGoghComputerVision/tree/main/data)
* **Paul Cézanne notebooks**: Run 4 experiments treating Cézanne as authentic
* **Inference notebooks**: Predictions on disputed works, including:
  * _Elimar_ (van Gogh attribution)
  * _Boy in a Red Vest_ (Cézanne attribution test)

**We also provide statistical tests to demonstrate our extensive validation work:** Comparisons of overfit vs. regularized Swin and unstable vs. stable EfficientNet models

***

## Repository Structure

```
│
├── .gitattributes            # Git configuration for attributes
├── .gitignore                # Git ignore rules
├── README.md                 # Main project documentation
├── SUMMARY.md                # Summary of the technical documents
├── requirements.txt          # Python dependencies
│
├── .gitbook/                 # GitBook assets (figures, plots)
├── data/                     # Data files (CSV, plots)
├── inference images/         # Sample images for inference/testing
├── models/                   # Trained model checkpoints (.pth files)
├── notebooks/                # Jupyter notebooks (experiments, analysis)
│   ├── Paul Cezanne experiment/
│   ├── experiments/
│   ├── exploration/
│   ├── inference/
│   └── statistical tests/
├── reproducibility/          # Resources for reproducing results
├── scripts/                  # Utility and training scripts
└── technical-discussion/     # Notes, explanations, and technical write-ups

```
