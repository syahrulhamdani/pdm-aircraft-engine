# Predictive Maintenance of Aircraft Engine

The goal of this project is to determine the remaining useful life of the aircraft engine given the engine's performance of each cycle with a number of engine's features

## Overview

In here, I use simulation data engine's provided by NASA which can be found [here](data-science-project-template.zip) and approach the problem using machine learning. Especially, by implementing Neural Network and WTTE-RNN, that is Weibull Time-To-Event RNN proposed by [Egil Martinsson](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/).

## Project Structure

```tree
.
├── READMME.md              <- README for developers using this project.
├── data
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data, ABT set for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── models                  <- Trained and serialized model, model prediction, or summaries
│
├── notebooks               <- Jupyter notebooks with default naming `number-initials-delimited-description`,
│                               e.g. `1.0-sbh-initial-exploratory`
│
├── references              <- Data dictionaries, category-class mapping, data headers,
│                               and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTex, etc.
│   ├── figures             <- Genereted graphics and figures to be used in reporting
│   └── models              <- Checkpoints of trained models
│
├── requirements.txt        <- The requirements file or library or packages for reproducing analysis environment,
│                               e.g. generated with `pip` or `conda` or directly using `docker`
│
├── utils                   <- Source code for use in in this project
│   ├── cli.py              <- Scripts to get user input via cli
│   ├── make_dataset.py     <- Scripts to prepare and/or generate dataset, including preprocessing steps
│   └── visualize.py        <- Scripts to create exploratory and results oriented visualizations
│
├── train.py                <- Scripts to train model
├── compare_plot            <- Scripts to estimate rul and create a comparison plot with original rul
└── predict.py              <- Scripts to infer and predict given data using saved trained model
```

## Usage

Still updating..
