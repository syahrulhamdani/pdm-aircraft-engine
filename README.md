# Predictive Maintenance of Aircraft Engine

The goal of this project is to determine the remaining useful life of the aircraft engine given the engine's performance of each cycle with a number of engine's features

## Overview

In here, I use simulation data engine's provided by NASA which can be found [here](data-science-project-template.zip) and approach the problem using machine learning. Especially, by implementing Neural Network and WTTE-RNN, that is Weibull Time-To-Event RNN proposed by [Egil Martinsson](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/).

## Project Structure

```tree
.
├── LICENSE
├── READMME.md          <- README for developers using this project.
├── data
│   ├── external        <- Data from third party sources.
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- The final, canonical data, ABT set for modeling.
│   ├── raw             <- The original, immutable data dump.
├── models              <- Trained and serialized model, model prediction, or summaries
│
├── notebooks           <- Jupyter notebooks with default naming `number-initials-delimited-description`,
│                          e.g. `1.0-sbh-initial-exploratory`
│
├── references          <- Data dictionaries, category-class mapping, data headers,
│                          and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTex, etc.
│   ├── figures         <- Genereted graphics and figures to be used in reporting
│
├── requirements.txt    <- The requirements file or library or packages for reproducing analysis environment,
│                          e.g. generated with `pip` or `conda` or directly using `docker`
│
└── src                 <- Source code for use in in this project
    ├── cli             <- Scripts to get user input via cli
    │   └── cli.py
    │
    ├── data            <- Scripts to download and/or generate dataset
    │   └── make_dataset.py
    │
    ├── models            <- Scripts to train the model and then trained models to make predictions
    │   └── train.py
    │   └── predict.py
    │
    └── visualization   <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```