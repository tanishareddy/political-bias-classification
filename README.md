# Political Bias Classification Using Machine Learning

## Project Overview

This project explores the use of Natural Language Processing (NLP) and machine learning to classify political tweets associated with the Democratic and Republican parties.

The project includes data preprocessing, text representation, model training, validation-based model selection, and final evaluation on a held-out test set. Three machine learning classifiers are compared to identify the best-performing model for this dataset.

## Objectives

- Clean and prepare political tweet data for classification.
- Explore text preprocessing variants.
- Train and compare multiple machine learning classifiers.
- Select a model using validation Macro F1-score.
- Evaluate the selected model on an unseen test set.
- Compare classification performance using standard evaluation metrics.

## Dataset

The project uses the [Democrat vs. Republican Tweets dataset on Kaggle](https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets).

The preprocessing workflow uses `ExtractedTweets.csv` and the `Party` label to distinguish between the two categories:

- **Democrat** — encoded as `0`
- **Republican** — encoded as `1`

The project focuses on party-associated tweets. These labels do not necessarily represent independently verified political bias or the factual accuracy of a tweet.

## Methodology

### 1. Data Preprocessing

The preprocessing notebook prepares the tweet data for machine learning. The workflow includes handling missing values, addressing conflicting labels, removing duplicate records, and generating alternative text variants.

The prepared dataset is divided into training, validation, and test partitions.

### 2. Text Representation

The model training notebook uses `tweet_variant_b` as its input text column.

Different text representations are used according to the classifier:

- Binary Count Vectorization for Bernoulli Naive Bayes.
- Count Vectorization for Multinomial Naive Bayes.
- TF-IDF Vectorization for LinearSVC.

The vectorizers are included within scikit-learn pipelines and are fitted on the training data.

### 3. Machine Learning Models

The following classifiers are compared:

| Model | Text representation |
|---|---|
| Bernoulli Naive Bayes | Binary Count Vectorization |
| Multinomial Naive Bayes | Count Vectorization |
| Linear Support Vector Classifier (LinearSVC) | TF-IDF |

### 4. Model Selection and Evaluation

The evaluation workflow follows three stages:

1. Fit each classification pipeline on the training set.
2. Compare model performance on the validation set and select the model with the highest Macro F1-score.
3. Evaluate the selected model on the held-out test set.

The evaluation metrics include accuracy, macro precision, macro recall, and macro F1-score.

## Experimental Results

The following values were reported for the corrected experiment. They should be confirmed against the executed notebook outputs before being treated as final results.

### Validation Performance

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Bernoulli Naive Bayes | 79.96% | 0.7989 |
| Multinomial Naive Bayes | 79.98% | 0.7992 |
| LinearSVC | 79.89% | 0.7986 |

Based on the reported validation results, **Multinomial Naive Bayes** was selected as the best-performing model.

### Final Test Performance

| Metric | Reported Result |
|---|---:|
| Accuracy | 79.94% |
| Macro Precision | 80.01% |
| Macro Recall | 79.86% |
| Macro F1-score | 0.7988 |

These results describe the reported experiment on the selected dataset and split. They should not be interpreted as a guarantee of performance on other datasets or political content.

## Repository Structure

```text
political-bias-classification/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_model_training_evaluation.ipynb
├── .gitignore
└── README.md
```

The notebooks contain the preprocessing workflow and the model training and evaluation pipeline.

## Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- Kaggle
- Git and GitHub

## How to Run

1. Open the project repository on GitHub.
2. Open the notebooks from the `notebooks/` directory in Google Colab.
3. Obtain the dataset from Kaggle and make it available in the Colab environment.
4. Configure Kaggle access using your own credentials. Do not upload credentials to GitHub.
5. Run `01_data_preprocessing.ipynb` first to generate the required dataset splits.
6. Run `02_model_training_evaluation.ipynb` to train the classifiers, compare validation performance, and evaluate the selected model.

The model evaluation notebook expects the following files to exist in the Colab environment:

- `/content/train_split.csv`
- `/content/val_split.csv`
- `/content/test_split.csv`

The selected model pipeline is saved as `/content/best_model_pipeline.joblib` during execution.

## Limitations

- Party-associated labels are not equivalent to independently verified political-bias labels.
- The dataset may reflect the language, accounts, and time period represented in the source data.
- Performance on this dataset may differ from performance on news articles or other social media content.
- The reported results apply to the particular preprocessing and evaluation setup used in the experiment.

## Reference

https://cs229.stanford.edu/proj2021spr/report2/81996990.pdf
The Stanford research paper provided for this project serves as background and motivation. This implementation uses a separate tweet dataset and its own machine learning workflow; it is not an exact reproduction of the paper.


