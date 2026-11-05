# SmishGuard: Smishing Detection System

A machine learning system that classifies SMS messages and URLs as **spam (smishing)** or **ham (legitimate)**.

## Introduction
Smishing ("SMS phishing") tricks people into clicking malicious links or sharing sensitive information through text messages. This project builds a text classifier that learns the patterns of fraudulent messages and links, and compares three algorithms to find which detects them best.

## Overview
- Combines an SMS dataset and a links dataset into a single dataframe
- Cleans the data (removes missing values) and converts text to features using **TF-IDF**
- Trains and compares **Multinomial Naive Bayes**, **Support Vector Machine**, and **K-Nearest Neighbors**
- Evaluates with accuracy, precision, recall, and F1-score
- Predicts on new, unseen messages and links

## Results
Evaluated on a held-out test set of 1,115 messages (80/20 split).

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 96.2% | 1.00 | 0.72 | 0.84 |
| Support Vector Machine | 97.9% | 0.98 | 0.86 | 0.92 |

SVM gave the best balance. Naive Bayes had perfect spam precision but missed more spam (lower recall).



## Tech Stack
Python, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn

## Dataset
Datasets are not included in this repository. Download them and place them in the project folder:
- `sms_spam.csv`: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle / UCI)
- `links_spam.csv`: a labeled spam/malicious URL dataset

If your column names differ, edit the `CONFIG` block at the top of `smishing_detection.py`.

## How to Run
```bash
git clone https://github.com/Aditi-adu-phen/smishing-detection.git
cd smishing-detection
pip install pandas numpy scikit-learn nltk matplotlib seaborn
python smishing_detection.py
```

## Project Structure
```
smishing_detection.py   - data loading, training, evaluation, prediction
README.md
results.png             - classifier accuracy comparison
```

## Future Improvements
- Improve spam recall with class balancing
- Add ROC curve and cross-validation
- Wrap the model in a small API for live message checks
