# 📘 NLP Master (NLP_DataScience)

> A collection of notebooks, models, and experiments focused on Natural Language Processing (NLP) — exploring approaches like BoW, TF-IDF, Word2Vec, Naïve Bayes, spam detection, sentiment analysis, and more.

---

## 📑 Table of Contents
1. [Introduction](#introduction)  
2. [Repository Contents](#repository-contents)  
3. [Features](#features)  
4. [Tech Stack](#tech-stack)  
5. [Setup & Installation](#setup--installation)  
6. [Usage](#usage)  
7. [Datasets](#datasets)  
8. [Results & Artifacts](#results--artifacts)  
9. [Future Enhancements](#future-enhancements)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Author](#author)  

---

## 🚀 Introduction
This repository explores a variety of **NLP techniques** and provides Jupyter notebooks, datasets, and trained models. It is designed for learners and practitioners who want to explore both traditional NLP methods and modern approaches.

The project covers:
- Text preprocessing and cleaning
- Feature engineering (BoW, n-grams, TF-IDF, embeddings)
- Classification models (Naïve Bayes, Logistic Regression, SVM)
- Use cases like **spam detection**, **sentiment analysis**, and **categorization**
- Initial experiments with **transformer-based models**

---

## 📂 Repository Contents

| File / Folder | Description |
|---------------|-------------|
| `text_preprocessing.ipynb` | Notebook showing text cleaning & preprocessing (tokenization, stopwords, etc.). |
| `Bag of Words (BoW).ipynb` | Implementation of Bag of Words and related experiments. |
| `BOW.pdf` | Slides / notes explaining Bag of Words. |
| `n-gram.ipynb` | N-gram model implementation and examples. |
| `tf=idf.ipynb` | TF-IDF implementation. |
| `sentiment with tfidf .ipynb` | Sentiment analysis using TF-IDF features. |
| `spam_detection_tfidf.ipynb` | Spam detection model with TF-IDF. |
| `svc_tfidf.ipynb` | Support Vector Classifier applied to text features. |
| `naivebayes_new_catagorization.ipynb` | Naïve Bayes text classification. |
| `Word2Vec.ipynb` | Word2Vec embeddings example. |
| `transformer_euri.ipynb` | Experiments with transformer models. |
| `Cleaned_Text_Dataset.csv` | Preprocessed dataset ready for modeling. |
| `news_data.csv` | Dataset of news articles used for classification. |
| `logistics_classification.ipynb` | Logistic Regression text classification task. |
| `logistics_spam.pkl`, `tfidf_spam.pkl` | Serialized trained models for reuse. |

---

## ✨ Features
- 🔤 Text preprocessing pipeline  
- 📊 Feature engineering (BoW, n-grams, TF-IDF, Word2Vec)  
- 🏷️ Classification (spam, sentiment, categorization)  
- 🤖 Traditional ML (Naïve Bayes, Logistic Regression, SVM)  
- 🧠 Early experiments with transformers  

---

## 🛠 Tech Stack
- **Language**: Python 3.8+  
- **Libraries**:  
  - `numpy`, `pandas` → data handling  
  - `nltk`, `spaCy` → preprocessing  
  - `scikit-learn` → ML models  
  - `gensim` → Word2Vec  
  - `matplotlib`, `seaborn` → visualization  
  - `transformers` (HuggingFace) → transformer models  

---

## ⚙️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ramesitexp/NLP_DataScience.git
   cd NLP_DataScience
