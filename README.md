# IcelandicDyslexiaClassifier
This is a respository for a final project in "NLP" at the University of Reykjavík during fall 2023.

The following code uses Scitkit-learn to train a logistic regression model and an SVM model to detect signs of dyslexia in writing. The training data are the subcorpora of the Icelandic Error Corpus; Student Essays from The Icelandic Error Corpus, The Icelandic Dyslexia Error Corpus and The Icelandic Children Error Corpus.

| Corpus    | Total Sentences | Correct Sentences | Incorrect Sentences | % Sentences with Errors |
|-----------|-----------------|-------------------|---------------------|-------------------------|
| Dyslexia  | 1838            | 284               | 1555                | 84.6%                   |
| General   | 7981            | 5102              | 2880                | 36.1%                   |
| Children  | 2070            | 413               | 1658                | 80.1%                   |


There are two scripts in this repository: 

icelandic_dyslexia_sklearn.py: logistics regression and svm models (using TF-IDF features) for General vs Dyslexia, General vs Dyslexia excluding punctuation error codes and General vs Dyslexia excluding all error codes. Furtheremore, the same setup is created for Children vs Dyslexia.

IceBERT_icelandic_dyslexia_sklearn.py: this is a script for a logistic regression models trained on General vs Dyslexia using IceBERT sentence embeddings. This is a separate script, because it cannot be run without a powerful GPU. 

