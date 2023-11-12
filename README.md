# IcelandicDyslexiaClassifier
This is a respository for a final project in "NLP" at the University of Reykjavík during fall 2023.

The following code uses Scitkit-learn to train a logistic regression model and an SVM model to detect signs of dyslexia in writing. The training data are student essays from the [Icelandic Error Corpus](https://github.com/antonkarl/iceErrorCorpus); The Icelandic Dyslexia Error Corpus and The Icelandic Children Error Corpus from the [Icelandic Specialized Error Corpora](https://github.com/antonkarl/iceErrorCorpusSpecialized). Below is an overview of the corpora:

| Corpus    | Total Sentences | Correct Sentences | Incorrect Sentences | % Sentences with Errors |
|-----------|-----------------|-------------------|---------------------|-------------------------|
| Dyslexia  | 1838            | 284               | 1555                | 84.6%                   |
| General   | 7981            | 5102              | 2880                | 36.1%                   |
| Children  | 2070            | 413               | 1658                | 80.1%                   |


There are two scripts in this repository: 

icelandic_dyslexia_sklearn.py: logistics regression and svm models (using TF-IDF features) for General vs Dyslexia, General vs Dyslexia excluding punctuation error codes and General vs Dyslexia excluding all error codes. Furtheremore, the same setup is created for Children vs Dyslexia. The performance of the models varies from 0.67 to 0.87 F1-score.

Icebert_icelandic_dyslexia_sklearn.py: this is a script for a logistic regression models trained on General vs Dyslexia using IceBERT sentence embeddings. This is a separate script, because it cannot be run without a powerful GPU. This model achieved a 0.77 F1-score, 0.71 precision and 0.84 recall.

/Naizeth Núñez Macías (naizeth23@ru.is), Ole Brehm (ole23@ru.is) & Annika Simonsen (annika22@ru.is).

