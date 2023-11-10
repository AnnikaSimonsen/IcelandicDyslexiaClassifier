import git
import glob
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import pandas as pd

## Clone Error Corpus and Error Corpus Specialized
!git clone https://github.com/antonkarl/iceErrorCorpusSpecialized.git
!git clone https://github.com/antonkarl/iceErrorCorpus.git

### Dyslexia Corpus
doc_dyslexia = glob.glob('/content/iceErrorCorpusSpecialized/iceErrorCorpusDyslexia/data/*.xml')
### Essays General Corpus
doc_general = glob.glob('/content/iceErrorCorpus/data/essays/*.xml')
### Childrens' corpus
doc_children = glob.glob('/content/iceErrorCorpusSpecialized/iceErrorCorpusChildLanguage/data/*.xml')

ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

print(len(doc_general))

# Dataset creation

ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

def parse_docs(corpus):
    docs = []
    for doc in corpus:
        root = ET.parse(doc).getroot()
        sents = root.findall('.//tei:s', ns)
        docs.append(sents)
    return docs

def extract_sentence(sent):
    sentence_parts = []
    for node in sent:
        if node.tag.endswith('w') and node.text:
            sentence_parts.append(node.text)
        elif node.tag.endswith('revision'):
            original_nodes = node.findall('.//tei:original/tei:w', ns)
            original_text = ' '.join([w.text for w in original_nodes if w.text])
            if original_text:
                sentence_parts.append(original_text)
    return " ".join(sentence_parts)

def create_dataset(corpus1, corpus2):
    dataset = []
    for corpus, writer_type in zip([corpus1, corpus2], ["atypical", "typical"]):
        docs = parse_docs(corpus)
        for doc in docs:
            for sent in doc:
                sentence_text = extract_sentence(sent)
                errors = [error.get('xtype') for error in sent.findall('.//tei:revision/tei:errors/tei:error', ns)] if sent.find('.//tei:revision', ns) else []
                dataset.append({
                    "sentence": sentence_text,
                    "errors": errors,
                    "writer": writer_type
                })
    return dataset

dataset = create_dataset(doc_dyslexia, doc_general)
dataset_children = create_dataset(doc_dyslexia, doc_children)


print(f'Dyslexia + General: {len(dataset)} sentences')
print(f'Dyslexia + Children: {len(dataset_children)} sentences')

print('\n')

print(dataset[:5])

# Dataset creation WITHOUT PUNCTUATION and STYLE ERROR CODES

## EXTRACTING A LIST OF ERROR CODES OF THE SUBCATEGORY 'PUNCTUATION'
## EXCEL FILE IN ONEDRIVE
all_ErrorCodes = pd.read_excel('/content/errorCodes.xlsx', sheet_name = 'ErrorCodesGeneral')

punctuation = all_ErrorCodes[(all_ErrorCodes['Subcategory'] == 'punctuation')]
style = all_ErrorCodes[(all_ErrorCodes['Subcategory'] == 'style')]

## only punctuation
# punctuation_errorCodes = list(punctuation['Error code'])

## only style
# style_errorCodes = list(style['Error code'])

## punctuation and style
punctuation_errorCodes = list(punctuation['Error code']) + list(style['Error code'])

def create_dataset_WITHOUT_PUNCTUATION(corpus1, corpus2):
    dataset = []
    for corpus, writer_type in zip([corpus1, corpus2], ["atypical", "typical"]):
        docs = parse_docs(corpus)
        for doc in docs:
            for sent in doc:
                sentence_text = extract_sentence(sent)
                errors = [error.get('xtype') for error in sent.findall('.//tei:revision/tei:errors/tei:error', ns) if error.get('xtype') not in punctuation_errorCodes] if sent.find('.//tei:revision', ns) else []
                dataset.append({
                    "sentence": sentence_text,
                    "errors": errors,
                    "writer": writer_type
                })
    return dataset

dataset_without_punctuation = create_dataset_WITHOUT_PUNCTUATION(doc_dyslexia, doc_general)
dataset_children_without_punctuation = create_dataset_WITHOUT_PUNCTUATION(doc_dyslexia, doc_children)

"""# New Section"""

# Training and evaluation

# General + dyslexic
    ## With punctuation codes (LR + SVM)
    ## Without punctuation codes (LR + SVM)

# Children + dyslexic
    ## With punctuation codes (LR + SVM)
    ## Without punctuation codes (LR + SVM)

# Pre-processing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# # Splitting data into sentences, errors, and labels ## DATASET
sentences = [entry["sentence"] for entry in dataset]
errors = [entry["errors"] for entry in dataset]
labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset]

# Splitting data into sentences, errors, and labels ### DATASET WITHOUT PUNCTUATION
#sentences = [entry["sentence"] for entry in dataset_without_punctuation]
#errors = [entry["errors"] for entry in dataset_without_punctuation]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_without_punctuation]

# Splitting data into sentences, errors, and labels ### CHILDREN
#sentences = [entry["sentence"] for entry in dataset_children]
#errors = [entry["errors"] for entry in dataset_children]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children]

# Splitting data into sentences, errors, and labels ### CHILDREN WITHOUT PUNCTUATION
#sentences = [entry["sentence"] for entry in dataset_children_without_punctuation]
#errors = [entry["errors"] for entry in dataset_children_without_punctuation]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children_without_punctuation]

# Splitting data into training and testing sets
X_sentences_train, X_sentences_test, X_errors_train, X_errors_test, y_train, y_test = train_test_split(sentences, errors, labels, test_size=0.2, random_state=42, stratify=labels)

# Transform the sentences and errors into feature vectors.
tfidf_vectorizer = TfidfVectorizer(max_features=5000) #### We could start with NONE to consider all the vocabulary
mlb = MultiLabelBinarizer()

# Transform sentences
X_sentences_train_tfidf = tfidf_vectorizer.fit_transform(X_sentences_train)
X_sentences_test_tfidf = tfidf_vectorizer.transform(X_sentences_test)

# Transform errors
X_errors_train_mlb = mlb.fit_transform(X_errors_train)
X_errors_test_mlb = mlb.transform(X_errors_test)

# Combine the features
X_train = np.hstack([X_sentences_train_tfidf.toarray(), X_errors_train_mlb])
X_test = np.hstack([X_sentences_test_tfidf.toarray(), X_errors_test_mlb])

## LOGISTIC REGRESSION ## DATASET
from sklearn.linear_model import LogisticRegression

# Build a classifier.
clf = LogisticRegression(max_iter=100, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced') #### changed max iteration to 100, it made no difference setting it to 1000

# Train and evaluate the classifier.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Typical", "Dyslexic"]).plot()
plt.title('Logistic Regression')
fig1 = plt.gcf()
plt.show()
fig1.savefig('LR_Typical_Dyslexic.png', dpi=100)

## SVM ### DATASET

from sklearn.svm import SVC

svm_clf = SVC(class_weight='balanced')
svm_clf.fit(X_train, y_train)

 # Classify a new example
y_pred = svm_clf.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Typical", "Dyslexic"]).plot()
plt.title('SVM')
fig2 = plt.gcf()
plt.show()
fig2.savefig('SVM_Typical_Dyslexic.png', dpi=100)

#Splitting data into sentences, errors, and labels ### DATASET WITHOUT PUNCTUATION


sentences = [entry["sentence"] for entry in dataset_without_punctuation]
errors = [entry["errors"] for entry in dataset_without_punctuation]
labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_without_punctuation]

# Splitting data into sentences, errors, and labels ### CHILDREN
#sentences = [entry["sentence"] for entry in dataset_children]
#errors = [entry["errors"] for entry in dataset_children]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children]

# Splitting data into sentences, errors, and labels ### CHILDREN WITHOUT PUNCTUATION
#sentences = [entry["sentence"] for entry in dataset_children_without_punctuation]
#errors = [entry["errors"] for entry in dataset_children_without_punctuation]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children_without_punctuation]

# Splitting data into training and testing sets
X_sentences_train, X_sentences_test, X_errors_train, X_errors_test, y_train, y_test = train_test_split(sentences, errors, labels, test_size=0.2, random_state=42, stratify=labels)

# Transform the sentences and errors into feature vectors.
tfidf_vectorizer = TfidfVectorizer(max_features=5000) #### We could start with NONE to consider all the vocabulary
mlb = MultiLabelBinarizer()

# Transform sentences
X_sentences_train_tfidf = tfidf_vectorizer.fit_transform(X_sentences_train)
X_sentences_test_tfidf = tfidf_vectorizer.transform(X_sentences_test)

# Transform errors
X_errors_train_mlb = mlb.fit_transform(X_errors_train)
X_errors_test_mlb = mlb.transform(X_errors_test)

# Combine the features
X_train = np.hstack([X_sentences_train_tfidf.toarray(), X_errors_train_mlb])
X_test = np.hstack([X_sentences_test_tfidf.toarray(), X_errors_test_mlb])

## LOGISTIC REGRESSION ## DATASET WITHOUT PUNCTUATION
from sklearn.linear_model import LogisticRegression

# Build a classifier.
clf = LogisticRegression(max_iter=100, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced') #### changed max iteration to 100, it made no difference setting it to 1000

# Train and evaluate the classifier.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Typical", "Dyslexic"]).plot()
plt.title('Logistic Regression \n Without Punctuation Error Codes')
fig3 = plt.gcf()
plt.show()
fig3.savefig('LR_Typical_Dyslexic_withoutPunctuation.png', dpi=100)

## SVM ### DATASET WITHOUT PUNCTUATION

from sklearn.svm import SVC

svm_clf = SVC(class_weight='balanced') #
svm_clf.fit(X_train, y_train)

 # Classify a new example
y_pred = svm_clf.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Typical", "Dyslexic"]).plot()
plt.title('SVM \n Without Punctuation Error Codes')
fig4 = plt.gcf()
plt.show()
fig4.savefig('SVM_Typical_Dyslexic_withoutPunctuation.png', dpi=100)

#plitting data into sentences, errors, and labels ### CHILDREN
sentences = [entry["sentence"] for entry in dataset_children]
errors = [entry["errors"] for entry in dataset_children]
labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children]

# Splitting data into sentences, errors, and labels ### CHILDREN WITHOUT PUNCTUATION
#sentences = [entry["sentence"] for entry in dataset_children_without_punctuation]
#errors = [entry["errors"] for entry in dataset_children_without_punctuation]
#labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children_without_punctuation]

# Splitting data into training and testing sets
X_sentences_train, X_sentences_test, X_errors_train, X_errors_test, y_train, y_test = train_test_split(sentences, errors, labels, test_size=0.2, random_state=42, stratify=labels)

# Transform the sentences and errors into feature vectors.
tfidf_vectorizer = TfidfVectorizer(max_features=5000) #### We could start with NONE to consider all the vocabulary
mlb = MultiLabelBinarizer()

# Transform sentences
X_sentences_train_tfidf = tfidf_vectorizer.fit_transform(X_sentences_train)
X_sentences_test_tfidf = tfidf_vectorizer.transform(X_sentences_test)

# Transform errors
X_errors_train_mlb = mlb.fit_transform(X_errors_train)
X_errors_test_mlb = mlb.transform(X_errors_test)

# Combine the features
X_train = np.hstack([X_sentences_train_tfidf.toarray(), X_errors_train_mlb])
X_test = np.hstack([X_sentences_test_tfidf.toarray(), X_errors_test_mlb])

## LOGISTIC REGRESSION ## CHILDREN
from sklearn.linear_model import LogisticRegression

# Build a classifier.
clf = LogisticRegression(max_iter=100, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced') #### changed max iteration to 100, it made no difference setting it to 1000

# Train and evaluate the classifier.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
plt.title('Logistic Regression')
fig5 = plt.gcf()
plt.show()
fig5.savefig('LG_Children_Dyslexic.png', dpi=100)

## SVM ### CHILDREN

from sklearn.svm import SVC

svm_clf = SVC(class_weight='balanced') #
svm_clf.fit(X_train, y_train)

 # Classify a new example
y_pred = svm_clf.predict(X_test)
#print(y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
plt.title('SVM')
fig6 = plt.gcf()
plt.show()
fig6.savefig('SVM_Children_Dyslexic.png', dpi=100)

#Splitting data into sentences, errors, and labels ### CHILDREN WITHOUT PUNCTUATION
sentences = [entry["sentence"] for entry in dataset_children_without_punctuation]
errors = [entry["errors"] for entry in dataset_children_without_punctuation]
labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset_children_without_punctuation]

# Splitting data into training and testing sets
X_sentences_train, X_sentences_test, X_errors_train, X_errors_test, y_train, y_test = train_test_split(sentences, errors, labels, test_size=0.2, random_state=42, stratify=labels)

# Transform the sentences and errors into feature vectors.
tfidf_vectorizer = TfidfVectorizer(max_features=5000) #### We could start with NONE to consider all the vocabulary
mlb = MultiLabelBinarizer()

# Transform sentences
X_sentences_train_tfidf = tfidf_vectorizer.fit_transform(X_sentences_train)
X_sentences_test_tfidf = tfidf_vectorizer.transform(X_sentences_test)

# Transform errors
X_errors_train_mlb = mlb.fit_transform(X_errors_train)
X_errors_test_mlb = mlb.transform(X_errors_test)

# Combine the features
X_train = np.hstack([X_sentences_train_tfidf.toarray(), X_errors_train_mlb])
X_test = np.hstack([X_sentences_test_tfidf.toarray(), X_errors_test_mlb])

## LOGISTIC REGRESSION ## CHILDREN WITHOUT PUNCTUATION
from sklearn.linear_model import LogisticRegression

# Build a classifier.
clf = LogisticRegression(max_iter=100, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced') #### changed max iteration to 100, it made no difference setting it to 1000

# Train and evaluate the classifier.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
plt.title('Logistic Regression \n Without Punctuation Error Codes')
fig7 = plt.gcf()
plt.show()
fig7.savefig('LG_Children_Dyslexic_WithoutPunctuation.png', dpi=100)

## SVM ### CHILDREN WITHOUT PUNCTUATION

from sklearn.svm import SVC

svm_clf = SVC(class_weight='balanced') #
svm_clf.fit(X_train, y_train)

 # Classify a new example
y_pred = svm_clf.predict(X_test)
#print(y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("Precision:", round(precision_score(y_test, y_pred),2))
print("Recall:", round(recall_score(y_test, y_pred),2))
print("F1:", round(f1_score(y_test, y_pred),2))
print(classification_report(y_test, y_pred))

# Plot matrix
ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
plt.title('SVM \n Without Punctuation Error Codes')
fig8 = plt.gcf()
plt.show()
fig8.savefig('SVM_Children_Dyslexic_WithoutPunctuation.png', dpi=100)

# Bonus:
  ## Just TF-IDF version

### PREPROCESSING
import numpy as np

def preprocess_no_error(data_set):
  # Splitting data into sentences, errors, and labels
  sentences = [entry["sentence"] for entry in data_set]
  errors = [entry["errors"] for entry in data_set]
  labels = [1 if entry["writer"] == "atypical" else 0 for entry in data_set]


  # 1. Transform the sentences and errors into feature vectors.
  tfidf_vectorizer = TfidfVectorizer(max_features=5000) #### We could start with NONE to consider all the vocabulary
  mlb = MultiLabelBinarizer()

  # Splitting data into training and testing sets
  X_sentences_train, X_sentences_test, X_errors_train, X_errors_test, y_train, y_test = train_test_split(sentences, errors, labels, test_size=0.2, random_state=42, stratify=labels)

  # Transform sentences
  X_sentences_train_tfidf = tfidf_vectorizer.fit_transform(X_sentences_train)
  X_sentences_test_tfidf = tfidf_vectorizer.transform(X_sentences_test)

  X_train = np.array(X_sentences_train_tfidf.toarray())
  X_test = np.array(X_sentences_test_tfidf.toarray())
  return [y_test, y_train, X_test, X_train]

## SVM
from sklearn.svm import SVC
def svm_auto(y_test_input, y_train_input, X_test_input, X_train_input):

  svm_clf = SVC(class_weight='balanced') #
  svm_clf.fit(X_train_input, y_train_input)

  # Classify a new example
  y_pred = svm_clf.predict(X_test_input)
  print(y_pred)

  cm = confusion_matrix(y_test_input, y_pred)
  tn, fp, fn, tp = cm.ravel()

  print("Accuracy:", round(accuracy_score(y_test_input, y_pred),2))
  print("Precision:", round(precision_score(y_test_input, y_pred),2))
  print("Recall:", round(recall_score(y_test_input, y_pred),2))
  print("F1:", round(f1_score(y_test_input, y_pred),2))
  print(classification_report(y_test_input, y_pred))

  # Plot matrix
  ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
  plt.title('SVM \n Classifier without Error Codes')
  fig9 = plt.gcf()
  plt.show()
  fig9.savefig('SVM_Children_Dyslexia_WithoutErrorCodes.png', dpi=100) ############# How to adapt the name???

ds = preprocess_no_error(dataset)
svm_auto(ds[0], ds[1], ds[2], ds[3])

ds = preprocess_no_error(dataset_children)
svm_auto(ds[0], ds[1], ds[2], ds[3])

## LOGISTIC REGRESSION automated
from sklearn.linear_model import LogisticRegression

def log_reg_auto(y_test_input, y_train_input, X_test_input, X_train_input):
  # Build a classifier.
  clf = LogisticRegression(max_iter=100, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced') #### changed max iteration to 100, it made no difference setting it to 1000

  # Train and evaluate the classifier.
  clf.fit(X_train_input, y_train_input)
  y_pred = clf.predict(X_test_input)

  print("Accuracy:", round(accuracy_score(y_test_input, y_pred_input),2))
  print("Precision:", round(precision_score(y_test_input, y_pred_input),2))
  print("Recall:", round(recall_score(y_test_input, y_pred_input),2))
  print("F1:", round(f1_score(y_test_input, y_pred_input),2))
  print(classification_report(y_test_input, y_pred_input))

  cm = confusion_matrix(y_test_input, y_pred_input)
  tn, fp, fn, tp = cm.ravel()

  # Plot matrix
  ConfusionMatrixDisplay(cm, display_labels = ["Children", "Dyslexic"]).plot()
  plt.title('Logistic Regression \n Without Punctuation Error Codes')
  fig7 = plt.gcf()
  plt.show()
  fig7.savefig('LG_Children_Dyslexic_WithoutPunctuation.png', dpi=100)
