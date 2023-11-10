import glob
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

### Dyslexia Corpus
doc_dyslexia = glob.glob('./content/iceErrorCorpusSpecialized/iceErrorCorpusDyslexia/data/*.xml')
### Essays General Corpus
doc_general = glob.glob('./content/iceErrorCorpus/data/essays/*.xml')

ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

# Dataset creation

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

### PREPROCESSING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# Splitting data into sentences, errors, and labels
sentences = [entry["sentence"] for entry in dataset]
errors = [entry["errors"] for entry in dataset]
labels = [1 if entry["writer"] == "atypical" else 0 for entry in dataset]


# Using IceBERT
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("mideind/IceBERT-igc") #"vesteinn/IceBERT""
model = RobertaModel.from_pretrained("mideind/IceBERT-igc") #"vesteinn/IceBERT"

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # Using the [CLS] token embedding as the sentence representation
    return outputs.last_hidden_state[0][0].detach().numpy()

sentences_bert = [get_bert_embedding(sentence) for sentence in sentences]

# Splitting the combined embeddings and labels into training and testing sets
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(sentences_bert, labels, test_size=0.2, random_state=42)

# Build a classifier
clf = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', multi_class='ovr', class_weight='balanced')

# Train the classifier
clf.fit(X_train_bert, y_train_bert)

# Predict on the test set
y_pred_bert = clf.predict(X_test_bert)

# Evaluate the classifier


print("Accuracy:", round(accuracy_score(y_test_bert, y_pred_bert),2))
print("Precision:", round(precision_score(y_test_bert, y_pred_bert),2))
print("Recall:", round(recall_score(y_test_bert, y_pred_bert),2))
print("F1:", round(f1_score(y_test_bert, y_pred_bert),2))
print(classification_report(y_test_bert, y_pred_bert))
