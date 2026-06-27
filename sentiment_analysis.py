import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.txt", delimiter = ";", names = ["text", "label"])
print(train.head())
print(train["label"].value_counts())

"""def custom_encoder(data):
    data.replace(to_replace = "surprise", value = 1, inplace = True)
    data.replace(to_replace = "love", value = 1, inplace = True)
    data.replace(to_replace = "fear", value = 0, inplace = True)
    data.replace(to_replace = "anger", value = 0, inplace = True)
    data.replace(to_replace = "sadness", value = 0, inplace = True)
    data.replace(to_replace = "joy", value = 1, inplace = True

custom_encoder(train["label"])"""

mapping = {"surprise":1, "love":1, "fear":0, "anger":0, "sadness":0, "joy":1}
train["label"] = train["label"].map(mapping)
print(train["label"].isnull().sum())
for label in train["label"].unique():
    print(repr(label))
train["label"] = train["label"].astype(int)



import re 
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()
stopwords = set(stopwords.words("english"))
def text_transformation(data):
    corpus = []
    for sentence in data:
        new_item = re.sub("[^a-zA-Z]", " ", str(sentence))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in stopwords]
        corpus.append(" ".join(str(x) for x in new_item))
    return corpus
#feature extraction (bag of words)
corpus = text_transformation(train["text"])
print(corpus[1])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range = (1,2))
X = cv.fit_transform(corpus)
y = train.label
#model + hyper parameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
parameters = {
    "max_features" : ["sqrt"],
    "n_estimators" : [100,200], 
    "max_depth" : [5, 10, None],
    "min_samples_leaf" : [1,2,5,10],
    "bootstrap" : [True, False]
}


from sklearn.model_selection import GridSearchCV
search_cv = GridSearchCV(RandomForestClassifier(), parameters, cv = 5, return_train_score = True, n_jobs = -1)
search_cv.fit(X,y)

print(search_cv.best_params_)


rfc = RandomForestClassifier(max_features = search_cv.best_params_["max_features"],
 max_depth = search_cv.best_params_["max_depth"],
 n_estimators = search_cv.best_params_["n_estimators"], 
 min_samples_leaf = search_cv.best_params_["min_samples_leaf"], 
 bootstrap = search_cv.best_params_["bootstrap"])

rfc.fit(X, y)
test_data = pd.read_csv("test.txt", delimiter = ";", names = ["text", "label"])
x_test = test_data.text
y_test = test_data.label
custom_encoder(y_test)
x_test = text_transformation(x_test)
x_text = cv.transform(x_test)

y_predict = rfc.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
a_s = accuracy_score(y_test, y_predict)
print("Accuracy score = ", a_s)

cr = classification_report(y_test, y_predict)
print("Classification report = ", cr)

def expression_check(input_str):
    if input_str == 1:
        print("input statement has positive sentiment")
    elif input_str == 0:
        print("input statement has negative sentiment")
    else:
        print("invalid output")
    

def sentiment_predictor(input):
    input_str = text_transformation(input)
    transform_text = cv.transform(input_str)
    prediction = rfc.predict(transform_text)
    expression_check(prediction)

input_1 = ["I am going to the shops with my freinds tommorow!"]
imput_2 = ["Sometimes i want to punch someone."]
sentiment_predictor(input_1)
sentiment_predictor(input_2)