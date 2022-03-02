
# Getting some hands-on experience with supervised machine learning

## Question 1. 
In this tutorial you will work with supervised machine learning. We will classify movie reviews from IMBD into positive and negative reviews.
To do this, first install and import the recquired packages and modules:

```python
conda install -c intel scikit-learn pandas
conda install -c conda-forge gensim eli5 keras tensorflow
conda install -c anaconda nltk

import os
import bz2
import pickle
import urllib.request
import re
import tarfile
import pandas as pd

from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import (LogisticRegression)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import joblib
import eli5
from nltk.sentiment import vader
```

## Question 2. 

As you noted when you read the article by Meppelink et al., there are a few steps that we need to take before we can use supervised machine learning. Namely:
    - Determine the sample criteria
    - Collect data
    - Develop a codebook and hand-code the data
    - Transform the text into vectors of numbers

In this tutorial, we focus on the actual machine learning part of the process. Hence, we will use a database created by Maas et al. (you can find it here: ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) which we will download in the next step. This data consists of movie reviews that were already given a label (positive or negative). Hence, we skip the first three steps of the process described above. 
To get the data, run the following code:

```python
filename = "reviewdata.pickle.bz2"
if os.path.exists(filename):
  print(f"Using cached file {filename}")
  with bz2.BZ2File(filename, "r") as zipfile:
    data = pickle.load(zipfile)
    text_train, text_test, y_train, y_test = data
else:
  url = "https://cssbook.net/d/aclImdb_v1.tar.gz"
  print(f"Downloading from {url}")
  fn, _headers = urllib.request.urlretrieve(url, 
                     filename=None)
  t = tarfile.open(fn, mode="r:gz")
  text_train,text_test = [], []
  y_train, y_test = [], []
  for f in t.getmembers():
    m=re.match("aclImdb/(\w+)/(pos|neg)/", f.name)
    if not m:
        # skip folder names, other categories
        continue
    dataset, label = m.groups()
    text = t.extractfile(f).read().decode("utf-8")
    if dataset == "train":
      text_train.append(text)
      y_train.append(label)
    elif dataset == "test":
      text_test.append(text)
      y_test.append(label)
  print(f"Saving to {filename}")
  with bz2.BZ2File(filename, "w") as zipfile:
    data = text_train, text_test, y_train, y_test
    pickle.dump(data, zipfile)
```

### Question 2a.
The code consists of an if and an else-statement.

What happens in the if-statement? Why?

### Question 2b.
What happens in the else-statement? Why?

## Question 3.
Now that we hopped over steps 1, 2, and 3, we will proceed to step 4: Transforming the text into numbers, or setting up a vectorizer. Let’s use a count vectorizer. Run this code:  

```python
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(text_train)
X_test = vectorizer.transform(text_test)
```

In the first line of code, you will see that the stopwords are defined (as a built-in stop word list). Why is that done?

## Question 4.
Now, let’s train our classifier, run it on the test data and request some information to evaluate it. Let’s run a Naïve Bayes classifer with our count vectorizer. Run this code:  

```python
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("    \tPrecision\tRecall")
for label in set(y_pred):
    pr = metrics.precision_score(y_test, y_pred, pos_label=label)
    re = metrics.recall_score(y_test,y_pred, pos_label=label)
    print(f"{label}:\t{pr:0.2f}\t\t{re:0.2f}")
```

For both positive and negative reviews, the code gives you the precision and the recall score. Based on these scores, would you say the classifier performs well? Which metric is most important for you to base your decision on? Why?

## Question 5.
Just like Meppelink and her colleagues, we can calculate an additional measure to evaluate our classifiers, namely the F1-score (also see the lecture slides). Take a close look at the lines of code used to calculate the precision score and the recall scores. Also have a look at the documentation for Scikit-learn and at the section about metric in particular (https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics). 

Can you write a line of code to get the F1-score and incorporate this into the code used in the previous question?

## Question 6.
Let’s see if some other classifiers would perform even better than the one based on Naïve Bayes and a count vectorizer. Let’s look at Naïve Bayes versus Logistic Regression and at a count vectorizer versus a tf·idf vectorizer. As you saw in the article by Meppelink et al., this results in four classifiers. 

We could simply copy-paste the code used in the previous question three times and adjust it for each of the classifiers. However, a cleaner approach is to write a function in which we define the specifics of each classifier. Run the following code to do so:

```python
configs = [
  ("NB-count",CountVectorizer(min_df=5,max_df=.5),
   MultinomialNB()),
  ("NB-TfIdf",TfidfVectorizer(min_df=5,max_df=.5),
   MultinomialNB()),
  ("LR-Count",CountVectorizer(min_df=5,max_df=.5),
   LogisticRegression(solver="liblinear")),
  ("LR-TfIdf",TfidfVectorizer(min_df=5,max_df=.5),
   LogisticRegression(solver="liblinear"))]
```

You will see that for each classifier, the min_df and the max_df are set to 5 and 0.5 respectively. Take a look at the documentation of scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). Can you figure out what these numbers mean? Would you like to change these numbers? Why?

## Question 7.
Let’s also request some measures to evaluate the four classifiers. Again, we could copy the code used before three times, but this would get rather messy, especially if we want to reuse our code later on. Can you transform the code used in Q5 into a function called short_classification_report?

## Question 8.
Now, we can create a loop that trains each classifier by calling the first function and then gives a classification report by calling the second function that we created. Run this code:

```python
for name, vectorizer, classifier in configs:
    print(name)
    X_train = vectorizer.fit_transform(text_train)
    X_test = vectorizer.transform(text_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    short_classification_report(y_pred)
    print("\n")
```

What classifier performs the best?

## Question 9.
Question 6 asked you what the numbers 5 and 0.5 that were set for each classifier mean. In addition to these two hyperparameters, there are many more hyperparameters that we can set. It is up to you what hyperparameters to change from the default value and what value to give them (take a look at the documentation to learn more about each hyperparameter than you can set https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). 

What if we want to figure out what the best score is for a few hyperparameters? We can do this by performing a gridsearch in which we cross-validate different model specifications. To perform a gridsearch, it is helpful to re-write our code into a pipeline with scikit-learn.

When you look at the code used in the previous question, you can see that it consists of two steps. First, we transform our data into a vector of numbers and second, we fit a classifier. With scikit-learn, we can combine these two steps into a ‘pipeline’. 

Run the code below. What strikes you about the output that it gives?

```python
for name, vectorizer, classifier in configs:
    print(name)
    pipe = make_pipeline(vectorizer, classifier)
    pipe.fit(text_train, y_train)
    y_pred = pipe.predict(text_test)
    short_classification_report(y_pred)
    print("\n") 
```

## Question 10.
Now, let’s perform a gridsearch to determine what the best value is for four parameters in our preferred classifier with Logistic regression and a tf·idf vectorizer. The parameters we examine are the ngram_range, max_df, min_df, and C (check the documentation if you want to learn what these hyperparameters mean). 

Run the code below. You may notice that it takes the computer quite some time to run this code. Why, do you think it takes your computer so long to run the code? 

```python
pipeline = Pipeline(steps = [
  ("vectorizer", TfidfVectorizer()), 
  ("classifier", LogisticRegression(
      solver="liblinear"))])
grid = {"vectorizer__ngram_range": [(1,1), (1,2)],
        "vectorizer__max_df": [0.5, 1.0],
        "vectorizer__min_df": [0, 5],
        "classifier__C": [0.01, 1, 100]
       }
search=GridSearchCV(estimator=pipeline, n_jobs=-1,
  param_grid=grid,scoring="accuracy", cv=5)
search.fit(text_train, y_train)
print(f"Best parameters: {search.best_params_}")
pred = search.predict(text_test)
print(short_classification_report(y_pred)) 
```
What are the best parameters to use according to the results of the gridsearch?

## Question 11.
We have now found the best classifier in terms of what model to use, what vectorizer to use and what values for four hyperparameters to set. We now want to use this model to predict the label for new data that we have not annotated (remember, this was the whole goal of SML)!

To do this, let’s save our classifier and our vectorizer to a file. If we don’t do this, we would need to re-train our model every time we want to use it. This is not so convenient, for example, we would always need to have our training data at hand. The code below shows you how to make a vectorizer and train a classifier (a repetition of what we did before to show you the whole process) and store them into a file.

In the code, you will see that both the classifier and the vectorizer are stored into a file. Why do you need to store both (why not just store the classifier only)?

```python

# Make a vectorizer and train a classifier
vectorizer=TfidfVectorizer(min_df=5, max_df=.5)
classifier=LogisticRegression(solver="liblinear")
X_train=vectorizer.fit_transform(text_train)
classifier.fit(X_train, y_train)

# Save them to disk
with open("myvectorizer.pkl",mode="wb") as f:
    pickle.dump(vectorizer, f)
with open("myclassifier.pkl",mode="wb") as f:
    joblib.dump(classifier, f)
  
# Later on, re-load this classifier and apply:
new_texts = ["This is a great movie", 
            "I hated this one.", 
            "What an awful fail"]

with open("myvectorizer.pkl",mode="rb") as f:
    myvectorizer = pickle.load(f)
with open("myclassifier.pkl",mode="rb") as f:
    myclassifier = joblib.load(f)
    
new_features = myvectorizer.transform(new_texts)
pred = myclassifier.predict(new_features)

for review, label in zip(new_texts, pred):
    print(f"'{review}' is probably '{label}'.")
```


### About this exercise:
The materials used in this exercise were created by Wouter van Atteveldt, Damian Trilling and Carlos Arcila Calderon and are reported in their book 'Computational Analysis of Communication' (Wiley-Blackwell).

```
