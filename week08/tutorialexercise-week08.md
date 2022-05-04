
# Getting some hands-on experience with supervised machine learning

## Question 1. 
In this tutorial you will work with supervised machine learning. We will classify movie reviews from IMBD into positive and negative reviews.
To do this, first install and import the recquired packages and modules:

```python
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (LogisticRegression)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.model_selection import GridSearchCV
import pickle
import joblib
```

## Question 2. 

As you noted when you read the article by Van Zoonen & Van der Meer (2016), there are a few steps that we need to take before we can use supervised machine learning. Namely:
- Collect data (in CS often texts, e.g., tweets)
- Develop a codebook and hand-code the data
- Transform the text into vectors of numbers

In this tutorial, we focus on the actual machine learning part of the process. Hence, we will use a database that already has a train subset and a test subset consisting of tweets and their labels. In this dataset, tweets are annotated according to six emotions (sadness, joy, fear, anger, love, and surprise). Hence, we skip the first three steps of the process described above.

Download the data for this exercise which consists of two files: test.txt and train.txt
These datafiles were retrieved from: https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp 


Can you write a script that opens each file and:
- Creates one list with the texts from the test-set
- Creates one list with the labels from the test-set
- Creates one list with the texts from the train-set
- Creates one list with the labels form the train set 

What could you do to check that this process went well? Can you explore the data a bit (e.g. by checking how often each label is present in the different datasets)?


Trouble figuring it out? (Note that a potential solution is provided here, but it is important that you go through it and make sure you understand what happens here and not merely copy and run the code)
```python
test = "test.txt"
train = "train.txt"

texts_test = []
labels_test = []

texts_train = []
labels_train = []

with open(test) as fi:
    data = csv.reader(fi, delimiter=';')
    for row in data:
        texts_test.append(row[0])
        labels_test.append(row[1])

with open(train) as fi:
    data = csv.reader(fi, delimiter=';')
    for row in data:
        texts_train.append(row[0])
        labels_train.append(row[1])

        
len(texts_test) == len(labels_test)
len(texts_train) == len(labels_train)


Counter(labels_train)
Counter(labels_test)

plt.bar(Counter(labels_test).keys(), Counter(labels_test).values())
```

## Question 3.
Now that we hopped over steps 1, 2, and 3, we will proceed to step 4: Transforming the text into numbers, or setting up a vectorizer. Let’s use a count vectorizer. Run this code:  

```python
countvectorizer = CountVectorizer(stop_words="english")
X_train = countvectorizer.fit_transform(texts_train)
X_test = countvectorizer.transform(texts_test)
```

In the first line of code, you will see that the stopwords are defined (as a built-in stop word list). Why is that done?

## Question 4.
Now, let’s train our classifier, run it on the test data and request some information to evaluate it. Let’s run a Naïve Bayes classifer with our count vectorizer. Run this code:  

```python
nb = MultinomialNB()
nb.fit(X_train, labels_train)

y_pred = nb.predict(X_test)

accuracy = nb.score(X_test, labels_test)
print(accuracy)

cm = confusion_matrix(labels_test, y_pred)
print(cm)

print(classification_report(labels_test, y_pred))
```

What does the output print? Based on this output, would you say the classifier performs well? Which metric is most important for you to base your decision on? Why?

## Question 5.
Let's try out some other classifiers as well to investigate which one would be best to use. Can you write a code that sets up a tf·idf vectorize and use this in a model based on Logistic Regression? Have a look at the code above and at the documentation of sklearn.


Trouble figuring it out? (Note that a potential solution is provided here, but try to go through it and compare it to the code used in Q3 and Q4 and spot the differences) 
```python
tfidfvectorizer = TfidfVectorizer(stop_words="english")
X_train = tfidfvectorizer.fit_transform(texts_train)
X_test = tfidfvectorizer.transform(texts_test)

logres = LogisticRegression()
logres.fit(X_train, labels_train)

y_pred = logres.predict(X_test)
print(classification_report(labels_test, y_pred))
```

## Question 6.
As you saw in the article by Meppelink et al., we can try different combinations of these models (Naïve Bayes and Logistic Regression) and vectorisers. This results in four classifiers. 

We could simply copy-paste the code used in the previous questions and adjust it for each of the classifiers. However, a cleaner approach is to write a function in which we define the specifics of each classifier. Run the following code to do so:

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

Now, we can create a loop that trains each classifier by calling the function that we build. Run this code:

```python
for name, vectorizer, classifier in configs:
    print(name)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    classifier.fit(X_train, labels_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(labels_test, y_pred))
    print("\n")  
```

What classifier performs the best? Why do you think this is the best classifier?


## Question 7.
In the first part of the code you are asked to run for question 6 (where you define the specifics of various classifiers) will see that for each classifier, the min_df and the max_df are set to 5 and 0.5 respectively. Take a look at the documentation of scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). Can you figure out what these numbers mean? Would you like to change these numbers? Why?

## Question 8.
Question 7 asked you what the numbers 5 and 0.5 that were set for each classifier mean. In addition to these two hyperparameters, there are many more hyperparameters that we can set. It is up to you what hyperparameters to change from the default value and what value to give them (take a look at the documentation to learn more about each hyperparameter than you can set https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). 

What if we want to figure out what the best score is for a few hyperparameters? We can do this by performing a gridsearch in which we cross-validate different model specifications. To perform a gridsearch, it is helpful to re-write our code into a pipeline with scikit-learn.

When you look at the code used in the previous question, you can see that it consists of two steps. First, we transform our data into a vector of numbers and second, we fit a classifier. With scikit-learn, we can combine these two steps into a ‘pipeline’. 

Run the code below. What strikes you about the output that it gives?

```python
for name, vectorizer, classifier in configs:
    print(name)
    pipe = make_pipeline(vectorizer, classifier)
    pipe.fit(texts_train, labels_train)
    y_pred = pipe.predict(texts_test)
    print(classification_report(labels_test, y_pred))
    print("\n") 
```

## Question 9.
Now, let’s perform a gridsearch to determine what the best value is for four parameters in a classifier with Logistic regression and a countvectorizer. The parameters we examine are the ngram_range, max_df, min_df, and C (check the documentation if you want to learn what these hyperparameters mean). 

Run the code below. You may notice that it takes the computer quite some time to run this code. Why, do you think it takes your computer so long to run the code? 

```python
pipeline = Pipeline(steps = [
  ("vectorizer", CountVectorizer()), 
  ("classifier", LogisticRegression(
      solver="liblinear"))])
grid = {"vectorizer__ngram_range": [(1,1), (1,2)],
        "vectorizer__max_df": [0.5, 1.0],
        "vectorizer__min_df": [0, 5],
        "classifier__C": [0.01, 1, 100]
       }
search=GridSearchCV(estimator=pipeline, n_jobs=-1,
  param_grid=grid,scoring="accuracy", cv=5)
search.fit(texts_train, labels_train)
print(f"Best parameters for:", name," {search.best_params_}")
pred = search.predict(texts_test)
print(classification_report(labels_test,search.best_estimator_.predict(texts_test))) 
```
What are the best parameters to use according to the results of the gridsearch?

## Question 10.
We have now found the best classifier in terms of what model to use, what vectorizer to use and what values for four hyperparameters to set. We now want to use this model to predict the label for new data that we have not annotated (remember, this was the whole goal of SML)!

To do this, let’s save our classifier and our vectorizer to a file. If we don’t do this, we would need to re-train our model every time we want to use it. This is not so convenient, for example, we would always need to have our training data at hand. The code below shows you how to make a vectorizer and train a classifier (a repetition of what we did before to show you the whole process) and store them into a file.

In the code, you will see that both the classifier and the vectorizer are stored into a file. Why do you need to store both (why not just store the classifier only)?

```python

# Make a vectorizer and train a classifier
vectorizer=TfidfVectorizer(min_df=5, max_df=.5)
classifier=LogisticRegression(solver="liblinear")
X_train=vectorizer.fit_transform(texts_train)
classifier.fit(X_train, labels_train)

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
This exercise is based on the materials developed and the texts written by Wouter van Atteveldt, Damian Trilling and Carlos Arcila Calderon as reported in their book 'Computational Analysis of Communication' (Wiley-Blackwell). The dataset used in this exercises is available on https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp and provided by https://www.kaggle.com/praveengovi.


