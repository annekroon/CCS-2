# Take-home exam: On Machine Learning
### Deadline: Friday 27 May, 5 pm.

This exam is handed our on Tuesday 24 May after the tutorial meetings. Submit your answers via Canvas before the 
deadline. Note that this is an individual exam and any evidence of co-operation will be reported to the exam committee. 

The exam consists of coding questions in which you will be asked to create code and of context questions in which you 
will be asked to use the literature that you have studied in the course to answer questions that help to contextualize 
the code you create.

  1.  **Machine learning: supervised and unsupervised**
      (context questions) In this course, we discusses various methods by placing them on a continuum ranging from rule-based approaches to automated approaches. Based on the literature and classes of this course, discuss the method of Machine Learning. In your answer, discuss the following:
      - What is supervised machine learning typically used for? Provided an example of a supervised machine learning application.
      - How does supervised machine learning differ from unsupervised machine learning?
      - Do you consider supervised machine learning to be a predominantly rule-based approach or an automated approach? Why?

  2.  **Exploring Twitter data**
      Download the data that is available on https://www.dropbox.com/sh/4mapojr85a6sc76/AABYMkjLVG-HhueAgd0qM9kwa?dl=0 (hatespeech_text_label_vote_RESTRICTED_100K.csv). This dataset consists of the texts of tweets and of a variable categorizing the tweet as being either normal, abusive, hateful, or spam. 
      - (coding question) Explore the dataset by calculating the amount of different tweets that are present in the dataset. 
      - (context question) Are some types of tweets present more often then others? Why is it relevant to check this?
  
  3.  **Building classifiers**
      (coding questions) A researcher wants to study how people express themselves on Twitter. To help her, create a (supervised) machine learning classifier that can be used to categorize tweets using the data you downloaded. Do so by:
      - Comparing at least two different classifiers (i.e., Logistic Regression, Naïve Bayes, SVM, Random Forest) 
      - Comparing a count vectorizer to a tfidf vectorizer
      
      To build your classifier, you will need to complete the following steps:
      - Split the dataset into a training set and a test set (hint: X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42) for which you will need to run: from sklearn.model_selection import train_test_split)
      - Set up the vectorizers
      - Training the classifiers
    
      Note that you have a few options here: you can build a machine that differentiates between the four different types of tweets, or you can choose to combine categories. You can play around with this and decide what approach you prefer.  
      Use the materials used in weeks 7 and 8 to find example code. 
      
                                               
  4.  **Validating the classifier**
      - (context question) How do your classifiers work: do they distinguish between the four different types of tweets, or did you decide to merge some categories? Why did you decide to do it in this way? 
      - (coding question) For each of your classifiers, calculate the precision, recall and the f1-score.
      - (context question) Based on the results you get, what classifier would you recommend the researcher to use? Why?
    

---

## Your answers to the take-home exam consist of the following:

  1. A text document (submit as PDF) 
  2. A .py file with your code 
  3. The output files for your code (if you work in Jupyter Notebooks, code and output can be integrated). 

The text document contains a list of all the questions and your answers. In case of coding questions, your answer consists of the line number that correspond to the relevant part of the code in the .py file.

Preferably, your code is well-documented and available in a github repo.


