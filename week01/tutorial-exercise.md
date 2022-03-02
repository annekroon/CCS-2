# Working with text: Warming up

Today, we will apply what we discussed in the lecture to Wikipedia texts about songs.

Download the file "WikipediaSongs.csv". 


## 1. Retrieving some basics.
First, focus on the text of the first song only.
- As a first step, let's see how long this text is. Figure out: how many words are included in the text? And how many sentences? Think about the following:
	- Tokenization
	- Punction
- What are the five most used (meaningful) words? 
- Based on the stopwords list included in the NLTK, how often does a stopword occur in this text?


## 2. Using regular expressions.

- A researcher wants to know how well-sources this Wikipedia text is. Using regular expressions, can you count how many times a reference is used in the text?
- In the text, different variations on the name "Molly Malone" are used, such as "Widow Malone". Can you list all the different variations of the name "Mally Malone" that occur in the text? How often does each variation occur?

Hint: If you are having trouble finding useful regular expressions, have a look here: https://docs.python.org/3/howto/regex.html#regex-howto 


## 3. About verbs.

- A linguist wants to know how many verbs (not considering their different conjugations) are used in this text. Can you count this for her? To do so, you need to use either stemming or lemmatization. Which one do you select and why?


## 4. Tyding things up.
Now, let's apply this to all the texts.
- A researcher wants to examine the Wikipedia texts of many songs. Can you build a function that performs some of the tasks above for each text that it receives as input? For each text the function needs to indicate:
	- How many words are included in the text
	- How many sentences are included in the text		
	- The five most commonly used (meaningful) words
 	- How many times a source is mentioned 
 	- How many verbs are used 

- Can you store this information neatly in a datafile?
