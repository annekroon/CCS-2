# Working with text: Warming up

Today, we will apply what we discussed in the lecture to Wikipedia texts about songs.

Download the file "WikipediaSongs.csv". 


## 1. Retrieving some basics.
First, focus on the text of the first song only.
- As a first step, let's see how long this text is. Figure out: how many words are included in the text? And how many sentences? Think about the following:
	- Tokenization
	- Capital letters
	- Punctuation
	- Verbs: Consider different conjugations of the same verb to be the same word 


## 2. Counting words.
- What are the five most used (meaningful) words? 
- Based on the stopwords list included in the NLTK, how often does a stopword occur in this text?
- Based on what you find, would you use the NLTK punctuation and stopword lists for this exercise straight away? If not, how would you modify them?


## 3. Using regular expressions.

- A researcher wants to know how well-sourced this first Wikipedia text is. Using regular expressions, can you count how many times references are used in the text?

Hint: If you are having trouble finding useful regular expressions, have a look here: https://docs.python.org/3/howto/regex.html#regex-howto and here: https://regexr.com/


## 3. Tidying things up.
Now, let's apply this to all the texts that are included in the datafile.
- Can you build a function that performs some of the tasks above for each text that it receives as input? For each text the function needs to indicate:
	- How many sentences are included in the text
	- How many words are included in the text		
	- The five most commonly used (meaningful) words
 	- How many times references are mentioned 

- Can you store this information neatly in a datafile?
