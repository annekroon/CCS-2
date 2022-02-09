## Build a content-based movie recommender system

Today, we will start working on building our own recommender system. For this assignment, we will work with movie data.
Download the following datasets [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata):
- `tmdb_5000_credits.csv`
- `tmdb_5000_movies.csv`

Place the files a `data/` folder.

### 1.  Explore the data

- As a first step, explore the datasets. Inspect what data you have at hand, what might be interesting variables and what not. Make a selection of interesting columns.
  -  Keep in mind that ultimatly, you want to build a content-based recommender systems. Hence, look for columns that might be suitable to use later on.
- Combine (merge) both datasets. Can you identify a variable that can be used for matching?
- Check whether the data is ready to use, or whether you need to transform or pre-processing your data somehow.


### 2.  Feature engineering

- Create, on the basis of step 1, a combined feature column. More specifically, combine (textual) data of several columns, that can be used later on.

*hint*:
If you want to 'glue' several columns with textual data together in `pandas`, you can do something like this:

```python
data['combined'] = data[['genres', 'overview']].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
```

- Make sure you are making an **informed** decision about merging these columns together. Why do think this is a good idea, when designing a recommender system?


### 3. Transform your data

- Think about a strategy for transforming your combined data column, as designed in step 2. More specifically, `fit_transform` the combined data column using `tfidf` or `count` vectorizer.
- when initializing the vectorizer, think about some of settings we've discussed in earlier weeks. Do you, for example, want to remove stopwords manually, or use pruning?


### 4. Calculate cosine similarity

- Using the vectorized (sparse) matrix, calculate cosine similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
```

### 5. Write a function that recommends movies on the basis of an inserted movie title.

- Think about ways to find the most similar movies.
- Think about what information of the most similar movies you want to return to the user (e.g., movie title, genre, etc.)
