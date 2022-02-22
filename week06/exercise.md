# Building your own recommender system

Today, we will start working on building our own recommender system. For this assignment, we will work with movie data.
Download the following datasets [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata):
- `tmdb_5000_credits.csv`
- `tmdb_5000_movies.csv`

Place the files a `data/` folder.

## 1. Explore and preprocess the data.


### a.  Explore the data

- As a first step, explore the datasets. Inspect what data you have at hand, what might be interesting variables and what not. Make a selection of interesting columns.
  -  Keep in mind that ultimatly, you want to build a knowledge-based and content-based recommender systems. Hence, look for columns that might be suitable to use later on.
- Combine (merge) both datasets. Can you identify a variable that can be used for matching?
Think about a good way to do this. Can you write a function that will return the merged data?

*hint*:

```python
def get_data():

    data1 = pd.read_csv('data/tmdb_5000_credits.csv')
    data2 = pd.read_csv('data/tmdb_5000_movies.csv')
    data2.rename(columns={'id': 'movie_id'}, inplace=True)

    data = pd.merge(data1,data2,  on=['movie_id', 'title'])
    data["original_title"] = data["original_title"].str.lower()

    return data
```

- Check whether the data is ready to use, or whether you need to transform or pre-processing your data somehow.

### b.  Pre-processing and feature engineering

As a first step, some data wrangling techniques are needed to get the data into the right shape.
- Think about relevant attributes of movies that you want to use later on when designing a recommender system.
- Can you convert `release_year` to a yearly-level variable?
- Can you clean up the `genres` column?

*hint*:
```python
from ast import literal_eval

def get_genres(x):
    return " ".join( [e['name'] for e in literal_eval(x)] )

data['genres'] = data['genres'].apply(get_genres)

```

## 2.   Create a knowledge-based recommender system

### a. Transform the data from wide to long

In order to create a knowledge-based recommender system, that leverages information on genre, we need to transform our data so that each genre is a single observation. Hence, we want a single genre in the rows. We will therefore transform the data from a wide to a long format:

```python
s = data.apply(lambda x: pd.Series(x['genres'].split()),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'single_genre'
data = data.join(s)

data[['single_genre', 'title', 'vote_average', 'vote_count', 'release_year']].head() #inspect the data to see whether all goes well.
```

#### b. Build a knowledge-based recommender system.

Next, build a simple knowledge-based recommender system based on the transformed data. You can find an example below. Try to adjust the code yourself; can you use different attributes of the movies to build your recommendation on?

```python
def knowledge_based_recommender(data):

    data = data[data['single_genre'].notna()]
    data['single_genre'] = data['single_genre'].str.lower()

    print(f"What type of genre do you like? \n\nYou can choose from the following:\n\n{set(data['single_genre'])}")

    genre = input().lower()

    print("What is the minimum release year of movies you are interested in? (e.g., how 'old' may a movie be?)" )

    release_year = int(input())

    movies = data[(data['single_genre'] == genre) &
    (data['release_year'] >= release_year) ]

    recommend_movies = movies.sort_values('vote_average', ascending=False)

    return recommend_movies[['title', 'vote_average', 'genres']].head(5)
```

We can further improve this algorithm by accounting for the fact that some movies have not been frequently rated. See for an improved scoring algorithm that is typically employed by IMDB [here](https://www.datacamp.com/community/tutorials/recommender-systems-python)


## 3. Create a content-based recommender system

Use the "wide" dataset (hence, before exploding the data to a long format, at step 1b).

### a. Create a combined feature column.
Create a combined feature column. More specifically, combine (textual) data of several columns, that can be used later on.

*hint*:
If you want to 'glue' several columns with textual data together in `pandas`, you can do something like this:

```python
data['combined'] = data[['genres', 'overview']].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
```

Make sure you are making an **informed** decision about merging these columns together. Why do think this is a good idea, when designing a recommender system?

### b. Transform your data

Think about a strategy for transforming your combined data column, as designed in step 2. More specifically, `fit_transform` the combined data column using `tfidf` or `count` vectorizer.
When initializing the vectorizer, think about some of settings we've discussed in earlier weeks. Do you, for example, want to remove stopwords manually, or use pruning?

### c. Calculate cosine similarity

- Using the vectorized (sparse) matrix, calculate cosine similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
```

### d. Write a function that recommends movies on the basis of an inserted movie title.

- Think about ways to find the most similar movies.
- Think about what information of the most similar movies you want to return to the user (e.g., movie title, genre, etc.)
