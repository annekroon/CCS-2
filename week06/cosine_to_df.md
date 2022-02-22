## Cosine similarity scores to df

This code example will provide an example of how to insert the output of `cosine_similarity` into a df.
Note that this is NOT something you would normally want to do, but it may help understand a bit better what happens under the hood.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

def get_data():
    data1 = pd.read_csv('data/tmdb_5000_credits.csv')
    data2 = pd.read_csv('data/tmdb_5000_movies.csv')
    data2.rename(columns={'id': 'movie_id'}, inplace=True)

    data = pd.merge(data1,data2,  on=['movie_id', 'title'])
    data["original_title"] = data["original_title"].str.lower()
    return data

def get_genres(x):
    return " ".join([e['name'].lower() for e in literal_eval(x)])

data= get_data()
data['genres'] = data['genres'].apply(get_genres)

df = data.sample(6) # take a random sample
df[['title', 'genres']]

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

df = df.set_index(['title','genres'])
v = cosine_similarity(tfidf_matrix)

print(v)  # this will return the similarity scores
```

now, combine `title`, `genre` and cosine similarity scores (`v`) into a single `df`:

```python
df = pd.DataFrame(v, columns=df.index.values, index=df.index)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=('title', 'genre'))
df.style.highlight_max(color = 'lightgreen', axis = 0)
```
