# Using random user input in a content-based recommender system

Some of you want to recommend data based on some keywords provided by users (e.g., think about asking what type of genres a person likes).

### 1. Get the data in the right shape (see code [here](build_a_recommender.ipynb))

```python
data = get_data(PATH)
data['release_year'] = pd.DatetimeIndex(data['release_date']).year
data['genres'] = data['genres'].apply(get_genres)

def combine_features(data):
    data['combined_features'] = data[['original_title', 'genres', 'overview', 'tagline']].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    return data

data = combine_features(data)
```
### 2. Get the input from the user.

Potentially, you need to preprocess this `str` further.

```python
QUERY = 'romance thriller Bike horse animal'.lower() #later, you can replace it with something like:
#print("Hello user! What type of genres or things in general do you like? You can just insert some key words!")
#QUERY = input()
```
### 3. Integrate the user input in your recommendation.

```python
tfidf = TfidfVectorizer(stop_words='english')
r = data['combined_features'].values.tolist()
r.append(QUERY) # add the user input to the list `r`

tfidf_matrix = tfidf.fit_transform(r)
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim[-1] # QUERY is now the last item in the list 'r'
sim_scores = list(enumerate(cosine_sim[index]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
movie_indices = [i[0] for i in sim_scores[1:10]]

data.iloc[movie_indices]['title']
```
