# Graph-Embedding-For-Recommendation-System
 Python based Graph Propagation algorithm, DeepWalk to evaluate and compare preference propagation algorithms in heterogeneous information networks from user item relation ship.

## Objective:
* Predict User's preference for some items, they have not yet rated using graph based Collaborative Filtering technique, DeepWalk on user-movie rating data set. 
* Firstly, using the movie review data set, a heterogeneous graph network with nodes as users, movies and its associated entities (actors, directors) were created.
* DeepWalk was used to generate a random walk over this graph. 
* Theses random walks were embedded in low dimensional space using Word2Vec. 
* The prediction for rating for a user-movie pair was done by finding the movie-rating node with the highest similarity to the user node.

## Requirements:
* numpy
* scipy

## Steps to Run:
Run the following command from root folder(not inside rec2vec)
```python
python -m rec2vec --walk-length 2 --number-walks 2 --workers 4
# ****arguments****
# walk-length
# number-walks
# workers
```

#### Ref : https://github.com/phanein/deepwalk
