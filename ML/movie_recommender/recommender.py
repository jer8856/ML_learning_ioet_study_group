
"""
ENVIRONMENT ALTERNATIVES
* Colab
* Jupyter
* Python installation

BOOKS
* https://learning.oreilly.com/library/view/machine-learning-with/9781801819312/Text/Chapter_1.xhtml
* https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/part01.html

PROJECTS
* ??

WHERE IT COMES FROM?
https://machinelearningmastery.com/machine-learning-for-programmers/


SYSTEMATIC PROCESS
1. Define the problem
    Step 1: What is the problem?
        Informal description of the problem
        - I need a program that will recommend movies based on the user's votes.
    Step 2: Why does the problem need to be solved?
        Motivation
            - I need to do this as a learning exercise
            - I need to do this as an example for the machine learning study group at ioet
        Benefits
            - I can present my work to others
            - I can use this as a learning exercise
            - I can use this as an example for the machine learning study group at ioet
        Use
            - Use this in react movie system
            - Use this in my github repo
    Step 3: How would I solve the problem? (flush domain knowledge)
        - I need to get the data and process it
        - Which metrics would I use??
        - I need to define the model (knn, decision tree, random forest, etc)
        - I'd write tests to make sure the model is working
        - I need to evaluate, improve, present the model
2. Prepare the data
    1. Select the data -> UCI, Kaggle, etc
        Consider what data is available, what data is missing and what data can be removed.
    2. Preprocess the data -> Formating, cleaning, sampling...
    3. Transform the data -> Scaling, Decomposition, Aggregation...
3. Spot check algorithms
    10 fold cross validation
    Define which algorithms to test
        C4.5 This is a decision tree algorithm and includes descendent methods like the famous C5.0 and ID3 algorithms.
        k-means. The go-to clustering algorithm.
        Support Vector Machines. This is really a huge field of study.
        Apriori. This is the go-to algorithm for rule extraction.
        EM. Along with k-means, go-to clustering algorithm.
        AdaBoost. This is really the family of boosting ensemble methods.
        knn (k-nearest neighbor). Simple and effective instance-based method.
        Naive Bayes. Simple and robust use of Bayes theorem on data.
        CART (classification and regression trees) another tree-based method.
4. Improve Results
    Algorithm Tuning: 
        where discovering the best models is treated like a search problem through model parameter space.
    Ensemble Methods:
        where the predictions made by multiple models are combined.
    Extreme Feature Engineering:
        where the attribute decomposition and aggregation seen in data preparation is pushed to the limits.
5. Present Results --> Deploy Model, Use predictions
    Context (Why):
        I need a program to present in the study group that recommends movies based on the user's votes.
    Problem (Question):
        Would it be possible to predict the satisfaction of a user based on the votes of other users?
    Solution (Answer):
        It was possible to predict the satisfaction using the k-nearest neighbor algorithm.
    Findings:
        Making simple the model was not enough.
        We needed to create a value in the data for creating the metrics.
        Is it reliable?
    Limitations:
        More data is needed.
        Use different attributes to train the model.
    Conclusions (Why+Question+Answer):
        We created a model that predicts the satisfaction of a user based on the votes of other users.
"""

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# data from uci, kaggle
# formatting the data
movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
# cleaning the data
final_dataset.fillna(0,inplace=True)

# Trial and Error experimentation to reduce Noises Off - Sampling

no_user_voted = ratings.groupby('movieId')['rating'].count()
no_movies_voted = ratings.groupby('userId')['rating'].count()

# N users should have voted a movie.
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
# M movies should have been voted by a user.
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

# dimensionality reduction??
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
# knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(final_dataset)
# knn.fit(csr_data)

try:
    movie_name = "Matrix"
    movie_list = movies[movies['title'].str.contains(movie_name)]
    movie_idx= movie_list.iloc[0]['movieId']
    selected_movie = final_dataset[final_dataset['movieId'] == movie_idx]
except IndexError:
    print("Movie not found")
    exit(0)

distances , indices = knn.kneighbors(selected_movie ,n_neighbors=11)  
moviess = sorted(zip(distances[0], indices[0]), key=lambda x: x[0])

for distance, index in moviess:
    movie_id = final_dataset.iloc[index].movieId
    title = movies.title[movies.movieId == movie_id].values[0]
    print(f'{title}, distance: {distance}\n')

"""
LINKS
https://machinelearningmastery.com/machine-learning-for-programmers/  - Machine Learning for Programmers
https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/  - Process for working through machine learning problems
https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/  - A Tour of Machine Learning Algorithms

https://towardsdatascience.com/how-to-build-a-movie-recommendation-system-67e321339109
https://techvidvan.com/tutorials/movie-recommendation-system-python-machine-learning/
"""