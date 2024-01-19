# Movie-Recommendation-System
...photo

This repository contains the code and resources used to build a recommendation system model. This model will be used to predict which genre and movies and likely to recieve high number of viewers based on the data provided which consists of different users and movies of various genres. The model will also be trained using the ratings data of different movies and genres.


# Problem Statement

Filamu Streaming Services is a streaming platform that has had recent complaints from their users over the lack of movie recommendation they like. The absense of a recommendation system on their platform is costing them business.  in order for them to solve this they have to come up with recommendation system  that will improve their user satisfaction.


# OVERVIEW
## Movie Industry Overview

The film industry, a global juggernaut, annually amasses billions in revenue. As per Statista, in 2019 alone, the global box office accumulated a staggering $42.5 billion.
Film production is the creative hub where movies, TV shows, and other visual content come to life, involving stakeholders such as studios, independent producers, and streaming platforms. 
The distribution segment markets and transports these creations to theaters, streaming services, and other outlets. Exhibition refers to the screening of movies and TV shows in cinemas
Movies possess a universal charm, forging connections among individuals from diverse backgrounds. Despite this collective appeal, our personal cinematic tastes exhibit.
uniqueness, spanning various genres such as thrillers, romance, or sci-fi, and often centering around preferred actors and directors.

# BUSINESS PROBLEM

In the world of streaming services, with the rise in users, competition among streaming services is at an all time high. With a high demand in content, the streaming services need to keep up with growing tastes in the consumer base and they need to push content that is relevant and up to their consumers standards.

Filamu Streaming Services aims to stay competitive in this industry by utilizing the data gotten from GroupLens research lab at the University of Minnesota to build a better and more advanced recommendation system.

Crafting a universal formula for movies that would captivate every individual proves challenging; however, through meticulous analysis of user data and movie-related data,valuable insights are extracted so as to provide users with movie recommendations that would best suit their preferences.

# COMPONENTS
Data from-https://grouplens.org/datasets/movielens/latest/
Presentation-https://grouplens.org/datasets/movielens/latest/
Github - Jupyter Notebook

## Data Understanding the Features 

MovieId - represents a unique identifier of the movie
Title -  name of the film containing the coinciding movieId
Genres - a stylistic or thematic category for motion pictures based on similarities either in the narrative elements, aesthetic approach, or the emotional response to the film
UserId -  a unique customer identifier by which an advertiser chooses to identify a user visiting their website.
Rating - a measurement of the quality or success of something
Timestamp - a digital record of the time of occurrence of a particular event.

## Data Sourcing and Preparation
Multiple datasets that had a common column, movieId, were merged into a pandas dataframe so as to make the data more comprehensible. 

The values in the “genre” column were a list which would pose a problem in correctly analyzing each different genre. A function to split the list into individual genres was created.

There were 0 null values and 0 duplicate entries  in the now almost complete dataset.


## Exploratory Data Analysis
Here we will explore the different features of the dataset to gain a better understanding of the data through univariate and bivariate analysis.
We will use data vizualization to uncover trends and patterns. 

## MODELLING.

### Types of Recommendation Systems
## Content Based Filtering
Content-based filtering is a recommendation strategy that suggests items similar to those a user has previously liked. It calculates similarity (often using cosine similarity) between the user’s preferences and item attributes, such as lead actors, directors, and genres. For example, if a user enjoys ‘The Prestige’, the system recommends movies with ‘Christian Bale’, ‘Thriller’ genre, or films by ‘Christopher Nolan’.
However, content-based filtering has drawbacks. It limits exposure to different products, preventing users from exploring a variety of items. This can hinder business expansion as users might not try out new types of products

## Collaborative Filtering
Collaborative filtering is a recommendation strategy that considers the user’s behavior and compares it with other users in the database. It uses the history of all users to influence the recommendation algorithm. Unlike content-based filtering, collaborative filtering relies on the interactions of multiple users with items to generate suggestions. It doesn’t solely depend on one user’s data for modeling. There are various approaches to implement collaborative filtering, but the key concept is the collective influence of multiple users on the recommendation outcome.
Functions were created to  automate the process of creating a model and hyperparameter tuning for collaborative models, providing an efficient way to experiment with different algorithms and settings.

## Building and Tuning
Here different models are constructed using surprise library’s algorithms, then evaluated and tuned according to collaborative filtering model methods.
Models : The models include 
- KNNBasic
- KNNBaseline
- KNNWithMeans
- SVD

Evaluation Metric: Root Mean Squared Error (RMSE)

## Evaluation
The SVD model consistently outperforms the other models, justifying its selection as the final model.
Hyperparameter tuning improves model performance.
User ratings influence the movie recommendations, making the system personalized.

....photo.


## Conclusion
- The top 3 movies are:
Forrest Gump(1994),The Shawshank Redemption(1994) and Pulp Fiction(1994)
- The bar plot shows the distribution of all the movie genres in the dataset.
- The 3 top genres are listed below starting from the most common one:Drama,Comedy and Action


## Recommendation
- Filamu should consider displaying the following movies as the most popular films in the their database: Forrest Gump(1994), Hoop Dreams(1994), Pulp Fiction(1994)
- Filamu should consider having more movies in the following four genres: Drama, Comedy, Action and Thriller, since they are the most popular.
- When deployed, the developed recommendation system will assist Filamu in suggesting films that users might like. This is due to the fact that the system will display to the users films similar to those they have already watched and enjoyed, as well as films that like users have enjoyed.

## Next steps
- Continuously monitor and retrain the deployed model with new data to ensure consistent predictive quality in real-world scenarios.
- Enhance scalability by implementing distributed computing frameworks like Spark to handle large datasets more efficiently.
- Create more comprehensive user and item profiles by incorporating features such as descriptions, demographics, and historical behavior.

## Deployment
Here we will make sure our model can predict.
The content based system was deployed to Streamlit.




 

