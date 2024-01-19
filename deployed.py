import streamlit as st
import pandas as pd
import pickle 
import requests

###
movie_rating_df = pd.read_csv('ml-latest-small/movie_rating_df.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# movie = movies[movies['genres'].str.contains('Adventure')].sample(1)
# print(movie)

movie_mapper = dict(zip(movies['title'], movies.index))
user_item_df = movie_rating_df[['userId', 'movieId', 'rating']]
ratings = [1,2,3,4,5,'n']
genres = ['Adventure',
         'Animation',
         'Children',
         'Comedy',
         'Fantasy',
         'Romance',
         'Drama',
         'Action',
         'Crime',
         'Thriller',
         'Horror',
         'Mystery',
         'Sci-Fi',
         'War',
         'Musical',
         'Documentary',
         'IMAX',
         'Western',
         'Film-Noir',
         '(no genres listed)']
# def movie_rater(movie_df=movies,num=3, genre=None):
#     userID = 1000
#     rating_list = []
#     genre = input('What movie genre have you watched before?')

#     while num > 0:
#         if genre:
#             global movie
#             movie = movie_df[movie_df['genres'].str.contains(genre)].sample(1)
#             st.text(str(movie['title'].values[0]))
#             rating = st.selectbox('How do you rate this movie on a scale of 1-5, select n if you have not seen :\n', ratings)
#             if rating == 'n':
#                 continue
#             else:
#                 rating_one_movie = {'userId':userID,'movieId':movie['movieId'].values[0], 'rating':rating}
#                 rating_list.append(rating_one_movie) 
#                 num -= 1
#                 return rating_list
#         else:
#             movie = movie_df.sample(1)
#         print(movie)
#         rating = input('How do you rate this movie on a scale of 1-5, press n if you have not seen :\n')
#         if rating == 'n':
#             continue
#         else:
#             rating_one_movie = {'userId':userID,'movieId':movie['movieId'].values[0], 'rating':rating}
#             rating_list.append(rating_one_movie) 
#             num -= 1
#     return rating_list

def movie_rater(movie_df=movies,num=3, genre=None):
    userID = 1000
    rating_list = []
    # genre = input('What movie genre have you watched before?')

    while num > 0:
        if genre:
            global movie
            movie = movie_df[movie_df['genres'].str.contains(genre)].sample(1)
            st.text(movie['title'].values[0])
            movie_name=movie['title'].values[0]
            rating = st.selectbox(f'How do you rate this {movie_name} on a scale of 1-5, select n if you have not seen :\n', ratings)
            if rating == 'n':
                continue
            else:
                rating_one_movie = {'userId':userID,'movieId':movie['movieId'].values[0], 'rating':rating}
                rating_list.append(rating_one_movie) 
                num -= 1
                return rating_list
        else:
            st.text('None')
            # movie = movie_df.sample(1)
            # print(movie)
            pass
        # st.text(movie)
        # selected_rating = st.selectbox('Select Rating', ratings)
        # rating = st.selectbox('How do you rate this movie on a scale of 1-5, press n if you have not seen :\n', ratings)
       
    return rating_list

def rank_movies(df, user_rating):
        ## add the new ratings to the original ratings DataFrame
    user_ratings = pd.DataFrame(user_rating)
    string_column = user_ratings.select_dtypes(include=[object]).columns
    for col in string_column:
        user_ratings[col] = pd.to_numeric(user_ratings[col], errors = 'coerce')
    user_ratings = user_ratings.dropna()
    new_ratings_df = pd.concat([df, user_ratings], axis=0)
    reader = Reader()
    new_data = Dataset.load_from_df(new_ratings_df, reader)

    # train a model using the new combined DataFrame
    svd_ = SVD(n_factors= 50, reg_all=0.05)
    svd_.fit(new_data.build_full_trainset())

    # make predictions for the user
    list_of_movies = []
    for m_id in movie_rating_df['movieId'].unique():
        list_of_movies.append( (m_id,svd_.predict(1000,m_id)[3]))
    # Order the predictions from highest to lowest rated
    ranked_movies = sorted(list_of_movies, key=lambda x:x[1], reverse=True)
    return ranked_movies
    
def recommended_movies(user_ratings, movie_title_df, n=5):
    recommended_movies_set = set()  # Keep track of recommended movies

    for idx, rec in enumerate(user_ratings):
        movie_id = int(rec[0])
        title_array = movie_title_df.loc[movie_title_df['movieId'] == movie_id, 'title'].values

        # Check if the array is not empty and the movie has not been recommended before
        if title_array.any() and title_array[0] not in recommended_movies_set:
            title = title_array[0]
            print('Recommendation #', idx+1, ':', title, '\n')
            recommended_movies_set.add(title)  # Add the movie to the set of recommended movies
            n -= 1

        if n == 0:
            break
content_matrix = pickle.load(open('content_matrix.sav','rb'))
# from sklearn.metrics.pairwise import cosine_similarity
# def get_movie_recommendations(movie, n=5):
#     idx = movie_mapper[movie]
#     print('Movie Index: ',idx)
#     movie_index = content_matrix[idx]
#     print('Index: ', type(movie_index))
#     scores = cosine_similarity(movie_index, content_matrix)
#     scores = scores.flatten()
#     print('Scores: ',scores)

#     recommended_idx = (-scores).argsort()[1:n+1]
#     print("Rec: ",recommended_idx)
#     return movies['title'].iloc[recommended_idx]
###
st.title('Filamu Movie Recommender')


selected_genre = st.selectbox("Select Movie Genre Of Interest", genres)

if st.button('Search'):
    rating_list = movie_rater(genre=selected_genre)
    ranked_movies = rank_movies(user_item_df,rating_list)
    rec_movies = recommended_movies(rating_list,movies,n=5)
# selected_movie_name = st.selectbox("Which movie are you interested in?",movies['title'].values)
# if st.button('Search'):
#     names = get_movie_recommendations(selected_movie_name)
#     for name in names:
#         st.text(name)
