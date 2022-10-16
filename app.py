import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

df_feat = pd.read_csv('train_valid_feat.csv')

tracks = df_feat['Track']
tracks = tracks.drop_duplicates()

columns = df_feat.columns
cat_index = []
index = 0
for col in columns:
    if df_feat[col].dtype == 'object':
        cat_index.append(index)
    index += 1


class SpotifyRecommender():
    def __init__(self, rec_data):
        # our class should understand which data to work with
        self.rec_data_ = rec_data.copy()

    # if we need to change data
    def change_data(self, rec_data):
        self.rec_data_ = rec_data

    # function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations_knn(self, song_name, amount=1):
        # choosing the data for our song
        try:
            song = self.rec_data_[(self.rec_data_.Track.str.lower() == song_name.lower())].head(1).values[
                0]  # vector of first time seen song
        except:
            print("This Song Doesn't Exsist")
            return
        # dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.Track.str.lower() != song_name.lower()]
        distances = []
        for r_song in tqdm(res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                # indeces of non-numerical columns
                if not col in cat_index:
                    # calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
        #         sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        res_data = res_data.drop_duplicates(subset=['Track'])
        columns = ['Track', 'URL']  # name -> uri
        # mapping
        return res_data[columns][:amount]

def predict(song_name):
    recomender_feat = SpotifyRecommender(df_feat)
    predicts = recomender_feat.get_recommendations_knn(song_name, 10)
    # print(predicts)
    indecies = predicts.index
    song_name_index = np.where(df_feat['Track'] == song_name)[0][0]
    indecies = indecies.insert(0, song_name_index)
    # print(df_feat.iloc[indecies, :])
    return predicts, df_feat.iloc[indecies, :]
st.set_page_config(
    page_title = 'Song Recommendation',
    page_icon = 'O',
)
def main():
    tracks_ = list(tracks)
    st.title("Spotify Songs Recommendation System")
    st.sidebar.title("Choose Track")
    st.markdown(" Welcome To Our Recommendation System")
    st.warning('Only Enter Track from the list provided')
    select = st.sidebar.selectbox('Choose Track', tracks_, key='1')
    submit = st.button('Recommend me 10 similar songs')
    if submit:
        if select:
            with st.spinner('Predicting...'):
                # time.sleep(2)
                song_name = select
                prediction, _ = predict(song_name)
                prediction.reset_index(drop=True, inplace=True)
                st.table(prediction)
        else:
            st.error('Please Enter All The Details')
if __name__ == '__main__':
    main()