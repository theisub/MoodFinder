from bs4 import BeautifulSoup
import re
import requests
import os
from utils import create_table, insert_dataframe

import pandas as pd

class Album():
    def __init__(self):
        self.album_name = ""
        self.artist_name =""
        self.album_mood=[]
        self.album_cover = ""

    def set_album_name(self,album_name):
        self.album_name = album_name

    def set_artist_name(self,artist_name):
        self.artist_name = artist_name

    def set_album_mood(self,album_mood):
        self.album_mood = album_mood

    def set_album_cover(self,album_cover):
        self.album_cover = album_cover



def get_album_info(soupik):
    album_info = Album()

    album_name = soupik.find(id="media_link_button_container_top").get("data-albums")
    album_info.set_album_name(album_name)

    artist_name = soupik.find(id="media_link_button_container_top").get("data-artists")
    album_info.set_artist_name(artist_name)

    album_mood = soupik.find(class_="release_pri_descriptors").get_text()
    album_mood = [x.strip() for x in album_mood.split(',')]
    album_info.set_album_mood(album_mood)

    album_cover = soupik.find(class_=re.compile("^coverart_")).find("img").attrs["src"][2:]
    album_info.set_album_cover(album_cover)
    return album_info


folder_name = "albums"
album_files = os.listdir(folder_name)

album_info_list = []

for item in album_files:
    with open(f"{folder_name}/{item}") as file:
        soupik = BeautifulSoup(file,'html.parser')
        album_info = get_album_info(soupik)
        album_info_list.append(album_info)



album_info_df = pd.DataFrame([x.__dict__ for x in album_info_list])
print(album_info_df.head())
# create_table()
    

# insert_dataframe(album_info_df)
# def add_info():