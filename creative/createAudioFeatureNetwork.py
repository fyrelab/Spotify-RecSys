import random
import numpy as np
from numpy import nan
import json
import os
import csv
import keras
import requests
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import pandas as pd

from spotify_utilities import get_golden_test_set, get_playlist_file, score, fill_top_list, song_clicks

def createCSV(DATA_PATH, MAX_FILES, AF_PATH):

    if os.path.isfile(AF_PATH):
        return

    top_list = []
    track_uri_to_id = {}
    track_id_to_uri = []

    CLIENT_ID = ""
    CLIENT_SECRET = ""

    TOKEN = ""

    import base64
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": TOKEN,
            "redirect_uri": "http://asciico.de"
        },
        headers={
            "Authorization": "Basic %s" % base64.b64encode(
                ("%s:%s" % (CLIENT_ID, CLIENT_SECRET)).encode("ascii")).decode("ascii")
        }
    )
    o = r.json()
    print(o)
    access_token = o["access_token"]

    for data in get_playlist_file(DATA_PATH, MAX_FILES):

        for playlist in data["playlists"]:
            # if playlist["pid"] in test_pids:
            #    continue

            track_uris = list(set([track["track_uri"] for track in playlist["tracks"]]))
            track_ids = []

            for track_uri in track_uris:
                if track_uri not in track_uri_to_id:
                    track_id = len(track_id_to_uri)
                    track_uri_to_id[track_uri] = track_id
                    track_id_to_uri.append(track_uri)
                    top_list.append(0)
                else:
                    track_id = track_uri_to_id[track_uri]

                top_list[track_id] += 1

    top_list = sorted([(i, count) for i, count in enumerate(top_list)], key=lambda x: x[1], reverse=True)

    batch_size = 100

    with open(AF_PATH, "a") as f:
        while i < len(top_list):
            j = 0
            songs = ""
            id_list = []
            the_track = track_id_to_uri[top_list[i][0]]
            song_id = str.split(the_track, ":")[2]
            songs += song_id + ","
            i += 1
            id_list.append(song_id)

            songs = songs.rstrip(",")

            # get response
            r = requests.get(
                "https://api.spotify.com/v1/audio-features/?ids=%s" % songs,
                headers={
                    "Authorization": "Bearer %s" % access_token
                }
            )
            response = r.json()

            exclude = ['type', 'id', 'uri', 'track_href', 'analysis_url']

            line = ""
            line += "feature_names" + ","

            ctn = 0
            for feature_list in response['audio_features']:
                line = ""
                line += id_list[ctn] + ","
                if feature_list is not None:
                    for key, value in feature_list.items():
                        if key not in exclude:
                            line += str(value) + ","
                line = line.rstrip(",")
                line += "\n"
                f.write(line)
                ctn += 1
            print(i)

    f.close()



def createNetwork(DATA_PATH, MAX_FILES, AF_PATH, MODEL_PATH):
    track_uri_to_id = {}
    track_id_to_uri = []

    top_list = []

    rowIndex = []
    columnIndex = []
    cellData = []
    full_playlists = []

    i = 0

    for data in get_playlist_file(DATA_PATH, MAX_FILES):

        for playlist in data["playlists"]:

            track_uris = list(set([track["track_uri"] for track in playlist["tracks"]]))
            track_ids = []

            for track_uri in track_uris:
                if track_uri not in track_uri_to_id:
                    track_id = len(track_id_to_uri)
                    track_uri_to_id[track_uri] = track_id
                    track_id_to_uri.append(track_uri)
                    top_list.append(0)
                else:
                    track_id = track_uri_to_id[track_uri]

                top_list[track_id] += 1

                track_ids.append(track_uri_to_id[track_uri])
            full_playlists.append(track_ids)

            i += 1
        print(i)

    num_songs = len(track_id_to_uri)

    top_list = sorted([(i, count) for i, count in enumerate(top_list)], key=lambda x: x[1], reverse=True)

    audio_features = pd.read_csv(AF_PATH, index_col=0)
    audio_features.replace('None', np.nan, inplace=True)
    audio_features_a = audio_features.applymap(pd.to_numeric)
    audio_features_np = audio_features_a.values

    af_index = {}

    i = 0
    for ind in audio_features_a.index:
        af_index[ind] = i
        i += 1

    training_set_size = 10000000
    X_train = np.empty((training_set_size, 26))
    Y_train = np.empty(((training_set_size), 1), dtype=int)

    i = 0

    playlist_batch_size = 6

    while i < training_set_size:
        playlist = random.choice(full_playlists)
        for j in range(0, playlist_batch_size):
            track = random.choice(playlist)
            t1 = str.split(track_id_to_uri[track], ":")[2]
            label = 0
            t2 = ""
            if (j % 2 == 0):
                while True:
                    track2 = random.choice(playlist)
                    t2 = str.split(track_id_to_uri[track2], ":")[2]
                    if t2 != t1:
                        break
                label = 1
            if (j % 2 == 1):
                while True:
                    track2 = random.choice(top_list)[0]
                    t2 = str.split(track_id_to_uri[track2], ":")[2]
                    if t2 not in [str.split(track_id_to_uri[track], ":")[2] for track in playlist]:
                        break
                label = 0
            t1_features = audio_features_np[af_index[t1], :]
            t2_features = audio_features_np[af_index[t2], :]
            if t1_features.size == 0 or t2_features.size == 0:
                continue
            if any(np.isnan(audio_features_np[af_index[t1], :])) or any(np.isnan(audio_features_np[af_index[t2], :])):
                continue
            X_train[i,] = np.concatenate((t1_features, t2_features), axis=0)
            Y_train[i,] = label
            i += 1
            if i == training_set_size:
                break

    model = Sequential()
    model.add(Dense(32, input_dim=26, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10, batch_size=4096)

    model.save(MODEL_PATH)