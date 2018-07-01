import random
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import json
import os

from spotify_utilities import get_playlist_file


def getData(DATA_PATH,MAX_FILES):
    full_model_fitting = True
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

                if full_model_fitting == False:
                    track_ids.append(track_uri_to_id[track_uri])

                if full_model_fitting == True:
                    track_id = track_uri_to_id[track_uri]
                    columnIndex.append(i)
                    rowIndex.append(track_id)
                    cellData.append(1)

            if full_model_fitting == False:
                full_playlists.append(track_ids)

            i += 1
        print(i)

    num_songs = len(track_id_to_uri)

    top_list = sorted([(i, count) for i, count in enumerate(top_list)], key=lambda x: x[1], reverse=True)

    if full_model_fitting == True:
        item_user_data = csr_matrix((cellData, (rowIndex, columnIndex)), dtype=np.float32)

        del rowIndex
        del columnIndex
        del cellData
        top_list = top_list[:1000]

    return track_uri_to_id, track_id_to_uri, top_list, num_songs, item_user_data


def build_model(item_user_data, latFactors, trainingIterations):
    print("start model fitting")
    model = implicit.als.AlternatingLeastSquares(factors=latFactors, iterations=trainingIterations, regularization=0.0)
    model.fit(item_user_data)
    print("finished model fitting")

    return model


def predictWithFM(model, seed, num_songs):
    rowIndex2 = []
    columnIndex2 = []
    data2 = []

    for track in seed:
        columnIndex2.append(0)
        rowIndex2.append(track)
        data2.append(1.0)

    user_item_data = csr_matrix((data2, (rowIndex2, columnIndex2)), shape=(num_songs, 1), dtype=np.float32)
    recommendations = model.recommend(0, user_item_data.T, N=500, recalculate_user=True)

    return [track_id for track_id, _ in recommendations if track_id not in seed]


import datetime
import json


def createMFRecs(CHALLENGE_PATH,DATA_PATH,MAX_FILES,FILE_PATH):
    with open(CHALLENGE_PATH, "r") as f:
        challenge = json.load(f)

        track_uri_to_id, track_id_to_uri, top_list, num_songs, item_user_data = getData(DATA_PATH,MAX_FILES)

        # file_name = "submission_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
        file_name = FILE_PATH

        model = build_model(item_user_data, 400, 15)

        with open(file_name, "w") as out:
            out.write("team_info,creative,Freshwater Sea,member@email.com\n")

            for playlist in challenge["playlists"]:
                # For less than 10 seed tracks, computing recommendations with FMs doesn't show good performance
                if playlist["num_samples"] >= 10:
                    print(playlist["pid"])

                    seed_ids = [
                        track_uri_to_id[track["track_uri"]]
                        for track in playlist["tracks"]
                        if track["track_uri"] in track_uri_to_id
                    ]

                    predictions = [track_id_to_uri[track_id] for track_id in predictWithFM(model, seed_ids, num_songs)]
                    out.write("%i,%s\n" % (playlist["pid"], ",".join(predictions)))