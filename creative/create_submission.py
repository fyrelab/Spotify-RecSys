import datetime, time, json
import os
import pandas as pd
import numpy as np
import json
import operator
import keras
from keras.models import load_model

from fast_functions import PredictionSystem
from matrix_factorization_recs import createMFRecs
from createAudioFeatureNetwork import createCSV, createNetwork

start = time.time()

DATA_PATH = "../../mpd.v1/data"
CHALLENGE_SET_PATH = "../../challenge.v1/challenge_set.json"
MAX_FILES = 1000
SUBMISSION_TYPE = "creative"
TEAM_NAME = "MY_TEAM_NAME"
MAIL_ADDR = "my@mail-addr.tld"
GB_FNAME = "submission_graph_based.csv"
FM_FNAME = "submission_fm_based.csv"
AF_PATH = "audio_features.csv"
MODEL_PATH = "feature_nn.h5"
OUTPUT_FILE_PATH = "submission_recsys.csv"

predicter = PredictionSystem(DATA_PATH, MAX_FILES)

print("Load time: %s" % (time.time() - start))

#Create recomendations based on graph based approach
start = time.time()

with open(CHALLENGE_SET_PATH, "r") as f:
    challenge = json.load(f)

    with open(GB_FNAME, "w") as out:
        out.write("team_info,{2},{0},{1}\n".format(TEAM_NAME, MAIL_ADDR, SUBMISSION_TYPE))
        
        i = 0

        for playlist in challenge["playlists"]:
            print(i)
            i += 1

            predictions = predicter.predict([track["track_uri"] for track in playlist["tracks"]], playlist.get("name", ""))
            out.write("%i,%s\n" % (playlist["pid"], ",".join(predictions)))

print("Graph based execution time: %s" % (time.time() - start))

start = time.time()

#Create recomendations based on factorization machines
createMFRecs(CHALLENGE_SET_PATH,DATA_PATH,MAX_FILES,FM_FNAME)

print("Matrix Factorization based execution time: %s" % (time.time() - start))

start = time.time()

#Crawl Audio Features

createCSV(DATA_PATH, MAX_FILES, AF_PATH)

#Create Network

createNetwork(DATA_PATH, MAX_FILES, AF_PATH, MODEL_PATH)

audio_features = pd.read_csv(AF_PATH, index_col=0)
audio_features.replace('None',np.nan,inplace=True)
audio_features_a = audio_features.applymap(pd.to_numeric)
audio_features_np = audio_features_a.values

af_index = {}

i = 0
for ind in audio_features_a.index:
    af_index[ind] = i
    i += 1

model = load_model(MODEL_PATH)

def getFeatureScores(seeds, gb_recs, fm_recs):
    scores = {}
    for t in (gb_recs + fm_recs):
        s_score = 0.0
        for seed in seeds:
            seed_t = t2 = str.split(seed, ":")[2]
            seed_ft = audio_features_np[af_index[seed_t], :]
            gb_recs_t = str.split(t, ":")[2]
            rec_ft = audio_features_np[af_index[gb_recs_t], :]
            ft_vec = np.concatenate((seed_ft, rec_ft), axis=0).reshape(1, 26)
            s_score += np.asscalar(model.predict(ft_vec, batch_size=None)[0])
        s_score = s_score / len(seeds)
        scores[t] = s_score

    return scores

#Merge submissions and boost tracks based on audio features
fm_data = {}
gb_data = {}

with open(FM_FNAME, "r") as f:
    for line in f:
        idx = line.find(",")
        pid = line[:idx]
        fm_data[pid] = line.rstrip().split(",")[1:]

with open(GB_FNAME, "r") as g:
    for line in g:
        idx = line.find(",")
        pid = line[:idx]
        gb_data[pid] = line.rstrip().split(",")[1:]

with open(CHALLENGE_SET_PATH, "r") as f:
    challenge = json.load(f)

    with open(OUTPUT_FILE_PATH, "w") as out:
        out.write("team_info,{2},{0},{1}\n".format(TEAM_NAME, MAIL_ADDR, SUBMISSION_TYPE))

        ctn = 0
        for playlist in challenge["playlists"]:
            pid = str(playlist["pid"])
            seed_len = playlist["num_samples"]
            seeds = [p['track_uri'] for p in playlist["tracks"]]
            num_holdouts = playlist["num_holdouts"]
            vote_counts = {}
            gb_tracks = gb_data[pid]
            if pid in fm_data:  # now we have to merge submissions
                fm_tracks = fm_data[pid]
                predictions = []

                i = 0
                j = 0
                gb_index = 0
                fm_index = 0
                while i < 500:
                    if j % 2 == 0:
                        if gb_tracks[gb_index] not in predictions:
                            predictions.append(gb_tracks[gb_index])
                            i += 1
                        gb_index += 1
                    elif fm_index < len(fm_tracks):
                        if fm_tracks[fm_index] not in predictions:
                            predictions.append(fm_tracks[fm_index])
                            i += 1
                        fm_index += 1
                    j += 1

            elif seed_len > 0:
                predictions = []
                if seed_len > 0:
                    ft_scores = getFeatureScores(seeds, gb_tracks, fm_tracks)
                    ft_list = [(t, (500 - i) * ft_scores[t]) for i, t in enumerate(gb_tracks)]
                    ft_list.sort(key=operator.itemgetter(1), reverse=True)

                    i = 0
                    j = 0
                    gb_index = 0
                    ft_index = 0
                    while i < 500:
                        if j % 2 == 0:
                            if gb_tracks[gb_index] not in predictions:
                                predictions.append(gb_tracks[gb_index])
                                i += 1
                            gb_index += 1
                        else:
                            if ft_list[ft_index][0] not in predictions:
                                predictions.append(ft_list[ft_index][0])
                                i += 1
                            ft_index += 1
                        j += 1
            else:
                predictions = gb_tracks
            print(ctn)
            ctn += 1
            out.write("%i,%s\n" % (playlist["pid"], ",".join(predictions)))

print("File merging time: %s" % (time.time() - start))