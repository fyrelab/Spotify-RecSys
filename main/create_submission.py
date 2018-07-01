import datetime, time, json
import os
import pandas as pd
import numpy as np
import json
import operator

from fast_functions import PredictionSystem
from matrix_factorization_recs import createMFRecs

start = time.time()

DATA_PATH = "../../mpd.v1/data"
CHALLENGE_SET_PATH = "../../challenge.v1/challenge_set.json"
MAX_FILES = 1000
SUBMISSION_TYPE = "main"
TEAM_NAME = "MY_TEAM_NAME"
MAIL_ADDR = "my@mail-addr.tld"
GB_FNAME = "submission_graph_based.csv"
FM_FNAME = "submission_fm_based.csv"
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
            else:
                predictions = gb_tracks
            print(ctn)
            ctn += 1
            out.write("%i,%s\n" % (playlist["pid"], ",".join(predictions)))

print("File merging time: %s" % (time.time() - start))