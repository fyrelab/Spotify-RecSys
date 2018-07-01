import os
import json
import itertools
import numpy as np
import math
import random

def r_precision(g, r):
    g_set = set(g)
    n = len(g_set)
    r_set = set(r[:n])
    return len(g_set & r_set) / n

def ndcg(g, r):
    g_set = set(g)
    intersec = set(r) & set(g)

    if len(intersec) < 1:
        return 0

    dcg = 0
    
    for i, rel in enumerate(r):
        dcg += (1 if rel in g_set else 0) / np.log2(i + 2)
    
    idcg = 0
    
    for i, _ in enumerate(g_set):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg

def song_clicks(g, r, max_n_predictions=500):
    g_set = set(g)
    
    for i, track in enumerate(r):
        if track in g_set:
            return float(int(i / 10))

    return float(max_n_predictions / 10.0 + 1)

def score(g, r):
    assert(len(set(r)) == 500)
    
    return np.array([
        r_precision(g, r),
        ndcg(g, r),
        song_clicks(g, r)
    ])

def get_playlist_file(data_path, max_files, exclude=[], include=None):
    i = 0

    if include is None:
        files = os.listdir(data_path)
    else:
        files = include

    for file_name in sorted(files):
        file_path = os.path.join(data_path, file_name)
    
        if file_name in exclude:
            continue
        
        if file_name.endswith(".json") and os.path.isfile(file_path):
            i += 1
            if i > max_files:
                break
            
            with open(file_path, "r") as f:
                yield json.load(f)


def pick_training_test(data_path, max_files, num_test=1):
    files = [file_name for file_name in os.listdir(data_path) if file_name.endswith(".json")]

    if len(files) - num_test <= 0:
        raise ValueError("Requested %i test files, but there is only a total of %i files" % (num_test, len(files)))
    
    random.shuffle(files)

    return (files[num_test:], files[:num_test])

def get_golden_test_set(data_path, golden_test_file="golden_test_set.json", max_files=1000):
    with open(golden_test_file, "r") as f:
        golden_test_test = json.load(f)
    
    reserved_pids = set()

    for playlist in golden_test_test:
        reserved_pids.add(playlist["pid"])
    
    return golden_test_test, reserved_pids



def get_challenge_test_set(data_path, files):
    playlists = []
    
    for data in get_playlist_file(data_path, len(files), include=files):
        for playlist in data["playlists"]:
            # get track uris deduplicated but keep the order
            track_uris_set = set()
            track_uris = []
            
            for track in playlist["tracks"]:
                track_uri = track["track_uri"]
                
                if track_uri not in track_uris_set:
                    track_uris_set.add(track_uri)
                    track_uris.append(track_uri)
                
            

            playlists.append((playlist["name"], track_uris)) 
    
    
    random.shuffle(playlists)
    
    groups = [
        (0, True, None),
        (1, True, "first"),
        (5, True, "first"),
        (5, False, "first"),
        (10, True, "first"),
        (10, False, "first"),
        (25, True, "first"),
        (25, True, "random"),
        (100, True, "first"),
        (100, True, "random"),
    ]
    
    groups = sorted(groups, key=lambda x: -x[0])
    group_counts = [0 for _ in groups]
    
    n = len(groups)
    k = math.floor(len(playlists) / n)
    
    test_set = []
    
    for name, track_uris in playlists:
        i = 0
        t = len(track_uris)
        
        while i < n - 1 and (t <= 1.1 * groups[i][0] or group_counts[i] == k):
            i += 1
        
        group_counts[i] += 1
        
        if groups[i][2] == "random":
            random.shuffle(track_uris)
        
        seeds = track_uris[:groups[i][0]]
        truth = track_uris[groups[i][0]:]
        
        test_set.append((
            name if groups[i][1] else "",
            seeds,
            truth
        ))
    
    return test_set


def get_test_set(data_path, files, seeds=[5]):
    seeds = sorted(seeds)
    seed_counts = [0 for _ in range(0,len(seeds))]

    test_set = []

    for data in get_playlist_file(data_path, len(files), include=files):
        for playlist in data["playlists"]:
            track_uris = list(set([track["track_uri"] for track in playlist["tracks"]]))
            random.shuffle(track_uris)

            # Pick a seed count that can be done with list
            # and at the same time has an even distribution between all seed counts
            MIN_TRUTH_TRACKS = 1
            max_seeds = len(track_uris) - MIN_TRUTH_TRACKS
            greatest_seed = find_limit(seeds, max_seeds)

            if greatest_seed > 0:
                seed_pos = np.argmin(seed_counts[:greatest_seed])
                seed_counts[seed_pos] += 1
                num_seeds = seeds[seed_pos]

                test_set.append((
                    playlist["name"],
                    [uri for uri in track_uris[:num_seeds]],
                    [uri for uri in track_uris[num_seeds:]]
                )) 
    
    return test_set


def find_limit(sorted_list, upper_bound):
    for i, item in enumerate(sorted_list):
        if item > upper_bound:
            return i
    
    return len(sorted_list)


def fill_top_list(top_list, existing, exclude):
    existing_set = set(existing)

    N = 500
    
    i = 500 - len(existing)
    j = 0
    fill_tracks = []
    
    while i > 0:
        top_track = top_list[j][0]
        if top_track not in existing_set and top_track not in exclude:
            fill_tracks.append(top_track)
            i -= 1
        
        j += 1
    
    return existing + fill_tracks