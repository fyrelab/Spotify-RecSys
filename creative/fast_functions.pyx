import numpy as np
import itertools
import collections

import spacy

from spotify_utilities import get_playlist_file


nlp = spacy.load('en', disable=["ner", "textcat", "parser", "tagger"])

def tokenize(name):
    return list(set([token.lemma_ for token in nlp(name.lower()) if not token.is_stop]))



class PredictionSystem:
    def __init__(self, DATA_PATH, MAX_FILES, includex=None, pid_blacklist=[]):
        self.song_to_playlists = []
        self.playlists_to_songs = []
        self.token_rankings = {}

        self.track_uri_to_id = {}
        self.track_id_to_uri = []

        stop_words = set(["playlist", "music", "for", "me", "to", "this", "song", "spotify", "i", "and"])

        print("Reading playlists...")

        for data in get_playlist_file(DATA_PATH, MAX_FILES, [], includex):
            for playlist in data["playlists"]:
                if playlist["pid"] not in pid_blacklist:
                    # Graph
                    track_uris = list(set([track["track_uri"] for track in playlist["tracks"]]))
                    track_ids = []

                    playlist_id = len(self.playlists_to_songs)

                    for track_uri in track_uris:
                        if track_uri not in self.track_uri_to_id:
                            track_id = len(self.track_id_to_uri)
                            self.track_uri_to_id[track_uri] = track_id
                            self.song_to_playlists.append([])
                            self.track_id_to_uri.append(track_uri)
                        else:
                            track_id = self.track_uri_to_id[track_uri]

                        track_ids.append(track_id)
                        self.song_to_playlists[track_id].append(playlist_id)

                    self.playlists_to_songs.append(track_ids)

                    # NLP
                    tokens = list(set(tokenize(playlist["name"])))
                    
                    for token in tokens:
                        if token not in stop_words:
                            if token not in self.token_rankings:
                                self.token_rankings[token] = {}

                            token_rank = self.token_rankings[token]

                            for track_id in track_ids:
                                token_rank[track_id] = token_rank.get(track_id, 0) + 1
        
        num_playlists = len(self.playlists_to_songs)

        print("Generating top list...")
        self.top_list = sorted(
            [
                (i, len(playlist_ids) / num_playlists)
                for i, playlist_ids in enumerate(itertools.islice(self.song_to_playlists, 1000))
            ],
            key=lambda x: x[1],
            reverse=True
        )

        print("Creating token rankings...")
        for token in self.token_rankings:
            token_ranking = self.token_rankings[token]
            self.token_rankings[token] = sorted(
                [(track, token_ranking[track]) for track in token_ranking if token_ranking[track] > 1],
                key=lambda x: x[1],
                reverse=True
            )
        
        print("Initialisation done")

    def predict(self, seeds, name):
        seeds = set([self.track_uri_to_id[seed] for seed in seeds if seed in self.track_uri_to_id])

        tracks = []

        if len(seeds) > 0:
            tracks = self._related_tracks_additive(seeds, seeds)
        
        if len(tracks) < 500:
            tracks += self.__related_by_name_log(name, seeds | set(tracks))
        
        if len(tracks) < 500:
            tracks = self._fill_top_list(tracks, seeds)
        
        return [self.track_id_to_uri[track_id] for track_id in tracks[:500]]

    def _take_most_likely(self, sorted_lists, exclude=[]):
        next_p = []
        next_track = []
        iters = [iter(sorted_list) for sorted_list in sorted_lists]

        picked = set()

        has_elements = 0

        for i, it in enumerate(iters):
            try:
                track, p = next(it)
                next_p.append(p)
                next_track.append(track)
                has_elements += 1
            except StopIteration:
                next_p.append(0)

        while has_elements > 0:
            max_idx = np.argmax(next_p)
            p = next_p[max_idx]
            track = next_track[max_idx]

            if p <= 0:
                return

            if track not in picked:
                yield track
                picked.add(track)

            try:
                while True:
                    trackn, pn = next(iters[max_idx])
                    
                    if trackn not in exclude:
                        break

                next_p[max_idx] = pn
                next_track[max_idx] = trackn

                if pn <= 0:
                    has_elements -= 1
            except StopIteration:
                next_p[max_idx] = 0
                next_track[max_idx] = None
                has_elements -= 1


    def _related_tracks_additive(self, seeds, exclude):
        tracks = {}
        
        n = len(seeds)

        for seed in seeds:
            playlists = self.song_to_playlists[seed]
            deg = len(playlists)
            
            neighbor_tracks = collections.Counter()
            
            for playlist in playlists:
                for track in self.playlists_to_songs[playlist]:
                    if track not in exclude:
                        neighbor_tracks[track] += 1

            for track in neighbor_tracks:
                p = neighbor_tracks[track] / deg
                if track in tracks or p > 0.01:
                    tracks[track] = tracks.get(track, 0) + p

        top_tracks = sorted(
            tracks.items(),
            key=lambda x: x[1],
            reverse=True
        )[:500]

        return list(itertools.islice(self._take_most_likely([
            top_tracks,
            self.top_list
        ], exclude=exclude), len(top_tracks)))

    def __related_by_name_log(self, name, exclude):
        tokens = tokenize(name)
        tracks = {}
        
        for token in tokens:
            if token in self.token_rankings:
                for track, count in self.token_rankings[token]:
                    if track not in exclude:
                        tracks[track] = tracks.get(track, 0) + np.log2(count)
        
        tracks = sorted(
            [track for track in tracks if tracks[track] > 1],
            key=lambda x: tracks[x],
            reverse=True
        )[:500]
        
        return tracks
    
    def _fill_top_list(self, existing, exclude):
        existing_set = set(existing)

        N = 500
        
        i = 500 - len(existing)
        j = 0
        fill_tracks = []
        
        while i > 0:
            top_track = self.top_list[j][0]
            if top_track not in existing_set and top_track not in exclude:
                fill_tracks.append(top_track)
                i -= 1
            
            j += 1
        
        return existing + fill_tracks
