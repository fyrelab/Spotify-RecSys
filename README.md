# Installation
This system requires Python 3. If Python 3 is not default `pip3` instead of `pip` and `python3` instead of `python` needs to be used for commands.

For each track run `pip install -r requirements.txt` from the track folder.

For the creative track afterwards run `python -m spacy download en` to download the English language model of spacy.

Now run for each track `python setup.py build_ext --inplace` to build the Cython target.

# Configuration
In each track's `create_submission.py` you need to specify both the mpd path and the challenge set path.
To run the system on a smaller subsample of the mpd you can alter the value of `MAX_FILES` within `create_submission.py`.

# Run
Run `python create_submission.py` in each folder respectively. `submission_recsys.csv` is the file that will be created with the final result that has been submitted. 