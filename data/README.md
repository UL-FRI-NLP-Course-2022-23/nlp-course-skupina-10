# Construct Dataset

## Usage

1. Download two sets of subtitles (https://www.podnapisi.net/), make sure that both subtitles have the same frame rate !
2. Merge the subtitles (https://subtitletools.com/merge-subtitles-online); use `Nearest cue` merging, chose a random color to style `merge` and `base` subtitles (we do this for easier parsing).
3. Run script `python3 construct_dataset.py`.

## TODO

1. Fix the character encoding issues.
2. Can we get a developer api for the merger (It seem like it is not free) ?
