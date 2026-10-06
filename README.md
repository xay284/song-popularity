# What Makes a Song Popular?

*Which audio features are associated with Spotify popularity, and how much can they actually predict?*

This project analyzes the relationship between Spotify track characteristics and popularity using 114,000 dataset rows, representing 89,740 distinct tracks after accounting for repeated track entries. The analysis compares audio features with genre information, controls for data leakage through grouped cross-validation, and explores how much of popularity remains unexplained.

**Stack:** Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, SHAP, Streamlit, Hugging Face Datasets, Git/GitHub

---

## Key findings

- **Genre predicts popularity much better than audio features alone.** Among tracks with non-zero popularity (80,393 tracks), LightGBM using genre alone reaches R² = 0.44 when cross-validation is grouped by lead artist, compared with R² = 0.17 for audio features alone.
- **Adding audio features to genre provides only a small additional gain.** The audio + genre model reaches R² = 0.46, an improvement of approximately +0.022 R² over genre alone. The gain is positive in all five folds.
- **Much of the apparent audio-popularity relationship is associated with genre composition.** Pooled rank correlations become substantially weaker within individual genres. For example, instrumentalness has a pooled Spearman correlation of −0.170 with popularity, compared with a median within-genre correlation of −0.046.
- **The explicit-content difference also becomes much smaller within genres.** The overall popularity gap of about 4 points between explicit and non-explicit tracks falls to roughly 1 point when comparing within genres.
- **Genre contributes more to model predictions than individual audio and track attributes.** SHAP analysis shows an average net contribution of about 8.1 popularity points from genre columns together, compared with 2.8 points from audio and other track attributes.
- **The conclusion is consistent across different treatments of low-popularity tracks.** The same overall pattern appears when using all tracks, tracks with popularity > 0, and tracks with popularity > 5.
- **A large part of popularity remains unexplained.** Even the audio + genre model leaves roughly half of the variance unexplained. Factors such as artist fame, marketing, playlists, social media exposure, and release timing are not included in the dataset.

---

## Data

The project uses the Spotify Tracks Dataset ([`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)) from Hugging Face.

The original dataset contains approximately 114,000 rows and information collected through the Spotify Web API. Each row includes a track, artist, genre, popularity score, and audio characteristics.

### Main features

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`
- `explicit`
- `key`
- `mode`
- `time_signature`
- `genre`

The target variable is Spotify popularity, ranging from 0 to 100. According to the dataset description, popularity is based mainly on the number of recent plays, so it should be interpreted as a snapshot of listening activity rather than a measure of a track's historical or artistic importance.

### Track duplication

The original dataset contains repeated entries for the same track because a track can be associated with multiple genres.

- Original rows: **114,000**
- Distinct `track_id`s: **89,740**

Because the audio features are identical across copies of the same track, rows were collapsed by `track_id`. Popularity was averaged where necessary, while the genre information was retained so that a track could still belong to multiple genres.

This step is important because treating every genre occurrence as an independent track would otherwise give repeated tracks disproportionate influence.

### Popularity = 0

Popularity contains a noticeable concentration of zero values.

For the original data:

- About **14.1%** of rows have popularity = 0.
- About **10.4%** of distinct tracks have popularity = 0.

These zero-popularity tracks are highly concentrated in some genres. For example, the original dataset contains high proportions of zero-popularity rows in genres such as jazz, Iranian, and romance.

These tracks have audio characteristics broadly similar to other tracks, while the low values from 1 to 5 are also over-represented in the same genres. This suggests a possible sampling artifact, although the analysis cannot establish the exact reason for the concentration.

For the main analysis, tracks with popularity = 0 are therefore excluded. Results are also checked using two sensitivity analyses:

- all tracks, including popularity = 0
- tracks with popularity > 5

---

## Method

### Exploratory analysis

The analysis first examines:

- Popularity distribution
- Audio-feature distributions
- Correlations between audio features and popularity
- Popularity across genres
- Explicit vs non-explicit tracks
- Duration
- Key and mode
- Time signature
- The concentration of low-popularity tracks across genres

Rank-based correlations are used because relationships between audio features and popularity are not necessarily linear.

### Features

The models use:

- 11 numeric audio/track features
- `explicit`
- `mode`
- `key`
- `time_signature`
- 114 genre indicators

A track can belong to several genres, so genre is represented using multiple binary indicators rather than forcing each track into a single genre.

### Models

Three types of models are compared:

1. **Mean baseline** — predicts the mean popularity for every track.
2. **Ridge regression** — provides a simple, regularized linear baseline.
3. **LightGBM** — captures nonlinear relationships and interactions between features.

The LightGBM settings are fixed rather than extensively tuned. The goal is to investigate the predictive signal in the data rather than maximize leaderboard-style performance.

### Leakage control

The same song can appear under multiple genres, so a random row-level train/test split could place the same track in both training and test data.

To avoid this, evaluation uses 5-fold `GroupKFold` cross-validation.

Two grouping strategies are used:

- **By song** — grouped using artist + title
- **By lead artist** — all tracks from the same lead artist remain in the same fold

The artist-level split is the stricter test because it asks whether the model can generalize to artists it has never seen during training.

### Metrics

The models are evaluated using:

- **R²** — proportion of variance in popularity explained by the model
- **MAE** — average absolute prediction error in popularity points
- **Spearman correlation** — how well predicted popularity rankings correspond to actual rankings

---

## Genre vs audio

A central question is:

> Does the sound of a track explain its popularity, or does genre carry most of the predictive information?

Three feature sets are compared:

**Audio only**

```
Audio features → Popularity
```

**Genre only**

```
Genre → Popularity
```

**Audio + genre**

```
Audio features + Genre → Popularity
```

This comparison helps determine how much additional information audio features provide once genre is already known.

---

## Results

### Popularity > 0 — 80,393 tracks

| Features | Model | R² by song | R² by artist | MAE by artist | Spearman by artist |
| --- | --- | ---: | ---: | ---: | ---: |
| None | Mean baseline | 0.000 | -0.004 | 15.16 | - |
| Audio | Ridge | 0.075 | 0.069 | 14.37 | 0.263 |
| Audio | LightGBM | 0.184 | 0.165 | 13.24 | 0.407 |
| Genre | LightGBM | 0.465 | 0.435 | 9.77 | 0.678 |
| Audio + genre | LightGBM | 0.490 | 0.457 | 9.56 | 0.698 |

The main result is clear:

> **Genre provides substantially more predictive information about popularity than audio features alone.**

Audio features still contribute some additional information, but the improvement after adding them to genre is relatively small.

### Sensitivity to low-popularity tracks

R² by artist, LightGBM:

| Subset | Tracks | Audio | Genre | Audio + genre |
| --- | ---: | ---: | ---: | ---: |
| All tracks | 89,740 | 0.109 | 0.343 | 0.358 |
| Popularity > 0 | 80,393 | 0.165 | 0.435 | 0.457 |
| Popularity > 5 | 77,690 | 0.206 | 0.564 | 0.586 |

These R² values are not directly comparable across subsets, because removing low-popularity tracks changes the distribution and variance of the target.

However, the qualitative conclusion is consistent: genre remains substantially more predictive than audio features alone.

---

## Model interpretation

SHAP is used to understand how the trained LightGBM model reaches its predictions.

The SHAP analysis shows that the genre variables collectively contribute substantially more to predictions than the audio and other track attributes.

For individual features, SHAP values indicate whether a feature pushes a prediction toward higher or lower popularity.

For example, higher instrumentalness generally pushes the model's predicted popularity downward in the fitted model.

These are model associations, not causal effects.

### Within-genre analysis

A feature can appear strongly associated with popularity in the full dataset simply because it is characteristic of certain genres.

For example:

```
        Genre
       ↙     ↘
   Audio     Popularity
```

To investigate this, correlations are also examined within individual genres.

The pooled relationships generally become weaker within genres. This suggests that part of the overall audio-popularity association may be explained by differences in genre composition rather than by a universal relationship between a particular audio feature and popularity.

This is why the analysis does not interpret a correlation such as:

> "More instrumental songs are less popular"

as a general causal rule.

---

## Limitations

### Association, not causation

The analysis identifies statistical associations.

It does not show that changing an audio feature would cause a song's popularity to change.

For example:

> Higher danceability is associated with higher popularity

does not mean:

> Making a song more danceable will make it more popular.

### Genre is a dataset variable

Genre is based on the labels provided by the dataset and does not necessarily represent objective or mutually exclusive musical categories.

The dataset contains approximately 1,000 tracks per genre in its original construction, and some genres contain unusually high concentrations of zero and near-zero popularity values.

Therefore, the genre finding should be interpreted as a finding within this dataset, not as a general ranking of real-world musical genres by popularity.

### Popularity is a snapshot

Spotify popularity is influenced by recent listening activity.

The dataset does not provide all the information that could explain popularity over time, such as:

- Artist fame
- Marketing and promotion
- Playlist placement
- Social media exposure
- Release date/timing
- Historical popularity

### Model limitations

LightGBM hyperparameters were fixed rather than extensively tuned.

The evaluation uses 5-fold cross-validation, where training folds overlap across different validation folds. Therefore, the small improvement from adding audio to genre should be interpreted as an estimated predictive gain, not as a formal statistical significance test.

### Spotify audio features

The audio features themselves are estimates produced by Spotify's systems rather than direct physical measurements. Some metadata values, such as unusual time-signature values, may also contain errors or uncertainty.

---

## Interactive explorer

The project includes a Streamlit application for exploring the dataset and model results.

The explorer allows users to interact with the data and investigate how track characteristics relate to predicted popularity.

To run it:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The compact data required by the explorer is stored in `app_data/`.

---

## Reproducing the analysis

The notebook in `notebooks/` downloads the dataset from Hugging Face, performs the preprocessing and analysis, and reproduces the reported results.

Additional result tables are stored in `results/`.

---

## Credits

Dataset: [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face.

The dataset was collected through the Spotify Web API. The dataset card reports a BSD license; the original dataset author is credited accordingly.

This project is an independent analysis of the dataset and does not claim to represent Spotify's own methodology for predicting or defining popularity.