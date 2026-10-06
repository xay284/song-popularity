from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="What makes a song popular?", layout="wide")

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]


@st.cache_data
def load():
    return pd.read_parquet("app_data/tracks_genre_rows.parquet")


df = load()
genres = sorted(df["track_genre"].unique())
default_idx = genres.index("pop") if "pop" in genres else 0

st.title("What makes a song popular?")
st.caption(
    "Spotify tracks dataset (114,000 rows, 114 genres). Popularity is a 0-100 score based "
    "mainly on recent plays. Explore how it differs by genre, and how little the audio "
    "features add once genre is known."
)

tab1, tab2, tab3 = st.tabs(["Genres", "Audio features", "Results"])

with tab1:
    genre = st.selectbox("Genre", genres, index=default_idx)
    g = df[df["track_genre"] == genre]
    nzg = g[g["popularity"] > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks", f"{len(g):,}")
    c2.metric("Popularity = 0", f"{(g['popularity'] == 0).mean():.0%}")
    c3.metric("Median (non-zero)", f"{nzg['popularity'].median():.0f}" if len(nzg) else "n/a")
    c4.metric("Popularity <= 5", f"{(g['popularity'] <= 5).mean():.0%}")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.arange(0, 102, 2)
    ax.hist(df["popularity"], bins=bins, density=True, alpha=0.4, label="all genres")
    ax.hist(g["popularity"], bins=bins, density=True, alpha=0.6, label=genre)
    ax.set_xlabel("popularity")
    ax.set_ylabel("density")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.write("Non-zero tracks only. Tracks are split into 10 groups (deciles) by the feature.")
    feat = st.selectbox("Audio feature", FEATURES, index=FEATURES.index("instrumentalness"))
    genre2 = st.selectbox("Compare with a genre", genres, index=default_idx, key="genre2")
    nz = df[df["popularity"] > 0]
    ng = nz[nz["track_genre"] == genre2]

    def decile_means(d):
        q = pd.qcut(d[feat].rank(method="first"), 10, labels=False)
        return d.groupby(q)["popularity"].mean().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, 11), decile_means(nz), marker="o", label="all genres pooled")
    if len(ng) >= 100:
        ax.plot(range(1, 11), decile_means(ng), marker="o", label=genre2)
    ax.set_xlabel(f"decile of {feat} (low to high)")
    ax.set_ylabel("mean popularity")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    rho_all = spearmanr(nz[feat], nz["popularity"])[0]
    msg = f"Spearman correlation with popularity, all genres pooled: {rho_all:+.3f}"
    if len(ng) >= 100:
        msg += f" | within {genre2}: {spearmanr(ng[feat], ng['popularity'])[0]:+.3f}"
    else:
        msg += f" | {genre2} has fewer than 100 non-zero tracks, so no within-genre estimate is shown"
    st.write(msg)

with tab3:
    st.markdown(
        "**Main finding:** among tracks with non-zero popularity, genre predicts popularity far "
        "better than audio features do (artist-grouped cross-validation, R2 about 0.44 vs 0.17), "
        "and audio adds only about 0.02 on top of genre. Genre is a dataset bucket, not a natural "
        "property, so this does not say which real-world genres are more popular."
    )
    path = Path("results/sensitivity_thresholds.csv")
    if path.exists():
        st.subheader("Same comparison with three ways of handling low-popularity tracks")
        st.dataframe(pd.read_csv(path), hide_index=True)
    path = Path("results/within_genre_correlations.csv")
    if path.exists():
        st.subheader("Audio correlations with popularity, pooled vs within genre")
        st.dataframe(pd.read_csv(path).round(3), hide_index=True)