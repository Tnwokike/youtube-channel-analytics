"""Pure-pandas analysis functions. No API calls here, so they are easy to unit test."""

from __future__ import annotations

import pandas as pd


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add engagement and timing columns to the video table."""
    df = df.copy()
    interactions = df["likes"].fillna(0) + df["comments"].fillna(0)
    # Engagement rate = (likes + comments) / views. Guard against 0-view videos.
    df["engagement_rate"] = (interactions / df["views"].where(df["views"] > 0)).fillna(0)
    df["is_short"] = df["duration_sec"] <= 60
    df["weekday"] = df["published_at"].dt.day_name()
    df["hour_utc"] = df["published_at"].dt.hour
    df["age_days"] = (pd.Timestamp.now(tz="UTC") - df["published_at"]).dt.days.clip(lower=1)
    df["views_per_day"] = df["views"] / df["age_days"]
    return df


def flag_spikes(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Mark videos that massively outperformed the channel's normal.

    We compare each video to the channel's *median* views, not the mean, because
    one viral video would drag the mean up and hide the very spikes we want to find.
    `spike_score` = views / median views. 3.0 means "3x a typical video".
    """
    df = df.copy()
    median = df["views"].median()
    df["spike_score"] = df["views"] / median if median else 0
    df["is_spike"] = df["spike_score"] >= threshold
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "videos_analysed": len(df),
        "median_views": int(df["views"].median()),
        "avg_engagement_rate": float(df["engagement_rate"].mean()),
        "spike_count": int(df["is_spike"].sum()),
        "shorts_share": float(df["is_short"].mean()),
    }


def channel_baseline(df: pd.DataFrame, channel_title: str, n_titles: int = 8) -> dict:
    """What a 'typical' video looks like on this channel, used as context for the AI."""
    typical = df[~df["is_spike"]]
    if typical.empty:
        typical = df
    # Titles of the videos closest to the median view count = most representative.
    median = df["views"].median()
    closest = typical.assign(_dist=(typical["views"] - median).abs()).nsmallest(n_titles, "_dist")
    return {
        "channel_title": channel_title,
        "median_views": int(median),
        "median_minutes": float(df["duration_sec"].median() / 60),
        "avg_engagement": float(df["engagement_rate"].mean()),
        "typical_titles": closest["title"].tolist(),
    }


def performance_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Median views and engagement grouped by a column (e.g. weekday, is_short)."""
    return (
        df.groupby(column)
        .agg(
            videos=("video_id", "count"),
            median_views=("views", "median"),
            avg_engagement=("engagement_rate", "mean"),
        )
        .reset_index()
    )
