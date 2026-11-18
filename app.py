"""Streamlit dashboard. Run with:  streamlit run app.py"""

import os

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.analysis import add_metrics, flag_spikes, performance_by, summary_stats
from src.youtube_client import ChannelNotFoundError, YouTubeClient

load_dotenv()
st.set_page_config(page_title="YouTube Channel Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def get_client() -> YouTubeClient:
    return YouTubeClient(os.getenv("YOUTUBE_API_KEY", ""))


# Cache API results for an hour so re-running the same channel costs no quota.
@st.cache_data(ttl=3600, show_spinner=False)
def load_channel(channel_input: str, max_videos: int):
    client = get_client()
    channel = client.get_channel(channel_input)
    videos = client.get_recent_videos(channel, max_videos=max_videos)
    return channel, videos


# ---------- sidebar ----------
st.sidebar.title("📊 Channel Analytics")
channel_input = st.sidebar.text_input("Channel URL or @handle", placeholder="@veritasium")
max_videos = st.sidebar.slider("Recent videos to analyse", 10, 200, 50, step=10)
spike_threshold = st.sidebar.slider("Spike threshold (× median views)", 1.5, 10.0, 3.0, 0.5)

if not channel_input:
    st.title("YouTube Channel Analytics")
    st.write("Enter a channel in the sidebar to see what's working on it.")
    st.stop()

try:
    with st.spinner("Fetching channel data…"):
        channel, videos = load_channel(channel_input, max_videos)
except (ChannelNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

if videos.empty:
    st.warning("This channel has no public videos.")
    st.stop()

df = flag_spikes(add_metrics(videos), threshold=spike_threshold)
stats = summary_stats(df)

# ---------- header ----------
col_img, col_txt = st.columns([1, 6])
col_img.image(channel.thumbnail_url, width=100)
col_txt.title(channel.title)
col_txt.caption(channel.description[:200] + ("…" if len(channel.description) > 200 else ""))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Subscribers", f"{channel.subscribers:,}" if channel.subscribers is not None else "Hidden")
k2.metric("Median views", f"{stats['median_views']:,}")
k3.metric("Avg engagement", f"{stats['avg_engagement_rate']:.2%}")
k4.metric("Spike videos", stats["spike_count"])
k5.metric("Shorts share", f"{stats['shorts_share']:.0%}")

# ---------- charts ----------
st.subheader("Views over time")
fig = px.scatter(
    df,
    x="published_at",
    y="views",
    color="is_spike",
    size="engagement_rate",
    hover_data={"title": True, "spike_score": ":.1f", "engagement_rate": ":.2%"},
    color_discrete_map={True: "#ef4444", False: "#6366f1"},
    labels={"published_at": "Published", "views": "Views", "is_spike": "Spike"},
)
fig.add_hline(y=stats["median_views"], line_dash="dot", annotation_text="median")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("By upload day")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = performance_by(df, "weekday").set_index("weekday").reindex(order).dropna().reset_index()
    st.plotly_chart(
        px.bar(by_day, x="weekday", y="median_views", hover_data=["videos"]),
        use_container_width=True,
    )
with c2:
    st.subheader("Shorts vs long-form")
    by_type = performance_by(df, "is_short").replace({"is_short": {True: "Shorts", False: "Long-form"}})
    st.plotly_chart(
        px.bar(by_type, x="is_short", y="median_views", hover_data=["videos", "avg_engagement"]),
        use_container_width=True,
    )

# ---------- spikes ----------
st.subheader("🔥 Breakout videos")
spikes = df[df["is_spike"]].sort_values("spike_score", ascending=False)
if spikes.empty:
    st.info("No videos above the spike threshold. Try lowering it in the sidebar.")
for _, v in spikes.iterrows():
    a, b = st.columns([1, 4])
    a.image(v["thumbnail_url"])
    b.markdown(f"**[{v['title']}]({v['url']})**")
    b.write(
        f"{v['views']:,} views · **{v['spike_score']:.1f}× the median** · "
        f"{v['engagement_rate']:.2%} engagement · {v['published_at']:%d %b %Y}"
    )

with st.expander("All videos (raw data)"):
    st.dataframe(
        df[["published_at", "title", "views", "likes", "comments", "engagement_rate", "spike_score"]]
        .sort_values("published_at", ascending=False),
        use_container_width=True,
    )
