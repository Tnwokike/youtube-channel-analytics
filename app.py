"""Streamlit dashboard. Run with:  streamlit run app.py"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.ai_insights import AIError, CommentAnalysis, GeminiAnalyzer, SpikeExplanation
from src.analysis import add_metrics, channel_baseline, flag_spikes, performance_by, summary_stats
from src.youtube_client import ChannelNotFoundError, YouTubeClient

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a key from Streamlit secrets (when deployed) or .env / environment (locally)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml locally -> fall back to env vars
        pass
    return os.getenv(name, default)
st.set_page_config(page_title="YouTube Channel Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def get_client() -> YouTubeClient:
    return YouTubeClient(get_secret("YOUTUBE_API_KEY", ""))


# Cache API results for an hour so re-running the same channel costs no quota.
@st.cache_data(ttl=3600, show_spinner=False)
def load_channel(channel_input: str, max_videos: int):
    client = get_client()
    channel = client.get_channel(channel_input)
    videos = client.get_recent_videos(channel, max_videos=max_videos)
    return channel, videos


@st.cache_data(ttl=3600, show_spinner=False)
def load_comments(video_id: str) -> list[str]:
    return get_client().get_top_comments(video_id, max_comments=100)


@st.cache_resource
def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer(get_secret("GEMINI_API_KEY", ""), get_secret("GEMINI_MODEL"))


# AI results are cached for a day: the same video never costs a second Gemini call.
@st.cache_data(ttl=86400, show_spinner=False)
def ai_comment_analysis(video_id: str, title: str) -> CommentAnalysis:
    return get_analyzer().analyze_comments(title, load_comments(video_id))


@st.cache_data(ttl=86400, show_spinner=False)
def ai_spike_explanation(video_id: str, video: dict, baseline: dict) -> SpikeExplanation:
    return get_analyzer().explain_spike(video, baseline, load_comments(video_id))


def ai_available() -> tuple[bool, str]:
    try:
        get_analyzer()
        return True, ""
    except AIError as e:
        return False, str(e)


def render_comment_analysis(a: CommentAnalysis) -> None:
    st.markdown(f"**Summary:** {a.overall_summary}")
    s = a.sentiment
    total = max(s.positive + s.neutral + s.negative, 1)
    sentiment_df = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Percent": [s.positive * 100 / total, s.neutral * 100 / total, s.negative * 100 / total],
        "row": "Comments",
    })
    fig = px.bar(
        sentiment_df, x="Percent", y="row", color="Sentiment", orientation="h",
        color_discrete_map={"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"},
        height=140,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None, xaxis_title="% of comments")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**What viewers are talking about**")
        icon = {"positive": "🟢", "neutral": "⚪", "negative": "🔴", "mixed": "🟡"}
        for t in a.themes:
            st.markdown(f"{icon.get(t.sentiment, '⚪')} **{t.name}**: {t.summary}")
            st.caption(f'“{t.example_comment}”')
    with right:
        st.markdown("**Viewer requests**")
        if a.viewer_requests:
            for r in a.viewer_requests:
                st.markdown(f"- {r}")
        else:
            st.caption("No clear requests in the top comments.")


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
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("By upload day")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = performance_by(df, "weekday").set_index("weekday").reindex(order).dropna().reset_index()
    st.plotly_chart(
        px.bar(by_day, x="weekday", y="median_views", hover_data=["videos"]),
        width="stretch",
    )
with c2:
    st.subheader("Shorts vs long-form")
    by_type = performance_by(df, "is_short").replace({"is_short": {True: "Shorts", False: "Long-form"}})
    st.plotly_chart(
        px.bar(by_type, x="is_short", y="median_views", hover_data=["videos", "avg_engagement"]),
        width="stretch",
    )

# ---------- spikes ----------
st.subheader("🔥 Breakout videos")
spikes = df[df["is_spike"]].sort_values("spike_score", ascending=False)
if spikes.empty:
    st.info("No videos above the spike threshold. Try lowering it in the sidebar.")
ai_ok, ai_msg = ai_available()
baseline = channel_baseline(df, channel.title)

for _, v in spikes.iterrows():
    a, b = st.columns([1, 4])
    a.image(v["thumbnail_url"])
    b.markdown(f"**[{v['title']}]({v['url']})**")
    b.write(
        f"{v['views']:,} views · **{v['spike_score']:.1f}× the median** · "
        f"{v['engagement_rate']:.2%} engagement · {v['published_at']:%d %b %Y}"
    )
    key = f"show_spike_{v['video_id']}"
    if b.button("🧠 Why did this spike?", key=f"btn_spike_{v['video_id']}", disabled=not ai_ok, help=ai_msg or None):
        st.session_state[key] = True
    if st.session_state.get(key):
        video = v[["title", "views", "spike_score", "engagement_rate", "duration_sec",
                   "is_short", "published_at"]].to_dict()
        try:
            with b, st.spinner("Asking Gemini…"):
                exp = ai_spike_explanation(v["video_id"], video, baseline)
            b.success(f"**{exp.headline}**")
            b.write(exp.explanation)
            b.markdown(" · ".join(f"`{f}`" for f in exp.key_factors) + f"  \n*Confidence: {exp.confidence}*")
        except AIError as e:
            b.error(str(e))

# ---------- AI comment analysis ----------
st.subheader("💬 Comment analysis")
if not ai_ok:
    st.info(ai_msg)
else:
    by_views = df.sort_values("views", ascending=False)
    options = by_views["video_id"].tolist()
    titles = dict(zip(by_views["video_id"], by_views["title"]))
    chosen = st.selectbox(
        "Pick a video", options, format_func=lambda vid: titles[vid], key="comment_video"
    )
    key = f"show_comments_{chosen}"
    if st.button("Analyse comments", type="primary"):
        st.session_state[key] = True
    if st.session_state.get(key):
        try:
            with st.spinner("Reading comments and asking Gemini…"):
                analysis = ai_comment_analysis(chosen, titles[chosen])
            render_comment_analysis(analysis)
        except AIError as e:
            st.error(str(e))

with st.expander("All videos (raw data)"):
    st.dataframe(
        df[["published_at", "title", "views", "likes", "comments", "engagement_rate", "spike_score"]]
        .sort_values("published_at", ascending=False),
        width="stretch",
    )
