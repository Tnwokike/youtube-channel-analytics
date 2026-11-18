"""Thin wrapper around the YouTube Data API v3.

Quota notes (default daily quota is 10,000 units):
- channels.list, playlistItems.list, videos.list, commentThreads.list each cost 1 unit.
- search.list costs 100 units, so we avoid it entirely by resolving channels
  via handle/ID and reading the channel's "uploads" playlist instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import isodate
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class ChannelNotFoundError(Exception):
    pass


class YouTubeAPIError(Exception):
    """A YouTube API failure, with a message that's safe and useful to show users."""


def _error_reasons(e: HttpError) -> set[str]:
    reasons = set()
    for d in getattr(e, "error_details", None) or []:
        if isinstance(d, dict):
            reasons.update(str(d.get(k, "")) for k in ("reason", "status"))
    return reasons


def _friendly_error(e: HttpError) -> YouTubeAPIError:
    reasons = _error_reasons(e)
    text = f"{e.reason or ''} {' '.join(reasons)}"
    if "API keys are not supported" in text:
        msg = ("YOUTUBE_API_KEY looks like a service-account-bound key (e.g. your Gemini key). "
               "Use the key restricted to YouTube Data API v3 instead.")
    elif "keyInvalid" in reasons or "API_KEY_INVALID" in text or "API key not valid" in text:
        msg = "YouTube rejected the API key. Check YOUTUBE_API_KEY is the current key, copied exactly."
    elif reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}:
        msg = "The YouTube API daily quota is used up. It resets at midnight Pacific time."
    elif "accessNotConfigured" in reasons or "SERVICE_DISABLED" in text:
        msg = "YouTube Data API v3 isn't enabled for this key's Google Cloud project."
    elif "BLOCKED" in text or "ipRefererBlocked" in reasons:
        msg = ("This API key's restrictions are blocking the request. In Google Cloud, set the key's "
               "Application restrictions to None and its API restriction to YouTube Data API v3.")
    else:
        msg = f"YouTube API error ({e.status_code}): {e.reason or 'unknown error'}"
    return YouTubeAPIError(msg)


def _execute(request):
    try:
        return request.execute()
    except HttpError as e:
        raise _friendly_error(e) from e


@dataclass
class ChannelInfo:
    channel_id: str
    title: str
    description: str
    subscribers: int | None  # None when the channel hides its count
    total_views: int
    video_count: int
    thumbnail_url: str
    uploads_playlist_id: str


_CHANNEL_ID_RE = re.compile(r"(UC[\w-]{22})")
_HANDLE_RE = re.compile(r"@([\w.\-]+)")


def parse_channel_input(text: str) -> tuple[str, str]:
    """Turn user input into ("id", value) or ("handle", value).

    Accepts:
      https://www.youtube.com/@mkbhd   ->  ("handle", "mkbhd")
      @mkbhd / mkbhd                   ->  ("handle", "mkbhd")
      https://youtube.com/channel/UC.. ->  ("id", "UC...")
    """
    text = text.strip()
    if not text:
        raise ValueError("Please enter a channel URL or handle.")
    if m := _CHANNEL_ID_RE.search(text):
        return "id", m.group(1)
    if m := _HANDLE_RE.search(text):
        return "handle", m.group(1)
    if "/" not in text and " " not in text:
        return "handle", text
    raise ValueError(
        "Couldn't read that. Try a link like youtube.com/@name or a handle like @name."
    )


class YouTubeClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing YOUTUBE_API_KEY. Add it to .env (local) or the app's Secrets settings (Streamlit Cloud).")
        self._yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    # ---------- channel ----------
    def get_channel(self, channel_input: str) -> ChannelInfo:
        kind, value = parse_channel_input(channel_input)
        params = {"part": "snippet,statistics,contentDetails"}
        params["id" if kind == "id" else "forHandle"] = value

        resp = _execute(self._yt.channels().list(**params))
        items = resp.get("items", [])
        if not items:
            raise ChannelNotFoundError(f"No channel found for '{channel_input}'.")

        c = items[0]
        stats = c["statistics"]
        return ChannelInfo(
            channel_id=c["id"],
            title=c["snippet"]["title"],
            description=c["snippet"].get("description", ""),
            subscribers=None
            if stats.get("hiddenSubscriberCount")
            else int(stats.get("subscriberCount", 0)),
            total_views=int(stats.get("viewCount", 0)),
            video_count=int(stats.get("videoCount", 0)),
            thumbnail_url=c["snippet"]["thumbnails"]["high"]["url"],
            uploads_playlist_id=c["contentDetails"]["relatedPlaylists"]["uploads"],
        )

    # ---------- videos ----------
    def get_recent_videos(self, channel: ChannelInfo, max_videos: int = 50) -> pd.DataFrame:
        """Return the channel's most recent uploads with stats, one row per video."""
        video_ids: list[str] = []
        page_token = None
        while len(video_ids) < max_videos:
            resp = _execute(
                self._yt.playlistItems().list(
                    part="contentDetails",
                    playlistId=channel.uploads_playlist_id,
                    maxResults=min(50, max_videos - len(video_ids)),
                    pageToken=page_token,
                )
            )
            video_ids += [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        rows = []
        for start in range(0, len(video_ids), 50):  # videos.list takes up to 50 IDs
            batch = video_ids[start : start + 50]
            resp = _execute(
                self._yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(batch))
            )
            for v in resp.get("items", []):
                s = v.get("statistics", {})
                rows.append(
                    {
                        "video_id": v["id"],
                        "title": v["snippet"]["title"],
                        "published_at": pd.to_datetime(v["snippet"]["publishedAt"]),
                        "duration_sec": int(
                            isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()
                        ),
                        "views": int(s.get("viewCount", 0)),
                        # likes/comments can be hidden or disabled -> keep as missing, not 0
                        "likes": int(s["likeCount"]) if "likeCount" in s else None,
                        "comments": int(s["commentCount"]) if "commentCount" in s else None,
                        "thumbnail_url": v["snippet"]["thumbnails"]["medium"]["url"],
                        "url": f"https://www.youtube.com/watch?v={v['id']}",
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("published_at").reset_index(drop=True)
        return df

    # ---------- comments ----------
    def get_top_comments(self, video_id: str, max_comments: int = 100) -> list[str]:
        """Top-level comments ordered by relevance. Returns [] if comments are disabled."""
        comments: list[str] = []
        page_token = None
        try:
            while len(comments) < max_comments:
                resp = (
                    self._yt.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        order="relevance",
                        textFormat="plainText",
                        maxResults=min(100, max_comments - len(comments)),
                        pageToken=page_token,
                    )
                    .execute()
                )
                comments += [
                    i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    for i in resp.get("items", [])
                ]
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as e:
            if "commentsDisabled" in _error_reasons(e):
                return []
            raise _friendly_error(e) from e
        return comments
