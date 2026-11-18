"""AI-generated insights using the Gemini API.

Design choices:
- Structured output: we hand Gemini a Pydantic schema and ask for JSON, so the
  dashboard gets typed fields (percentages, lists) instead of free text to parse.
- Comments are trimmed before sending, which keeps requests small, fast, and
  well inside the free-tier limits.
"""

from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_COMMENT_CHARS = 300   # long comments get truncated
MAX_COMMENTS = 100


class AIError(Exception):
    """Raised with a user-friendly message when the Gemini call fails."""


# ---------- output schemas ----------
class SentimentSplit(BaseModel):
    positive: int = Field(description="Percent of comments that are positive, 0-100")
    neutral: int = Field(description="Percent of comments that are neutral, 0-100")
    negative: int = Field(description="Percent of comments that are negative, 0-100")


class Theme(BaseModel):
    name: str = Field(description="Short theme label, 2-5 words")
    summary: str = Field(description="One sentence on what viewers are saying")
    example_comment: str = Field(description="A real comment from the input that shows this theme, copied verbatim")
    sentiment: Literal["positive", "neutral", "negative", "mixed"]


class CommentAnalysis(BaseModel):
    overall_summary: str = Field(description="One or two sentences summarising audience reaction")
    sentiment: SentimentSplit
    themes: list[Theme] = Field(description="The 3-5 most common themes, most common first")
    viewer_requests: list[str] = Field(
        description="Specific things viewers ask the creator to make or do next. Empty list if none."
    )


class SpikeExplanation(BaseModel):
    headline: str = Field(description="One short sentence: the main reason this video broke out")
    explanation: str = Field(description="2-3 sentences explaining why it likely outperformed")
    key_factors: list[str] = Field(description="2-4 short factors, e.g. 'Timely topic', 'Curiosity-gap title'")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How confident you are given only this data"
    )


# ---------- helpers ----------
def _clean_comments(comments: list[str]) -> list[str]:
    cleaned = []
    for c in comments[:MAX_COMMENTS]:
        c = " ".join(c.split())  # collapse newlines / extra spaces
        if c:
            cleaned.append(c[:MAX_COMMENT_CHARS])
    return cleaned


def _format_comments(comments: list[str]) -> str:
    return "\n".join(f"- {c}" for c in _clean_comments(comments))


def build_comment_prompt(video_title: str, comments: list[str]) -> str:
    return (
        f'Video title: "{video_title}"\n\n'
        f"Below are the top comments on this YouTube video, ordered by relevance.\n"
        f"Analyse the audience reaction. Base everything only on these comments; "
        f"the example_comment for each theme must be copied from the list.\n\n"
        f"Comments:\n{_format_comments(comments)}"
    )


def build_spike_prompt(video: dict, baseline: dict, comments: list[str]) -> str:
    minutes = video["duration_sec"] / 60
    return (
        "A YouTube video significantly outperformed its channel's usual results. "
        "Explain the most likely reasons it broke out, using only the data below. "
        "Compare it to the channel's typical video. Be specific and avoid generic advice.\n\n"
        f"Channel: {baseline['channel_title']}\n"
        f"Typical (median) views per video: {baseline['median_views']:,}\n"
        f"Typical video length: {baseline['median_minutes']:.1f} min\n"
        f"Channel's average engagement rate: {baseline['avg_engagement']:.2%}\n"
        f"Titles of some typical videos for comparison:\n"
        + "\n".join(f"- {t}" for t in baseline["typical_titles"])
        + "\n\nBreakout video:\n"
        f'Title: "{video["title"]}"\n'
        f"Views: {video['views']:,} ({video['spike_score']:.1f}x the median)\n"
        f"Engagement rate: {video['engagement_rate']:.2%}\n"
        f"Length: {minutes:.1f} min ({'Short' if video['is_short'] else 'long-form'})\n"
        f"Published: {video['published_at']:%A %d %B %Y}, {video['published_at']:%H:%M} UTC\n\n"
        f"Top comments on the breakout video:\n{_format_comments(comments[:40]) or '(comments unavailable)'}"
    )


# ---------- client ----------
class GeminiAnalyzer:
    def __init__(self, api_key: str, model: str | None = None):
        if not api_key:
            raise AIError("Missing GEMINI_API_KEY. Add it to .env (local) or the app's Secrets settings (Streamlit Cloud) to enable AI insights.")
        self._client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def _generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a sharp YouTube analytics assistant. You write concise, "
                        "specific, evidence-based insights for content strategists."
                    ),
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.3,  # low = more consistent, factual output
                ),
            )
        except errors.ClientError as e:
            if e.code == 429:
                raise AIError("Gemini rate limit reached. Wait a minute and try again.") from e
            if e.code in (400, 401, 403) and "API key" in str(e):
                raise AIError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.") from e
            if e.code == 404:
                raise AIError(
                    f"Model '{self.model}' isn't available. Set GEMINI_MODEL in .env to a current model."
                ) from e
            raise AIError(f"Gemini request failed: {e.message or e}") from e
        except errors.ServerError as e:
            raise AIError("Gemini is temporarily unavailable. Try again shortly.") from e

        if not resp.text:
            raise AIError("Gemini returned an empty response (it may have been blocked by safety filters).")
        return schema.model_validate_json(resp.text)

    def analyze_comments(self, video_title: str, comments: list[str]) -> CommentAnalysis:
        if not comments:
            raise AIError("This video has no comments to analyse (they may be turned off).")
        return self._generate(build_comment_prompt(video_title, comments), CommentAnalysis)

    def explain_spike(self, video: dict, baseline: dict, comments: list[str]) -> SpikeExplanation:
        return self._generate(build_spike_prompt(video, baseline, comments), SpikeExplanation)
